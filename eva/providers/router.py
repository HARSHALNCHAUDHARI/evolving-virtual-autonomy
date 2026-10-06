"""Model router — selects and routes requests to AI model providers.

Provides deterministic, capability-aware model selection based on task type,
required capabilities, provider/model preferences, latency/cost preferences,
and availability with fallback handling and explainable routing decisions.
"""

from __future__ import annotations

from typing import Any, TypeVar

from eva.logging import get_logger
from eva.providers.base import ModelProvider
from eva.providers.mock import MockProvider
from eva.providers.models import (
    CandidateEvaluation,
    ModelCapability,
    ModelRequest,
    ModelResponse,
    ModelSpec,
    NoEligibleModelError,
    ProviderNotFoundError,
    RouteRequest,
    RoutingDecision,
    TaskType,
)
from eva.providers.registry import ModelRegistry

logger = get_logger(__name__)

T = TypeVar("T")


class ModelRouter:
    """Capability-aware model router for intelligent provider dispatch.

    Decouples the cognitive core from individual providers.
    Performs deterministic candidate filtering, multi-factor scoring,
    fallback selection, and produces explainable routing decisions.
    """

    def __init__(
        self,
        providers: dict[str, ModelProvider] | None = None,
        *,
        default_provider: str = "mock",
        default_model: str = "mock-v1",
        registry: ModelRegistry | None = None,
    ) -> None:
        self._registry = registry or ModelRegistry()
        self._providers: dict[str, ModelProvider] = {}
        self._default_provider = default_provider
        self._default_model = default_model
        self._last_decision: RoutingDecision | None = None
        self._task_routes: dict[TaskType, str] = {
            TaskType.CONVERSATION: default_provider,
            TaskType.UNDERSTANDING: default_provider,
            TaskType.REASONING: default_provider,
            TaskType.STRUCTURED_OUTPUT: default_provider,
            TaskType.GENERAL: default_provider,
        }

        if providers:
            for p in providers.values():
                self.register_provider(p)
        else:
            # Register a default mock provider if none supplied
            self.register_provider(MockProvider(name=default_provider))

        logger.debug(
            "ModelRouter initialized (default_provider=%s, default_model=%s)",
            self._default_provider,
            self._default_model,
        )

    @property
    def registry(self) -> ModelRegistry:
        """Return the central model registry."""
        return self._registry

    @property
    def default_provider_name(self) -> str:
        """Return the default provider name."""
        return self._default_provider

    @property
    def default_model_name(self) -> str:
        """Return the default model name."""
        return self._default_model

    @property
    def last_decision(self) -> RoutingDecision | None:
        """Return the most recent routing decision."""
        return self._last_decision

    def register_provider(
        self,
        provider: ModelProvider,
        spec: ModelSpec | None = None,
    ) -> None:
        """Register a model provider instance and optional model specification."""
        self._providers[provider.name.lower()] = provider
        if spec is not None:
            self._registry.register_model(spec)
        elif hasattr(provider, "get_spec") and not self._registry.has_provider(provider.name):
            try:
                self._registry.register_model(provider.get_spec())
            except Exception as exc:
                logger.warning(
                    "Failed to register model specification for provider '%s': %s",
                    provider.name,
                    exc,
                )
        logger.debug("Registered provider: %s", provider.name)

    def unregister_provider(self, name: str) -> None:
        """Unregister a provider by name."""
        self._providers.pop(name.lower(), None)

    def get_provider(self, name: str) -> ModelProvider:
        """Retrieve a registered provider by name.

        Raises:
            ProviderNotFoundError: If the provider is not registered.
        """
        normalized = name.lower()
        provider = self._providers.get(normalized)
        if provider is None:
            raise ProviderNotFoundError(f"Provider '{name}' is not registered or supported.")
        return provider

    def has_provider(self, name: str) -> bool:
        """Check if a provider is registered."""
        return name.lower() in self._providers

    def list_providers(self) -> list[str]:
        """List names of all registered providers."""
        return list(self._providers.keys())

    def register_route(self, task_type: TaskType | str, provider_name: str) -> None:
        """Map a task type to a specific provider name."""
        if isinstance(task_type, str):
            task_type = TaskType(task_type.lower())
        self._task_routes[task_type] = provider_name.lower()
        logger.debug("Mapped task %s -> provider %s", task_type.value, provider_name)

    def _normalize_criteria(
        self,
        criteria: RouteRequest | ModelRequest | TaskType | str | None,
    ) -> RouteRequest:
        """Convert arbitrary routing criteria into a normalized RouteRequest."""
        if criteria is None:
            return RouteRequest(task_type=TaskType.GENERAL)

        if isinstance(criteria, RouteRequest):
            return criteria

        if isinstance(criteria, str):
            lowered = criteria.lower()
            if lowered in self._providers:
                return RouteRequest(
                    provider=lowered,
                    preferred_provider=lowered,
                    task_type=TaskType.GENERAL,
                )
            try:
                tt = TaskType(lowered)
                return RouteRequest(task_type=tt)
            except ValueError:
                return RouteRequest(
                    provider=lowered,
                    preferred_provider=lowered,
                    task_type=TaskType.GENERAL,
                )

        if isinstance(criteria, TaskType):
            return RouteRequest(task_type=criteria)

        if isinstance(criteria, ModelRequest):
            return RouteRequest(
                task_type=TaskType.GENERAL,
                provider=criteria.provider,
                preferred_provider=criteria.provider,
                model=criteria.model,
                preferred_model=criteria.model,
            )

        return RouteRequest(task_type=TaskType.GENERAL)

    def _score_candidate(
        self,
        spec: ModelSpec,
        req: RouteRequest,
    ) -> tuple[float, dict[str, float], list[str]]:
        """Compute deterministic score, component breakdown, and reasons for an eligible candidate."""
        score = 0.0
        breakdown: dict[str, float] = {}
        reasons: list[str] = []

        # 1. Matched required capabilities
        if req.required_capabilities:
            matched_count = len(req.required_capabilities)
            cap_score = matched_count * 10.0
            score += cap_score
            breakdown["required_capabilities"] = cap_score
            for cap in sorted([str(c.value if hasattr(c, "value") else c) for c in req.required_capabilities]):
                reasons.append(f"required {cap} capability matched")

        # 2. Extra capabilities bonus (richer models get a slight advantage)
        extra_caps = len(spec.capabilities) - len(req.required_capabilities)
        if extra_caps > 0:
            extra_score = float(extra_caps) * 1.0
            score += extra_score
            breakdown["extra_capabilities"] = extra_score

        # 3. Explicit or preferred provider
        if req.provider and spec.provider.lower() == req.provider.lower():
            score += 100.0
            breakdown["explicit_provider"] = 100.0
            reasons.append(f"matched explicitly requested provider '{spec.provider}'")
        elif req.preferred_provider and spec.provider.lower() == req.preferred_provider.lower():
            score += 50.0
            breakdown["preferred_provider"] = 50.0
            reasons.append(f"matched preferred provider '{spec.provider}'")

        # 4. Explicit model preference
        if req.preferred_model and spec.model.lower() == req.preferred_model.lower():
            score += 40.0
            breakdown["preferred_model"] = 40.0
            reasons.append(f"matched preferred model '{spec.model}'")

        # 5. Task route match
        task_route_provider = self._task_routes.get(req.task_type, self._default_provider).lower()
        if spec.provider.lower() == task_route_provider:
            score += 20.0
            breakdown["task_route"] = 20.0
            reasons.append(f"matched default route for {req.task_type.value}")

        # 6. Priority tier
        if spec.priority > 0:
            priority_score = float(spec.priority) * 5.0
            score += priority_score
            breakdown["priority"] = priority_score
            reasons.append(f"priority tier {spec.priority}")

        # 7. Latency preference
        if req.latency_preference:
            lat_pref = req.latency_preference.lower()
            if lat_pref == "low":
                if spec.latency_tier == "low":
                    score += 15.0
                    breakdown["latency"] = 15.0
                    reasons.append("matched low latency preference")
                elif spec.latency_tier == "medium":
                    score += 5.0
                    breakdown["latency"] = 5.0
            elif lat_pref == "medium":
                if spec.latency_tier in ("low", "medium"):
                    score += 10.0
                    breakdown["latency"] = 10.0
                    reasons.append("matched acceptable latency tier")

        # 8. Cost preference
        if req.cost_preference:
            cost_pref = req.cost_preference.lower()
            if cost_pref in ("free", "low"):
                if spec.provider.lower() == "mock" or (spec.cost and spec.cost.input_cost_per_million == 0.0):
                    score += 15.0
                    breakdown["cost"] = 15.0
                    reasons.append("matched free/low cost preference")
                elif spec.cost and spec.cost.input_cost_per_million is not None and spec.cost.input_cost_per_million <= 1.0:
                    score += 10.0
                    breakdown["cost"] = 10.0
                    reasons.append("matched low cost tier")

        # 9. Availability bonus
        if spec.available:
            score += 30.0
            breakdown["availability"] = 30.0
            reasons.append("provider available")

        # 10. Default provider baseline
        if spec.provider.lower() == self._default_provider.lower():
            score += 5.0
            breakdown["default_provider"] = 5.0

        return score, breakdown, reasons

    def select_model(
        self,
        criteria: RouteRequest | ModelRequest | TaskType | str | None = None,
    ) -> RoutingDecision:
        """Select the optimal model via capability filtering, deterministic scoring, and fallback.

        Capability filtering is strictly applied prior to scoring: any model
        missing a required capability is immediately disqualified.
        """
        req = self._normalize_criteria(criteria)

        # Check if an explicit provider was requested that is not registered
        if req.provider and not self.has_provider(req.provider):
            err = f"Provider '{req.provider}' is not registered or supported."
            decision = RoutingDecision(
                selected_model=None,
                success=False,
                error=err,
                candidates_evaluated=[],
                request=req,
            )
            self._last_decision = decision
            return decision

        # 1. Gather candidate specifications from registry
        candidate_specs = self._registry.list_models()

        # Ensure registered providers without an explicit spec have a synthesized baseline
        registered_spec_providers = {s.provider.lower() for s in candidate_specs}
        for prov_name, prov_inst in self._providers.items():
            if prov_name not in registered_spec_providers:
                if hasattr(prov_inst, "get_spec"):
                    candidate_specs.append(prov_inst.get_spec())
                else:
                    candidate_specs.append(
                        ModelSpec(
                            provider=prov_name,
                            model=getattr(prov_inst, "model_name", "default"),
                            capabilities={ModelCapability.TEXT},
                            available=True,
                        )
                    )

        evaluations: list[CandidateEvaluation] = []
        eligible: list[CandidateEvaluation] = []

        # 2. Hard Capability Filtering & Constraint Check
        for spec in candidate_specs:
            model_id = spec.model_id
            provider_name = spec.provider.lower()

            # Constraint: Provider must be registered in router
            if provider_name not in self._providers:
                evaluations.append(
                    CandidateEvaluation(
                        model_id=model_id,
                        provider=spec.provider,
                        model=spec.model,
                        model_spec=spec,
                        eligible=False,
                        elimination_reason=f"Provider '{spec.provider}' is not registered in router.",
                    )
                )
                continue

            # Constraint: Minimum priority
            if req.min_priority > 0 and spec.priority < req.min_priority:
                evaluations.append(
                    CandidateEvaluation(
                        model_id=model_id,
                        provider=spec.provider,
                        model=spec.model,
                        model_spec=spec,
                        eligible=False,
                        elimination_reason=f"Priority {spec.priority} below required minimum {req.min_priority}.",
                    )
                )
                continue

            # Constraint: Privacy preference (local_only)
            if req.privacy_preference == "local_only":
                is_local = (
                    spec.metadata.get("local", False)
                    or provider_name in ("mock", "local")
                )
                if not is_local:
                    evaluations.append(
                        CandidateEvaluation(
                            model_id=model_id,
                            provider=spec.provider,
                            model=spec.model,
                            model_spec=spec,
                            eligible=False,
                            elimination_reason="Model is not local (privacy_preference='local_only').",
                        )
                    )
                    continue

            # Hard Gate: Required capabilities check
            missing_caps = [
                cap for cap in req.required_capabilities if not spec.has_capability(cap)
            ]
            if missing_caps:
                missing_str = ", ".join(
                    sorted([str(c.value if hasattr(c, "value") else c) for c in missing_caps])
                )
                evaluations.append(
                    CandidateEvaluation(
                        model_id=model_id,
                        provider=spec.provider,
                        model=spec.model,
                        model_spec=spec,
                        eligible=False,
                        elimination_reason=f"Missing required capabilities: {missing_str}.",
                    )
                )
                continue

            # Candidate is eligible for scoring
            score, breakdown, reasons = self._score_candidate(spec, req)
            eval_record = CandidateEvaluation(
                model_id=model_id,
                provider=spec.provider,
                model=spec.model,
                model_spec=spec,
                eligible=True,
                score=score,
                score_breakdown=breakdown,
                reasons=reasons,
            )
            evaluations.append(eval_record)
            eligible.append(eval_record)

        # 3. Check if any candidates survived filtering
        if not eligible:
            if req.preferred_provider and not self.has_provider(req.preferred_provider):
                err = f"Provider '{req.preferred_provider}' is not registered or supported."
            elif req.required_capabilities:
                req_caps_str = ", ".join(
                    sorted([str(c.value if hasattr(c, "value") else c) for c in req.required_capabilities])
                )
                err = f"No eligible model found matching required capabilities: [{req_caps_str}]."
            else:
                err = "No eligible model found matching routing criteria."

            decision = RoutingDecision(
                selected_model=None,
                success=False,
                error=err,
                candidates_evaluated=evaluations,
                request=req,
            )
            self._last_decision = decision
            return decision

        # 4. Deterministic Ranking
        # Sort key: descending score, descending priority, ascending provider, ascending model
        ranked_eligible = sorted(
            eligible,
            key=lambda c: (
                -c.score,
                -c.model_spec.priority,
                c.model_spec.provider.lower(),
                c.model_spec.model.lower(),
            ),
        )

        # 5. Selection and Fallback Handling
        primary_candidate = ranked_eligible[0]
        selected_eval = primary_candidate
        fallback_used = False
        fallback_from = None

        if not primary_candidate.model_spec.available:
            if req.fallback_allowed:
                # Select the next available eligible candidate
                fallback_candidate = next(
                    (c for c in ranked_eligible[1:] if c.model_spec.available), None
                )
                if fallback_candidate is not None:
                    fallback_used = True
                    fallback_from = primary_candidate.model_id
                    selected_eval = fallback_candidate
                    selected_eval.reasons.append(
                        f"primary candidate '{fallback_from}' was unavailable; fallback selected"
                    )
                else:
                    err = (
                        f"Primary model '{primary_candidate.model_id}' is unavailable "
                        "and no available fallback candidate exists."
                    )
                    decision = RoutingDecision(
                        selected_model=None,
                        success=False,
                        error=err,
                        candidates_evaluated=evaluations,
                        request=req,
                    )
                    self._last_decision = decision
                    return decision
            else:
                err = (
                    f"Primary model '{primary_candidate.model_id}' is unavailable "
                    "and fallback is disabled."
                )
                decision = RoutingDecision(
                    selected_model=None,
                    success=False,
                    error=err,
                    candidates_evaluated=evaluations,
                    request=req,
                )
                self._last_decision = decision
                return decision

        decision = RoutingDecision(
            selected_model=selected_eval.model_spec,
            provider_name=selected_eval.model_spec.provider,
            model_name=selected_eval.model_spec.model,
            success=True,
            score=selected_eval.score,
            reasons=selected_eval.reasons,
            candidates_evaluated=evaluations,
            fallback_used=fallback_used,
            fallback_from=fallback_from,
            request=req,
        )
        self._last_decision = decision
        return decision

    def route(
        self,
        criteria: RouteRequest | ModelRequest | TaskType | str | None = None,
    ) -> ModelProvider:
        """Select a ModelProvider based on capability-aware routing criteria.

        Raises:
            ProviderNotFoundError: If a specific provider was requested but is not registered.
            NoEligibleModelError: If no candidate satisfies the required capabilities.
        """
        decision = self.select_model(criteria)

        if not decision.success:
            req = self._normalize_criteria(criteria)
            target_prov = req.provider or req.preferred_provider
            if target_prov and not self.has_provider(target_prov):
                raise ProviderNotFoundError(
                    f"Provider '{target_prov}' is not registered or supported."
                )
            raise NoEligibleModelError(decision.error or "No eligible model found.")

        return self.get_provider(decision.provider_name)

    async def generate(
        self,
        request: ModelRequest,
        *,
        task_type: TaskType = TaskType.GENERAL,
        required_capabilities: set[ModelCapability | str] | None = None,
        fallback_allowed: bool = True,
    ) -> ModelResponse:
        """Route and execute a text generation request with runtime fallback."""
        criteria = RouteRequest(
            task_type=task_type,
            provider=request.provider,
            model=request.model,
            required_capabilities=required_capabilities or set(),
            fallback_allowed=fallback_allowed,
        )
        decision = self.select_model(criteria)

        if not decision.success:
            return ModelResponse(
                content="",
                model=request.model or self._default_model,
                provider=request.provider or self._default_provider,
                request_id=request.request_id,
                success=False,
                error=decision.error,
                finish_reason="error",
            )

        provider = self.get_provider(decision.provider_name)
        target_model = decision.model_name or request.model or self._default_model
        if request.model is None:
            request.model = target_model

        # Runtime availability check with dispatch-time fallback
        is_avail = await provider.is_available()
        if not is_avail:
            fallback_found = False
            if fallback_allowed:
                for cand_eval in decision.candidates_evaluated:
                    if (
                        cand_eval.eligible
                        and cand_eval.provider != decision.provider_name
                        and self.has_provider(cand_eval.provider)
                    ):
                        alt_provider = self.get_provider(cand_eval.provider)
                        if await alt_provider.is_available():
                            logger.info(
                                "Primary provider '%s' unavailable at runtime, falling back to '%s'",
                                decision.provider_name,
                                alt_provider.name,
                            )
                            decision.fallback_used = True
                            decision.fallback_from = f"{decision.provider_name}:{decision.model_name}"
                            provider = alt_provider
                            target_model = cand_eval.model
                            request.model = target_model
                            fallback_found = True
                            break

            if not fallback_found:
                return ModelResponse(
                    content="",
                    model=target_model,
                    provider=provider.name,
                    request_id=request.request_id,
                    success=False,
                    error=f"Selected provider '{provider.name}' is unavailable and no fallback succeeded.",
                    finish_reason="error",
                )

        response = await provider.generate(request)
        if isinstance(response, str):
            response = ModelResponse(
                content=response,
                model=target_model,
                provider=provider.name,
                request_id=request.request_id,
            )

        # Attach explainable routing decision metadata
        response.metadata.extra["routing_decision"] = decision.model_dump()
        return response

    async def generate_structured(
        self,
        request: ModelRequest,
        schema: type[T],
        *,
        task_type: TaskType = TaskType.STRUCTURED_OUTPUT,
        fallback_allowed: bool = True,
    ) -> ModelResponse:
        """Route and execute a structured generation request with schema capability enforcement."""
        criteria = RouteRequest(
            task_type=task_type,
            provider=request.provider,
            model=request.model,
            required_capabilities={ModelCapability.STRUCTURED_OUTPUT},
            fallback_allowed=fallback_allowed,
        )
        decision = self.select_model(criteria)

        if not decision.success:
            return ModelResponse(
                content="",
                model=request.model or self._default_model,
                provider=request.provider or self._default_provider,
                request_id=request.request_id,
                success=False,
                error=decision.error,
                finish_reason="error",
            )

        provider = self.get_provider(decision.provider_name)
        target_model = decision.model_name or request.model or self._default_model
        if request.model is None:
            request.model = target_model

        # Runtime availability check with dispatch-time fallback
        is_avail = await provider.is_available()
        if not is_avail:
            fallback_found = False
            if fallback_allowed:
                for cand_eval in decision.candidates_evaluated:
                    if (
                        cand_eval.eligible
                        and cand_eval.provider != decision.provider_name
                        and self.has_provider(cand_eval.provider)
                    ):
                        alt_provider = self.get_provider(cand_eval.provider)
                        if await alt_provider.is_available():
                            logger.info(
                                "Primary provider '%s' unavailable at runtime, falling back to '%s'",
                                decision.provider_name,
                                alt_provider.name,
                            )
                            decision.fallback_used = True
                            decision.fallback_from = f"{decision.provider_name}:{decision.model_name}"
                            provider = alt_provider
                            target_model = cand_eval.model
                            request.model = target_model
                            fallback_found = True
                            break

            if not fallback_found:
                return ModelResponse(
                    content="",
                    model=target_model,
                    provider=provider.name,
                    request_id=request.request_id,
                    success=False,
                    error=f"Selected provider '{provider.name}' is unavailable and no fallback succeeded.",
                    finish_reason="error",
                )

        response = await provider.generate_structured(request, schema)
        response.metadata.extra["routing_decision"] = decision.model_dump()
        return response
