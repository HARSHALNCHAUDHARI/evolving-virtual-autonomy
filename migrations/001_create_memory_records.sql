-- Migration: 001_create_memory_records.sql
-- Description: Create persistent memory_records table and indexes for EVA memory subsystem.
-- Authoritative Model: eva.memory.models.MemoryRecord
-- Scope: Milestone 1E.3 — PostgreSQL Persistent Storage

CREATE TABLE IF NOT EXISTS memory_records (
    memory_id UUID PRIMARY KEY,
    memory_type VARCHAR(64) NOT NULL,
    content TEXT NOT NULL,
    structured_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    source VARCHAR(64) NOT NULL,
    confidence DOUBLE PRECISION NOT NULL DEFAULT 1.0 CHECK (confidence >= 0.0 AND confidence <= 1.0),
    importance DOUBLE PRECISION NOT NULL DEFAULT 0.5 CHECK (importance >= 0.0 AND importance <= 1.0),
    relevance DOUBLE PRECISION NOT NULL DEFAULT 0.0 CHECK (relevance >= 0.0 AND relevance <= 1.0),
    project_id VARCHAR(255),
    session_id VARCHAR(255),
    provenance JSONB NOT NULL,
    embedding_ref JSONB,
    access_metadata JSONB NOT NULL,
    retention_metadata JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);

-- Indexes for deterministic querying and filtering
CREATE INDEX IF NOT EXISTS idx_memory_records_type ON memory_records (memory_type);
CREATE INDEX IF NOT EXISTS idx_memory_records_project ON memory_records (project_id);
CREATE INDEX IF NOT EXISTS idx_memory_records_session ON memory_records (session_id) WHERE session_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_memory_records_source ON memory_records (source);
CREATE INDEX IF NOT EXISTS idx_memory_records_created_at ON memory_records (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_memory_records_updated_at ON memory_records (updated_at DESC);
