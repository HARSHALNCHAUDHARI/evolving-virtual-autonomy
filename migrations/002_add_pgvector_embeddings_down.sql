-- Migration: 002_add_pgvector_embeddings_down.sql
-- Description: Revert embedding vector column and pgvector extension.
-- Scope: Milestone 1E.6 — Semantic Memory Retrieval with pgvector

-- Drop the embedding column if it exists
ALTER TABLE memory_records DROP COLUMN IF EXISTS embedding;
