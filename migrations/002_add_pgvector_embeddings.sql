-- Migration: 002_add_pgvector_embeddings.sql
-- Description: Enable pgvector extension and add embedding vector column to memory_records.
-- Scope: Milestone 1E.6 — Semantic Memory Retrieval with pgvector

-- 1. Enable the pgvector extension if available
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Add embedding column supporting dense vector representations
-- Defined as generic `vector` to support configurable embedding dimensions (e.g. 768, 1536, 384)
ALTER TABLE memory_records ADD COLUMN IF NOT EXISTS embedding vector;

-- 3. Vector Indexing Note:
-- Exact nearest-neighbor search (`ORDER BY embedding <=> $1`) on unconstrained vectors
-- delivers 100% recall with sub-millisecond execution for typical personal agent datasets (< 100k records).
-- When a deployment standardizes on a fixed embedding dimension (e.g. 768), an HNSW index can be created:
-- CREATE INDEX IF NOT EXISTS idx_memory_records_embedding ON memory_records
--     USING hnsw ((embedding::vector(768)) vector_cosine_ops)
--     WITH (m = 16, ef_construction = 64);
