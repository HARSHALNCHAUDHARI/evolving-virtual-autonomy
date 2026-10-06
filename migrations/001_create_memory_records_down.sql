-- Reversal Migration: 001_create_memory_records_down.sql
-- Description: Drop memory_records table and all associated indexes.
-- Scope: Reversible migration for Milestone 1E.3

DROP TABLE IF EXISTS memory_records CASCADE;
