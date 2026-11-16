-- Atlas Postgres schema — source of truth. Apply via app.db.init.
CREATE TABLE IF NOT EXISTS papers (
  id            TEXT PRIMARY KEY,
  title         TEXT NOT NULL,
  authors       TEXT[] NOT NULL,
  venue         TEXT,
  year          SMALLINT,
  citations     INTEGER NOT NULL DEFAULT 0,
  doi           TEXT UNIQUE,
  arxiv_id      TEXT,
  s2_id         TEXT UNIQUE,
  openalex_id   TEXT UNIQUE,
  topic         TEXT NOT NULL,
  url           TEXT NOT NULL,
  pdf_url       TEXT,
  pdf_sha256    TEXT,
  ingested_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
  summarized_at TIMESTAMPTZ,
  why_kept      TEXT NOT NULL,
  extraction    TEXT NOT NULL DEFAULT 'abstract_only',
  CONSTRAINT citations_nonneg CHECK (citations >= 0)
);
CREATE INDEX IF NOT EXISTS ix_papers_topic_citations ON papers (topic, citations DESC);
CREATE INDEX IF NOT EXISTS ix_papers_ingested_at ON papers (ingested_at);
CREATE INDEX IF NOT EXISTS ix_papers_year ON papers (year);

CREATE TABLE IF NOT EXISTS chunks (
  id                TEXT PRIMARY KEY,
  paper_id          TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
  n                 SMALLINT NOT NULL,
  section           TEXT NOT NULL,
  text              TEXT NOT NULL,
  qdrant_point_id   TEXT UNIQUE,
  UNIQUE (paper_id, n)
);
CREATE INDEX IF NOT EXISTS ix_chunks_paper_id ON chunks (paper_id);

CREATE TABLE IF NOT EXISTS chunk_embeddings_cache (
  sha256      TEXT PRIMARY KEY,
  model_id    TEXT NOT NULL,
  dim         SMALLINT NOT NULL,
  created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS eval_questions (
  id          TEXT PRIMARY KEY,
  question    TEXT NOT NULL,
  filters     JSONB NOT NULL,
  gold_paper_ids TEXT[] NOT NULL,
  created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS topic_state (
  topic       TEXT PRIMARY KEY,
  min_citations INTEGER NOT NULL,
  last_polled_at TIMESTAMPTZ,
  cursor      TEXT
);

CREATE TABLE IF NOT EXISTS paper_duplicates (
  kept_id     TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
  duplicate_id TEXT NOT NULL,
  score       DOUBLE PRECISION NOT NULL,
  reason      TEXT NOT NULL,
  created_at  TIMESTAMPTZ DEFAULT now(),
  PRIMARY KEY (kept_id, duplicate_id)
);
