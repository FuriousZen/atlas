from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _dsn() -> str:
    from app.config import get_settings

    url = get_settings().database_url
    if url.startswith("postgresql+"):
        url = "postgresql://" + url.split("://", 1)[1]
    return url


def get_conn() -> Any:
    """Open a psycopg connection from DATABASE_URL (Settings.database_url).

    Raises if Postgres is unreachable; callers must treat this as best-effort
    and continue on the in-memory path.
    """
    import psycopg

    return psycopg.connect(_dsn(), connect_timeout=3)


def init_db(schema_path: str = "app/db/schema.sql") -> None:
    """Execute the DDL file statement-by-statement. Raises on failure."""
    import psycopg

    text = Path(schema_path).read_text(encoding="utf-8")
    statements = [part.strip() for part in text.split(";") if part.strip()]
    with psycopg.connect(_dsn(), connect_timeout=3) as conn:
        with conn.cursor() as cur:
            for statement in statements:
                cur.execute(statement)
        conn.commit()


def load_topic_state(conn: Any) -> dict:
    """Return {topic: {"min_citations": int, "cursor": str | None}}."""
    cur = conn.cursor()
    cur.execute("SELECT topic, min_citations, cursor FROM topic_state")
    rows = cur.fetchall()
    return {row[0]: {"min_citations": row[1], "cursor": row[2]} for row in rows}


def save_topic_state(conn: Any, topic: str, min_citations: int, cursor: str | None) -> None:
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO topic_state (topic, min_citations, cursor) VALUES (%s, %s, %s)"
        " ON CONFLICT (topic) DO UPDATE SET min_citations = EXCLUDED.min_citations,"
        " cursor = EXCLUDED.cursor",
        (topic, min_citations, cursor),
    )
    conn.commit()


def record_duplicates(conn: Any, pairs: list[dict[str, object]]) -> None:
    """Best-effort persist of dedupe pairs; never raises."""
    if not pairs:
        return
    try:
        cur = conn.cursor()
        for pair in pairs:
            cur.execute(
                "INSERT INTO paper_duplicates (kept_id, duplicate_id, score, reason)"
                " VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING",
                (pair.get("kept_id"), pair.get("duplicate_id"), pair.get("score"), pair.get("reason")),
            )
        conn.commit()
    except Exception:
        logger.warning("record_duplicates failed; continuing in-memory", exc_info=True)
