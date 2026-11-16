from __future__ import annotations

import argparse
import logging

from app.agent.graphs import run_ingestion
from app.ingest.sources import SemanticScholarClient
from app.config import get_settings


logger = logging.getLogger(__name__)


def _best_effort_db_topic_state(topic: str, min_citations: int) -> None:
    try:
        from app.db.session import get_conn, init_db, load_topic_state, save_topic_state

        try:
            init_db()
        except Exception:
            logger.warning("postgres init_db unavailable; continuing in-memory", exc_info=True)
            return
        conn = get_conn()
        try:
            load_topic_state(conn)
            save_topic_state(conn, topic, min_citations, None)
        finally:
            try:
                conn.close()
            except Exception:
                logger.debug("postgres conn close failed", exc_info=True)
    except Exception:
        logger.warning("postgres unavailable; continuing in-memory", exc_info=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", required=True)
    parser.add_argument("--since")
    parser.add_argument("--min-citations", type=int, required=True)
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()
    settings = get_settings()
    _best_effort_db_topic_state(args.topic, args.min_citations)
    candidates = SemanticScholarClient(settings.atlas_s2_base, settings.s2_api_key).search(args.topic, args.topic, args.since, min(args.limit, 100))
    print(run_ingestion(candidates, args.topic, args.min_citations))


if __name__ == "__main__":
    main()
