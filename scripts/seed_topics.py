from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone

from app.agent.graphs import run_ingestion
from app.ingest.sources import CandidatePaper


def make_papers(topics: list[tuple[str, int]], count: int) -> list[CandidatePaper]:
    base = datetime.now(timezone.utc)
    papers: list[CandidatePaper] = []
    for index in range(count):
        topic, threshold = topics[index % len(topics)]
        papers.append(CandidatePaper(id=f"seed:{topic}:{index}", title=f"{topic.title()} methods and evidence {index}", authors=["Atlas Researcher"], venue="Synthetic Journal", year=2025, citations=threshold + (index % 40), topic=topic, url=f"https://example.invalid/papers/{topic}/{index}", abstract=f"A synthetic study of {topic}, retrieval, reranking, and limitations. The results compare baseline methods and report negative findings where appropriate.", source="seed", published_at=base - timedelta(days=index)))
    return papers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topics", default="reranking:80,retrieval:50")
    parser.add_argument("--papers", type=int, default=30, help="synthetic papers to ingest per topic (not total across topics)")
    args = parser.parse_args()
    topics = []
    for value in args.topics.split(","):
        topic, threshold = value.split(":", 1)
        topics.append((topic, int(threshold)))
    for topic, threshold in topics:
        result = run_ingestion(make_papers([(topic, threshold)], args.papers), topic, threshold)
        print(result)


if __name__ == "__main__":
    main()
