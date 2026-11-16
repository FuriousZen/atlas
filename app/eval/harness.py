from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

from app.agent.graphs import run_query


def load_dataset(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def hit_at_k(ranked: list[str], expected: set[str], k: int) -> bool:
    """Set-overlap hit within the top-k cited paper ids, in citation order."""
    return bool(set(ranked[:k]) & expected)


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return float(ordered[low] * (1 - frac) + ordered[high] * frac)


def summarize_metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Pure aggregation over per-query results (no network; unit-testable).

    Each item carries: id, ranked (ordered paper ids), expected (gold ids),
    elapsed_ms, faithfulness, model.
    """
    total = len(items)
    latencies = [float(item.get("elapsed_ms", 0.0)) for item in items]
    faithfulness = [float(item.get("faithfulness", 0.0)) for item in items]
    models = sorted({str(item.get("model", "unknown")) for item in items})
    return {
        "questions": total,
        "hit_at_1": sum(hit_at_k(item["ranked"], set(item["expected"]), 1) for item in items) / max(1, total),
        "hit_at_5": sum(hit_at_k(item["ranked"], set(item["expected"]), 5) for item in items) / max(1, total),
        "hit_at_10": sum(hit_at_k(item["ranked"], set(item["expected"]), 10) for item in items) / max(1, total),
        "faithfulness_mean": sum(faithfulness) / max(1, total),
        "p50_ms": percentile(latencies, 50),
        "p95_ms": percentile(latencies, 95),
        "provider": models[0] if len(models) == 1 else models,
        "models": models,
    }


def run_one(item: dict[str, Any]) -> dict[str, Any]:
    t0 = time.perf_counter()
    result = run_query(item["question"], item.get("filters", {}))
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)
    ranked = [citation.get("paper_id") for citation in result.get("citations", []) if isinstance(citation, dict) and citation.get("paper_id")]
    expected = list(item.get("gold_paper_ids", []))
    return {
        "id": item.get("id"),
        "ranked": ranked,
        "hit_at_1": hit_at_k(ranked, set(expected), 1),
        "hit_at_5": hit_at_k(ranked, set(expected), 5),
        "hit_at_10": hit_at_k(ranked, set(expected), 10),
        "hit": bool(set(ranked) & set(expected)),
        "found": sorted(set(ranked)),
        "expected": sorted(expected),
        "faithfulness": float(result.get("faithfulness", 0.0) or 0.0),
        "model": str(result.get("model", "unknown")),
        "elapsed_ms": elapsed_ms,
    }


async def _run_all(dataset: list[dict[str, Any]], concurrency: int) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def _one(item: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            return await asyncio.to_thread(run_one, item)

    return list(await asyncio.gather(*(_one(item) for item in dataset)))


def run(dataset: list[dict[str, Any]], concurrency: int = 1) -> dict[str, Any]:
    start = time.perf_counter()
    if concurrency is not None and concurrency >= 2:
        rows = asyncio.run(_run_all(dataset, concurrency))
    else:
        rows = [run_one(item) for item in dataset]
    wall_ms = round((time.perf_counter() - start) * 1000, 3)
    summary = summarize_metrics(rows)
    return {**summary, "elapsed_ms": wall_ms, "per_query_ms": [row["elapsed_ms"] for row in rows], "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Atlas retrieval eval: hit@k, faithfulness, latency over atlas_qa.jsonl.",
        epilog="Gold paper_ids reference seed:* synthetics; run `make seed` (scripts/seed_topics.py with defaults) before `make eval`.",
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--concurrent", type=int, default=1, help="queries in flight via asyncio gather (default 1, sequential)")
    args = parser.parse_args()
    result = run(load_dataset(args.dataset), concurrency=args.concurrent)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
