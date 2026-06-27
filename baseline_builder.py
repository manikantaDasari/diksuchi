#!/usr/bin/env python3
"""
baseline_builder.py — Generates model_ratings_baseline.json from model_benchmarks_raw.json.

Usage:
    python baseline_builder.py              # writes model_ratings_baseline.json
    python baseline_builder.py --dry-run    # prints JSON to stdout, no file write
    python baseline_builder.py --verify     # prints a per-model score table for spot-checks

This file is the ONLY tool that should modify model_ratings_baseline.json.
model_ratings_baseline.json is a BUILD ARTIFACT — treat it like a compiled binary:
  - DO NOT edit it manually
  - COMMIT the generated file (it ships with Diksuchi for offline use)
  - To update scores, edit model_benchmarks_raw.json and re-run this script

The pipeline mirrors the _BUCKET_WEIGHTS weighting matrix used by rating_fetcher.py
for HF leaderboard scores, so bundled baseline and live HF data are computed
on the same scale and blend cleanly.

_BUCKET_WEIGHTS vector slot → benchmark:
  [0] IFEval    — instruction following
  [1] BBH       — big bench hard (reasoning chains)
  [2] MATH      — competition math
  [3] GPQA      — graduate-level science (Diamond subset if available)
  [4] MUSR      — multistep soft reasoning
  [5] MMLU      — broad academic knowledge
  [6] avg       — average of all HF benchmark slots

Benchmark → slot mapping (see _BENCHMARK_TO_SLOT):
  Direct mappings (use score as-is):
    IFEval           → slot 0
    BBH              → slot 1
    MATH             → slot 2
    GPQA_Diamond     → slot 3  (preferred, higher bar)
    GPQA             → slot 3  (fallback)
    MUSR             → slot 4
    MMLU             → slot 5
    MMLU_PRO         → slot 5  (preferred, harder version)

  Proxy mappings (lower-confidence estimates, flagged):
    HumanEval        → slot 1  (code correctness ≈ BBH reasoning; scale: HE * 0.80)
    MBPP             → slot 1  (similar to HumanEval; scale: MBPP * 0.80)
    AIME             → slot 2  (competition math; scale: AIME, same 0-100 range)

  Imputed (derived when slot has no measured value):
    slot 0 (IFEval)  ← MMLU * 0.88        (IFEval ~88% of MMLU empirically)
    slot 1 (BBH)     ← MMLU * 0.92        (BBH ~92% of MMLU empirically)
    slot 4 (MUSR)    ← mean(BBH, GPQA) * 0.85
                       OR MMLU * 0.72     (if no BBH/GPQA)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ─── File paths ───────────────────────────────────────────────────────────────

HERE            = Path(__file__).parent
RAW_FILE        = HERE / "model_benchmarks_raw.json"
BASELINE_FILE   = HERE / "model_ratings_baseline.json"

# ─── The same weights matrix as rating_fetcher.py ─────────────────────────────
# DO NOT CHANGE without also updating rating_fetcher._BUCKET_WEIGHTS.
_BUCKET_WEIGHTS = {
    #            IFEval  BBH   MATH  GPQA  MUSR  MMLU  avg
    "simple":   [0.40,  0.10, 0.05, 0.05, 0.05, 0.25, 0.10],
    "coding":   [0.35,  0.35, 0.10, 0.05, 0.05, 0.05, 0.05],
    "reasoning":[0.05,  0.25, 0.25, 0.25, 0.15, 0.00, 0.05],
    "creative": [0.50,  0.05, 0.00, 0.00, 0.00, 0.20, 0.25],
    "critical": [0.05,  0.15, 0.25, 0.35, 0.10, 0.00, 0.10],
}

# ─── Benchmark → slot + scale factor ─────────────────────────────────────────
# Each entry: (slot_index, scale_factor, is_proxy)
# scale_factor adjusts for benchmark difficulty relative to HF standard slot.
# is_proxy=True means the mapping is approximate; noted in provenance output.
_BENCHMARK_TO_SLOT: dict[str, tuple[int, float, bool]] = {
    "IFEval":       (0, 1.00, False),
    "BBH":          (1, 1.00, False),
    "MATH":         (2, 1.00, False),
    "GPQA_Diamond": (3, 1.00, False),   # preferred — harder Diamond subset
    "GPQA":         (3, 0.95, False),   # fallback — slightly easier full set
    "MUSR":         (4, 1.00, False),
    "MMLU_PRO":     (5, 1.00, False),   # preferred — harder
    "MMLU":         (5, 0.92, False),   # fallback — slightly easier
    "HumanEval":    (1, 0.80, True),    # proxy for BBH (code ≈ reasoning)
    "MBPP":         (1, 0.80, True),    # proxy for BBH
    "AIME":         (2, 1.00, True),    # proxy for MATH (both competition math)
    "MGSM":         (0, 0.90, True),    # proxy for IFEval (multilingual instruction)
    "SWE-bench":    (1, 0.70, True),    # proxy for BBH (real-world code tasks, harder)
}


def _build_slots(benchmarks: dict[str, float]) -> tuple[list[float | None], list[str]]:
    """
    Given a {benchmark_name: score_0_to_100} dict, fill vector slots [0..6].

    Returns:
        slots      — list of 7 floats (0–1), None where slot not measured/imputed
        provenance — list of human-readable notes about what filled each slot
    """
    # Priority: direct > proxy > imputed
    # For each slot, track the best available source
    #   priority: 0=direct, 1=proxy, 2=imputed — lower wins
    slot_val:  list[float | None] = [None] * 7
    slot_pri:  list[int]          = [999]  * 7
    slot_note: list[str]          = [""]   * 7

    for bname, score in benchmarks.items():
        if bname not in _BENCHMARK_TO_SLOT:
            continue
        slot_idx, scale, is_proxy = _BENCHMARK_TO_SLOT[bname]
        priority = 1 if is_proxy else 0
        if priority < slot_pri[slot_idx]:
            slot_val[slot_idx]  = min(score * scale / 100.0, 1.0)
            slot_pri[slot_idx]  = priority
            slot_note[slot_idx] = f"{'proxy:' if is_proxy else ''}{bname}={score:.1f}{'*'+str(scale) if scale != 1.0 else ''}"

    # ── Imputation pass ─────────────────────────────────────────────────────
    # Only fill still-None slots; never overwrite measured/proxy values.

    # slot 0 (IFEval) from MMLU if available
    if slot_val[0] is None and slot_val[5] is not None:
        slot_val[0]  = round(slot_val[5] * 0.88, 4)
        slot_note[0] = f"imputed:MMLU*0.88"

    # slot 1 (BBH) from MMLU if available
    if slot_val[1] is None and slot_val[5] is not None:
        slot_val[1]  = round(slot_val[5] * 0.92, 4)
        slot_note[1] = f"imputed:MMLU*0.92"

    # slot 4 (MUSR): try mean(BBH, GPQA) * 0.85, fallback to MMLU * 0.72
    if slot_val[4] is None:
        if slot_val[1] is not None and slot_val[3] is not None:
            slot_val[4]  = round(((slot_val[1] + slot_val[3]) / 2.0) * 0.85, 4)
            slot_note[4] = "imputed:mean(BBH,GPQA)*0.85"
        elif slot_val[5] is not None:
            slot_val[4]  = round(slot_val[5] * 0.72, 4)
            slot_note[4] = "imputed:MMLU*0.72"

    # slot 3 (GPQA) from MMLU * 0.55 (GPQA is much harder)
    if slot_val[3] is None and slot_val[5] is not None:
        slot_val[3]  = round(slot_val[5] * 0.55, 4)
        slot_note[3] = "imputed:MMLU*0.55"

    # slot 2 (MATH) from MMLU * 0.80 if still missing
    if slot_val[2] is None and slot_val[5] is not None:
        slot_val[2]  = round(slot_val[5] * 0.80, 4)
        slot_note[2] = "imputed:MMLU*0.80"

    # slot 6 (avg): arithmetic mean of all available slots 0–5
    available = [v for v in slot_val[:6] if v is not None]
    if available:
        slot_val[6]  = round(sum(available) / len(available), 4)
        slot_note[6] = f"avg_of_{len(available)}_slots"

    provenance = [f"slot{i}={slot_note[i] or 'missing'}" for i in range(7)]
    return slot_val, provenance


def _slots_to_bucket_scores(slots: list[float | None]) -> dict[str, float]:
    """Apply _BUCKET_WEIGHTS matrix to the 7-element slot vector → 5 bucket scores."""
    # Replace any remaining None with 0.0 for the dot product
    fields = [v if v is not None else 0.0 for v in slots]
    scores: dict[str, float] = {}
    for bucket, weights in _BUCKET_WEIGHTS.items():
        scores[bucket] = round(sum(f * w for f, w in zip(fields, weights)), 4)
    return scores


def _overall_from_slots(slots: list[float | None]) -> float:
    """Overall score = slot 6 (avg) or mean of available slots."""
    if slots[6] is not None:
        return round(slots[6], 4)
    available = [v for v in slots[:6] if v is not None]
    return round(sum(available) / len(available), 4) if available else 0.0


def build_baseline(raw_data: dict) -> dict:
    """
    Transform model_benchmarks_raw.json → baseline dict compatible with
    model_ratings_baseline.json schema.

    Output schema per model entry:
    {
      "simple": 0.xxx,
      "coding": 0.xxx,
      "reasoning": 0.xxx,
      "creative": 0.xxx,
      "critical": 0.xxx,
      "overall": 0.xxx,
      "data_quality": "official|paper|community|estimated",
      "source_url": "...",
      "date": "YYYY-MM",
      "_provenance": ["slot0=MMLU*0.88", ...]   # debug info only
    }
    """
    output: dict[str, Any] = {}
    skipped: list[str] = []

    for entry in raw_data.get("models", []):
        # Skip section headers (they're dicts with only a _section key)
        if "_section" in entry:
            continue

        model_id: str = entry.get("model", "")
        if not model_id:
            continue

        benchmarks: dict = entry.get("benchmarks", {})
        if not benchmarks:
            skipped.append(model_id)
            continue

        slots, provenance = _build_slots(benchmarks)
        bucket_scores     = _slots_to_bucket_scores(slots)
        overall           = _overall_from_slots(slots)

        record: dict[str, Any] = {
            **bucket_scores,
            "overall":      overall,
            "data_quality": entry.get("data_quality", "estimated"),
            "source_url":   entry.get("source_url", ""),
            "date":         entry.get("date", ""),
            "_provenance":  provenance,
        }
        if entry.get("_note"):
            record["_note"] = entry["_note"]

        output[model_id] = record

        # Also register aliases under their own keys (same record, note origin)
        for alias in entry.get("aliases", []):
            if alias and alias != model_id:
                alias_record = dict(record)
                alias_record["_alias_of"] = model_id
                output[alias] = alias_record

    return output, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Build model_ratings_baseline.json")
    parser.add_argument("--dry-run",  action="store_true", help="Print JSON to stdout, no file write")
    parser.add_argument("--verify",   action="store_true", help="Print score table for spot-check")
    parser.add_argument("--raw",      default=str(RAW_FILE),      help="Input raw benchmarks JSON")
    parser.add_argument("--out",      default=str(BASELINE_FILE), help="Output baseline JSON path")
    args = parser.parse_args()

    raw_path = Path(args.raw)
    out_path = Path(args.out)

    if not raw_path.exists():
        print(f"ERROR: raw benchmarks file not found: {raw_path}", file=sys.stderr)
        sys.exit(1)

    with open(raw_path, encoding="utf-8") as f:
        raw_data = json.load(f)

    baseline, skipped = build_baseline(raw_data)

    # ── Wrap with metadata ──────────────────────────────────────────────────
    output_doc: dict[str, Any] = {
        "_comment": [
            "AUTO-GENERATED — do not edit manually.",
            "Source: model_benchmarks_raw.json",
            "Generator: baseline_builder.py",
            "To regenerate: python baseline_builder.py   OR   make baseline",
        ],
        "_generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "_source_file":  str(raw_path.name),
        "_model_count":  len([k for k, v in baseline.items() if "_alias_of" not in v]),
        "_alias_count":  len([k for k, v in baseline.items() if "_alias_of" in v]),
        **baseline,
    }

    # ── Dry-run / verify mode ───────────────────────────────────────────────
    if args.dry_run:
        print(json.dumps(output_doc, indent=2))
        return

    if args.verify:
        _print_verify_table(baseline)
        if skipped:
            print(f"\nSkipped (no benchmarks data): {', '.join(skipped)}")
        return

    # ── Write file ──────────────────────────────────────────────────────────
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_doc, f, indent=2)
        f.write("\n")

    # Count unique models (non-aliases)
    n_models = output_doc["_model_count"]
    n_aliases = output_doc["_alias_count"]
    print(f"✅  Wrote {out_path.name}  ({n_models} models, {n_aliases} aliases)")
    if skipped:
        print(f"   ⚠️  Skipped {len(skipped)} entries with no benchmark data: {skipped}")


def _print_verify_table(baseline: dict) -> None:
    """Print a human-readable score table for spot-checking."""
    # Skip aliases for the table
    models = {k: v for k, v in baseline.items() if "_alias_of" not in v}

    header = f"{'Model':<55} {'simple':>7} {'coding':>7} {'reason':>7} {'creatv':>7} {'critcl':>7} {'overall':>7} {'qual':<10}"
    print(header)
    print("-" * len(header))

    for model_id, scores in sorted(models.items(), key=lambda x: -x[1].get("overall", 0)):
        short = model_id[-55:] if len(model_id) > 55 else model_id
        q = scores.get("data_quality", "?")[:9]
        print(
            f"{short:<55} "
            f"{scores.get('simple',0):>7.3f} "
            f"{scores.get('coding',0):>7.3f} "
            f"{scores.get('reasoning',0):>7.3f} "
            f"{scores.get('creative',0):>7.3f} "
            f"{scores.get('critical',0):>7.3f} "
            f"{scores.get('overall',0):>7.3f} "
            f"{q:<10}"
        )


if __name__ == "__main__":
    main()
