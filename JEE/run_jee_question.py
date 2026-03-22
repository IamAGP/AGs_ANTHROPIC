"""
JEE Advanced 2025 — Test questions against claude-opus-4-6
with extended thinking (effort=max). Stores every byte the API returns.

Usage:
  uv run JEE/run_jee_question.py               # run all questions
  uv run JEE/run_jee_question.py Q16           # run a single question
  uv run JEE/run_jee_question.py Q9 Q11        # run specific questions
  uv run JEE/run_jee_question.py --skip Q16    # all except Q16
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import anthropic
from dotenv import load_dotenv

# ── Config ────────────────────────────────────────────────────────────────────
MODEL = "claude-opus-4-6"
MAX_TOKENS = 16000
EFFORT = "max"
RESULTS_DIR = Path(__file__).parent / "results"
JSONL_PATH = Path(__file__).parent / "jee_advanced_2025_p2.jsonl"

SYSTEM_PROMPT = (
    "You are an expert mathematician solving JEE Advanced problems. "
    "Show your complete reasoning step by step, then state the final numerical answer clearly."
)

# ── Load questions ─────────────────────────────────────────────────────────────
def load_all_questions() -> list[dict]:
    questions = []
    with open(JSONL_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions

def load_questions(question_ids: list[str]) -> list[dict]:
    all_q = load_all_questions()
    id_set = set(question_ids)
    found = [q for q in all_q if q["question_id"] in id_set]
    missing = id_set - {q["question_id"] for q in found}
    if missing:
        raise ValueError(f"Question IDs not found: {missing}")
    return found

# ── Run one question ──────────────────────────────────────────────────────────
def run_question(client: anthropic.Anthropic, question: dict) -> dict:
    print(f"\n{'='*60}")
    print(f"Model   : {MODEL}  |  Effort: {EFFORT}")
    print(f"Question: {question['question_id']} | {question['section']}")
    print(f"{'='*60}")
    print(f"\n{question['unicode_version']}\n")
    print(f"Ground truth: {question['ground_truth']}\n")
    print("Calling API...\n")

    t_start = time.perf_counter()
    wall_start = datetime.now(timezone.utc).isoformat()

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        thinking={"type": "adaptive"},
        output_config={"effort": EFFORT},
        messages=[{"role": "user", "content": question["latex_version"]}],
    )

    t_end = time.perf_counter()
    wall_end = datetime.now(timezone.utc).isoformat()
    elapsed_sec = round(t_end - t_start, 3)

    # ── Parse content blocks ──────────────────────────────────────────────────
    thinking_blocks, text_blocks = [], []
    for block in response.content:
        if block.type == "thinking":
            thinking_blocks.append({
                "type": "thinking",
                "thinking": block.thinking,
                "signature": getattr(block, "signature", None),
            })
        elif block.type == "text":
            text_blocks.append({"type": "text", "text": block.text})

    # ── Usage & cost ──────────────────────────────────────────────────────────
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "cache_creation_input_tokens": getattr(response.usage, "cache_creation_input_tokens", 0),
        "cache_read_input_tokens": getattr(response.usage, "cache_read_input_tokens", 0),
    }
    input_cost  = usage["input_tokens"]  / 1_000_000 * 5
    output_cost = usage["output_tokens"] / 1_000_000 * 25
    total_cost  = round(input_cost + output_cost, 6)

    # ── Build record ──────────────────────────────────────────────────────────
    record = {
        "meta": {
            "run_at_utc": wall_start,
            "run_end_utc": wall_end,
            "elapsed_seconds": elapsed_sec,
            "model": MODEL,
            "effort": EFFORT,
            "max_tokens": MAX_TOKENS,
        },
        "question": question,
        "api_response": {
            "id": response.id,
            "model": response.model,
            "type": response.type,
            "role": response.role,
            "stop_reason": response.stop_reason,
            "stop_sequence": response.stop_sequence,
            "usage": usage,
            "estimated_cost_usd": total_cost,
            "thinking_blocks": thinking_blocks,
            "text_blocks": text_blocks,
        },
    }

    # ── Save ──────────────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    qid = question["question_id"]
    out_path = RESULTS_DIR / f"{qid}_{MODEL}_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"Elapsed          : {elapsed_sec}s")
    print(f"Input tokens     : {usage['input_tokens']:,}")
    print(f"Output tokens    : {usage['output_tokens']:,}")
    thinking_chars = sum(len(b["thinking"]) for b in thinking_blocks)
    print(f"Thinking length  : {thinking_chars:,} chars")
    print(f"Estimated cost   : ${total_cost:.4f} USD")
    print(f"Stop reason      : {response.stop_reason}")
    print(f"\nGround truth     : {question['ground_truth']}")
    print("\n--- MODEL ANSWER ---")
    for tb in text_blocks:
        print(tb["text"])
    print(f"\nSaved → {out_path}")

    return {"qid": qid, "ground_truth": question["ground_truth"],
            "cost": total_cost, "elapsed": elapsed_sec, "record": record}

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in .env")

    client = anthropic.Anthropic(api_key=api_key)

    # Determine which questions to run
    # Usage: no args = all | Q9 Q11 = specific | --skip Q16 = all except
    args = sys.argv[1:]
    if "--skip" in args:
        skip_idx = args.index("--skip")
        skip_ids = set(args[skip_idx + 1:])
        questions = [q for q in load_all_questions() if q["question_id"] not in skip_ids]
    elif args:
        questions = load_questions(args)
    else:
        questions = load_all_questions()

    print(f"\nRunning {len(questions)} question(s): {[q['question_id'] for q in questions]}")

    results = []
    total_cost = 0.0
    for q in questions:
        r = run_question(client, q)
        results.append(r)
        total_cost += r["cost"]

    # ── Final scorecard (only meaningful when running multiple) ───────────────
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("SCORECARD")
        print(f"{'='*60}")
        print(f"{'QID':<6} {'Ground Truth':<20} {'Elapsed':>10} {'Cost':>10}")
        print("-" * 50)
        for r in results:
            print(f"{r['qid']:<6} {str(r['ground_truth']):<20} {r['elapsed']:>9}s {r['cost']:>9.4f}$")
        print("-" * 50)
        print(f"{'TOTAL':<6} {'':<20} {sum(r['elapsed'] for r in results):>9.1f}s {total_cost:>9.4f}$")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
