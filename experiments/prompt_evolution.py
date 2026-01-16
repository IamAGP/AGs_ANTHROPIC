#!/usr/bin/env python3
"""
Prompt Evolution Experiment: Side-by-side comparison of prompt variants.

This experiment runs the same eval dataset with different prompt variants to show
how prompt engineering affects agent behavior in production.

NO ASSUMPTIONS: Uses real eval runs, no fudging of results.
"""
import asyncio
import json
from pathlib import Path
from typing import Dict, Any, List
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from src.eval.eval_runner import run_evaluation


async def run_prompt_comparison(
    prompt_variants: List[str],
    num_questions: int = None,
    output_dir: str = "experiments/results"
) -> Dict[str, Any]:
    """
    Run evaluation with different prompt variants side-by-side.

    Args:
        prompt_variants: List of prompt variant names to compare
        num_questions: Number of questions to test (None = all questions)
        output_dir: Where to save results

    Returns:
        Dictionary mapping variant name to evaluation results
    """
    results = {}

    print("=" * 80)
    print("PROMPT EVOLUTION EXPERIMENT")
    print("=" * 80)
    print(f"\nTesting {len(prompt_variants)} prompt variants:")
    for variant in prompt_variants:
        print(f"  - {variant}")
    print(f"\nQuestions to test: {num_questions if num_questions else 'ALL'}")
    print("=" * 80)

    # Run evaluation for each prompt variant
    for i, variant in enumerate(prompt_variants, 1):
        print(f"\n[{i}/{len(prompt_variants)}] Running evaluation with '{variant}' prompt...")
        print("-" * 80)

        try:
            eval_results = await run_evaluation(
                prompt_variant=variant,
                num_questions=num_questions
            )
            results[variant] = eval_results

            # Show quick summary
            metrics = eval_results["metrics"]
            print(f"\n✓ Completed '{variant}':")
            print(f"  Accuracy: {metrics['accuracy']:.1%}")

            avg_prec = metrics['retrieval_metrics']['avg_precision']
            avg_rec = metrics['retrieval_metrics']['avg_recall']
            print(f"  Avg Precision: {avg_prec if avg_prec is not None else 'N/A'}")
            print(f"  Avg Recall: {avg_rec if avg_rec is not None else 'N/A'}")

        except Exception as e:
            print(f"\n✗ Error running '{variant}': {e}")
            results[variant] = {"error": str(e)}

    # Save raw results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results_file = output_path / "prompt_comparison_raw.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Raw results saved to: {results_file}")

    return results


def generate_comparison_table(results: Dict[str, Any]) -> str:
    """
    Generate a markdown comparison table from results.

    Args:
        results: Dictionary mapping variant name to evaluation results

    Returns:
        Markdown formatted comparison table
    """
    table = "# Prompt Evolution: Side-by-Side Comparison\n\n"
    table += "## Summary Metrics\n\n"
    table += "| Prompt Variant | Accuracy | Avg Precision | Avg Recall | Answer w/o Retrieval | Tool Usage Rate |\n"
    table += "|---------------|----------|---------------|------------|---------------------|----------------|\n"

    for variant, result in results.items():
        if "error" in result:
            table += f"| {variant} | ERROR | - | - | - | - |\n"
            continue

        metrics = result["metrics"]
        failure_breakdown = metrics["failure_mode_breakdown"]

        # Calculate tool usage rate (1.0 - answer_without_retrieval_rate)
        answer_without_retrieval = failure_breakdown.get("answer_without_retrieval", {}).get("count", 0)
        total = metrics["total_questions"]
        tool_usage_rate = 1.0 - (answer_without_retrieval / total if total > 0 else 0)

        avg_precision = metrics["retrieval_metrics"]["avg_precision"]
        avg_recall = metrics["retrieval_metrics"]["avg_recall"]

        table += f"| {variant} | {metrics['accuracy']:.1%} | "
        table += f"{avg_precision if avg_precision is not None else 'N/A'} | "
        table += f"{avg_recall if avg_recall is not None else 'N/A'} | "
        table += f"{answer_without_retrieval}/{total} | "
        table += f"{tool_usage_rate:.1%} |\n"

    # Add failure mode breakdown
    table += "\n## Failure Mode Breakdown\n\n"
    table += "| Prompt Variant | Success | Retrieval Failure | Prompt Following | Answer w/o Retrieval | No Search |\n"
    table += "|---------------|---------|-------------------|------------------|---------------------|----------|\n"

    for variant, result in results.items():
        if "error" in result:
            continue

        breakdown = result["metrics"]["failure_mode_breakdown"]
        table += f"| {variant} | "
        table += f"{breakdown.get('success', {}).get('count', 0)} | "
        table += f"{breakdown.get('retrieval_failure', {}).get('count', 0)} | "
        table += f"{breakdown.get('prompt_following_failure', {}).get('count', 0)} | "
        table += f"{breakdown.get('answer_without_retrieval', {}).get('count', 0)} | "
        table += f"{breakdown.get('no_search_attempted', {}).get('count', 0)} |\n"

    return table


async def main():
    """Run prompt evolution experiment."""

    # Define which prompts to compare
    # Include misleading prompt to test answer_without_retrieval failure mode
    prompt_variants = ["misleading", "broken", "default", "verbose"]

    print("\nStarting Prompt Evolution Experiment...")
    print("This will run the SAME eval dataset with different prompts.\n")

    # Default to 15 questions for meaningful comparison
    num_questions = 15

    print(f"\nRunning with {num_questions} questions per variant...")
    print(f"Total evals to run: {len(prompt_variants)} variants × {num_questions} questions = {len(prompt_variants) * num_questions} evals\n")

    # Run the comparison
    results = await run_prompt_comparison(
        prompt_variants=prompt_variants,
        num_questions=num_questions
    )

    # Generate comparison table
    table = generate_comparison_table(results)

    # Save markdown report
    report_path = Path("experiments/results/prompt_comparison_report.md")
    with open(report_path, 'w') as f:
        f.write(table)

    print(f"\n✓ Comparison report saved to: {report_path}")
    print("\n" + "=" * 80)
    print("RESULTS PREVIEW")
    print("=" * 80)
    print(table)


if __name__ == "__main__":
    asyncio.run(main())
