#!/usr/bin/env python3
"""
Demo script for Agentic RAG Evaluation Framework

Usage:
    python demo.py --help                    # Show help
    python demo.py --num 5                   # Run on 5 questions
    python demo.py --full                    # Run on all questions
    python demo.py --show-failures           # Show detailed failure reports
    python demo.py --compare-prompts         # Compare different prompt variants
"""
import argparse
import asyncio
import json
import logging
from pathlib import Path
from dotenv import load_dotenv
from src.eval.eval_runner import EvaluationRunner
from src.eval.trajectory_analyzer import FailureMode

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('debug.log')
    ]
)


async def run_demo(args):
    """Run the evaluation demo."""
    print("\n" + "="*80)
    print("AGENTIC RAG EVALUATION FRAMEWORK DEMO")
    print("Evaluating Agentic RAG in Production: Lessons from 1.5 Years in the Trenches")
    print("="*80 + "\n")

    if args.compare_prompts:
        print("Running prompt variant comparison...")
        await compare_prompt_variants(args.num)
        return

    # Standard evaluation
    print(f"Configuration:")
    print(f"  - Model: {args.model}")
    print(f"  - Prompt variant: {args.prompt_variant}")
    print(f"  - Questions: {'All' if args.full else args.num}")
    print(f"  - Show failures: {args.show_failures}")
    print()

    # Run evaluation
    runner = EvaluationRunner(
        prompt_variant=args.prompt_variant,
        model=args.model
    )

    num_questions = None if args.full else args.num
    results = await runner.run_evaluation(
        num_questions=num_questions,
        save_results=True,
        output_path=args.output
    )

    # Print summary
    runner.print_summary(results["metrics"])

    # Show detailed failure reports if requested
    if args.show_failures:
        show_failure_examples(results["detailed_results"])


async def compare_prompt_variants(num_questions: int = 10):
    """Compare different prompt variants."""
    variants = ["default", "verbose", "concise", "broken"]
    results_by_variant = {}

    print("\n" + "="*80)
    print("PROMPT VARIANT COMPARISON")
    print("="*80 + "\n")

    for variant in variants:
        print(f"\nEvaluating with prompt variant: {variant}")
        print("-"*80)

        try:
            runner = EvaluationRunner(prompt_variant=variant)
            results = await runner.run_evaluation(
                num_questions=num_questions,
                save_results=True,
                output_path=f"results/eval_results_{variant}.json"
            )

            results_by_variant[variant] = results["metrics"]

        except Exception as e:
            print(f"Error with variant {variant}: {e}")
            results_by_variant[variant] = {"error": str(e)}

    # Print comparison table
    print("\n" + "="*80)
    print("PROMPT VARIANT COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Variant':<15} {'Accuracy':<12} {'Retrieval Fail':<18} {'Prompt Fail':<15}")
    print("-"*80)

    for variant, metrics in results_by_variant.items():
        if "error" in metrics:
            print(f"{variant:<15} ERROR: {metrics['error']}")
        else:
            accuracy = metrics["accuracy"] * 100
            retrieval_fail = metrics["failure_mode_breakdown"].get(FailureMode.RETRIEVAL_FAILURE.value, {}).get("percentage", 0)
            prompt_fail = metrics["failure_mode_breakdown"].get(FailureMode.PROMPT_FOLLOWING_FAILURE.value, {}).get("percentage", 0)

            print(f"{variant:<15} {accuracy:>6.1f}%      {retrieval_fail:>6.1f}%            {prompt_fail:>6.1f}%")

    print("="*80 + "\n")


def show_failure_examples(results: list):
    """Show detailed failure reports for different failure modes."""
    print("\n" + "="*80)
    print("EXAMPLE FAILURE REPORTS")
    print("="*80 + "\n")

    # Group results by failure mode
    by_failure_mode = {}
    for result in results:
        if result["error"] is None:
            mode = result["failure_analysis"]["failure_mode"]
            if mode not in by_failure_mode:
                by_failure_mode[mode] = []
            by_failure_mode[mode].append(result)

    # Show one example of each failure type
    for mode in [FailureMode.RETRIEVAL_FAILURE, FailureMode.PROMPT_FOLLOWING_FAILURE, FailureMode.SUCCESS]:
        mode_str = mode.value
        if mode_str in by_failure_mode and len(by_failure_mode[mode_str]) > 0:
            example = by_failure_mode[mode_str][0]

            print(f"\nExample: {mode_str.upper().replace('_', ' ')}")
            print("-"*80)

            if "failure_report" in example["failure_analysis"]:
                print(example["failure_analysis"]["failure_report"])
            else:
                # For success cases, show a summary
                print(f"Question: {example['question']}")
                print(f"\nExpected: {example['expected_answer']}")
                print(f"\nAgent's Answer: {example['agent_answer']}")
                print(f"\nResult: ✓ CORRECT")
                print("-"*80 + "\n")


async def interactive_mode():
    """Run in interactive mode for testing single questions."""
    from src.agents.faq_agent import FAQAgent
    from src.tools.vector_store import get_vector_store

    print("\n" + "="*80)
    print("INTERACTIVE MODE")
    print("="*80 + "\n")

    # Initialize
    print("Initializing agent and vector store...")
    agent = FAQAgent()
    vector_store = get_vector_store()

    print("\nReady! Ask questions about our FAQ (or type 'quit' to exit)")
    print("-"*80 + "\n")

    while True:
        question = input("Question: ").strip()

        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if not question:
            continue

        try:
            # Get agent response
            response = await agent.ask(question)

            # Analyze trajectory
            trajectory_analysis = agent.analyze_trajectory(response["trajectory"])

            print(f"\nAnswer: {response['answer']}")
            print(f"\nRetrieved docs: {trajectory_analysis['retrieved_docs']}")
            print(f"Tool calls: {len(trajectory_analysis['tool_calls'])}")
            print("-"*80 + "\n")

        except Exception as e:
            print(f"Error: {e}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Demo script for Agentic RAG Evaluation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--num",
        type=int,
        default=5,
        help="Number of questions to evaluate (default: 5)"
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help="Run evaluation on all questions in dataset"
    )

    parser.add_argument(
        "--prompt-variant",
        type=str,
        default="default",
        choices=["default", "verbose", "concise", "broken"],
        help="Which prompt variant to use (default: default)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-5-20250929",
        help="Claude model to use (default: claude-sonnet-4-5-20250929)"
    )

    parser.add_argument(
        "--show-failures",
        action="store_true",
        help="Show detailed failure reports"
    )

    parser.add_argument(
        "--compare-prompts",
        action="store_true",
        help="Compare different prompt variants"
    )

    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode to test individual questions"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="results/eval_results.json",
        help="Output path for results JSON (default: results/eval_results.json)"
    )

    args = parser.parse_args()

    # Run in appropriate mode
    if args.interactive:
        asyncio.run(interactive_mode())
    else:
        asyncio.run(run_demo(args))


if __name__ == "__main__":
    main()
