"""
Evaluation runner for agentic RAG system.
"""
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from tqdm import tqdm
import time

from ..agents.faq_agent import FAQAgent
from ..tools.vector_store import get_vector_store
from .graders import get_graders
from .trajectory_analyzer import analyze_trajectory_for_failure, FailureMode

logger = logging.getLogger(__name__)


class EvaluationRunner:
    """Main evaluation harness for the FAQ agent."""

    def __init__(
        self,
        test_dataset_path: str = "data/eval_dataset.json",
        prompt_variant: str = "default",
        model: str = "claude-sonnet-4-5-20250929"
    ):
        """
        Initialize the evaluation runner.

        Args:
            test_dataset_path: Path to test dataset JSON
            prompt_variant: Which prompt variant to use
            model: Claude model to use
        """
        self.test_dataset_path = test_dataset_path
        self.prompt_variant = prompt_variant
        self.model = model

        # Load test dataset
        with open(test_dataset_path, 'r', encoding='utf-8') as f:
            self.test_cases = json.load(f)

        print(f"Loaded {len(self.test_cases)} test cases")

        # Initialize agent
        self.agent = FAQAgent(prompt_variant=prompt_variant, model=model)

        # Initialize graders
        self.graders = get_graders()

        # Initialize vector store (for retrieving full documents)
        self.vector_store = get_vector_store()

        # Results storage
        self.results: List[Dict[str, Any]] = []

    async def run_evaluation(
        self,
        num_questions: Optional[int] = None,
        save_results: bool = True,
        output_path: str = "results/eval_results.json"
    ) -> Dict[str, Any]:
        """
        Run evaluation on the test dataset.

        Args:
            num_questions: If specified, only run on first N questions
            save_results: Whether to save detailed results to file
            output_path: Path to save results JSON

        Returns:
            Dictionary with aggregate metrics and detailed results
        """
        # Select questions to evaluate
        questions_to_eval = self.test_cases[:num_questions] if num_questions else self.test_cases

        print(f"\nRunning evaluation on {len(questions_to_eval)} questions...")
        print(f"Model: {self.model}")
        print(f"Prompt variant: {self.prompt_variant}\n")

        self.results = []
        start_time = time.time()

        # Run each test case
        for test_case in tqdm(questions_to_eval, desc="Evaluating"):
            result = await self._evaluate_single_question(test_case)
            self.results.append(result)

        total_time = time.time() - start_time

        # Calculate aggregate metrics
        metrics = self._calculate_metrics()
        metrics["total_time_seconds"] = total_time
        metrics["avg_time_per_question"] = total_time / len(questions_to_eval)

        # Save results if requested
        if save_results:
            self._save_results(output_path, metrics)

        return {
            "metrics": metrics,
            "detailed_results": self.results
        }

    async def _evaluate_single_question(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate agent on a single question.

        Args:
            test_case: Test case dictionary from dataset

        Returns:
            Dictionary with evaluation results for this question
        """
        question = test_case["question"]
        expected_answer = test_case["expected_answer"]
        expected_doc_ids = test_case["expected_doc_ids"]
        answerable = test_case["answerable"]

        logger.info(f"[EVAL] Evaluating question: '{question}'")

        # Run agent
        try:
            logger.info(f"[EVAL] Calling agent.ask()")
            agent_response = await self.agent.ask(question)
            logger.info(f"[EVAL] Agent returned response")
            agent_answer = agent_response["answer"]

            # Analyze trajectory
            trajectory_analysis = self.agent.analyze_trajectory(agent_response["trajectory"])
            agent_response["trajectory_analysis"] = trajectory_analysis

            # Get full documents that were retrieved
            retrieved_docs = [
                self.vector_store.get_document(doc_id)
                for doc_id in trajectory_analysis["retrieved_docs"]
                if self.vector_store.get_document(doc_id) is not None
            ]

            # Grade retrieval
            retrieval_grade = self.graders["retrieval"].grade(
                retrieved_doc_ids=trajectory_analysis["retrieved_docs"],
                expected_doc_ids=expected_doc_ids,
                tool_calls=trajectory_analysis["tool_calls"]
            )

            # Grade context usage
            context_usage_grade = self.graders["context_usage"].grade(
                agent_answer=agent_answer,
                retrieved_docs=retrieved_docs,
                expected_answer=expected_answer,
                question=question
            )

            # Grade answer quality
            answer_quality_grade = self.graders["answer_quality"].grade(
                agent_answer=agent_answer,
                expected_answer=expected_answer,
                question=question,
                answerable=answerable
            )

            grading_results = {
                "retrieval": retrieval_grade,
                "context_usage": context_usage_grade,
                "answer_quality": answer_quality_grade
            }

            # Classify failure mode
            failure_analysis = analyze_trajectory_for_failure(
                question=question,
                expected_answer=expected_answer,
                expected_doc_ids=expected_doc_ids,
                answerable=answerable,
                agent_response=agent_response,
                grading_results=grading_results
            )

            return {
                "test_id": test_case["id"],
                "question": question,
                "expected_answer": expected_answer,
                "agent_answer": agent_answer,
                "answerable": answerable,
                "grading": grading_results,
                "failure_analysis": failure_analysis,
                "metadata": agent_response.get("metadata", {}),
                "error": None
            }

        except Exception as e:
            logger.error(f"[EVAL] Error evaluating question '{question}': {e}", exc_info=True)
            print(f"\nError evaluating question '{question}': {e}")
            return {
                "test_id": test_case["id"],
                "question": question,
                "expected_answer": expected_answer,
                "agent_answer": None,
                "answerable": answerable,
                "grading": None,
                "failure_analysis": {
                    "failure_mode": "error",
                    "reasoning": str(e)
                },
                "metadata": {},
                "error": str(e)
            }

    def _calculate_metrics(self) -> Dict[str, Any]:
        """Calculate aggregate metrics from all results."""
        total = len(self.results)
        errors = sum(1 for r in self.results if r["error"] is not None)
        valid_results = [r for r in self.results if r["error"] is None]

        if len(valid_results) == 0:
            return {"error": "No valid results to calculate metrics"}

        # Overall accuracy
        correct = sum(
            1 for r in valid_results
            if r["grading"]["answer_quality"]["correct"]
        )
        accuracy = correct / len(valid_results)

        # Failure mode breakdown
        failure_modes = {}
        for mode in FailureMode:
            count = sum(
                1 for r in valid_results
                if r["failure_analysis"]["failure_mode"] == mode.value
            )
            failure_modes[mode.value] = {
                "count": count,
                "percentage": (count / len(valid_results)) * 100
            }

        # Retrieval metrics (for answerable questions only)
        answerable_results = [r for r in valid_results if r["answerable"]]
        if answerable_results:
            avg_retrieval_precision = sum(
                r["grading"]["retrieval"]["precision"]
                for r in answerable_results
                if r["grading"]["retrieval"]["precision"] is not None
            ) / len(answerable_results)

            avg_retrieval_recall = sum(
                r["grading"]["retrieval"]["recall"]
                for r in answerable_results
                if r["grading"]["retrieval"]["recall"] is not None
            ) / len(answerable_results)
        else:
            avg_retrieval_precision = None
            avg_retrieval_recall = None

        # Context usage (for answerable questions where context was present)
        context_present_results = [
            r for r in answerable_results
            if r["grading"]["context_usage"]["context_present"]
        ]
        if context_present_results:
            avg_groundedness = sum(
                r["grading"]["context_usage"]["groundedness_score"]
                for r in context_present_results
            ) / len(context_present_results)
        else:
            avg_groundedness = None

        # Semantic similarity
        avg_semantic_similarity = sum(
            r["grading"]["answer_quality"]["semantic_similarity"]
            for r in valid_results
        ) / len(valid_results)

        return {
            "total_questions": total,
            "errors": errors,
            "valid_results": len(valid_results),
            "accuracy": accuracy,
            "correct_answers": correct,
            "failure_mode_breakdown": failure_modes,
            "retrieval_metrics": {
                "avg_precision": avg_retrieval_precision,
                "avg_recall": avg_retrieval_recall
            },
            "context_usage_metrics": {
                "avg_groundedness": avg_groundedness
            },
            "semantic_similarity": {
                "average": avg_semantic_similarity
            }
        }

    def _save_results(self, output_path: str, metrics: Dict[str, Any]):
        """Save evaluation results to file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        results_data = {
            "config": {
                "model": self.model,
                "prompt_variant": self.prompt_variant,
                "test_dataset": self.test_dataset_path,
                "num_questions": len(self.results)
            },
            "metrics": metrics,
            "detailed_results": self.results
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved to {output_path}")

    def print_summary(self, metrics: Dict[str, Any]):
        """Print a formatted summary of evaluation results."""
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)

        # Handle error case
        if "error" in metrics:
            print(f"ERROR: {metrics['error']}")
            print("="*80 + "\n")
            return

        print(f"Total Questions: {metrics['total_questions']}")
        print(f"Errors: {metrics['errors']}")
        print(f"Valid Results: {metrics['valid_results']}")
        print(f"\nAccuracy: {metrics['accuracy']:.2%} ({metrics['correct_answers']}/{metrics['valid_results']})")

        print("\n" + "-"*80)
        print("FAILURE MODE BREAKDOWN")
        print("-"*80)
        for mode, stats in metrics['failure_mode_breakdown'].items():
            print(f"{mode:30s}: {stats['count']:3d} ({stats['percentage']:5.1f}%)")

        if metrics['retrieval_metrics']['avg_precision'] is not None:
            print("\n" + "-"*80)
            print("RETRIEVAL METRICS")
            print("-"*80)
            print(f"Average Precision: {metrics['retrieval_metrics']['avg_precision']:.3f}")
            print(f"Average Recall:    {metrics['retrieval_metrics']['avg_recall']:.3f}")

        if metrics['context_usage_metrics']['avg_groundedness'] is not None:
            print("\n" + "-"*80)
            print("CONTEXT USAGE METRICS")
            print("-"*80)
            print(f"Average Groundedness: {metrics['context_usage_metrics']['avg_groundedness']:.3f}")

        print("\n" + "-"*80)
        print("PERFORMANCE")
        print("-"*80)
        print(f"Total Time: {metrics['total_time_seconds']:.1f}s")
        print(f"Avg Time per Question: {metrics['avg_time_per_question']:.1f}s")
        print("="*80 + "\n")


async def run_evaluation(
    num_questions: Optional[int] = None,
    prompt_variant: str = "default",
    model: str = "claude-sonnet-4-5-20250929"
) -> Dict[str, Any]:
    """
    Convenience function to run evaluation.

    Args:
        num_questions: Number of questions to evaluate (None for all)
        prompt_variant: Which prompt variant to use
        model: Claude model to use

    Returns:
        Dictionary with metrics and results
    """
    runner = EvaluationRunner(
        prompt_variant=prompt_variant,
        model=model
    )

    results = await runner.run_evaluation(num_questions=num_questions)
    runner.print_summary(results["metrics"])

    return results
