"""
Trajectory analyzer for failure mode classification.

Implements the key debugging heuristic from production experience:
- IF correct answer AND used retrieval → Success
- IF correct answer BUT no retrieval → Answer without retrieval (process violation)
- IF wrong answer AND context NOT in trajectory → Retrieval issue
- IF wrong answer AND context IN trajectory → Prompting issue
- IF no search attempted → Critical failure
"""
from typing import Dict, Any, List, Literal
from enum import Enum


class FailureMode(str, Enum):
    """Failure mode classification."""
    SUCCESS = "success"
    RETRIEVAL_FAILURE = "retrieval_failure"
    PROMPT_FOLLOWING_FAILURE = "prompt_following_failure"
    MIXED_FAILURE = "mixed_failure"
    NO_SEARCH_ATTEMPTED = "no_search_attempted"
    ANSWER_WITHOUT_RETRIEVAL = "answer_without_retrieval"  # Agent answered correctly but didn't use retrieval tools


class TrajectoryAnalyzer:
    """
    Analyzes agent trajectories to classify failure modes.

    This is the key innovation: systematic classification of failures to enable
    targeted debugging and improvements.
    """

    def classify_failure_mode(
        self,
        retrieval_grade: Dict[str, Any],
        context_usage_grade: Dict[str, Any],
        answer_quality_grade: Dict[str, Any],
        answerable: bool
    ) -> Dict[str, Any]:
        """
        Classify the failure mode based on grading results.

        Core Logic:
        1. If answer is correct AND search was called → SUCCESS
        2. If answer is correct BUT no search was called → ANSWER_WITHOUT_RETRIEVAL (process violation)
        3. If answer is wrong and no search was attempted → NO_SEARCH_ATTEMPTED
        4. If answer is wrong:
           a. If expected context NOT in trajectory → RETRIEVAL_FAILURE
           b. If expected context IN trajectory but not used correctly → PROMPT_FOLLOWING_FAILURE
           c. If both issues present or unclear → MIXED_FAILURE

        Args:
            retrieval_grade: Results from RetrievalGrader
            context_usage_grade: Results from ContextUsageGrader
            answer_quality_grade: Results from AnswerQualityGrader
            answerable: Whether question should be answerable

        Returns:
            Dictionary with:
            - failure_mode: One of FailureMode enum values
            - reasoning: Explanation of the classification
            - actionable_fix: What should be fixed to improve
        """
        correct = answer_quality_grade["correct"]
        called_search = retrieval_grade["called_search_tool"]
        retrieval_success = retrieval_grade["retrieval_success"]
        context_present = context_usage_grade["context_present"]
        context_used_correctly = context_usage_grade.get("context_used_correctly")

        # ANSWER WITHOUT RETRIEVAL (Correct but didn't search - used pre-training knowledge)
        if correct and not called_search:
            return {
                "failure_mode": FailureMode.ANSWER_WITHOUT_RETRIEVAL,
                "reasoning": "Agent provided correct answer but did NOT use any retrieval tools. This suggests the agent relied on its pre-training knowledge instead of the FAQ database.",
                "actionable_fix": "This is a process violation. Strengthen system prompt to enforce mandatory tool usage. In production, relying on pre-training knowledge can lead to outdated or incorrect information."
            }

        # SUCCESS CASE (Correct answer AND used retrieval)
        if correct:
            return {
                "failure_mode": FailureMode.SUCCESS,
                "reasoning": "Agent provided correct answer using retrieved information from FAQ database",
                "actionable_fix": None
            }

        # NO SEARCH ATTEMPTED (Critical failure)
        if not called_search:
            return {
                "failure_mode": FailureMode.NO_SEARCH_ATTEMPTED,
                "reasoning": "Agent did not call any search tools. Cannot answer without searching the FAQ database.",
                "actionable_fix": "Improve system prompt to ensure agent always searches before answering. This is a prompt-following issue."
            }

        # For answerable questions, classify based on context and usage
        if answerable:
            # Case 1: Context not retrieved (RETRIEVAL FAILURE)
            if not retrieval_success or not context_present:
                return {
                    "failure_mode": FailureMode.RETRIEVAL_FAILURE,
                    "reasoning": f"Expected documents were not retrieved. Retrieved {retrieval_grade['num_retrieved']} docs, but none of the {retrieval_grade['num_expected']} expected docs were found.",
                    "actionable_fix": "Improve retrieval: better embeddings, chunking strategy, or search relevance. Consider query expansion or multi-hop retrieval."
                }

            # Case 2: Context retrieved but not used correctly (PROMPT FOLLOWING FAILURE)
            if context_present and context_used_correctly == False:
                return {
                    "failure_mode": FailureMode.PROMPT_FOLLOWING_FAILURE,
                    "reasoning": f"Expected documents WERE retrieved (recall={retrieval_grade['recall']:.2f}), but agent failed to use them correctly to answer the question.",
                    "actionable_fix": "Improve system prompt: add more explicit instructions about using retrieved context, provide examples of good answers, or reduce prompt complexity."
                }

            # Case 3: Context retrieved but used ambiguously (could be either issue)
            if context_present and context_used_correctly is None:
                return {
                    "failure_mode": FailureMode.MIXED_FAILURE,
                    "reasoning": "Context was retrieved but answer is incorrect. Unable to determine if context was used correctly (LLM judge failed).",
                    "actionable_fix": "Manual review needed. Check if context has the answer (retrieval issue) or if prompt needs improvement (prompt-following issue)."
                }

            # Case 4: Context retrieved and seems used correctly, but still wrong
            # This is tricky - likely the "expected answer" is too strict or the retrieved context
            # doesn't actually contain the answer
            if context_present and context_used_correctly == True:
                return {
                    "failure_mode": FailureMode.MIXED_FAILURE,
                    "reasoning": "Context was retrieved and agent attempted to use it, but answer judged incorrect. Possible issues: retrieved context incomplete, expected answer too strict, or subtle reasoning error.",
                    "actionable_fix": "Manual review needed. Check: (1) Does retrieved context fully answer the question? (2) Is expected answer too strict? (3) Is there a reasoning error?"
                }

        # For unanswerable questions
        else:
            # Agent should recognize it's unanswerable
            # If it searched but said "I don't know" → correct (already handled above)
            # If it searched and gave a wrong answer → prompt following issue
            if called_search:
                return {
                    "failure_mode": FailureMode.PROMPT_FOLLOWING_FAILURE,
                    "reasoning": "Agent searched but provided an answer instead of admitting the information is not available. This is an unanswerable question.",
                    "actionable_fix": "Improve prompt to better handle unanswerable questions: emphasize admitting limitations, provide examples of appropriate 'I don't know' responses."
                }

        # Default fallback
        return {
            "failure_mode": FailureMode.MIXED_FAILURE,
            "reasoning": "Unable to clearly classify failure mode.",
            "actionable_fix": "Manual review needed."
        }

    def generate_failure_report(
        self,
        question: str,
        expected_answer: str,
        agent_answer: str,
        failure_mode: str,
        reasoning: str,
        actionable_fix: str,
        retrieval_grade: Dict[str, Any],
        trajectory_analysis: Dict[str, Any]
    ) -> str:
        """
        Generate a human-readable failure report.

        Args:
            question: Original question
            expected_answer: Expected answer
            agent_answer: Agent's actual answer
            failure_mode: Classified failure mode
            reasoning: Reasoning for classification
            actionable_fix: Suggested fix
            retrieval_grade: Retrieval grading results
            trajectory_analysis: Trajectory analysis with tool calls

        Returns:
            Formatted failure report as string
        """
        sep = '=' * 80
        dash = '-' * 80

        precision_str = f"{retrieval_grade['precision']:.2f}" if retrieval_grade['precision'] is not None else 'N/A'
        recall_str = f"{retrieval_grade['recall']:.2f}" if retrieval_grade['recall'] is not None else 'N/A'

        report = f"""
{sep}
FAILURE REPORT
{sep}

FAILURE MODE: {failure_mode}
{reasoning}

QUESTION:
{question}

EXPECTED ANSWER:
{expected_answer}

AGENT'S ANSWER:
{agent_answer}

{sep}
RETRIEVAL ANALYSIS
{sep}
- Called search tool: {retrieval_grade['called_search_tool']}
- Documents retrieved: {retrieval_grade['num_retrieved']}
- Expected documents: {retrieval_grade['num_expected']}
- Retrieved doc IDs: {retrieval_grade['retrieved_doc_ids']}
- Expected doc IDs: {retrieval_grade['expected_doc_ids']}
- Retrieval precision: {precision_str}
- Retrieval recall: {recall_str}

{sep}
TOOL CALLS
{sep}
"""
        for i, tool_call in enumerate(trajectory_analysis['tool_calls'], 1):
            report += f"\n{i}. {tool_call['tool_name']}"
            report += f"\n   Input: {tool_call['tool_input']}"

        report += f"""

{sep}
ACTIONABLE FIX
{sep}
{actionable_fix if actionable_fix else 'No fix needed - answer was correct.'}

{sep}
"""
        return report


def analyze_trajectory_for_failure(
    question: str,
    expected_answer: str,
    expected_doc_ids: List[str],
    answerable: bool,
    agent_response: Dict[str, Any],
    grading_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Convenience function to analyze trajectory and classify failure mode.

    Args:
        question: Original question
        expected_answer: Ground truth answer
        expected_doc_ids: Expected document IDs
        answerable: Whether question is answerable
        agent_response: Response from agent.ask()
        grading_results: Results from all graders

    Returns:
        Dictionary with failure analysis
    """
    analyzer = TrajectoryAnalyzer()

    # Classify failure mode
    classification = analyzer.classify_failure_mode(
        retrieval_grade=grading_results["retrieval"],
        context_usage_grade=grading_results["context_usage"],
        answer_quality_grade=grading_results["answer_quality"],
        answerable=answerable
    )

    # Generate report if it's a failure (including ANSWER_WITHOUT_RETRIEVAL which is a process violation)
    if classification["failure_mode"] not in [FailureMode.SUCCESS]:
        trajectory_analysis = agent_response.get("trajectory_analysis", {})
        report = analyzer.generate_failure_report(
            question=question,
            expected_answer=expected_answer,
            agent_answer=agent_response["answer"],
            failure_mode=classification["failure_mode"],
            reasoning=classification["reasoning"],
            actionable_fix=classification["actionable_fix"],
            retrieval_grade=grading_results["retrieval"],
            trajectory_analysis=trajectory_analysis
        )
        classification["failure_report"] = report

    return classification
