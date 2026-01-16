"""
Multi-dimensional graders for evaluating agent responses.
"""
from typing import Dict, Any, List, Set
from anthropic import Anthropic
import os
import re
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from ..tools.embeddings import get_embedding_model


class RetrievalGrader:
    """
    Code-based grader for retrieval quality.

    Evaluates:
    - Did the agent call search tools?
    - Were the expected documents retrieved?
    - Precision and recall of retrieval
    """

    def grade(
        self,
        retrieved_doc_ids: List[str],
        expected_doc_ids: List[str],
        tool_calls: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Grade retrieval quality.

        Args:
            retrieved_doc_ids: List of document IDs that were retrieved
            expected_doc_ids: List of document IDs that should have been retrieved
            tool_calls: List of tool calls made by the agent

        Returns:
            Dictionary with retrieval grading results
        """
        # Check if agent called search tools
        # Note: Tool names include MCP prefix like "mcp__faq-tools__search_faq"
        search_tools = ["search_faq", "get_document", "search_by_category"]
        called_search = any(
            any(search_tool in tc.get("tool_name", "") for search_tool in search_tools)
            for tc in tool_calls
        )

        # Calculate precision and recall
        retrieved_set = set(retrieved_doc_ids)
        expected_set = set(expected_doc_ids)

        if len(expected_set) == 0:
            # For unanswerable questions, no docs should be marked as "correct"
            # Good behavior is to retrieve something but recognize it doesn't answer
            precision = None
            recall = None
            retrieval_success = len(retrieved_set) > 0  # Did try to search
        else:
            # Calculate set intersection
            correct_retrievals = retrieved_set.intersection(expected_set)

            precision = len(correct_retrievals) / len(retrieved_set) if len(retrieved_set) > 0 else 0.0
            recall = len(correct_retrievals) / len(expected_set) if len(expected_set) > 0 else 0.0

            # Success if recall > 0 (at least one expected doc was retrieved)
            retrieval_success = recall > 0

        return {
            "called_search_tool": called_search,
            "num_retrieved": len(retrieved_doc_ids),
            "num_expected": len(expected_doc_ids),
            "retrieved_doc_ids": retrieved_doc_ids,
            "expected_doc_ids": expected_doc_ids,
            "precision": precision,
            "recall": recall,
            "retrieval_success": retrieval_success
        }


class ContextUsageGrader:
    """
    Hybrid grader (code + model-based) for context usage.

    Evaluates:
    - Was necessary context present in retrieved documents?
    - Did the agent use the retrieved context correctly?
    """

    def __init__(self):
        """Initialize with Anthropic client for LLM-as-judge."""
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def grade(
        self,
        agent_answer: str,
        retrieved_docs: List[Dict[str, str]],
        expected_answer: str,
        question: str
    ) -> Dict[str, Any]:
        """
        Grade context usage.

        Args:
            agent_answer: The agent's final answer
            retrieved_docs: Full documents that were retrieved
            expected_answer: Ground truth expected answer
            question: Original question

        Returns:
            Dictionary with context usage grading results
        """
        # Check if context was present
        context_present = len(retrieved_docs) > 0

        if not context_present:
            return {
                "context_present": False,
                "context_used_correctly": None,
                "groundedness_score": 0.0
            }

        # Combine retrieved document content
        retrieved_context = "\n\n".join([
            f"Document {doc['id']}: {doc['content']}"
            for doc in retrieved_docs
        ])

        # Use LLM to judge if context was used correctly
        context_usage_result = self._judge_context_usage(
            question=question,
            retrieved_context=retrieved_context,
            agent_answer=agent_answer,
            expected_answer=expected_answer
        )

        # Use LLM to judge groundedness
        groundedness_score = self._judge_groundedness(
            agent_answer=agent_answer,
            retrieved_context=retrieved_context
        )

        return {
            "context_present": True,
            "context_used_correctly": context_usage_result,
            "groundedness_score": groundedness_score
        }

    def _judge_context_usage(
        self,
        question: str,
        retrieved_context: str,
        agent_answer: str,
        expected_answer: str
    ) -> bool:
        """
        Use LLM to judge if the agent used the retrieved context correctly.

        Returns:
            True if context was used correctly, False otherwise
        """
        prompt = f"""You are evaluating whether an AI agent correctly used retrieved context to answer a question.

Question: {question}

Retrieved Context:
{retrieved_context}

Expected Answer: {expected_answer}

Agent's Answer: {agent_answer}

Did the agent correctly use the information from the retrieved context to answer the question?

Consider:
- Did the agent reference information from the context?
- Is the answer consistent with the context?
- Did the agent ignore relevant information that was available?

Respond with ONLY "YES" or "NO"."""

        try:
            response = self.client.messages.create(
                model="claude-3-5-haiku-20241022",  # Fast, cheap model for grading
                max_tokens=10,
                messages=[{"role": "user", "content": prompt}]
            )

            answer = response.content[0].text.strip().upper()
            return "YES" in answer

        except Exception as e:
            print(f"Error in LLM context usage judgment: {e}")
            return None

    def _judge_groundedness(self, agent_answer: str, retrieved_context: str) -> float:
        """
        Use LLM to judge how well the answer is grounded in retrieved context.

        Returns:
            Score from 0.0 to 1.0 indicating groundedness
        """
        prompt = f"""You are evaluating whether an AI answer is grounded in (supported by) the retrieved context.

Retrieved Context:
{retrieved_context}

Agent's Answer: {agent_answer}

Is the agent's answer fully supported by the retrieved context? Rate the groundedness on a scale:
1.0 = Fully grounded, all claims are supported by context
0.75 = Mostly grounded, minor unsupported details
0.5 = Partially grounded, some claims not in context
0.25 = Minimally grounded, mostly unsupported
0.0 = Not grounded, answer contradicts or ignores context

Respond with ONLY a number (0.0, 0.25, 0.5, 0.75, or 1.0)."""

        try:
            response = self.client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=10,
                messages=[{"role": "user", "content": prompt}]
            )

            score_text = response.content[0].text.strip()
            # Extract first number from response
            match = re.search(r'(0\.\d+|1\.0|0|1)', score_text)
            if match:
                return float(match.group(1))
            return 0.5  # Default to middle score if parsing fails

        except Exception as e:
            print(f"Error in LLM groundedness judgment: {e}")
            return 0.5


class AnswerQualityGrader:
    """
    Hybrid grader for answer quality.

    Uses multiple methods:
    - Exact match (strict)
    - Semantic similarity (embedding-based)
    - LLM-as-judge (flexible, rubric-based)
    """

    def __init__(self):
        """Initialize with embedding model and Anthropic client."""
        self.embedding_model = get_embedding_model()
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def grade(
        self,
        agent_answer: str,
        expected_answer: str,
        question: str,
        answerable: bool
    ) -> Dict[str, Any]:
        """
        Grade answer quality using multiple methods.

        Args:
            agent_answer: The agent's final answer
            expected_answer: Ground truth expected answer
            question: Original question
            answerable: Whether the question should be answerable from corpus

        Returns:
            Dictionary with answer quality grading results
        """
        # Exact match (case-insensitive, whitespace normalized)
        exact_match = self._check_exact_match(agent_answer, expected_answer)

        # Semantic similarity
        semantic_similarity = self._calculate_semantic_similarity(agent_answer, expected_answer)

        # LLM-as-judge
        llm_correct, llm_reasoning = self._llm_judge(
            question=question,
            agent_answer=agent_answer,
            expected_answer=expected_answer,
            answerable=answerable
        )

        # Overall correctness (LLM judge is the source of truth)
        correct = llm_correct

        return {
            "correct": correct,
            "exact_match": exact_match,
            "semantic_similarity": semantic_similarity,
            "llm_judge_correct": llm_correct,
            "llm_judge_reasoning": llm_reasoning
        }

    def _check_exact_match(self, answer1: str, answer2: str) -> bool:
        """Check if two answers match exactly (normalized)."""
        norm1 = " ".join(answer1.lower().split())
        norm2 = " ".join(answer2.lower().split())
        return norm1 == norm2

    def _calculate_semantic_similarity(self, answer1: str, answer2: str) -> float:
        """Calculate semantic similarity between two answers using embeddings."""
        try:
            emb1 = self.embedding_model.embed(answer1)
            emb2 = self.embedding_model.embed(answer2)
            similarity = self.embedding_model.cosine_similarity(emb1, emb2)
            return float(similarity)
        except Exception as e:
            print(f"Error calculating semantic similarity: {e}")
            return 0.0

    def _llm_judge(
        self,
        question: str,
        agent_answer: str,
        expected_answer: str,
        answerable: bool
    ) -> tuple[bool, str]:
        """
        Use LLM as judge to determine if answer is correct.

        Returns:
            Tuple of (is_correct: bool, reasoning: str)
        """
        if answerable:
            prompt = f"""You are evaluating whether an AI agent answered a customer support question correctly.

Question: {question}

Expected Answer (Ground Truth): {expected_answer}

Agent's Answer: {agent_answer}

Is the agent's answer correct? The answer doesn't need to match word-for-word, but it should convey the same key information.

Respond in this format:
CORRECT: [YES or NO]
REASONING: [Brief explanation of your judgment]"""
        else:
            # For unanswerable questions, check if agent admitted lack of knowledge
            prompt = f"""You are evaluating whether an AI agent handled an unanswerable question correctly.

Question: {question}

This question is NOT answerable from the FAQ database.

Agent's Answer: {agent_answer}

Did the agent correctly indicate that they don't have this information? Look for phrases like:
- "I don't have information about..."
- "This is not covered in our FAQ"
- "I cannot find..."

The agent should NOT make up an answer or provide unrelated information.

Respond in this format:
CORRECT: [YES or NO]
REASONING: [Brief explanation]"""

        try:
            response = self.client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}]
            )

            result_text = response.content[0].text.strip()

            # Parse response
            correct = "CORRECT: YES" in result_text.upper()

            # Extract reasoning
            reasoning_match = re.search(r'REASONING:\s*(.+)', result_text, re.IGNORECASE | re.DOTALL)
            reasoning = reasoning_match.group(1).strip() if reasoning_match else result_text

            return correct, reasoning

        except Exception as e:
            print(f"Error in LLM judge: {e}")
            return False, f"Error: {str(e)}"


def get_graders() -> Dict[str, Any]:
    """
    Get instances of all graders.

    Returns:
        Dictionary mapping grader names to grader instances
    """
    return {
        "retrieval": RetrievalGrader(),
        "context_usage": ContextUsageGrader(),
        "answer_quality": AnswerQualityGrader()
    }
