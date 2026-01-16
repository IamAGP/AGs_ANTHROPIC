"""
FAQ Agent using claude-agent-sdk with MCP tools.
"""
import os
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from claude_agent_sdk import (
    query, create_sdk_mcp_server, ClaudeAgentOptions,
    AssistantMessage, UserMessage, SystemMessage, ResultMessage,
    TextBlock, ToolUseBlock, ToolResultBlock, ThinkingBlock
)
from .prompts import get_system_prompt
from ..tools.retrieval_tools import FAQ_TOOLS
import json

logger = logging.getLogger(__name__)


class FAQAgent:
    """FAQ chatbot agent using Claude Code with MCP retrieval tools."""

    def __init__(
        self,
        prompt_variant: str = "default",
        model: str = "claude-sonnet-4-5-20250929",
        max_tokens: int = 4000
    ):
        """
        Initialize the FAQ agent.

        Args:
            prompt_variant: Which system prompt variant to use
            model: Claude model to use
            max_tokens: Maximum tokens for response
        """
        self.prompt_variant = prompt_variant
        self.model = model
        self.max_tokens = max_tokens
        self.system_prompt = get_system_prompt(prompt_variant)

        # Create MCP server with retrieval tools
        self.mcp_server = create_sdk_mcp_server(
            name="faq-retrieval-tools",
            version="1.0.0",
            tools=FAQ_TOOLS
        )

    async def ask(self, question: str) -> Dict[str, Any]:
        """
        Ask the FAQ agent a question and get a response with full trajectory.

        Args:
            question: User's question

        Returns:
            Dictionary containing:
            - question: The original question
            - answer: Agent's final answer
            - trajectory: Full conversation trajectory including tool calls
            - metadata: Cost, tokens, duration, etc.
        """
        logger.info(f"[AGENT] ask() called with question: '{question}'")

        try:
            # Create agent options with MCP server
            logger.info(f"[AGENT] Creating ClaudeAgentOptions with MCP server")
            options = ClaudeAgentOptions(
                mcp_servers={"faq-tools": self.mcp_server},
                allowed_tools=[
                    "mcp__faq-tools__search_faq",
                    "mcp__faq-tools__get_document",
                    "mcp__faq-tools__search_by_category",
                    "mcp__faq-tools__list_categories"
                ],
                system_prompt=self.system_prompt
            )

            # Collect all messages
            messages = []
            final_text = ""
            result_metadata = {}

            # Workaround for Issue #386: Convert string prompt to async generator
            # String prompts cause "ProcessTransport is not ready for writing" error with SDK MCP servers
            async def prompt_generator():
                yield {"type": "user", "message": {"role": "user", "content": question}}

            # Run query - it returns an AsyncIterator
            logger.info(f"[AGENT] Starting query() call")
            async for message in query(prompt=prompt_generator(), options=options):
                logger.info(f"[AGENT] Received message type: {type(message).__name__}")
                messages.append(message)

                # Extract final text from assistant messages
                if hasattr(message, 'content'):
                    for block in message.content:
                        if hasattr(block, 'text'):
                            final_text += block.text

                # Extract metadata from result message
                if hasattr(message, 'total_cost_usd'):
                    result_metadata = {
                        "total_cost_usd": message.total_cost_usd,
                        "usage": message.usage if hasattr(message, 'usage') else None,
                        "turn_count": message.num_turns if hasattr(message, 'num_turns') else None,
                        "duration_ms": message.duration_ms if hasattr(message, 'duration_ms') else None
                    }

            # Extract trajectory from collected messages
            logger.info(f"[AGENT] Extracting trajectory from {len(messages)} messages")
            trajectory = self._extract_trajectory_from_messages(messages)

            # Format response
            response = {
                "question": question,
                "answer": final_text,
                "trajectory": trajectory,
                "metadata": {
                    "model": self.model,
                    "prompt_variant": self.prompt_variant,
                    **result_metadata
                }
            }

            logger.info(f"[AGENT] Successfully completed ask()")
            return response

        except Exception as e:
            logger.error(f"[AGENT] ERROR in ask(): {e}", exc_info=True)
            raise

    def _extract_trajectory_from_messages(self, messages: List) -> List[Dict[str, Any]]:
        """
        Extract the full trajectory from collected messages.

        This includes all messages, tool calls, and tool results.

        Args:
            messages: List of messages from query() iterator

        Returns:
            List of trajectory steps
        """
        trajectory = []

        # Process each message
        for i, msg in enumerate(messages):
            logger.info(f"[AGENT] Processing message {i}: type={type(msg).__name__}")
            step = None
            content_blocks = []

            # Handle AssistantMessage
            if isinstance(msg, AssistantMessage):
                step = {
                    "role": "assistant",
                    "content": []
                }
                # AssistantMessage.content is a list of ContentBlock objects
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        content_blocks.append({
                            "type": "text",
                            "text": block.text
                        })
                    elif isinstance(block, ToolUseBlock):
                        content_blocks.append({
                            "type": "tool_use",
                            "tool_name": block.name,
                            "tool_input": block.input,
                            "tool_use_id": block.id
                        })
                    elif isinstance(block, ToolResultBlock):
                        content_blocks.append({
                            "type": "tool_result",
                            "tool_use_id": block.tool_use_id,
                            "content": block.content
                        })
                    elif isinstance(block, ThinkingBlock):
                        content_blocks.append({
                            "type": "thinking",
                            "thinking": block.thinking
                        })

            # Handle UserMessage
            elif isinstance(msg, UserMessage):
                step = {
                    "role": "user",
                    "content": []
                }
                if isinstance(msg.content, str):
                    content_blocks.append({
                        "type": "text",
                        "text": msg.content
                    })
                elif isinstance(msg.content, list):
                    for block in msg.content:
                        if isinstance(block, TextBlock):
                            content_blocks.append({
                                "type": "text",
                                "text": block.text
                            })
                        elif isinstance(block, ToolResultBlock):
                            content_blocks.append({
                                "type": "tool_result",
                                "tool_use_id": block.tool_use_id,
                                "content": block.content
                            })

            # Handle SystemMessage (no content blocks)
            elif isinstance(msg, SystemMessage):
                step = {
                    "role": "system",
                    "content": []
                }

            # Handle ResultMessage (no content blocks)
            elif isinstance(msg, ResultMessage):
                step = {
                    "role": "result",
                    "content": []
                }

            # Add step to trajectory if we have one
            if step is not None:
                step["content"] = content_blocks
                trajectory.append(step)
                logger.info(f"[AGENT] Added step with {len(content_blocks)} content blocks")

        return trajectory

    def analyze_trajectory(self, trajectory: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze the trajectory to extract key information for evaluation.

        Args:
            trajectory: Agent trajectory from ask()

        Returns:
            Dictionary with trajectory analysis:
            - tool_calls: List of all tool calls made
            - retrieved_docs: List of all document IDs retrieved
            - reasoning_steps: List of thinking blocks
        """
        analysis = {
            "tool_calls": [],
            "retrieved_docs": [],
            "reasoning_steps": []
        }

        logger.info(f"[AGENT] analyze_trajectory: processing {len(trajectory)} steps")
        for step in trajectory:
            content_blocks = step.get("content", [])
            logger.info(f"[AGENT] Step has {len(content_blocks)} content blocks")
            for content_block in content_blocks:
                block_type = content_block.get("type", "UNKNOWN")
                logger.info(f"[AGENT] Content block type: {block_type}")
                if content_block["type"] == "tool_use":
                    tool_call = {
                        "tool_name": content_block.get("tool_name"),
                        "tool_input": content_block.get("tool_input"),
                        "tool_use_id": content_block.get("tool_use_id")
                    }
                    analysis["tool_calls"].append(tool_call)

                elif content_block["type"] == "tool_result":
                    # Parse tool result to extract document IDs
                    try:
                        result_content = content_block.get("content")
                        if result_content:
                            # Tool results might be text or list
                            if isinstance(result_content, str):
                                result_json = json.loads(result_content)
                            elif isinstance(result_content, list) and len(result_content) > 0:
                                # Sometimes tool results are wrapped in a list with text blocks
                                if isinstance(result_content[0], dict) and 'text' in result_content[0]:
                                    result_json = json.loads(result_content[0]['text'])
                                else:
                                    result_json = result_content
                            else:
                                result_json = result_content

                            # Extract document IDs from results
                            if isinstance(result_json, dict) and "results" in result_json:
                                for doc in result_json["results"]:
                                    if "doc_id" in doc:
                                        analysis["retrieved_docs"].append(doc["doc_id"])
                            elif isinstance(result_json, dict) and "doc_id" in result_json:
                                analysis["retrieved_docs"].append(result_json["doc_id"])
                    except (json.JSONDecodeError, TypeError, AttributeError):
                        pass  # Skip if we can't parse the result

                elif content_block["type"] == "thinking":
                    analysis["reasoning_steps"].append(content_block.get("thinking", ""))

        return analysis


async def run_faq_agent_query(question: str, prompt_variant: str = "default") -> Dict[str, Any]:
    """
    Convenience function to run a single FAQ agent query.

    Args:
        question: User's question
        prompt_variant: Which prompt variant to use

    Returns:
        Response dictionary with answer and trajectory
    """
    agent = FAQAgent(prompt_variant=prompt_variant)
    return await agent.ask(question)
