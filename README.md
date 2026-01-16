# Agentic RAG Evaluation Framework

**Evaluating Agentic RAG in Production: Lessons from 1.5 Years in the Trenches**

A systematic evaluation framework for agentic RAG systems, demonstrating trajectory-based debugging to separate retrieval failures from prompt-following failures.

## The Key Insight

From 1.5 years of production experience with agentic RAG workflows (RFP response generation, customer FAQ chatbots), we developed a simple but powerful debugging heuristic:

```
IF context present in trajectory BUT wrong answer → Prompting issue
IF context NOT in trajectory → Retrieval issue
IF correct answer → Success
```

This framework systematizes that insight with automated grading and failure classification.

## Architecture

- **Agent**: Claude Code (ReAct-style reasoning via claude-agent-sdk)
- **Tools**: Custom retrieval tools (vector search, document lookup) via MCP
- **Knowledge Base**: Synthetic FAQ documents (e-commerce customer support)
- **Evaluation**: Multi-dimensional grading with trajectory analysis

## Features

### Multi-Dimensional Grading

1. **Retrieval Grader** (Code-based)
   - Did agent call search tools?
   - Were expected documents retrieved?
   - Precision and recall of retrieval

2. **Context Usage Grader** (Model-based)
   - Was necessary context present in trajectory?
   - Did agent use retrieved context correctly?
   - Groundedness score

3. **Answer Quality Grader** (Hybrid)
   - Exact match
   - Semantic similarity
   - LLM-as-judge for correctness

### Failure Mode Classification

Automatic classification into:
- `SUCCESS`: Correct answer
- `RETRIEVAL_FAILURE`: Expected docs not retrieved
- `PROMPT_FOLLOWING_FAILURE`: Context retrieved but not used correctly
- `NO_SEARCH_ATTEMPTED`: Agent didn't search
- `MIXED_FAILURE`: Multiple issues or unclear

Each failure mode includes actionable fixes:
- Retrieval failures → Improve embeddings, chunking, search relevance
- Prompt failures → Improve system prompt, add examples, reduce complexity

## Quick Start

### Installation

```bash
# Dependencies already installed via uv
# See pyproject.toml for full list
```

### Run Evaluation

```bash
# Quick test (5 questions)
uv run python demo.py --num 5

# Full evaluation (all 25 questions)
uv run python demo.py --full

# Show detailed failure reports
uv run python demo.py --num 10 --show-failures

# Compare prompt variants
uv run python demo.py --compare-prompts

# Interactive mode
uv run python demo.py --interactive
```

## Project Structure

```
.
├── src/
│   ├── data/
│   │   ├── faq_corpus.py          # Generate synthetic FAQ docs
│   │   └── test_questions.py      # Generate test dataset
│   ├── tools/
│   │   ├── embeddings.py          # Embedding generation
│   │   ├── vector_store.py        # In-memory vector store
│   │   └── retrieval_tools.py     # MCP tools for retrieval
│   ├── agents/
│   │   ├── prompts.py             # System prompts
│   │   └── faq_agent.py           # FAQ agent with claude-agent-sdk
│   └── eval/
│       ├── graders.py             # Multi-dimensional graders
│       ├── trajectory_analyzer.py # Failure mode classification
│       └── eval_runner.py         # Main evaluation harness
├── data/
│   ├── faq_docs.json              # FAQ corpus (50 documents)
│   └── eval_dataset.json          # Test questions (25 questions)
├── demo.py                         # Main demo script
└── README.md
```

## Dataset

### FAQ Corpus (50 documents)
- **Returns & Refunds**: 10 docs
- **Shipping & Tracking**: 10 docs
- **Account Management**: 10 docs
- **Product Information**: 10 docs
- **Payment & Billing**: 10 docs

### Test Questions (25 questions)
- **Single-hop**: 15 questions (60%) - Answerable from single document
- **Multi-hop**: 5 questions (20%) - Require reasoning across multiple documents
- **Unanswerable**: 5 questions (20%) - Not in FAQ corpus (should say "I don't know")

## Example Output

```
================================================================================
EVALUATION SUMMARY
================================================================================
Total Questions: 25
Errors: 0
Valid Results: 25

Accuracy: 84.00% (21/25)

--------------------------------------------------------------------------------
FAILURE MODE BREAKDOWN
--------------------------------------------------------------------------------
success                       : 21 ( 84.0%)
retrieval_failure            :  2 (  8.0%)
prompt_following_failure     :  2 (  8.0%)

--------------------------------------------------------------------------------
RETRIEVAL METRICS
--------------------------------------------------------------------------------
Average Precision: 0.889
Average Recall:    0.933

--------------------------------------------------------------------------------
CONTEXT USAGE METRICS
--------------------------------------------------------------------------------
Average Groundedness: 0.925
================================================================================
```

## Key Concepts from Anthropic's Evals Blog

This framework implements several concepts from Anthropic's "Demystifying Evals for AI Agents" blog:

1. **Trajectory Analysis**: Read the full agent trace, not just final outputs
2. **Multi-dimensional Grading**: Separate retrieval, reasoning, and output quality
3. **LLM-as-Judge**: Use models for nuanced evaluation
4. **Failure Mode Classification**: Systematic categorization of errors
5. **Actionable Fixes**: Each failure mode suggests specific improvements

## Extending the Framework

### Add New Prompt Variants

Edit `src/agents/prompts.py`:

```python
FAQ_AGENT_PROMPT_VARIANTS["my_variant"] = """
Your custom prompt here...
"""
```

Then run:
```bash
uv run python demo.py --prompt-variant my_variant
```

### Add New Test Cases

Edit `src/data/test_questions.py` and regenerate:

```bash
uv run python src/data/test_questions.py
```

### Use Different Embedding Models

Edit `src/tools/embeddings.py`:

```python
EmbeddingModel(model_name="all-mpnet-base-v2")  # Larger, more accurate
```

## Blog Post Outline

**Title**: "Evaluating Agentic RAG in Production: Lessons from 1.5 Years in the Trenches"

1. **The Problem**: Manual testing doesn't scale, prompt changes break things
2. **The Key Insight**: Trajectory-based debugging (retrieval vs prompting failures)
3. **Building the Framework**: Multi-dimensional grading, automated classification
4. **Real Examples**: Show actual trajectories and failure modes
5. **Results**: Metrics and lessons learned
6. **Takeaways**: Start small, read transcripts, treat evals as living artifacts

## Future Enhancements

- [ ] Statistical metrics (pass@k, pass^k) for non-determinism
- [ ] MS MARCO validation (show framework generalizes)
- [ ] Continuous monitoring integration
- [ ] A/B testing framework
- [ ] Production deployment guide

## Credits

Based on 1.5 years of production experience with LangGraph agentic RAG workflows for RFP response generation and customer FAQ chatbots.

Inspired by Anthropic's "Demystifying Evals for AI Agents" blog post (January 2026).

Built with:
- [Anthropic Claude](https://anthropic.com) - LLM and agent capabilities
- [claude-agent-sdk](https://github.com/anthropics/claude-agent-sdk-python) - Agent framework
- [sentence-transformers](https://www.sbert.net/) - Embeddings

## License

MIT
