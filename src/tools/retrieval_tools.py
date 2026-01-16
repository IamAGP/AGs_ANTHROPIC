"""
MCP tools for FAQ retrieval using claude-agent-sdk.
"""
from claude_agent_sdk import tool
from .vector_store import get_vector_store
from typing import List, Dict, Any
import json
import logging

logger = logging.getLogger(__name__)


@tool("search_faq", "Search the FAQ database for documents relevant to the query using semantic search", {"query": str, "top_k": int})
async def search_faq(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Search the FAQ database for documents relevant to the query using semantic search.
    """
    logger.info(f"[TOOL] search_faq called with args: {args}")
    try:
        query = args["query"]
        top_k = args.get("top_k", 3)
        logger.info(f"[TOOL] search_faq - query: '{query}', top_k: {top_k}")

        # Validate top_k
        if top_k < 1:
            top_k = 1
        if top_k > 10:
            top_k = 10

        # Get vector store
        store = get_vector_store()

        # Perform search
        results = store.search(query, top_k=top_k)
        logger.info(f"[TOOL] search_faq - found {len(results)} results")

        # Format results for the agent
        formatted_results = {
            "query": query,
            "num_results": len(results),
            "results": []
        }

        for result in results:
            formatted_results["results"].append({
                "doc_id": result['id'],
                "title": result['title'],
                "category": result['category'],
                "content": result['content'],
                "similarity_score": round(result['similarity_score'], 4),
                "rank": result['rank']
            })

        logger.info(f"[TOOL] search_faq - returning results")
        return {
            "content": [{
                "type": "text",
                "text": json.dumps(formatted_results, indent=2)
            }]
        }
    except Exception as e:
        logger.error(f"[TOOL] search_faq ERROR: {e}", exc_info=True)
        raise


@tool("get_document", "Retrieve a specific FAQ document by its ID", {"doc_id": str})
async def get_document(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Retrieve a specific FAQ document by its ID.
    """
    doc_id = args["doc_id"]
    store = get_vector_store()
    doc = store.get_document(doc_id)

    if doc is None:
        result = {
            "error": f"Document with ID '{doc_id}' not found",
            "doc_id": doc_id
        }
    else:
        result = {
            "doc_id": doc['id'],
            "title": doc['title'],
            "category": doc['category'],
            "content": doc['content']
        }

    return {
        "content": [{
            "type": "text",
            "text": json.dumps(result, indent=2)
        }]
    }


@tool("search_by_category", "Search for FAQ documents within a specific category", {"category": str, "limit": int})
async def search_by_category(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Search for FAQ documents within a specific category.
    """
    category = args["category"]
    limit = args.get("limit", 5)

    # Validate limit
    if limit < 1:
        limit = 1
    if limit > 10:
        limit = 10

    store = get_vector_store()

    # Validate category
    valid_categories = store.get_all_categories()
    if category not in valid_categories:
        result = {
            "error": f"Invalid category '{category}'",
            "valid_categories": valid_categories
        }
    else:
        # Get documents in category
        results = store.search_by_category(category, top_k=limit)

        # Format results
        result = {
            "category": category,
            "num_results": len(results),
            "results": []
        }

        for doc in results:
            result["results"].append({
                "doc_id": doc['id'],
                "title": doc['title'],
                "content": doc['content']
            })

    return {
        "content": [{
            "type": "text",
            "text": json.dumps(result, indent=2)
        }]
    }


@tool("list_categories", "List all available FAQ categories in the knowledge base", {})
async def list_categories(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    List all available FAQ categories in the knowledge base.
    """
    store = get_vector_store()
    categories = store.get_all_categories()

    result = {
        "categories": categories,
        "count": len(categories)
    }

    return {
        "content": [{
            "type": "text",
            "text": json.dumps(result, indent=2)
        }]
    }


# Export tools for MCP server creation
FAQ_TOOLS = [search_faq, get_document, search_by_category, list_categories]
