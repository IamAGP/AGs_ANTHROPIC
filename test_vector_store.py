"""Quick test of vector store functionality."""
from src.tools.vector_store import initialize_vector_store

# Initialize vector store
print("Initializing vector store...")
store = initialize_vector_store("data/faq_docs.json")

# Test search
print("\n" + "="*60)
print("Test 1: Search for 'return policy'")
print("="*60)
results = store.search("What is your return policy?", top_k=3)
for result in results:
    print(f"\nDoc ID: {result['id']}")
    print(f"Title: {result['title']}")
    print(f"Similarity: {result['similarity_score']:.4f}")
    print(f"Content preview: {result['content'][:100]}...")

# Test retrieval by ID
print("\n" + "="*60)
print("Test 2: Get document by ID (faq_022)")
print("="*60)
doc = store.get_document("faq_022")
if doc:
    print(f"Title: {doc['title']}")
    print(f"Content: {doc['content']}")

# Test category search
print("\n" + "="*60)
print("Test 3: Search by category (account_management)")
print("="*60)
cat_results = store.search_by_category("account_management", top_k=3)
for doc in cat_results:
    print(f"  - {doc['id']}: {doc['title']}")

print("\n" + "="*60)
print("All tests completed!")
print("="*60)
