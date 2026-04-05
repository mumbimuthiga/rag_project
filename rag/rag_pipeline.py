from rag.embedder import embed_text
from rag.vector_store import VectorStore

vector_store = None

def build_index(chunks):
    global vector_store
    
    embeddings = embed_text(chunks)
    dimension = embeddings.shape[1]
    
    vector_store = VectorStore(dimension)
    vector_store.add(embeddings, chunks)

def query_rag(question):
    query_embedding = embed_text([question])
    
    results = vector_store.search(query_embedding)
    
    context = "\n".join(results)
    
    # Simple response (no LLM yet)
    return f"Context:\n{context}\n\nQuestion: {question}"