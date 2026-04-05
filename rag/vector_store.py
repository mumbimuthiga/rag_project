import faiss  #handles vector similarity search
import numpy as np #works with arrays (vectors)

class VectorStore: #stores text chunks and their embeddings for retrieval (custom db for vectors)
    def __init__(self, dimension): #dimension = size of each embedding vector e.g[0.1, 0.2, 0.3] → dimension = 3
        self.index = faiss.IndexFlatL2(dimension) #L2 = Euclidean distance (measures similarity between vectors) #Creates a FAISS index for storing and searching vectors based on L2 distance
        self.text_chunks = [] #stores the original text chunks corresponding to the embeddings (used for retrieval). Because FAISS only stores numbers (not text)

    def add(self, embeddings, texts):
        self.index.add(np.array(embeddings)) #Adds the provided embeddings to the FAISS index for future similarity search. Converts the list of embeddings into a NumPy array before adding it to the index.
        self.text_chunks.extend(texts) #Adds the corresponding text chunks to the text_chunks list for later retrieval. Uses extend to add multiple texts at once.

    def search(self, query_embedding, k=3): #Searches for the top k most similar text chunks based on the provided query embedding. query_embedding is the embedding vector for the search query, and k is the number of top results to return.
        D, I = self.index.search(query_embedding, k) #Performs a search in the FAISS index using the provided query embedding and retrieves the distances (D) and indices (I) of the top k most similar embeddings. D contains the distances to the nearest neighbors, and I contains their corresponding indices in the text_chunks list.
        return [self.text_chunks[i] for i in I[0]] #Returns the original text chunks corresponding to the top k most similar embeddings based on the indices retrieved from the FAISS search. I[0] is used to access the first (and only) row of indices since we are searching with a single query embedding.