from . import config
from sentence_transformers import SentenceTransformer
from typing import Literal, List
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np

class EmbeddingVectorizer:
    """Word embeddings vectorizer using SentenceTransformer"""

    def __init__(self, model_name: str = None, normalize: bool = True):
        model_name = model_name or config.EMBEDDING_MODEL
        self.model = SentenceTransformer(model_name)
        self.normalize = normalize

    def _format_inputs(self, texts: List[str], mode: Literal['query', 'passage']) -> List[str]:
        if mode not in {"query", "passage"}:
            raise ValueError("Mode must be either 'query' or 'passage'")
        return [f"{mode}: {text.strip()}" for text in texts]
    
    def transform(self, texts: List[str], mode: Literal['query', 'passage'] = 'query'):
        inputs = self._format_inputs(texts, mode)
        embeddings = self.model.encode(inputs, normalize_embeddings=self.normalize)
        return embeddings
    
def vectorize_data(X_train, X_test):
    """Vectorize data using BoW, TF-IDF, and Embeddings"""
    print("Vectorizing data")
    
    # Bag of Words
    print("Bag of Words")
    bow_vectorizer = CountVectorizer()
    X_train_bow = bow_vectorizer.fit_transform(X_train).toarray()
    X_test_bow = bow_vectorizer.transform(X_test).toarray()
    
    # TF-IDF
    print("TF-IDF")
    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train).toarray()
    X_test_tfidf = tfidf_vectorizer.transform(X_test).toarray()
    
    # Embeddings
    print("Embeddings")
    embedding_vectorizer = EmbeddingVectorizer()
    X_train_embeddings = embedding_vectorizer.transform(X_train)
    X_test_embeddings = embedding_vectorizer.transform(X_test)

    # Convert to numpy arrays
    X_train_bow = np.array(X_train_bow)
    X_test_bow = np.array(X_test_bow)
    X_train_tfidf = np.array(X_train_tfidf)
    X_test_tfidf = np.array(X_test_tfidf)
    X_train_embeddings = np.array(X_train_embeddings)
    X_test_embeddings = np.array(X_test_embeddings)
    
    print(f"Vectorized shapes:")
    print(f"BoW: {X_train_bow.shape} / {X_test_bow.shape}")
    print(f"TF-IDF: {X_train_tfidf.shape} / {X_test_tfidf.shape}")
    print(f"Embeddings: {X_train_embeddings.shape} / {X_test_embeddings.shape}")
    
    return {
        'bow': (X_train_bow, X_test_bow),
        'tfidf': (X_train_tfidf, X_test_tfidf),
        'embeddings': (X_train_embeddings, X_test_embeddings)
    }