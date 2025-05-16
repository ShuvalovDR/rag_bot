from typing import List

import numpy as np
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def init_ensemble_retriever(file_path: str, device: str = "cpu", k: int=5, weights: List[float]=[0.5, 0.5]):
    """Initialize custom retriever
    Args:
        file_path (str): path to csv file with data for rag
        device (str): device for embedding model (default: "cpu")
        k: (int): the number of documents to find (default: 5)
        weights: (List[float]): score weights of ensemble retriever
    """
    assert np.isclose(sum(weights), 1.0, rtol=1e-6, atol=1e-6), \
        f"Sum of weights is: {sum(weights)}, but sum must be equal to 1.0"
    assert len(weights) == 2, f"Len of weights array must be 2, now lenght is {len(weights)}"
    faiss_retriever = init_faiss_retriever(file_path, k)
    bm25_retriever = init_bm25_retriever(file_path, k)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever], weights=weights
    )
    return ensemble_retriever

def init_faiss_retriever(file_path: str, k: int, device: str = "cpu") -> VectorStoreRetriever:
    """Initialize faiss retriever
    Args:
        file_path (str): path to csv file
        device (str): device for embedding model (default: "cpu")
        k: (int): the number of documents to find (default: 5)
    Returns:
        VectorStoreRetriever: faiss retriever
    """
    loader = CSVLoader(
        file_path=file_path,
        encoding="utf-8",
        source_column="Question",
        metadata_columns=["Products_name"],
        csv_args={
            "fieldnames": ["Question", "Products_name"]
        }
    )
    documents = loader.load()

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True} 
    )

    vector_db = FAISS.from_documents(
        documents= documents,
        embedding=embeddings
    )
    
    return vector_db.as_retriever(search_kwargs={'k': k})

def init_bm25_retriever(file_path: str, k: int) -> VectorStoreRetriever:
    """Initialize bm25 retriever
    Args:
        file_path (str): path to csv file
        k: (int): the number of documents to find
    Returns:
        VectorStoreRetriever: bm25 retriever
    """
    loader = CSVLoader(
        file_path=file_path,
        encoding="utf-8",
        source_column="Question",
        metadata_columns=["Products_name"],
        csv_args={
            "fieldnames": ["Question", "Products_name"]
        }
    )
    documents = loader.load()
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = k
    return bm25_retriever
    




    







