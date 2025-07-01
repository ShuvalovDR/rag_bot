from typing import Tuple
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_models() -> Tuple[SentenceTransformer, SentenceTransformer]:
    logger.info("Downloading speech recognition model...")
    asr_model = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small",
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    logger.info("Downloading text embedding model...")
    embedding_model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device="cuda:0" if torch.cuda.is_available() else "cpu"
    )

    logger.info("All models downloaded successfully!")
    return asr_model, embedding_model

if __name__ == "__main__":
    download_models()
