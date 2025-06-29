FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    aiogram==3.20.0.post0 \
    faiss-cpu==1.11.0 \
    fastapi==0.115.12 \
    huggingface-hub==0.31.1 \
    langchain==0.3.25 \
    langchain-community==0.3.23 \
    langchain-core==0.3.59 \
    langchain-huggingface==0.2.0 \
    langchain-openai==0.3.17 \
    langchain-redis==0.2.1 \
    langgraph==0.3.34 \
    langsmith==0.3.42 \
    numpy==1.26.4 \
    openai==1.79.0 \
    pydantic==2.11.4 \
    python-dotenv==1.1.0 \
    rank-bm25==0.2.2 \
    redis==5.3.0 \
    scikit-learn==1.6.1 \
    scipy==1.15.3 \
    sentence-transformers==4.1.0 \
    transformers==4.51.3 \
    uvicorn==0.34.2

WORKDIR /app

ENV HF_XET_NUM_CONCURRENT_RANGE_GETS=32
ENV HF_XET_MAX_CONCURRENT_DOWNLOADS=4

COPY . /app
RUN python3 download_models.py

WORKDIR /app

ENTRYPOINT ["python3", "bot.py"]
