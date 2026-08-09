                  Yelp Dataset
                       │
                       ▼
                 S3 Silver Layer
                       │
                 Spark on EMR
                       │
                       ▼
          Gold RAG Documents (Already Done)
                       │
                       ▼
              Document Chunking (Spark)
                       │
                       ▼
          Chunked Parquet Files (S3 Gold)
                       │
────────────────────────────────────────────
               Data Engineering Ends
────────────────────────────────────────────
                       │
                       ▼
          Generate Embeddings (Python)
                       │
                       ▼
             Vector Database (FAISS/Chroma)
                       │
                       ▼
               Retrieval + LLM
                       │
                       ▼
                 Streamlit Chatbot
