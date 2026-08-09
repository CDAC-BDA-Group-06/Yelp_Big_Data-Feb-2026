

# import pandas as pd
# import pyarrow
# import duckdb
# import sentence_transformers
# import faiss
# import openai
# import google.protobuf

# print("Pandas:", pd.__version__)
# print("Protobuf:", google.protobuf.__version__)
# print("Setup successful")

# from google.colab import drive

# drive.mount("/content/drive")

from pathlib import Path

# Main project folder visible in Google Drive
PROJECT_ROOT = Path("C:\\Users\\MIHIR ZOPE\\Desktop\\RAG\\data\\rag")

# Input Parquet folders
BUSINESS_DOCUMENTS_PATH = (
    PROJECT_ROOT / "business_documents"
)

REVIEW_DOCUMENTS_PATH = (
    PROJECT_ROOT / "review_documents"
)

# Generated chunks will be stored here
CHUNKS_PATH = (
    PROJECT_ROOT / "chunks"
)

# Generated FAISS indexes and metadata will be stored here
MODEL_OUTPUT_PATH = (
    PROJECT_ROOT / "models" / "faiss_index"
)

# Create output folders
CHUNKS_PATH.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_OUTPUT_PATH.mkdir(
    parents=True,
    exist_ok=True,
)

# Display and verify paths
print("Project root:", PROJECT_ROOT)
print("Business data:", BUSINESS_DOCUMENTS_PATH)
print("Review data:", REVIEW_DOCUMENTS_PATH)
print("Chunk output:", CHUNKS_PATH)
print("FAISS output:", MODEL_OUTPUT_PATH)

print("\nPath verification:")
print(
    "Business folder exists:",
    BUSINESS_DOCUMENTS_PATH.exists(),
)
print(
    "Review folder exists:",
    REVIEW_DOCUMENTS_PATH.exists(),
)
print(
    "Chunks folder exists:",
    CHUNKS_PATH.exists(),
)
print(
    "FAISS folder exists:",
    MODEL_OUTPUT_PATH.exists(),
)

business_files = list(
    BUSINESS_DOCUMENTS_PATH.glob("*.parquet")
)

review_files = list(
    REVIEW_DOCUMENTS_PATH.glob("*.parquet")
)

print("Business Parquet files:", len(business_files))
print("Review Parquet files:", len(review_files))

assert business_files, "Business Parquet files not found"
assert review_files, "Review Parquet files not found"

import pyarrow.dataset as ds

business_dataset = ds.dataset(
    str(BUSINESS_DOCUMENTS_PATH),
    format="parquet",
)

review_dataset = ds.dataset(
    str(REVIEW_DOCUMENTS_PATH),
    format="parquet",
)

print("BUSINESS SCHEMA")
print(business_dataset.schema)

print("\nREVIEW SCHEMA")
print(review_dataset.schema)

business_count = business_dataset.count_rows()
review_count = review_dataset.count_rows()

print("\nComplete business documents:", f"{business_count:,}")
print("Complete review documents:", f"{review_count:,}")

# ### ***2: Chunk the complete dataset***

import gc
import shutil
import pyarrow as pa
import pyarrow.parquet as pq

from tqdm.auto import tqdm


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
PARQUET_BATCH_SIZE = 5000

BUSINESS_CHUNKS_PATH = (
    CHUNKS_PATH / "business_chunks"
)

REVIEW_CHUNKS_PATH = (
    CHUNKS_PATH / "review_chunks"
)

print("Chunk size:", CHUNK_SIZE)
print("Overlap:", CHUNK_OVERLAP)
print("Batch size:", PARQUET_BATCH_SIZE)

def chunk_text(
    text,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP,
):
    if text is None:
        return []

    text = str(text).strip()

    if not text:
        return []

    if overlap >= chunk_size:
        raise ValueError(
            "Overlap must be smaller than chunk size"
        )

    if len(text) <= chunk_size:
        return [text]

    step_size = chunk_size - overlap
    chunks = []

    for start in range(0, len(text), step_size):
        end = min(start + chunk_size, len(text))

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

    return chunks

sample_text = "Yelp RAG document chunking test. " * 100

sample_chunks = chunk_text(sample_text)

print("Input characters:", len(sample_text))
print("Chunks created:", len(sample_chunks))
print("First chunk length:", len(sample_chunks[0]))

def clean_metadata_value(value):
    if value is None:
        return None

    if hasattr(value, "isoformat"):
        return value.isoformat()

    if isinstance(value, list):
        return ", ".join(str(item) for item in value)

    return value


def write_chunk_records(
    records,
    output_folder,
    part_number,
):
    if not records:
        return

    output_file = (
        output_folder
        / f"part-{part_number:05d}.parquet"
    )

    table = pa.Table.from_pylist(records)

    pq.write_table(
        table,
        output_file,
        compression="snappy",
    )


def chunk_parquet_dataset(
    input_folder,
    output_folder,
    document_type,
    metadata_columns,
):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)

    # Delete previously generated chunks to prevent duplicates.
    if output_folder.exists():
        shutil.rmtree(output_folder)

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset = ds.dataset(
        str(input_folder),
        format="parquet",
    )

    available_columns = set(dataset.schema.names)

    selected_columns = [
        column
        for column in (
            ["document_id", "document_text"]
            + metadata_columns
        )
        if column in available_columns
    ]

    print("Selected columns:", selected_columns)

    total_documents = dataset.count_rows()

    scanner = dataset.scanner(
        columns=selected_columns,
        batch_size=PARQUET_BATCH_SIZE,
    )

    total_processed_documents = 0
    total_generated_chunks = 0
    part_number = 0

    progress_bar = tqdm(
        total=total_documents,
        desc=f"Chunking {document_type}",
    )

    for record_batch in scanner.to_batches():
        rows = record_batch.to_pylist()
        chunk_records = []

        for row in rows:
            document_id = str(
                row.get("document_id", "")
            )

            document_text = row.get(
                "document_text",
                "",
            )

            chunks = chunk_text(document_text)

            for chunk_number, chunk in enumerate(chunks):
                record = {
                    "chunk_id": (
                        f"{document_id}_{chunk_number}"
                    ),
                    "document_id": document_id,
                    "chunk_number": chunk_number,
                    "total_chunks": len(chunks),
                    "chunk_text": chunk,
                    "document_type": document_type,
                }

                for column in metadata_columns:
                    if column in row:
                        record[column] = (
                            clean_metadata_value(
                                row.get(column)
                            )
                        )

                chunk_records.append(record)

        write_chunk_records(
            records=chunk_records,
            output_folder=output_folder,
            part_number=part_number,
        )

        part_number += 1
        total_processed_documents += len(rows)
        total_generated_chunks += len(chunk_records)

        progress_bar.update(len(rows))

        del rows
        del chunk_records
        gc.collect()

    progress_bar.close()

    print("\nChunking completed")
    print(
        "Documents processed:",
        f"{total_processed_documents:,}",
    )
    print(
        "Chunks generated:",
        f"{total_generated_chunks:,}",
    )
    print("Output folder:", output_folder)

    return {
        "documents": total_processed_documents,
        "chunks": total_generated_chunks,
    }

business_metadata_columns = [
    "business_id",
    "business_name",
    "address",
    "city",
    "state",
    "postal_code",
    "primary_category",
    "business_rating",
    "review_count",
    "price_range",
    "is_open",
]

business_metrics = chunk_parquet_dataset(
    input_folder=BUSINESS_DOCUMENTS_PATH,
    output_folder=BUSINESS_CHUNKS_PATH,
    document_type="business",
    metadata_columns=business_metadata_columns,
)

review_metadata_columns = [
    "review_id",
    "business_id",
    "user_id",
    "business_name",
    "city",
    "state",
    "primary_category",
    "review_date",
    "stars",
    "sentiment",
    "review_length",
]

review_metrics = chunk_parquet_dataset(
    input_folder=REVIEW_DOCUMENTS_PATH,
    output_folder=REVIEW_CHUNKS_PATH,
    document_type="review",
    metadata_columns=review_metadata_columns,
)

business_chunk_dataset = ds.dataset(
    str(BUSINESS_CHUNKS_PATH),
    format="parquet",
)

review_chunk_dataset = ds.dataset(
    str(REVIEW_CHUNKS_PATH),
    format="parquet",
)

print(
    "Business chunks:",
    f"{business_chunk_dataset.count_rows():,}",
)

print(
    "Review chunks:",
    f"{review_chunk_dataset.count_rows():,}",
)

sample_chunks = (
    business_chunk_dataset
    .head(3)
    .to_pandas()
)

display(sample_chunks)

# # ***3: Generate embeddings from all chunks***

import torch
import numpy as np

from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

EMBEDDING_BATCH_SIZE = 256

print("Device:", DEVICE)
print("Loading model:", EMBEDDING_MODEL_NAME)

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME,
    device=DEVICE,
)

EMBEDDING_DIMENSION = (
    embedding_model
    .get_sentence_embedding_dimension()
)

print("Embedding dimension:", EMBEDDING_DIMENSION)

import json
import faiss


def convert_for_json(value):
    if value is None:
        return None

    if isinstance(value, np.generic):
        return value.item()

    if hasattr(value, "isoformat"):
        return value.isoformat()

    if isinstance(value, list):
        return [
            convert_for_json(item)
            for item in value
        ]

    return value


def build_faiss_index(
    chunks_folder,
    index_output_file,
    metadata_output_file,
    embedding_model,
):
    chunks_dataset = ds.dataset(
        str(chunks_folder),
        format="parquet",
    )

    total_chunks = chunks_dataset.count_rows()

    index = faiss.IndexFlatIP(
        EMBEDDING_DIMENSION
    )

    scanner = chunks_dataset.scanner(
        batch_size=EMBEDDING_BATCH_SIZE,
    )

    metadata_count = 0

    with open(
        metadata_output_file,
        "w",
        encoding="utf-8",
    ) as metadata_file:

        progress_bar = tqdm(
            total=total_chunks,
            desc=f"Embedding {Path(chunks_folder).name}",
        )

        for record_batch in scanner.to_batches():
            rows = record_batch.to_pylist()

            texts = [
                str(row.get("chunk_text", ""))
                for row in rows
            ]

            embeddings = embedding_model.encode(
                texts,
                batch_size=EMBEDDING_BATCH_SIZE,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )

            embeddings = np.asarray(
                embeddings,
                dtype=np.float32,
            )

            index.add(embeddings)

            for row in rows:
                metadata_record = {
                    key: convert_for_json(value)
                    for key, value in row.items()
                }

                metadata_file.write(
                    json.dumps(
                        metadata_record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                metadata_count += 1

            progress_bar.update(len(rows))

            del rows
            del texts
            del embeddings
            gc.collect()

        progress_bar.close()

    faiss.write_index(
        index,
        str(index_output_file),
    )

    print("\nIndex successfully created")
    print("Vectors:", f"{index.ntotal:,}")
    print("Metadata rows:", f"{metadata_count:,}")
    print("Index file:", index_output_file)
    print("Metadata file:", metadata_output_file)

    assert index.ntotal == metadata_count

    return index.ntotal

BUSINESS_INDEX_FILE = (
    MODEL_OUTPUT_PATH / "business.index"
)

BUSINESS_METADATA_FILE = (
    MODEL_OUTPUT_PATH / "business_metadata.jsonl"
)

business_vector_count = build_faiss_index(
    chunks_folder=BUSINESS_CHUNKS_PATH,
    index_output_file=BUSINESS_INDEX_FILE,
    metadata_output_file=BUSINESS_METADATA_FILE,
    embedding_model=embedding_model,
)

REVIEW_INDEX_FILE = (
    MODEL_OUTPUT_PATH / "review.index"
)

REVIEW_METADATA_FILE = (
    MODEL_OUTPUT_PATH / "review_metadata.jsonl"
)

review_vector_count = build_faiss_index(
    chunks_folder=REVIEW_CHUNKS_PATH,
    index_output_file=REVIEW_INDEX_FILE,
    metadata_output_file=REVIEW_METADATA_FILE,
    embedding_model=embedding_model,
)

## 4: Load FAISS and query the RAG

def load_jsonl(file_path):
    records = []

    with open(
        file_path,
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            records.append(json.loads(line))

    return records


business_index = faiss.read_index(
    str(BUSINESS_INDEX_FILE)
)

review_index = faiss.read_index(
    str(REVIEW_INDEX_FILE)
)

business_metadata = load_jsonl(
    BUSINESS_METADATA_FILE
)

review_metadata = load_jsonl(
    REVIEW_METADATA_FILE
)

assert business_index.ntotal == len(
    business_metadata
)

assert review_index.ntotal == len(
    review_metadata
)

print(
    "Business vectors:",
    f"{business_index.ntotal:,}",
)

print(
    "Review vectors:",
    f"{review_index.ntotal:,}",
)

def metadata_matches(
    metadata,
    city=None,
    state=None,
    sentiment=None,
    min_stars=None,
):
    if city:
        metadata_city = str(
            metadata.get("city", "")
        ).lower()

        if metadata_city != city.lower():
            return False

    if state:
        metadata_state = str(
            metadata.get("state", "")
        ).lower()

        if metadata_state != state.lower():
            return False

    if sentiment:
        metadata_sentiment = str(
            metadata.get("sentiment", "")
        ).lower()

        if metadata_sentiment != sentiment.lower():
            return False

    if min_stars is not None:
        rating = metadata.get(
            "business_rating",
            metadata.get("stars"),
        )

        if rating is None:
            return False

        try:
            if float(rating) < float(min_stars):
                return False
        except (ValueError, TypeError):
            return False

    return True


def search_single_index(
    query_vector,
    index,
    metadata,
    source_type,
    candidate_count,
    city=None,
    state=None,
    sentiment=None,
    min_stars=None,
):
    candidate_count = min(
        candidate_count,
        index.ntotal,
    )

    scores, indices = index.search(
        query_vector,
        candidate_count,
    )

    results = []

    for score, index_position in zip(
        scores[0],
        indices[0],
    ):
        if index_position < 0:
            continue

        record = metadata[index_position]

        if not metadata_matches(
            metadata=record,
            city=city,
            state=state,
            sentiment=sentiment,
            min_stars=min_stars,
        ):
            continue

        results.append(
            {
                "score": float(score),
                "source_type": source_type,
                "metadata": record,
                "text": record.get(
                    "chunk_text",
                    "",
                ),
            }
        )

    return results


def search_rag(
    query,
    document_type="all",
    city=None,
    state=None,
    sentiment=None,
    min_stars=None,
    top_k=8,
    candidate_count=500,
):
    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32,
    )

    results = []

    if document_type in ("all", "business"):
        results.extend(
            search_single_index(
                query_vector=query_embedding,
                index=business_index,
                metadata=business_metadata,
                source_type="business",
                candidate_count=candidate_count,
                city=city,
                state=state,
                min_stars=min_stars,
            )
        )

    if document_type in ("all", "review"):
        results.extend(
            search_single_index(
                query_vector=query_embedding,
                index=review_index,
                metadata=review_metadata,
                source_type="review",
                candidate_count=candidate_count,
                city=city,
                state=state,
                sentiment=sentiment,
                min_stars=min_stars,
            )
        )

    results.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return results[:top_k]

query = (
    "Italian restaurants with delicious pasta "
    "and friendly service"
)

results = search_rag(
    query=query,
    document_type="all",
    top_k=8,
)

for number, result in enumerate(results, start=1):
    metadata = result["metadata"]

    print("=" * 80)
    print("Result:", number)
    print("Score:", round(result["score"], 4))
    print("Type:", result["source_type"])
    print(
        "Business:",
        metadata.get("business_name"),
    )
    print(
        "Location:",
        metadata.get("city"),
        metadata.get("state"),
    )
    print("Text:")
    print(result["text"][:500])

results = search_rag(
    query="food quality and customer service",
    document_type="review",
    city="Philadelphia",
    sentiment="positive",
    min_stars=4,
    top_k=10,
    candidate_count=2000,
)

for result in results:
    metadata = result["metadata"]

    print(
        round(result["score"], 4),
        metadata.get("business_name"),
        metadata.get("city"),
        metadata.get("stars"),
        metadata.get("sentiment"),
    )

# 5: Complete RAG answer generation
# Cell 1 â€” Configure OpenAI key securely
# In Colab:
# Left sidebar â†’ Secrets â†’ Add new secret
# Name: OPENAI_API_KEY

from google.colab import userdata
from openai import OpenAI

OPENAI_API_KEY = userdata.get(
    "OPENAI_API_KEY"
)

openai_client = OpenAI(
    api_key=OPENAI_API_KEY
)

GENERATION_MODEL = "gpt-5.6-sol"

def build_context(results):
    context_sections = []

    for source_number, result in enumerate(
        results,
        start=1,
    ):
        metadata = result["metadata"]

        context_sections.append(
            f"""
SOURCE {source_number}

Source type: {result["source_type"]}
Business: {metadata.get("business_name", "Unknown")}
Business ID: {metadata.get("business_id", "")}
Location: {metadata.get("city", "")}, {metadata.get("state", "")}
Category: {metadata.get("primary_category", "")}
Rating: {
    metadata.get(
        "business_rating",
        metadata.get("stars", "")
    )
}
Sentiment: {metadata.get("sentiment", "")}

Content:
{result["text"]}
""".strip()
        )

    return "\n\n---\n\n".join(
        context_sections
    )

def ask_rag(
    question,
    document_type="all",
    city=None,
    state=None,
    sentiment=None,
    min_stars=None,
    top_k=8,
):
    retrieved_results = search_rag(
        query=question,
        document_type=document_type,
        city=city,
        state=state,
        sentiment=sentiment,
        min_stars=min_stars,
        top_k=top_k,
        candidate_count=2000,
    )

    if not retrieved_results:
        return {
            "answer": (
                "No relevant information was found "
                "for the supplied filters."
            ),
            "sources": [],
        }

    context = build_context(
        retrieved_results
    )

    response = openai_client.responses.create(
        model=GENERATION_MODEL,
        instructions="""
You are a Yelp data RAG assistant.

Answer using only the retrieved context.

Rules:
1. Never invent missing information.
2. If the context is insufficient, clearly say so.
3. Distinguish business facts from customer opinions.
4. If customer reviews disagree, mention the disagreement.
5. Cite evidence using [Source 1], [Source 2], etc.
6. Mention relevant business names.
7. Answer in the same language as the user's question.
8. Do not claim that retrieved examples represent every review.
""",
        input=f"""
Question:
{question}

Retrieved Yelp context:
{context}
""",
    )

    return {
        "answer": response.output_text,
        "sources": retrieved_results,
    }

# Cell 4 â€” Ask questions

result = ask_rag(
    question=(
        "Which Italian restaurants have good pasta "
        "and friendly customer service?"
    ),
    document_type="all",
    top_k=10,
)

print(result["answer"])

result = ask_rag(
    question=(
        "What complaints do customers have "
        "about service?"
    ),
    document_type="review",
    city="Philadelphia",
    sentiment="negative",
    top_k=10,
)

print(result["answer"])

# Cell 5 â€” Interactive chatbot

print("Yelp RAG chatbot is ready.")
print("Enter 'exit' to stop.\n")

while True:
    question = input("You: ").strip()

    if question.lower() in {
        "exit",
        "quit",
        "stop",
    }:
        print("Chatbot stopped.")
        break

    if not question:
        continue

    result = ask_rag(
        question=question,
        document_type="all",
        top_k=10,
    )

    print("\nAssistant:")
    print(result["answer"])
    print()
