#!/usr/bin/env python3
"""
RAG precompute script:
- typically done in conjection of each prompt, but would require the loading
of 2 models i.e. will need 48GB RAM. In this case we will run it before actually prompting
- Retrieves top-k contexts of each crop description from a FAISS index + metadata using cosine similarity
- Saves a pickle file of contexts
"""

import os
import json
import pickle
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer



class PrecomputedRAG:
    def __init__(self, metadata_pkl: Path, faiss_index_path: Path, model_name: str):
        # Load precomputed metadata (texts, source_rows, chunk_ids, etc.)
        with open(metadata_pkl, "rb") as f:
            data = pickle.load(f)
        self.texts = data["texts"]
        self.source_rows = data.get("source_rows", None)
        self.chunk_ids = data.get("chunk_ids", None)

        # Load FAISS index
        self.index = faiss.read_index(str(faiss_index_path))

        # Load embedding model (for query only)
        self.model = SentenceTransformer(model_name)
        print(f"Loaded {len(self.texts)} precomputed embeddings from metadata.")
        print(f"FAISS index: {faiss_index_path}")
        print(f"Query encoder: {model_name}")

    def retrieve_context(self, query: str, top_k: int = 5):
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        # faiss expects float32
        distances, indices = self.index.search(
            query_embedding.astype("float32"), top_k
        )
        contexts = []
        for idx in indices[0]:
            if idx != -1:
                contexts.append(
                    {
                        "text": self.texts[idx],
                        "source_row": None if self.source_rows is None else self.source_rows[idx],
                        # "score": float(distances[0][i])  
                    }
                )
        return contexts

    def get_context_string(self, query: str, top_k: int = 5, max_length: int = 2000) -> str:
        contexts = self.retrieve_context(query, top_k=top_k)
        parts, total = [], 0
        for ctx in contexts:
            t = ctx["text"]
            if total + len(t) > max_length:
                break
            parts.append(t)
            total += len(t)
        return "\n\n".join(parts)


def main():
    parser = argparse.ArgumentParser(
        description="Generate context strings from a FAISS+metadata store for each name in a CSV."
    )

    # Required paths
    parser.add_argument(
        "--country-csv",
        required=True,
        type=Path,
        help="Path to the CSV containing a column of names.",
    )
    parser.add_argument(
        "--output-pkl",
        required=True,
        type=Path,
        help="Where to save the resulting contexts list (pickle).",
    )
    parser.add_argument(
        "--faiss-index",
        required=True,
        type=Path,
        help="Path to the FAISS index file (.index).",
    )
    parser.add_argument(
        "--metadata-pkl",
        required=True,
        type=Path,
        help="Path to the metadata pickle with texts",
    )

    # Column
    parser.add_argument(
        "--name-column",
        default="original_name",
        help="CSV column to read names from. Default: original_name",
    )

    # Retrieval params
    parser.add_argument("--top-k", type=int, default=5, help="Top-K contexts. Default: 5")
    parser.add_argument(
        "--max-length",
        type=int,
        default=1000,
        help="Max characters for concatenated context. Default: 1000",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-Embedding-8B",
        help="Embedding model name. Default: Qwen/Qwen3-Embedding-8B",
    )


    args = parser.parse_args()



    # Load CSV and names
    df = pd.read_csv(args.country_csv)
    names = df[args.name_column]

    # Build retriever
    retriever = PrecomputedRAG(
        metadata_pkl=args.metadata_pkl,
        faiss_index_path=args.faiss_index,
        model_name=args.model,
    )

    # Retrieve contexts
    contexts_list = []
    for name in names:
        context_s = retriever.get_context_string(
            name, top_k=args.top_k, max_length=args.max_length
        )
        contexts_list.append(context_s)

    # Save output
    args.output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_pkl, "wb") as f:
        pickle.dump(contexts_list, f)

    print("Saved ", len(contexts_list), "contexts to ", args.output_pkl)


if __name__ == "__main__":
    main()