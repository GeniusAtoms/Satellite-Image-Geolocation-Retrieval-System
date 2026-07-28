"""
Query engine. Given a satellite image, returns the 5 most visually
similar locations from the FAISS index.

Each result contains:
    - rank       : 1 (best match) to 5
    - latitude   : float
    - longitude  : float

Usage:
    python locate.py --query /path/to/query.png --index_dir ./index_store
"""

import os
import json
import argparse
import faiss

from embedder import ClipEmbedder


def locate(query_path, index_dir, top_k=5, nprobe=10):
    index = faiss.read_index(os.path.join(index_dir, "geo_index.faiss"))
    with open(os.path.join(index_dir, "metadata.json")) as f:
        metadata = json.load(f)

    # nprobe only applies to IVF index — ignored by flat and hnsw
    if isinstance(index, faiss.IndexIVFFlat):
        index.nprobe = nprobe

    embedder = ClipEmbedder()
    query_emb = embedder.embed_image(query_path).reshape(1, -1)

    similarities, indices = index.search(query_emb, top_k)

    results = []
    for rank in range(top_k):
        idx = int(indices[0][rank])
        if idx == -1:
            break
        match = metadata[idx]
        results.append({
            "rank":      rank + 1,
            "latitude":  match["lat"],
            "longitude": match["lon"],
        })

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query",     required=True, help="Path to query image")
    parser.add_argument("--index_dir", default="./index_store")
    parser.add_argument("--top_k",     type=int, default=5)
    parser.add_argument("--nprobe",    type=int, default=10)
    args = parser.parse_args()

    results = locate(args.query, args.index_dir, args.top_k, args.nprobe)
    print(json.dumps(results, indent=2))
