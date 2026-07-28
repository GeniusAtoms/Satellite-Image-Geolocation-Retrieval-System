"""
Builds a FAISS index over a dataset of satellite images.
Run once per dataset. Saves geo_index.faiss + metadata.json.

Usage:
    python build_index.py --dataset /path/to/images --out_dir ./index_store
    python build_index.py --dataset ./images --out_dir ./index_store --index_type ivf --nlist 256
"""

import os
import json
import argparse
import numpy as np
import faiss
from tqdm import tqdm

from embedder import ClipEmbedder
from geo_parse import parse_lat_lon

VALID_EXT = (".png", ".jpg", ".jpeg")


def build(dataset_dir, out_dir, batch_size=32, index_type="flat", nlist=100):
    os.makedirs(out_dir, exist_ok=True)

    files = [f for f in sorted(os.listdir(dataset_dir)) if f.lower().endswith(VALID_EXT)]
    if not files:
        raise RuntimeError(f"No images found in {dataset_dir}")

    embedder = ClipEmbedder()
    all_embeddings = []
    metadata = []

    for i in tqdm(range(0, len(files), batch_size), desc="Embedding images"):
        batch_files = files[i:i + batch_size]
        batch_paths = [os.path.join(dataset_dir, f) for f in batch_files]

        try:
            batch_emb = embedder.embed_image_batch(batch_paths)
        except Exception as e:
            print(f"Skipping batch at {batch_files[0]}: {e}")
            continue

        for f, emb in zip(batch_files, batch_emb):
            try:
                lat, lon = parse_lat_lon(f)
            except ValueError as e:
                print(f"Skipping {f}: {e}")
                continue
            all_embeddings.append(emb)
            metadata.append({"lat": lat, "lon": lon})

    if not all_embeddings:
        raise RuntimeError("No valid embeddings produced. Check filenames and image files.")

    embeddings = np.vstack(all_embeddings).astype("float32")
    dim = embeddings.shape[1]

    if index_type == "flat":
        # Exact search — best for datasets up to ~100k images
        index = faiss.IndexFlatIP(dim)

    elif index_type == "ivf":
        # Approximate search via clustering — scales to millions of images
        # Tune accuracy vs speed at query time using nprobe
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)

    elif index_type == "hnsw":
        # Approximate graph-based search — very fast queries, higher memory usage
        index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)

    else:
        raise ValueError(f"Unknown index_type: {index_type}")

    index.add(embeddings)

    faiss.write_index(index, os.path.join(out_dir, "geo_index.faiss"))
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)

    print(f"\nDone. Indexed {len(metadata)} images -> {out_dir}")
    print(f"Index type: {index_type}  |  Embedding dim: {dim}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    required=True,  help="Folder of dataset images")
    parser.add_argument("--out_dir",    default="./index_store")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--index_type", choices=["flat", "ivf", "hnsw"], default="flat")
    parser.add_argument("--nlist",      type=int, default=100, help="Clusters for IVF index")
    args = parser.parse_args()

    build(args.dataset, args.out_dir, args.batch_size, args.index_type, args.nlist)
