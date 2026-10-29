#!/usr/bin/env python3
"""Reproduce CHI3242 week 8: one source file = one document."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

import jieba
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
from PIL import Image
from qhchina import load_fonts
from qhchina.helpers import fonts as qh_fonts, github as qh_github
from qhchina.analytics.stylometry import Stylometry
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from download_corpora import COLLECTIONS
from interactive_plot import write_interactive_html

# qhchina 0.2.7 has no configurable cache path. Keep both font-cache references
# in this project, so running the exercise does not write to the home directory.
qh_github.CACHE_BASE = ROOT / ".cache" / "qhchina"
qh_fonts._CACHE_DIR = qh_github.CACHE_BASE / "fonts"

COLORS = {"呐喊": "#c0553e", "彷徨": "#d5a441", "故事新编": "#7556a4", "沈从文": "#2d8190"}
Z_LABEL = "collection mean of story z-scores; corpus-wide qhchina standardization (equal weight per story, ddof=0)"


def is_letter_or_number(char: str) -> bool:
    return unicodedata.category(char)[0] in {"L", "N"}


def clean_text(text: str) -> str:
    # Replacing (rather than deleting) separators prevents words joining across punctuation.
    return "".join(char if is_letter_or_number(char) else " " for char in text)


def tokenize(text: str, tokenizer: jieba.Tokenizer) -> list[str]:
    return [word for word in tokenizer.cut(clean_text(text), cut_all=False, HMM=True)
            if word and all(is_letter_or_number(char) for char in word)]


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    # BOM makes Chinese labels readable in Excel as well as ordinary UTF-8 readers.
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_documents(output: Path) -> list[dict]:
    tokenizer = jieba.Tokenizer()
    tokenizer.tmp_dir = str(ROOT / ".cache")
    tokenizer.initialize()
    documents, errors = [], []
    author_indices = Counter()
    provenance = json.loads((ROOT / "data-provenance.json").read_text(encoding="utf-8"))
    expected_hashes = {f["path"]: f["sha256"] for c in provenance["collections"] for f in c["files"]}
    for author, collection, directory in COLLECTIONS:
        files = sorted((ROOT / directory).rglob("*.txt"))
        if not files:
            errors.append({"source_file": directory, "error": "No .txt files found"})
        for source in files:
            relative = source.relative_to(ROOT).as_posix()
            try:
                content = source.read_bytes()
                digest = hashlib.sha256(content).hexdigest()
                if expected_hashes.get(relative) != digest:
                    raise ValueError("Corpus does not match data-provenance.json")
                text = content.decode("utf-8")
                tokens = tokenize(text, tokenizer)
                if not tokens:
                    raise ValueError("No letter/number tokens after preprocessing")
                author_indices[author] += 1
                documents.append({
                    "document_id": f"{author}_{author_indices[author]}",
                    "author": author, "collection": collection,
                    "title": source.stem, "source_file": relative,
                    "character_count": len(text),
                    "letter_number_character_count": sum(is_letter_or_number(c) for c in text),
                    "token_count": len(tokens), "sha256": digest, "tokens": tokens,
                })
            except (UnicodeError, OSError, ValueError) as error:
                errors.append({"source_file": relative, "error": str(error)})
    actual = {d["source_file"] for d in documents} | {e["source_file"] for e in errors}
    for missing in sorted(set(expected_hashes) - actual):
        errors.append({"source_file": missing, "error": "Expected source file missing"})
    write_json(output / "preprocessing_errors.json", errors)
    if errors:
        raise RuntimeError(f"Preprocessing failed for {len(errors)} file(s); see preprocessing_errors.json")
    with (output / "story_tokens.jsonl").open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(json.dumps(document, ensure_ascii=False) + "\n")
    fields = [key for key in documents[0] if key != "tokens"]
    write_csv(output / "story_metadata.csv", fields,
              [{key: d[key] for key in fields} for d in documents])
    return documents


def fit_analysis(documents: list[dict], n_features: int, output: Path) -> dict:
    corpus = {}
    for document in documents:
        corpus.setdefault(document["author"], []).append(document["tokens"])
    stylo = Stylometry(n_features=n_features, distance="cosine")
    stylo.fit_transform(corpus)
    vectors = np.asarray(stylo.document_vectors, dtype=float)
    assert list(stylo.document_ids) == [d["document_id"] for d in documents], "Document ID order mismatch"
    assert list(stylo.document_labels) == [d["author"] for d in documents], "Document author order mismatch"
    features = list(stylo.features)
    assert len(features) == n_features
    assert vectors.shape == (len(documents), n_features)
    assert np.isfinite(vectors).all()
    counts = [Counter(d["tokens"]) for d in documents]
    # Independently check z-scores of each document's relative token frequencies.
    relative = np.array([[count[word] / d["token_count"] for word in features]
                         for count, d in zip(counts, documents)])
    standard_deviation = relative.std(axis=0, ddof=0)
    standard_deviation[standard_deviation < 1e-10] = 1.0
    expected = (relative - relative.mean(axis=0)) / standard_deviation
    np.testing.assert_allclose(vectors, expected, atol=1e-10, rtol=1e-8)
    pca = PCA(n_components=2, svd_solver="full")
    coordinates = pca.fit_transform(vectors)
    assert coordinates.shape == (len(documents), 2)
    assert np.isfinite(coordinates).all()
    stories = []
    for d, point in zip(documents, coordinates):
        row = {key: value for key, value in d.items() if key not in {"tokens", "sha256", "letter_number_character_count"}}
        row.update(pc1=float(point[0]), pc2=float(point[1]))
        stories.append(row)
    write_csv(output / f"story_coordinates_{n_features}.csv", list(stories[0]), stories)
    global_counts = sum(counts, Counter())
    feature_rows = [{"feature_index": i + 1, "word": word, "corpus_token_count": global_counts[word]}
                    for i, word in enumerate(features)]
    write_csv(output / f"selected_features_{n_features}.csv", list(feature_rows[0]), feature_rows)
    np.savez_compressed(output / f"model_vectors_{n_features}.npz", vectors=vectors,
                        relative_frequencies=relative, features=np.array(features),
                        document_ids=np.array(stylo.document_ids),
                        pca_components=pca.components_, pca_mean=pca.mean_,
                        explained_variance_ratio=pca.explained_variance_ratio_)
    top_features = []
    collection_stats = []
    for author, label, directory in COLLECTIONS:
        indices = [i for i, d in enumerate(documents) if d["collection"] == label]
        totals = sum((counts[i] for i in indices), Counter())
        total_tokens = sum(documents[i]["token_count"] for i in indices)
        means = vectors[indices].mean(axis=0)
        ranked = sorted(enumerate(features), key=lambda item: (-totals[item[1]], item[0]))
        for rank, (index, word) in enumerate(ranked, 1):
            top_features.append({"collection": label, "author": author, "rank": rank, "word": word,
                                 "count": totals[word], "total_collection_tokens": total_tokens,
                                 "frequency_per_10000": totals[word] / total_tokens * 10000,
                                 "collection_mean_corpus_zscore": float(means[index]),
                                 "zscore_definition": Z_LABEL})
        collection_stats.append({"collection": label, "author": author, "document_count": len(indices),
                                 "character_count": sum(documents[i]["character_count"] for i in indices),
                                 "token_count": total_tokens,
                                 "pca_centroid": coordinates[indices].mean(axis=0).tolist(),
                                 "mean_distance_to_collection_centroid": float(np.linalg.norm(coordinates[indices] - coordinates[indices].mean(axis=0), axis=1).mean()),
                                 "high_mean_zscore": sorted([r for r in top_features if r["collection"] == label], key=lambda r: -r["collection_mean_corpus_zscore"])[:8],
                                 "low_mean_zscore": sorted([r for r in top_features if r["collection"] == label], key=lambda r: r["collection_mean_corpus_zscore"])[:8]})
    write_csv(output / f"top_features_by_collection_{n_features}.csv", list(top_features[0]), top_features)
    loading_rows = [{"word": word, "pc1_loading": float(pca.components_[0, i]),
                     "pc2_loading": float(pca.components_[1, i])} for i, word in enumerate(features)]
    write_csv(output / f"pca_loadings_{n_features}.csv", list(loading_rows[0]), loading_rows)
    # A descriptive flag in the 2D picture, not an authorship classifier or a test-set score.
    authors = [d["author"] for d in documents]
    author_centroids = {a: coordinates[np.array(authors) == a].mean(axis=0) for a in corpus}
    opposite_side = []
    for d, point in zip(documents, coordinates):
        distances = {a: float(np.linalg.norm(point - centroid)) for a, centroid in author_centroids.items()}
        closest = min(distances, key=distances.get)
        if closest != d["author"]:
            opposite_side.append({"title": d["title"], "document_id": d["document_id"],
                                  "author": d["author"], "collection": d["collection"],
                                  "closer_author_centroid": closest, "distance_to_author_centroids": distances})
    analysis = {"n_features": n_features, "document_count": len(documents),
                "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                "explained_variance_ratio_total": float(pca.explained_variance_ratio_.sum()),
                "author_silhouette_in_full_zscore_space": float(silhouette_score(vectors, authors, metric="cosine")),
                "author_silhouette_in_pca_plane": float(silhouette_score(coordinates, authors)),
                "collections": collection_stats, "author_centroids": {k: v.tolist() for k, v in author_centroids.items()},
                "closer_to_other_author_centroid_in_pca_plane": opposite_side,
                "centroid_flag_definition": "Euclidean distance to two in-sample author centroids in the PCA plane; descriptive only, no held-out validation.",
                "stories": stories}
    write_json(output / f"analysis_summary_{n_features}.json", analysis)
    return analysis


def plot_png(analysis: dict, destination: Path) -> None:
    fig, ax = plt.subplots(figsize=(15, 10), constrained_layout=True)
    texts = []
    for collection, color in COLORS.items():
        rows = [s for s in analysis["stories"] if s["collection"] == collection]
        ax.scatter([s["pc1"] for s in rows], [s["pc2"] for s in rows],
                   label=f"{collection}（{len(rows)} 个文档）", color=color, s=58, edgecolor="white", linewidth=0.7, zorder=3)
        for story in rows:
            texts.append(ax.text(story["pc1"], story["pc2"], story["title"], fontsize=8, alpha=0.9))
    variance = analysis["explained_variance_ratio"]
    ax.set(xlabel=f"PC1（解释方差 {variance[0]:.2%}）", ylabel=f"PC2（解释方差 {variance[1]:.2%}）",
           title=f"鲁迅与沈从文的文体计量 · {analysis['n_features']} 个高频词\n{analysis['document_count']} 个源文件，各一文档；保留原文简繁写法")
    ax.axhline(0, color="#999999", linewidth=0.5, alpha=0.45)
    ax.axvline(0, color="#999999", linewidth=0.5, alpha=0.45)
    ax.grid(alpha=0.12)
    ax.legend(loc="best", frameon=True)
    ax.margins(x=0.10, y=0.10)
    fig.canvas.draw()
    np.random.seed(3242)
    adjust_text(texts, ax=ax, iter_lim=400,
                arrowprops={"arrowstyle": "-", "color": "#9a9a9a", "lw": 0.5},
                expand=(1.12, 1.25))
    fig.savefig(destination, dpi=180, facecolor="#fffdf8")
    plt.close(fig)
    with Image.open(destination) as image:
        image.verify()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", nargs="+", type=int, default=[30, 100, 300])
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    if any(n < 2 for n in args.features):
        parser.error("Each feature count must be at least 2")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    versions = {name: version(name) for name in ["qhchina", "jieba", "matplotlib", "scikit-learn", "numpy", "plotly", "Pillow", "adjustText"]}
    write_json(output / "environment.json", {"python": platform.python_version(), "packages": versions,
               "preprocessing": "Keep Unicode categories L/N; replace every other code point with a space before jieba.cut(cut_all=False,HMM=True); no stopwords or script conversion.",
               "character_count_definition": "len(original UTF-8 decoded text), including whitespace/punctuation; letter_number_character_count is exported separately.",
               "pca": "sklearn PCA(n_components=2, svd_solver='full') fitted to qhchina document_vectors; distance='cosine' is not applied as a PCA distance matrix."})
    print("Verified imports:", versions, flush=True)
    documents = read_documents(output)
    for _, label, _ in COLLECTIONS:
        rows = [d for d in documents if d["collection"] == label]
        print(f"{label}: {len(rows)} documents, {sum(d['token_count'] for d in rows):,} tokens", flush=True)
    analyses = []
    # qhchina's bundled Chinese font must be registered before any plotting.
    load_fonts()
    for count in sorted(set(args.features)):
        analysis = fit_analysis(documents, count, output)
        plot_png(analysis, output / f"pca_{count}.png")
        analyses.append(analysis)
        print(f"{count} features: {analysis['document_count']} PCA points; explained variance {analysis['explained_variance_ratio_total']:.2%}", flush=True)
    write_interactive_html(analyses, output / "pca_interactive.html")
    print(f"Outputs: {output}", flush=True)


if __name__ == "__main__":
    main()
