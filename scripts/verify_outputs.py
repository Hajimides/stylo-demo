#!/usr/bin/env python3
"""Validate saved exercise artifacts independently of the rendering pipeline."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import unicodedata

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs"


def rows(name):
    with (OUTPUT / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    provenance = json.loads((ROOT / "data-provenance.json").read_text(encoding="utf-8"))
    expected_sources = {f["path"]: f for c in provenance["collections"] for f in c["files"]}
    docs = [json.loads(line) for line in (OUTPUT / "story_tokens.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(docs) == len(expected_sources) == provenance["total_txt_files"]
    assert {d["source_file"] for d in docs} == set(expected_sources)
    assert len({d["document_id"] for d in docs}) == len(docs)
    assert not json.loads((OUTPUT / "preprocessing_errors.json").read_text(encoding="utf-8"))
    metadata = rows("story_metadata.csv")
    assert len(metadata) == len(docs)
    counts = []
    for doc, meta in zip(docs, metadata):
        raw = (ROOT / doc["source_file"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == expected_sources[doc["source_file"]]["sha256"] == doc["sha256"]
        assert len(raw.decode("utf-8")) == doc["character_count"]
        assert len(doc["tokens"]) == doc["token_count"] > 0
        assert all(t and all(unicodedata.category(c)[0] in "LN" for c in t) for t in doc["tokens"])
        for key in meta:
            assert str(doc[key]) == meta[key], (key, doc["source_file"])
        counts.append(Counter(doc["tokens"]))
    global_counts = sum(counts, Counter())
    checks = []
    for n in (30, 100, 300):
        coordinates = rows(f"story_coordinates_{n}.csv")
        features = rows(f"selected_features_{n}.csv")
        top_features = rows(f"top_features_by_collection_{n}.csv")
        assert len(coordinates) == len(docs)
        assert [r["document_id"] for r in coordinates] == [d["document_id"] for d in docs]
        names = [f["word"] for f in features]
        assert len(names) == len(set(names)) == n
        assert set(names) == {word for word, count in global_counts.most_common(n)}
        assert len(top_features) == 4 * n
        saved = np.load(OUTPUT / f"model_vectors_{n}.npz", allow_pickle=False)
        relative = np.array([[count[word] / d["token_count"] for word in names] for d, count in zip(docs, counts)])
        np.testing.assert_allclose(relative, saved["relative_frequencies"])
        std = relative.std(axis=0)
        std[std < 1e-10] = 1
        z = (relative - relative.mean(axis=0)) / std
        np.testing.assert_allclose(z, saved["vectors"], atol=1e-10)
        xy = np.array([[float(row["pc1"]), float(row["pc2"])] for row in coordinates])
        np.testing.assert_allclose(xy, (z - saved["pca_mean"]) @ saved["pca_components"].T, atol=1e-9)
        assert np.isfinite(xy).all()
        for collection in provenance["collections"]:
            label = collection["collection"]
            selected = [r for r in top_features if r["collection"] == label]
            assert len(selected) == n and {r["word"] for r in selected} == set(names)
            assert [int(r["rank"]) for r in selected] == list(range(1, n + 1))
            assert [int(r["count"]) for r in selected] == sorted((int(r["count"]) for r in selected), reverse=True)
            indices = [i for i, d in enumerate(docs) if d["collection"] == label]
            total_tokens = sum(docs[i]["token_count"] for i in indices)
            mean_z = z[indices].mean(axis=0)
            for row in selected:
                index = names.index(row["word"])
                count = sum(counts[i][row["word"]] for i in indices)
                assert int(row["count"]) == count
                assert int(row["total_collection_tokens"]) == total_tokens
                np.testing.assert_allclose(float(row["frequency_per_10000"]), count / total_tokens * 10000)
                np.testing.assert_allclose(float(row["collection_mean_corpus_zscore"]), mean_z[index], atol=1e-10)
                assert "corpus-wide" in row["zscore_definition"]
        with Image.open(OUTPUT / f"pca_{n}.png") as image:
            image.load()
            assert image.width >= 1500 and image.height >= 1000
        checks.append({"features": n, "coordinate_rows": len(coordinates), "feature_rows": len(features), "collection_feature_rows": len(top_features), "numerical_checks": "passed"})
    html = (OUTPUT / "pca_interactive.html").read_text(encoding="utf-8")
    assert not re.search(r"__(?:COLLECTION_CARDS|FEATURE_BUTTONS|PLOT_HTML|ANALYSIS_DATA|HOVER_TEMPLATE)__", html)
    assert not re.search(r"<script\b[^>]*\bsrc=", html, re.I), "External JS prevents offline use"
    match = re.search(r'<script id="analysis-data" type="application/json">(.*?)</script>', html, re.S)
    embedded = json.loads(match.group(1))
    assert {a["n_features"] for a in embedded} == {30, 100, 300}
    for analysis in embedded:
        assert len(analysis["stories"]) == len(docs)
        assert [s["document_id"] for s in analysis["stories"]] == [d["document_id"] for d in docs]
    log = (OUTPUT / "run.log").read_text(encoding="utf-8")
    assert not re.search(r"Glyph .* missing from font", log)
    report = {"status": "passed", "document_count": len(docs), "source_hashes": "passed", "token_categories": "passed", "png_decode": "passed", "offline_html_and_placeholders": "passed", "analyses": checks}
    (OUTPUT / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
