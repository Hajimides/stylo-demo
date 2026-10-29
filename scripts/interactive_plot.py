"""Build the self-contained, offline PCA explorer used by the exercise.

The input coordinates must already have been computed from qhchina document
vectors. This module only renders results; it never refits or rescales them.
"""

from __future__ import annotations

import html
import json
import math
from collections import Counter
from pathlib import Path


COLLECTIONS = ("呐喊", "彷徨", "故事新编", "沈从文")
COLORS = ("#c0553e", "#d5a441", "#7556a4", "#2d8190")


def _safe_json(value: object) -> str:
    """Escape characters that can terminate an HTML script element."""
    return (
        json.dumps(value, ensure_ascii=False, allow_nan=False)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def write_interactive_html(analyses: list[dict], output_path: Path) -> None:
    """Write an offline explorer, preferring 100 features when available.

    Each analysis contains ``n_features``, ``explained_variance_ratio`` and
    ``stories``. Each story contains ``document_id``, ``author``, ``collection``,
    ``source_file``, ``title``, ``token_count``, ``character_count``, ``pc1`` and
    ``pc2``. Feature settings may differ, but must use the same story documents.
    """
    import plotly.graph_objects as go
    import plotly.io as pio

    by_features = {int(item["n_features"]): item for item in analyses}
    if not by_features or len(by_features) != len(analyses):
        raise ValueError("Analyses must be nonempty and have unique feature counts.")
    initial_features = 100 if 100 in by_features else min(by_features)
    initial = by_features[initial_features]
    document_ids = [story["document_id"] for story in initial["stories"]]
    if not document_ids or len(document_ids) != len(set(document_ids)):
        raise ValueError("Expected a nonempty set of unique story document IDs.")
    for analysis in analyses:
        if [story["document_id"] for story in analysis["stories"]] != document_ids:
            raise ValueError("All analyses must contain the same stories in the same order.")
        if len(analysis["explained_variance_ratio"]) != 2:
            raise ValueError("Expected exactly two explained variance ratios.")
        for story in analysis["stories"]:
            if story["collection"] not in COLLECTIONS:
                raise ValueError(f"Unexpected collection: {story['collection']!r}")
            if not all(math.isfinite(float(story[key])) for key in ("pc1", "pc2")):
                raise ValueError("All PCA coordinates must be finite.")

    hover_template = (
        "<b>《%{customdata[0]}》</b><br>"
        "作者：%{customdata[1]}<br>"
        "文集：%{customdata[2]}<br>"
        "文档 ID：%{customdata[3]}<br>"
        "原文字符数：%{customdata[4]:,}<br>"
        "分词数：%{customdata[5]:,}<br>"
        "PC1：%{x:.4f}<br>PC2：%{y:.4f}<br>"
        "来源：%{customdata[6]}<extra></extra>"
    )

    def make_trace(collection: str, color: str):
        stories = [s for s in initial["stories"] if s["collection"] == collection]
        return go.Scatter(
            x=[s["pc1"] for s in stories],
            y=[s["pc2"] for s in stories],
            name=collection,
            legendgroup=collection,
            mode="markers",
            text=[html.escape(s["title"]) for s in stories],
            textposition="top center",
            textfont={"size": 10},
            marker={"color": color, "size": 10, "opacity": 0.85,
                    "line": {"color": "#fffdf8", "width": 1}},
            customdata=[
                [html.escape(s["title"]), html.escape(s["author"]),
                 html.escape(s["collection"]), html.escape(s["document_id"]),
                 s["character_count"], s["token_count"],
                 html.escape(s["source_file"])]
                for s in stories
            ],
            hovertemplate=hover_template,
        )

    variance = initial["explained_variance_ratio"]
    fig = go.Figure([make_trace(c, color) for c, color in zip(COLLECTIONS, COLORS)])
    fig.update_layout(
        template="plotly_white",
        font={"family": '"PingFang SC", "Microsoft YaHei", sans-serif', "color": "#263b3c"},
        paper_bgcolor="#fffdf8",
        plot_bgcolor="#fffdf8",
        margin={"l": 64, "r": 30, "t": 32, "b": 65},
        hovermode="closest",
        dragmode="pan",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02,
                "xanchor": "left", "x": 0, "title": {"text": "小说集"}},
        xaxis={"title": f"PC1 · {variance[0]:.2%}", "zerolinecolor": "#c2c9c5",
               "gridcolor": "#e9ece7"},
        yaxis={"title": f"PC2 · {variance[1]:.2%}", "zerolinecolor": "#c2c9c5",
               "gridcolor": "#e9ece7"},
    )
    plot_html = pio.to_html(
        fig, full_html=False, include_plotlyjs=True, div_id="pca-plot",
        default_width="100%", default_height="640px",
        config={"responsive": True, "displaylogo": False, "scrollZoom": True,
                "displayModeBar": True, "modeBarButtonsToRemove": ["select2d", "lasso2d"],
                "toImageButtonOptions": {"filename": "stylometry-pca", "scale": 2}},
    )
    counts = Counter(s["collection"] for s in initial["stories"])
    cards = "".join(
        f'<div class="collection-card"><span class="dot" style="background:{color}"></span>'
        f'<span>{html.escape(c)}</span><strong>{counts[c]}</strong><small>个文档</small></div>'
        for c, color in zip(COLLECTIONS, COLORS)
    )
    buttons = "".join(
        f'<button class="feature-button" type="button" data-features="{n}" '
        f'aria-pressed="{str(n == initial_features).lower()}">{n} 个词</button>'
        for n in sorted(by_features)
    )
    document = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>鲁迅与沈从文 · 文体计量 PCA</title>
<style>
:root{color-scheme:light;--ink:#263b3c;--muted:#657572;--paper:#fffdf8;--line:#dce1d8;--accent:#215a56}
*{box-sizing:border-box}body{margin:0;background:#f1f3eb;color:var(--ink);font:15px/1.75 "PingFang SC","Microsoft YaHei",sans-serif}
main{max-width:1260px;margin:0 auto;padding:40px 32px 28px}.eyebrow{font-size:12px;letter-spacing:.16em;color:var(--accent);font-weight:700}
h1{font-family:"Songti SC","SimSun",serif;font-size:clamp(26px,4vw,42px);font-weight:600;letter-spacing:.04em;line-height:1.3;margin:10px 0 14px}
.intro{color:var(--muted);max-width:850px;margin:0 0 24px}.collections{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:0 0 24px}
.collection-card{display:flex;align-items:center;gap:10px;background:var(--paper);border:1px solid var(--line);border-radius:10px;padding:14px 18px}
.collection-card strong{font-size:25px;margin-left:auto;font-variant-numeric:tabular-nums}.collection-card small{color:var(--muted)}.dot{width:10px;height:10px;border-radius:50%;flex:none}
.plot-card{background:var(--paper);border:1px solid var(--line);border-radius:14px;overflow:hidden;box-shadow:0 8px 28px #263b3c08}
.toolbar{display:flex;justify-content:space-between;gap:14px;align-items:center;padding:18px 22px;border-bottom:1px solid var(--line);flex-wrap:wrap}
.group{display:flex;align-items:center;gap:10px;flex-wrap:wrap}.group-label{font-size:13px;color:var(--muted);margin-right:4px}
button{font:inherit;font-size:13px;cursor:pointer;border:1px solid var(--line);background:#fffdf8;color:var(--ink);padding:7px 14px;border-radius:7px;line-height:1.5}
button:hover{border-color:var(--accent)}button:focus-visible,input:focus-visible{outline:3px solid #8bb8ac;outline-offset:3px}
button[aria-pressed=true]{background:var(--accent);border-color:var(--accent);color:white}label{font-size:13px;display:flex;align-items:center;gap:6px;cursor:pointer}
input{accent-color:var(--accent)}.plot-summary{padding:12px 22px 0;color:var(--muted);font-size:13px;min-height:36px;font-variant-numeric:tabular-nums}
#pca-plot{width:100%;min-height:480px}.help{padding:12px 22px 16px;margin:0;border-top:1px solid var(--line);color:var(--muted);font-size:13px}
.notes{display:grid;grid-template-columns:1fr 1fr;gap:26px;padding:24px 2px 8px;font-size:14px;color:var(--muted)}.notes h2{font-size:15px;color:var(--ink);margin:0 0 5px}.notes p{margin:0}
footer{border-top:1px solid var(--line);margin-top:20px;padding-top:14px;font-size:12px;color:var(--muted)}a{color:var(--accent)}
@media(max-width:720px){main{padding:24px 14px}.collections{grid-template-columns:1fr 1fr;gap:8px}.collection-card{padding:10px 12px;gap:7px}.collection-card strong{font-size:22px}.toolbar{padding:14px}.notes{grid-template-columns:1fr;gap:16px}#pca-plot{min-height:520px}.help,.plot-summary{padding-left:14px;padding-right:14px}}
</style>
</head>
<body>
<main>
<header><div class="eyebrow">CHI3242 · 第 8 周 · 文体计量学</div>
<h1>鲁迅与沈从文的文体坐标</h1>
<p class="intro">每个点是一个源 .txt 文档，每个文件保留为整体。用高频词的相对频率，经 qhchina 的全语料 z 分数标准化后，投影到两个主成分；切换特征数量，观察结构是否稳定。</p></header>
<div class="collections" aria-label="各小说集文档数">__COLLECTION_CARDS__</div>
<section class="plot-card" aria-label="文档 PCA 交互图">
<div class="toolbar"><div class="group"><span class="group-label">高频词特征</span>__FEATURE_BUTTONS__</div>
<div class="group"><label><input id="show-titles" type="checkbox">显示篇名</label><button id="reset-view" type="button">重置视图</button></div></div>
<div id="plot-summary" class="plot-summary" role="status" aria-live="polite"></div>
__PLOT_HTML__
<p class="help">悬停查看文档详情；拖动平移，滚轮缩放。点击图例隐藏或显示文集，双击图例只查看该文集。右上角工具栏可框选缩放、恢复比例和下载图片。</p>
</section>
<div class="notes"><section><h2>如何读这张图</h2><p>点靠近，表示它们在所选特征的二维投影中较相似。坐标轴百分比是该主成分解释的方差；未展示的维度仍包含信息。不同特征设置分别拟合 PCA，坐标轴的方向和比例可能改变。</p></section>
<section><h2>解释边界</h2><p>这是探索性比较，不能直接当作作者判定。文集、题材、成书时期、篇幅和分词方法都可能影响位置。原始《故事新编》为繁体，其他集合为简体，本分析未做简繁转换；字形与分词差异也是混杂因素。源文件还包含序言，以及《孔乙己》文件串接三篇作品等问题，详见<a href="corpus_quality_notes.md">语料质量说明</a>。在图上靠近另一位作者，也不等于归属错误。每个文档权重相同。</p></section></div>
<footer>本文件已内嵌全部坐标与 Plotly，保存后可离线打开。__FEATURE_SETTINGS_LABEL__；方法为 qhchina 默认 z-score → scikit-learn PCA。<a href="https://mcjkurz.github.io/teaching/courses/2026-2027/CHI3242/notes/week-08/">课程练习</a></footer>
</main>
<script id="analysis-data" type="application/json">__ANALYSIS_DATA__</script>
<script>
"use strict";
(() => {
  const analyses = JSON.parse(document.getElementById("analysis-data").textContent);
  const byFeatures = new Map(analyses.map(a => [a.n_features, a]));
  const collections = ["呐喊", "彷徨", "故事新编", "沈从文"];
  const colors = ["#c0553e", "#d5a441", "#7556a4", "#2d8190"];
  const plot = document.getElementById("pca-plot");
  const initialFeatures = __INITIAL_FEATURES__;
  const state = {n_features: initialFeatures, point_count: byFeatures.get(initialFeatures).stories.length, show_titles: false};
  window.__STYLO_DATA__ = analyses;
  window.__STYLO_STATE__ = state;
  const escapeHTML = text => String(text).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  function updateSummary(analysis) {
    const variance = analysis.explained_variance_ratio;
    document.getElementById("plot-summary").textContent = `${analysis.n_features} 个高频词 · ${analysis.stories.length} 个文档 · PC1 ${(variance[0]*100).toFixed(2)}% · PC2 ${(variance[1]*100).toFixed(2)}% · 合计 ${((variance[0]+variance[1])*100).toFixed(2)}%`;
  }
  async function showAnalysis(nFeatures) {
    const analysis = byFeatures.get(nFeatures);
    const visibility = new Map(plot.data.map(trace => [trace.name, trace.visible]));
    const traces = collections.map((collection, index) => {
      const stories = analysis.stories.filter(s => s.collection === collection);
      return {type:"scatter", x:stories.map(s=>s.pc1), y:stories.map(s=>s.pc2), name:collection,
        legendgroup:collection, visible:visibility.get(collection), mode:state.show_titles ? "markers+text" : "markers",
        text:stories.map(s=>escapeHTML(s.title)), textposition:"top center", textfont:{size:10},
        marker:{color:colors[index],size:10,opacity:0.85,line:{color:"#fffdf8",width:1}},
        customdata:stories.map(s=>[escapeHTML(s.title),escapeHTML(s.author),escapeHTML(s.collection),escapeHTML(s.document_id),s.character_count,s.token_count,escapeHTML(s.source_file)]),
        hovertemplate:__HOVER_TEMPLATE__};
    });
    const layout = Object.assign({}, plot.layout, {
      xaxis:Object.assign({}, plot.layout.xaxis,{title:{text:`PC1 · ${(analysis.explained_variance_ratio[0]*100).toFixed(2)}%`},autorange:true}),
      yaxis:Object.assign({}, plot.layout.yaxis,{title:{text:`PC2 · ${(analysis.explained_variance_ratio[1]*100).toFixed(2)}%`},autorange:true})
    });
    await Plotly.react(plot, traces, layout);
    state.n_features = nFeatures;
    state.point_count = analysis.stories.length;
    document.querySelectorAll("[data-features]").forEach(button => button.setAttribute("aria-pressed", String(Number(button.dataset.features) === nFeatures)));
    updateSummary(analysis);
  }
  document.querySelectorAll("[data-features]").forEach(button => button.addEventListener("click", () => showAnalysis(Number(button.dataset.features))));
  document.getElementById("show-titles").addEventListener("change", event => {
    state.show_titles = event.target.checked;
    Plotly.restyle(plot, {mode:state.show_titles ? "markers+text" : "markers"});
  });
  document.getElementById("reset-view").addEventListener("click", () => {
    Plotly.restyle(plot,{visible:true});
    Plotly.relayout(plot,{"xaxis.autorange":true,"yaxis.autorange":true});
  });
  updateSummary(byFeatures.get(initialFeatures));
})();
</script>
</body>
</html>
"""
    for placeholder, replacement in (
        ("__COLLECTION_CARDS__", cards),
        ("__FEATURE_BUTTONS__", buttons),
        ("__PLOT_HTML__", plot_html),
        ("__ANALYSIS_DATA__", _safe_json(analyses)),
        ("__HOVER_TEMPLATE__", _safe_json(hover_template)),
        ("__INITIAL_FEATURES__", _safe_json(initial_features)),
        ("__FEATURE_SETTINGS_LABEL__", "本页提供 " + "／".join(str(n) for n in sorted(by_features))
         + f" 个特征设置，默认 {initial_features} 个特征"),
    ):
        document = document.replace(placeholder, replacement)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")
