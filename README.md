# CHI3242 第 8 周：鲁迅与沈从文的文体计量

已完成[课程讲义](https://mcjkurz.github.io/teaching/courses/2026-2027/CHI3242/notes/week-08/)的四步练习及 30／300 特征比较。主分析为 `Stylometry(n_features=100, distance="cosine")` 的默认 z-score，随后对文档向量做 scikit-learn PCA。

## 查看结果

- [分析报告及四道观察题的回答](outputs/练习报告.md)
- [离线交互图](outputs/pca_interactive.html)：用浏览器打开，可切换 30／100／300 特征、悬停详情、平移缩放、开关文集、显示篇名、重置。
- [100 特征 PCA 图片](outputs/pca_100.png)
- [100 特征文档坐标 CSV](outputs/story_coordinates_100.csv)
- [100 个全局特征](outputs/selected_features_100.csv)
- [四个文集的特征统计 CSV](outputs/top_features_by_collection_100.csv)：每集 100 行，总计 400 行。
- [语料质量说明](outputs/corpus_quality_notes.md)

四个指定目录共有 43 个 `.txt` 文件，总计 231,457 个保留词元。每文件为一文档，保留自序、序言及“未命名”文件。`孔乙己.txt` 实际连接了三篇作品；《故事新编》原文主要为繁体，其他集合主要为简体。本次保留原始课程语料，以上因素均在报告中说明。

## 重新运行

在项目根目录使用 Python 3.12（本次为 3.12.14）：

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -u scripts/analyze.py > outputs/run.log 2>&1
.venv/bin/python scripts/verify_outputs.py
.venv/bin/python -m unittest discover -s tests -v
```

语料已下载在 `corpora/authors/`，不必重复下载。若要重新获取本次同一上游版本：

```sh
.venv/bin/python scripts/download_corpora.py --commit a62187705288592cc150bf57bd0b4c7d50f4f693
```

下载器使用 shallow sparse clone，仅复制四个指定目录中的 `.txt`，并保护本地已有不同内容。每文件的 SHA-256 与来源提交见 [data-provenance.json](data-provenance.json)。首次绘图时 `qhchina.load_fonts()` 会联网下载中文字体至项目 `.cache/qhchina/fonts/`；缓存完成后可再次使用。输出的 HTML 已内嵌 Plotly 与坐标，浏览时无需网络或 Python。

默认一次生成 30、100、300 特征三组结果。也可以单独运行，结果建议放入另一个目录：

```sh
.venv/bin/python scripts/analyze.py --features 30 --output /private/tmp/stylo-30
```

完整浏览器验证另需 Node.js、Playwright 与 Chromium；它们不属于分析所需的 Python 依赖：

```sh
node scripts/verify_interactive.mjs
```

若 Playwright 不在 Node 的默认模块路径，可将 `PLAYWRIGHT_MODULE` 指向其 `index.mjs` 后运行。浏览器检查会生成 `outputs/browser_validation.json` 和 `outputs/pca_interactive.png`。

## 方法与文件

`scripts/analyze.py` 以 UTF-8 读原文件，将非 Unicode 字母／数字字符全部替换为空格后用 jieba 分词。不去停用词，不转换简繁，不拆分文件；遇到读取失败、散列不符或空词表会报错，记录于 `outputs/preprocessing_errors.json`。

`outputs/story_tokens.jsonl` 保存每个文档的完整词表与元数据。`character_count` 为原始解码文本的字符数，包含标点和空白，另提供仅字母／数字的字符数。CSV 使用带 BOM 的 UTF-8，方便 Excel 读取中文。

同一全局特征集在每个文集内按 token 总数排名。每万词频的分母为文集全部保留词元；`collection_mean_corpus_zscore` 是该文集中各文档 z 向量的等权均值，标准化基准为全部 43 个文档。两者的加权口径不同。`distance="cosine"` 不会使 PCA 的输入变成距离矩阵。

每个特征设置另存 `model_vectors_*.npz`、`pca_loadings_*.csv` 与 `analysis_summary_*.json`，可审计相对词频、z-score、PCA 载荷和解释方差。[verify_outputs.py](scripts/verify_outputs.py) 根据保存的词表独立重算 z-score、坐标投影和文集统计，同时核验来源散列、CSV 行数、PNG 解码及 HTML 数据完整性。

`requirements.txt` 固定直接依赖，`requirements-lock.txt` 记录本次全部 Python 包版本。[environment.json](outputs/environment.json)、[run.log](outputs/run.log)、[validation.json](outputs/validation.json) 与 [browser_validation.json](outputs/browser_validation.json) 保存运行及验证证据。

方法参考：[qhchina Stylometry 官方文档](https://www.qhchina.org/docs/stylometry/)。
