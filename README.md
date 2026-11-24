# spectral-network-gui

Native `egui/eframe` viewer for completed spectral-matcher analyses. The GUI does not run the matcher or require its service.

## Run

Install a stable Rust toolchain and run from the repository root:

```bash
cargo run -p spectral-network-gui
```

Generate results from the **spectral-matcher** repository root:

```bash
cargo run -- analyze --config config/analyze.00283.toml
```

In the GUI, under **Input**, click **Open result directory…** and select the configured `output_dir` folder. One selection loads the graph, original network spectrum peaks, library-search results, and node attributes (one row per network feature, including features without library hits); LOTUS metadata loads in the background when configured. Select a graph node to see its top-ranked structure first. **Node details**, **Ranked library hits**, **Spectrum**, and **Peak Filter** are collapsible and start closed; expand them for the original spectrum, all ranked hits with scores and taxonomy, or filtering controls. No separate JSON or TSV selection is required.

In the attributes table, the first column always shows the network **Feature ID** as a link (or its spectrum ID when the input has no feature ID). The **Search mode** button switches between feature-ID search across the full network and all-attribute filtering of visible rows. Clicking a feature link clears graph visibility filters, selects that node, and zooms the graph with the node centered.

Under the left **Style** section, choose a numeric attribute such as peak height or peak area for **Node size column**. Set minimum and maximum node radii in pixels; **Intensity pivot** is the fraction of the observed value range that maps to the midpoint radius (`0.5` gives linear scaling). Nodes without a numeric value retain degree-based size. **Degree (default)** restores the original sizing.

When the result includes sample metadata, **Pie grouping** in **Style** selects which `ATTRIBUTE_` column supplies the pie slices for each node; **Off** restores plain nodes. The legend names each category. **Pie palette** assigns a color scale to the selected grouping: **Contrasting** (categorical default), **Sequential**, or **Diverging**. Each grouping retains its own palette when switched. Slice area is proportional to that feature's category mean peak area; nodes without measured positive areas use the normal fill color. Node size and selection rings also apply to pies.

The result directory must contain `result.toml` and the referenced artifacts:

```toml
network = "network.json"
search = "search.json"
lotus_csv = "/absolute/path/to/lotus.csv" # optional when taxonomy was not run
feature_quant_csv = "/absolute/path/to/quant.csv"    # optional paired inputs
sample_metadata_csv = "/absolute/path/to/metadata.csv"
```

`network` and `search` are paths relative to the result directory. `lotus_csv`, when present, is an absolute CSV path. Provide `feature_quant_csv` and `sample_metadata_csv` together as absolute CSV paths to enable grouped attributes and pies; `analyze` records them from top-level config fields (or from the `[taxonomy]` pair when top-level paths are absent). Network and search JSON are authoritative; the search JSON already embeds TSV attributes. Missing, invalid, or inconsistent manifest/artifacts produce an error instead of showing a partial result. Failure to parse LOTUS is reported separately while the already-loaded network and search hits remain available.

The quantification CSV joins its `row ID` to the MGF `FEATURE_ID`; each `<filename> Peak area` header joins the metadata CSV's `filename`. Only rows with `sample_type = sample` contribute when that column exists. Each nonempty `ATTRIBUTE_` metadata column becomes a grouping, with one table column per category named `<ATTRIBUTE_column>:<category>`. Values are arithmetic means of **observed** sample peak areas (including measured zero; missing observations are excluded). Categories with no observation show a blank table cell, and missing or zero areas contribute no pie slice. Category names are sorted alphabetically per grouping; the selected grouping's positive category means determine each node's proportions.

The web build cannot choose a local result directory; use the native app for this workflow.
