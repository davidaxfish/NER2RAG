# current
 
```python
# ───────────────────────────────────────────────────────────────────────────────
# Interactive PCA Dashboard w/ Silhouette & Suggestions
# Requirements: pandas, holoviews, panel, bokeh, scikit-learn
#   pip install pandas holoviews panel bokeh scikit-learn
# ───────────────────────────────────────────────────────────────────────────────
 
import os, json
import pandas as pd
import holoviews as hv
import panel as pn
from bokeh.palettes import Category20c
from itertools import cycle
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
 
# Initialize
hv.extension('bokeh')
pn.extension()
 
# ───────────────────────────────────────────────────────────────────────────────
# Configuration
# JSON_PATH     = r"C:\Users\e182868\OneDrive - Applied Materials\script_cloud\result\cleaned_synonym_entities\synonym_dict_ici_count_100_80_cleaned.json"
# JSON_PATH     ="./result/cleaned_synonym_entities/synonym_dict_nin2_50_90_cleaned.json"
JSON_PATH     ="./result/cleaned_synonym_entities/hierarchical_entities_100_95_cleaned_1.json"
POINT_SIZE    = 6
PLOT_WIDTH    = 800
PLOT_HEIGHT   = 800
PALETTE_FULL  = Category20c[max(Category20c.keys())]
LEGEND_HEIGHT = 300
TOP_K         = 10
# ───────────────────────────────────────────────────────────────────────────────
 
def load_df(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    raw = json.loads(open(path,'r',encoding='utf8').read())
    ents = raw.get('entities')
    if not isinstance(ents, list):
        raise ValueError("JSON must contain top-level 'entities': [...]")
    rows = []
    for e in ents:
        if not isinstance(e, dict): continue
        ent, coords = e.get('entity'), e.get('coords')
        cluster, parent = e.get('cluster'), e.get('parent',None)
        if (isinstance(ent,str)
            and isinstance(coords,list) and len(coords)>=2
            and all(isinstance(v,(int,float)) for v in coords[:2])):
            rows.append({
                'entity': ent.strip(),
                'x': coords[0], 'y': coords[1],
                'cluster': str(cluster),
                'parent': str(parent) if parent else 'ROOT'
            })
    if not rows:
        raise ValueError("No valid entities found in JSON")
    return pd.DataFrame(rows)
 
def scatter_panel(df, color_by, colors):
    cmap = dict(zip(sorted(df[color_by].unique()), colors))
    dfc = df.assign(color=df[color_by].map(cmap))
    pts = hv.Points(dfc, ['x','y'], ['entity','parent','color'])
    return pts.opts(
        width=PLOT_WIDTH, height=PLOT_HEIGHT,
        tools=['hover','pan','wheel_zoom','box_zoom','reset','save'],
        color='color', cmap=None,
        size=POINT_SIZE,
        xlabel='PC1', ylabel='PC2',
        title=f"PCA Scatter colored by {color_by.capitalize()}",
        show_legend=False, active_tools=['wheel_zoom']
    )
 
def legend_panel(categories, colors):
    items = []
    for cat,col in zip(categories,colors):
        html = (
            f"<div style='display:flex;align-items:center;margin:2px 0;'>"
            f"<div style='width:15px;height:15px;background:{col};"
            f"border:1px solid #000;margin-right:5px;'></div>"
            f"<div>{cat}</div></div>"
        )
        items.append(pn.pane.HTML(html,margin=0))
    return pn.Column(*items, height=LEGEND_HEIGHT, scroll=True, width=200)
 
def stats_panel(df, color_by):
    counts = df.groupby(color_by).size().reset_index(name='count').sort_values('count',ascending=False)
    table = pn.widgets.DataFrame(counts, width=300, height=LEGEND_HEIGHT)
    topk = counts.head(TOP_K)
    bars = hv.Bars(topk, kdims=[color_by], vdims=['count']).opts(
        width=300, height=200, tools=['hover'],
        title=f"Top {TOP_K} by {color_by.capitalize()}", xrotation=45
    )
    return pn.Column("## Counts & Top Categories", table, bars)
 
def create_dashboard():
    df = load_df(JSON_PATH)
    selector = pn.widgets.RadioButtonGroup(name='Color by', options=['cluster','parent'], value='cluster')
 
    @pn.depends(selector.param.value, watch=False)
    def view(color_by):
        # compute color mapping
        cats = sorted(df[color_by].unique())
        n = len(cats)
        colors = PALETTE_FULL[:n] if n<=len(PALETTE_FULL) else [c for _,c in zip(range(n),cycle(PALETTE_FULL))]
        # scatter, legend, stats
        scatter = scatter_panel(df, color_by, colors)
        legend  = legend_panel(cats, colors)
        stats   = stats_panel(df, color_by)
        # silhouette
        # encode labels and compute; fallback if only one category
        labels = LabelEncoder().fit_transform(df[color_by])
        sil = silhouette_score(df[['x','y']], labels) if len(set(labels))>1 else None
        if sil is None:
            sugg = "Only one category—no silhouette."
            sil_text = "Silhouette: N/A"
        else:
            sil_text = f"Silhouette: {sil:.3f}"
            if sil < 0.3:
                sugg = "Low separation: consider merging similar categories or raising ICI thresholds."
            elif sil < 0.5:
                sugg = "Moderate separation: fine-tune cluster filters or synonym groups."
            else:
                sugg = "Good separation—cleaning looks robust."
 
        # display silhouette + suggestion
        sil_panel = pn.pane.Markdown(f"**{sil_text}**\n\n*Suggestion:* {sugg}", width=300)
 
        return pn.Column(
            pn.Row(scatter, pn.Column(legend, stats)),
            pn.Spacer(height=10),
            sil_panel
        )
 
    return pn.Column(
        "# Semantic Entity PCA Explorer",
        "Switch coloring and view stats below; silhouette indicates cluster cohesion.",
        selector,
        view
    )
 
dashboard = create_dashboard()
dashboard.servable()
 
```
 
# cord
 

```python
 
import os, json, re, math
from collections import Counter
import pandas as pd
import holoviews as hv
import panel as pn
from bokeh.palettes import Category20c
from itertools import cycle
hv.extension('bokeh')
pn.extension()
# JSON_PATH = r"C:\Users\e182868\OneDrive - Applied Materials\script_cloud\result\cleaned_synonym_entities\synonym_dict_nin2_50_90_cleaned.json"  # your cleaned entities JSON
JSON_PATH ="./result/cleaned_synonym_entities/hierarchical_entities_100_95_cleaned_1.json"
RAW_CORPUS_PATH  =  r"C:\Users\e182868\OneDrive - Applied Materials\script_cloud\result\cleantext\my_text_file_100_80_mid.txt"
# raw text, one document per line
# ───────────────────────────────────────────────────────────────────────────────
# Configuration (tunable)
EXCLUDE_LIST     = []             # exact strings to drop
PMI_THRESHOLD    = 11          # require PMI ≥ threshold
WINDOW_SPLIT_RE  = r"[.?!]\s+"    # sentence splitter regex
CHORD_WIDTH      = 1200
CHORD_HEIGHT     = 1200
PALETTE_FULL     = Category20c[max(Category20c.keys())]
# ───────────────────────────────────────────────────────────────────────────────
 
def load_labels(json_path, exclude):
    raw = json.loads(open(json_path,'r',encoding='utf8').read())
    parents, entities = set(), set()
    for e in raw.get('entities', []):
        if not isinstance(e, dict): continue
        p = (e.get('parent') or 'ROOT').strip()
        t = e.get('entity','').strip()
        if p and t and not p.isdigit() and not t.isdigit():
            if p not in exclude and t not in exclude:
                parents.add(p); entities.add(t)
    return sorted(parents), sorted(entities)
 
def load_corpus(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return [line.strip() for line in open(path,'r',encoding='utf8') if line.strip()]
 
def compute_cooccurrence(parents, entities, docs, split_re):
    """
    Count total windows N, parent counts, entity counts, and joint counts,
    but skip any p==e pairs.
    """
    N = 0
    p_counts, e_counts, pe_counts = Counter(), Counter(), Counter()
    patterns = {lbl: re.compile(rf"\b{re.escape(lbl)}\b", re.IGNORECASE)
                for lbl in parents+entities}
    for doc in docs:
        windows = re.split(split_re, doc)
        for w in windows:
            N += 1
            found_p = {p for p in parents if patterns[p].search(w)}
            found_e = {e for e in entities if patterns[e].search(w)}
            for p in found_p:
                p_counts[p] += 1
            for e in found_e:
                e_counts[e] += 1
            # only count joint if p != e
            for p in found_p:
                for e in found_e:
                    if p != e:
                        pe_counts[(p, e)] += 1
    return N, p_counts, e_counts, pe_counts
 
def build_pmi_df(N, p_counts, e_counts, pe_counts, threshold):
    rows = []
    for (p, e), j in pe_counts.items():
        if j == 0: continue
        denom = p_counts[p] * e_counts[e]
        if denom == 0: continue
        val = math.log2((j * N) / denom)
        if val >= threshold:
            rows.append({'parent': p, 'entity': e, 'count': j, 'pmi': val})
    if not rows:
        raise ValueError("No pairs pass the PMI threshold")
    return pd.DataFrame(rows)
 
def make_chord_and_table(pmi_df):
    labels = list(pd.unique(pmi_df[['parent','entity']].values.ravel()))
    n = len(labels)
    palette = PALETTE_FULL[:n] if n<=len(PALETTE_FULL) else [c for _,c in zip(range(n), cycle(PALETTE_FULL))]
    nodes_df = pd.DataFrame({'name': labels})
    nodes_df['col'] = nodes_df['name'].map(dict(zip(labels, palette)))
    chord = hv.Chord((pmi_df.rename(columns={'parent':'source','entity':'target','pmi':'value'}),
                      hv.Dataset(nodes_df,'name')))
    chord_plot = chord.opts(
        width=CHORD_WIDTH, height=CHORD_HEIGHT,
        labels='name',
        label_text_font_size='14pt',
        node_color='col', node_cmap=None,
        edge_color_index='value', edge_cmap='Viridis',
        edge_line_width=hv.dim('value')*2,
        edge_alpha=0.8, node_size=15,
        tools=['hover','tap','wheel_zoom','reset'],
        title=f"Parent→Entity Chord (PMI ≥ {PMI_THRESHOLD})"
    )
    table = pn.widgets.DataFrame(pmi_df[['parent','entity','count','pmi']],
                                 name="Relation Counts & PMI",
                                 width=700, height=300)
    return chord_plot, table
 
# ───────────────────────────────────────────────────────────────────────────────
# Execute pipeline
parents, entities = load_labels(JSON_PATH, EXCLUDE_LIST)
docs = load_corpus(RAW_CORPUS_PATH)
N, p_counts, e_counts, pe_counts = compute_cooccurrence(parents, entities, docs, WINDOW_SPLIT_RE)
pmi_df = build_pmi_df(N, p_counts, e_counts, pe_counts, PMI_THRESHOLD)
chord_plot, rel_table = make_chord_and_table(pmi_df)
 
dashboard = pn.Column(
    "# PMI-Driven Parent→Entity Co-occurrence (Excluding p==e)",
    pn.pane.Markdown(f"_Total windows: {N}, PMI threshold: {PMI_THRESHOLD}_"),
    chord_plot,
    rel_table
)
 
dashboard.servable()
 
```
 

