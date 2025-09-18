# PCA engaged
 
```python
import os, sys, json, re
from tqdm import tqdm
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition        import PCA
from sklearn.cluster              import AgglomerativeClustering
import inspect
import matplotlib.pyplot as plt
 
# ── Parameters ──────────────────────────────────────────────────────────────
# INPUT_JSON_PATH      = r'.\result\pair_set\bestpair_SVO\hierarchical_entities_100_95.json'
# OUTPUT_JSON_PATH     = "./result/cleaned_synonym_entities/hierarchical_entities_100_95_cleaned.json"
# INPUT_JSON_PATH = r"C:\Users\e182868\OneDrive - Applied Materials\script_cloud\result\pair_set\bestpair_SVO\hierarchical_entities_100_95.json"
INPUT_JSON_PATH = HIERARCHY_PATH
OUTPUT_JSON_PATH = "./result/cleaned_synonym_entities/hierarchical_entities_100_95_cleaned_1.json"
 
MIN_COUNT            = 20
MIN_ICI              = 0.001
MAX_NGRAM_LENGTH     = 4
CLUSTER_DISTANCE     = 1.0
PCA_COMPONENTS       = 2
ALPHA                = 0.5
 
# ── Monkey-patch thinc.compat so spaCy won’t crash on missing cublas/cupy ──
try:
    import thinc.compat as _compat
    _compat.cublas = None
    _compat.cupy   = None
    _compat.cupyx  = None
except ImportError:
    # either thinc or compat isn’t even present – spaCy import will fail later
    pass
 
# ── Load spaCy now that our patch is in place ───────────────────────────────
try:
    import spacy, spacy.cli
    # disable GPU probing
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    try:
        nlp = spacy.load("en_core_web_sm", disable=["parser","ner"])
    except OSError:
        print("[spaCy] downloading model en_core_web_sm…")
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm", disable=["parser","ner"])
    print("[spaCy] loaded `en_core_web_sm` successfully; POS filtering active")
    USE_NLP = True
except Exception as e:
    print(f"[ERROR] spaCy could not load: {type(e).__name__}: {e}")
    sys.exit(1)  # fatal now that you insist on not bypassing
 
# ── Helpers ────────────────────────────────────────────────────────────────
def load_data(path):
    """
    Flatten a hierarchical JSON of { parent_key: {entity, ici, count, children:[…]}, … }
    into a list of dicts: [{"entity":…, "ici_score":…, "count":…}, …].
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
 
    out = []
    # if top-level "entities" list present
    if isinstance(data, dict) and "entities" in data and isinstance(data["entities"], list):
        data = data["entities"]
 
    # handle list-of-dicts
    if isinstance(data, list):
        for item in data:
            ent = item.get("entity") or item.get("name") or None
            ici = item.get("ici_score") or item.get("ici") or 0
            cnt = item.get("count", 0)
            if ent:
                out.append({"entity": ent, "ici_score": float(ici), "count": int(cnt)})
        return out
 
    # handle dict-of-parents
    if isinstance(data, dict):
        for parent in data.values():
            ent = parent.get("entity")
            ici = parent.get("ici_score", parent.get("ici", 0))
            cnt = parent.get("count", 0)
            out.append({"entity": ent, "ici_score": float(ici), "count": int(cnt)})
 
            for child in parent.get("children", []):
                ent_c = child.get("entity")
                ici_c = child.get("ici_score", child.get("ici", 0))
                cnt_c = child.get("count", 0)
                out.append({"entity": ent_c, "ici_score": float(ici_c), "count": int(cnt_c)})
        return out
 
    raise ValueError(f"Unrecognized JSON format at {path}")
 
def normalize(text: str) -> str:
    text = text.lower().strip()
    return re.sub(r"[^\w\s]", "", text)
 
def pos_filter(candidates):
    filtered = []
    for item in candidates:
        txt = normalize(item["entity"])
        # length filter
        if len(txt.split()) > MAX_NGRAM_LENGTH:
            continue
        # POS filter
        doc = nlp(txt)
        if any(tok.pos_.startswith("VERB") for tok in doc):
            continue
        # ICI/count thresholds
        if item["ici_score"] < MIN_ICI or item["count"] < MIN_COUNT:
            continue
        filtered.append({**item, "entity": txt})
    return filtered
 
def compute_significance(items):
    ici = np.array([it["ici_score"] for it in items])
    cnt = np.array([it["count"]     for it in items])
    n_ici = (ici - ici.min()) / (ici.ptp() or 1)
    n_cnt = (cnt - cnt.min()) / (cnt.ptp() or 1)
    sig   = ALPHA * n_ici + (1 - ALPHA) * n_cnt
    for i, it in enumerate(items):
        it["significance"] = float(sig[i])
    return items
 
def cluster_entities(items):
    texts  = [it["entity"] for it in items]
    mat    = TfidfVectorizer().fit_transform(texts).toarray()
    coords = PCA(n_components=PCA_COMPONENTS).fit_transform(mat)
 
    sig    = inspect.signature(AgglomerativeClustering.__init__)
    params= {"n_clusters": None,
             "distance_threshold": CLUSTER_DISTANCE,
             "linkage": "ward"}
    if "metric" in sig.parameters:
        params["metric"] = "euclidean"
    else:
        params["affinity"] = "euclidean"
 
    clusterer = AgglomerativeClustering(**params)
    labels    = clusterer.fit_predict(coords)
 
    for i, it in enumerate(items):
        it["coords"]  = coords[i].tolist()
        it["cluster"] = int(labels[i])
    return items
 
def build_hierarchy(items):
    by_ent = {it["entity"]: it for it in items}
    for it in items:
        # pick the shortest parent token that appears in this entity
        it["parent"] = next(
            (cand for cand in sorted(by_ent, key=lambda x: len(x.split()))
             if cand != it["entity"] and cand in it["entity"].split()),
            None
        )
    return items
 
def save_cleaned(items, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"entities": items}, f, indent=2)
 
# ── Pipeline ───────────────────────────────────────────────────────────────
print("1) Loading & flattening JSON…")
cands = load_data(INPUT_JSON_PATH)
print(f"   → {len(cands)} total entities (including children)")
 
print("2) POS + threshold filtering…")
filt  = pos_filter(cands)
print(f"   → {len(filt)} remain after POS / ICI / count")
 
print("3) Computing significance…")
sig   = compute_significance(filt)
 
print("4) Clustering via PCA+Agglomerative…")
clust = cluster_entities(sig)
 
print("5) Building parent hierarchy…")
hier  = build_hierarchy(clust)
 
print(f"6) Saving cleaned entities → {OUTPUT_JSON_PATH}")
save_cleaned(hier, OUTPUT_JSON_PATH)
 
print("7) Visualizing clusters…")
xs = [it["coords"][0] for it in hier]
ys = [it["coords"][1] for it in hier]
lbl= [it["cluster"]    for it in hier]
 
fig, ax = plt.subplots(figsize=(8,6))
scatter = ax.scatter(xs, ys, c=lbl, cmap="tab10", s=50, alpha=0.7)
ax.set_title("Entity Clusters (PCA 2D)")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
plt.show()
 
print("✅ Pipeline completed successfully!")
 

```