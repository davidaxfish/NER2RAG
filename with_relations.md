```python
# ENT_JSON_PATH = r"C:\Users\e182868\OneDrive - Applied Materials\script_cloud\result\synonym_dict_nin2_50_90.json"    # your entity hierarchy JSON
ENT_JSON_PATH = "./result/cleaned_synonym_entities/hierarchical_entities_100_95_cleaned_1.json"
CORPUS_PATH   = r"C:\Users\e182868\OneDrive - Applied Materials\script_cloud\result\cleantext\my_text_file_100_80_mid.txt"     # raw text, one document per line
```
 

```python
# ───────────────────────────────────────────────────────────────────────────────
# Enhanced Joint Entity–Relation Extraction w/ Safe Chunking
# Requirements: spacy, rapidfuzz, tqdm, networkx, pandas, matplotlib
# ───────────────────────────────────────────────────────────────────────────────
 
import os, json, math
from collections import defaultdict, Counter
from itertools import islice
 
from rapidfuzz import process, fuzz
from tqdm.auto import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import spacy
import networkx as nx
 
# ───────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ENT_JSON_PATH   = "628JSON.txt"
# CORPUS_PATH     = "corpus.txt"
# how many characters per chunk (must be <= nlp.max_length)
CHAR_CHUNK_SIZE = 80_000
# overlap so we don’t cut sentences in half
CHAR_OVERLAP    = 1_000
# sentences per NLP window
SENT_WINDOW     = 5
# fuzzy thresholds
FUZZY_THRESH    = 90
OVERLAP_THRESH  = 85
CONF_BASE       = 0.6
# ensure spacy will never choke on raw docs
nlp = spacy.load("en_core_web_sm")
nlp.max_length = max(nlp.max_length, CHAR_CHUNK_SIZE + 1)
# ───────────────────────────────────────────────────────────────────────────────
 
def load_entities(path):
    data = json.loads(open(path,'r',encoding='utf8').read())
    parent_map, ents = {}, set()
    flat = data.get("entities") or data.get("entity")
    if isinstance(flat, list):
        for it in flat:
            if isinstance(it, dict) and "entity" in it and "parent" in it:
                e, p = it["entity"].strip(), str(it["parent"]).strip() or "ROOT"
                ents.add(e); parent_map[e] = p
        if ents:
            return data, sorted(ents, key=len, reverse=True), parent_map
    if isinstance(data, dict):
        for p,lst in data.items():
            if isinstance(lst, list):
                for it in lst:
                    if isinstance(it, dict) and "entity" in it:
                        e = it["entity"].strip()
                        ents.add(e); parent_map[e] = p.strip() or "ROOT"
        if ents:
            return data, sorted(ents, key=len, reverse=True), parent_map
    raise ValueError("Entity JSON format not recognized")
 
def load_corpus(path):
    return [ln.strip() for ln in open(path,'r',encoding='utf8') if ln.strip()]
 
def char_chunks(text, size=CHAR_CHUNK_SIZE, overlap=CHAR_OVERLAP):
    start = 0
    L = len(text)
    while start < L:
        end = min(start + size, L)
        yield text[start:end]
        start = max(end - overlap, end)
 
def segment_windows(text):
    """Sentence‐based windows on a safe‐sized chunk."""
    doc = nlp(text)
    sents = [s.text for s in doc.sents]
    for i in range(0, len(sents), SENT_WINDOW):
        yield " ".join(islice(sents, i, i + SENT_WINDOW))
 
def filter_similar(src, tgt):
    sim = fuzz.token_sort_ratio(src, tgt)
    return sim < OVERLAP_THRESH and src.lower() not in tgt.lower() and tgt.lower() not in src.lower()
 
def extract_relations(docs, entities, parent_map):
    rels = defaultdict(lambda: {"conf":0, "evid":[]})
 
    for did, full_doc in enumerate(tqdm(docs, desc="Docs")):
        # break the doc into safe chunks
        for chunk in char_chunks(full_doc):
            for win in segment_windows(chunk):
                if not win: continue
 
                # 1) detect fuzzy mentions
                found = {m[0] for m in process.extract(
                    win, entities,
                    scorer=fuzz.partial_ratio,
                    score_cutoff=FUZZY_THRESH
                )}
                mentions = sorted(found, key=len, reverse=True)
                pairs = [(s,t) for i,s in enumerate(mentions)
                         for t in mentions[i+1:]
                         if filter_similar(s,t)]
                if not pairs:
                    continue
 
                # 2) build dependency graph once
                sp = nlp(win)
                G = nx.Graph()
                for tok in sp:
                    for ch in tok.children:
                        G.add_edge(tok.i, ch.i)
 
                # 3) extract subj–verb–obj relations
                for tok in sp:
                    if tok.pos_ == "VERB":
                        subs = [c.text for c in tok.children if c.dep_ in ("nsubj","nsubjpass")]
                        objs = [c.text for c in tok.children if c.dep_ in ("dobj","pobj","dative")]
                        for src,tgt in pairs:
                            if any(src.lower() in s.lower() for s in subs) and any(tgt.lower() in o.lower() for o in objs):
                                rel = tok.lemma_
                                conf = min(0.9, CONF_BASE + 0.1)
                                key = (src,tgt,rel)
                                rec = rels[key]
                                rec["conf"] = max(rec["conf"], conf)
                                rec["evid"].append({"doc_id":f"doc{did}", "window":win})
 
    return [
        {"source":s,"target":t,"type":r,"confidence":round(v["conf"],2),"evidence":v["evid"]}
        for (s,t,r),v in rels.items()
    ]
 
def compute_pmi(rels):
    df = pd.DataFrame(rels)[["source","target"]]
    cnt = df.groupby(["source","target"]).size().reset_index(name="count")
    total = cnt["count"].sum()
    pc = cnt.groupby("source")["count"].sum()
    tc = cnt.groupby("target")["count"].sum()
    def pmi_row(r):
        j,p,t = r["count"], pc[r["source"]], tc[r["target"]]
        return math.log2((j*total)/(p*t)) if j and p and t else float("-inf")
    cnt["pmi"] = cnt.apply(pmi_row,axis=1)
    print("PMI = log2(joint*N/(source*N * target*N))")
    return cnt
 
def visualize(rels):
    freqs = Counter(r["type"] for r in rels)
    plt.figure(figsize=(20,8))
    plt.bar(freqs.keys(), freqs.values())
    plt.xticks(rotation=45,ha="right")
    plt.title("Relation Types")
    plt.tight_layout(); plt.show()
 
def merge_and_save(ent_json, rels, out="with_relations.json"):
    ent_json["relations"] = rels
    with open(out,"w",encoding="utf8") as f:
        json.dump(ent_json,f,indent=2,ensure_ascii=False)
    print("Saved →", out)
 
def main():
    ent_json, entities, parent_map = load_entities(ENT_JSON_PATH)
    docs = load_corpus(CORPUS_PATH)
    rels = extract_relations(docs, entities, parent_map)
    pmi_df = compute_pmi(rels)
    print(pmi_df.head())
    visualize(rels)
    merge_and_save(ent_json, rels)
 
if __name__=="__main__":
    main()
 

```
 