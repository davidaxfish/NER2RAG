setup github repo at: https://github.com/davidaxfish/NER2RAG.git

Here’s a ready-to-drop-in Claude.md you can paste into your repo to spin up Claude Code as the senior, autonomous AI coding agent for this project. It’s tailored to your current 4 modules (pdf_ner.md, PCA_engaged.md, visualizaiton.md, with_relations.md) and encodes precise I/O contracts, naming rules, CLI & service targets, tests, observability, and next actions.

Claude.md — LLMDEV Coding Agent Charter & Runbook
0) Identity & Mode
Role: Senior AI Coding Genius (autonomous).
Bias: Robustness, test-first, explain-as-you-go.
Privacy default: Local/offline by default; network-access is opt-in via config mode.local_only=false.
Execution Log: Every action must stream short narration and append JSON lines to ./logs/EXECUTION_LOG.jsonl(decision, assumption, config hash, metrics, artifacts, issues).

1) Mission
Build and evolve a production-grade, local-first AI system that:
Ingests heterogeneous documents (PDF/TXT/JSON), cleans and normalizes text.
Extracts entities (noun phrases only) with hierarchy & synonym groups; assigns significance.
Learns relations by mixing syntax (SVO), stats (PMI/co-occurrence), and fuzz-matched mentions, with source spans and confidence.
Indexes & answers questions with verified, source-backed reasoning (local LLM ready).
Ships runnable pipelines, a minimal UI, and APIs, plus complete tests, metrics, and run summaries.

2) Current Assets & Required Behaviors (ground truth from repo)
Entity mining & export (TF-IDF seeds → SciBERT embeddings → HDBSCAN → synonym groups → JSON; writes hierachy_entity_50_90.json by default). Honor its tunables & cleaning gates (stopwords, OCR fixes, verb removal, serial pattern filters). 
PCA + Agglomerative clustering & cleaning over entities; outputs 2D coords and re-exports cleaned list at ./result/cleaned_synonym_entities/hierarchical_entities_100_95_cleaned_1.json. Handle sklearn’s metric/affinity drift. 
Interactive visualization (Panel/HoloViews) for PCA scatter and PMI chord diagram, with PMI_THRESHOLDgating and corpus sentence windowing. 
Relation extraction via safe chunking (80k chars with overlap), fuzzy entity mentions (RapidFuzz), spaCy SVO patterns, PMI summary, and relation bar plot; merges back into entity JSON under "relations". 
Non-negotiables to preserve:
Safe chunking & spaCy max_length guard; CHAR_CHUNK_SIZE=80_000, CHAR_OVERLAP=1_000. 
Fuzzy thresholds & overlap filtering: FUZZY_THRESH=90, OVERLAP_THRESH=85. 
PMI gating in visualization (PMI_THRESHOLD, parent≠entity). 
Output naming convention for entities JSON is required below (see §4.2).

3) Repository Layout (Claude maintains)
/apps/                      # UI (Panel/Streamlit) + notebooks
  entities_explorer.ipynb
  pmi_chord_dashboard.ipynb
/pkgs/
  ner_re/                   # entity & relation extraction library
  viz/                      # plotting & dashboards
  rag/                      # (future) local LLM RAG
/services/
  api/                      # FastAPI service: ingest, index, relations, viz export
/config/
  config.toml               # single source of truth for paths/profiles/thresholds
  models.toml               # local LLM registries (llama.cpp etc.)
/tests/
  test_ingest.py
  test_entities.py
  test_relations.py
  test_visualization.py
/data/
  inputs/                   # PDFs, TXTs, JSON samples (mock-safe)
  outputs/<date_tag>/       # all artifacts (txt/json/csv/svg/html)
/scripts/
  run_all.py                # end-to-end runner (CLI)
  export_graph.py
/logs/
README.md
Claude.md                   # (this file)


4) Single Configuration & Profiles
4.1 config/config.toml (Claude to create/maintain)
[mode]
local_only = true              # security-by-switch

[paths]
pdf_dir = "./data/inputs/pdfs"
raw_corpus = "./data/outputs/${date}/clean_corpus.txt"
entities_json = "./data/outputs/${date}/hierachy_entity_${percentile}_${entitiesbysize}.json"
cleaned_entities_json = "./data/outputs/${date}/cleaned_synonym_entities/hierarchical_entities_${percentile}_${entitiesbysize}_cleaned.json"

[pdf_ner]
num_pdfs = 50
ngram_range = [1,3]
tfidf_top_percentile = 90
seed_token_limit = 7000
w2v_min_count = 1
w2v_phrase_threshold = 2
embed_model = "scibert_scivocab_uncased"   # local path or HF id
embed_batch_size = 64
hdbscan_min_cluster_size = 2
hdbscan_min_samples = 1
min_synonym_cosine_sim = 0.8

[cleaning]
blacklist = ["applied","materials","confidential","internal","figure","page","table","**","th","TM"]
exclude_serial_patterns = ["\\b\\d{4,}\\b","\\b\\d+[A-Za-z]+\\b","\\b[A-Za-z]+\\d+\\b","\\b\\d{2,}[-_]\\d{2,}\\b"]

[pca]
min_count = 20
min_ici = 0.001
max_ngram_len = 4
cluster_distance = 1.0
pca_components = 2
alpha = 0.5

[relations]
char_chunk_size = 80000
char_overlap = 1000
sent_window = 5
fuzzy_thresh = 90
overlap_thresh = 85
conf_base = 0.6

[viz]
pmi_threshold = 11

4.2 Output File Naming (must match UI & CLIs)
When user sets percentile={P} and entitiesbysize={E}, save entities file as:
hierachy_entity_{P}_{E}.json

and persist a cleaned/coords variant under /cleaned_synonym_entities/…. The raw corpus must also be persisted as clean_corpus.txt. (Map your current code that writes hierachy_entity_50_90.json and Panel dashboards that read cleaned entities; keep consistent.) 

5) Stable I/O Contracts
5.1 Clean Corpus
Path: ./data/outputs/<date>/clean_corpus.txt
Format: one document per line (preprocessed & de-noised).
Produced by: pdf_ner after cleaning & OCR fixes (wordninja/wordnet), de-verbing (spaCy), serial filters, blacklist. 
5.2 Entities JSON (baseline)
{
  "entities": [
    {
      "entity": "load lock gate valve",
      "ici_score": 2.31,
      "count": 47,
      "coords": [1.234, -0.567],         // added by PCA step
      "cluster": 3,                       // added by PCA step
      "parent": "gate valve"              // added by hierarchy step
    }
  ]
}

Producer: pdf_ner → PCA_engaged pipeline. 
5.3 Entities+Relations JSON (merge)
{
  "entities": [ ... ],
  "relations": [
    {
      "source": "vacuum pump",
      "target": "load lock",
      "type": "supply",                   // verb lemma or typed pattern
      "confidence": 0.82,
      "evidence": [{"doc_id":"doc17","window":"...source sentence window..."}]
    }
  ]
}

Producer: relation extractor with SVO + fuzzy mention + PMI report; chunking guards must remain (CHAR_CHUNK_SIZE, CHAR_OVERLAP), spaCy EN model. 

6) Pipelines, CLIs & Endpoints
6.1 End-to-End Runner (CLI)
python scripts/run_all.py \
  --pdf_dir ./data/inputs/pdfs \
  --percentile 50 \
  --entitiesbysize 90 \
  --mode local_offline \
  --save_dir ./data/outputs/$(date +%Y%m%d)

Stages (must stream to EXECUTION_LOG):
Ingest & Clean → clean_corpus.txt (preserve current OCR/verb filters). 
Seed & Synonyms → hierachy_entity_{P}_{E}.json. 
PCA+Cluster+Hierarchy → .../cleaned_synonym_entities/..._cleaned_1.json with coords, cluster, parent. 
Relations → merge "relations" and save with_relations.json next to cleaned entities. Respect fuzzy & chunk settings. 
Viz Exports → save entities_pca.html & pmi_chord.html for offline viewing; PMI excludes p==e and uses threshold. 
6.2 FastAPI Service (local by default)
POST /ingest (pdf|txt|json), POST /entities, POST /relations, GET /viz/pca, GET /viz/pmi
Echo config hash in response; attach artifact paths; stream SSE logs.
6.3 Minimal UI (Panel; local-only)
Entities Explorer (PCA): color by cluster|parent, silhouette text, counts table. 
PMI Chord: parent→entity arcs, table of (parent, entity, count, pmi). 

7) Tests (must pass before “green”)
Create pytest tests + tiny mock fixtures (/data/inputs/mock/).
7.1 Ingest/Clean
Removes noise & dates; splits sticky tokens; filters verbs.
Assert blacklist tokens like "**" and "th" are removed; serial patterns ignored. 
7.2 Entities
Naming contract: outputs hierachy_entity_{P}_{E}.json.
Schema contract: entity, ici_score, count present; no verbs; lengths ≤ max_ngram_len. 
7.3 PCA/Clusters/Hierarchy
coords length=2; cluster is int; parent is shortest super-token where applicable; handles sklearn metric/affinity compatibility. 
7.4 Relations
Safety: respects CHAR_CHUNK_SIZE and CHAR_OVERLAP; FUZZY_THRESH and anti-overlap guard.
Evidence: each relation has ≥1 window span; PMI computes without division by zero. 
7.5 Visualization
PMI excludes p==e; threshold applied; raises clean error if no pairs pass. 

8) Observability & Run Summary
Logs: ./logs/EXECUTION_LOG.jsonl (structured; one line per step).
Progress: tqdm for long loops (embedding, PDF IO, relation windows).
RUN_SUMMARY.md: emit at end with:
config hash + profile, artifact paths, timing (per stage), counters (#docs, #entities, #relations), key metrics (silhouette, PMI pairs), open issues.

9) Interfaces to Implement (code-first tasks)
Use the Error-Proof Delivery Protocol:
PLAN → PATCH (unified diff) → RUN (exact commands / one-cell demo).
Task A — Wrap Ingest→Entities (from pdf_ner.md)
Goal: Unify PDF/TXT/JSON ingestion; always write:
clean_corpus.txt
hierachy_entity_{P}_{E}.json
Constraints to preserve: TF-IDF percentile seeds; OCR & verb filters; SciBERT embed; HDBSCAN; synonym grouping. 
Task B — Clean & Embed Coordinates (from PCA_engaged.md)
Goal: Enrich entities with coords, cluster, parent, respecting AgglomerativeClustering API differences (metric/affinity). 
Task C — Relations (from with_relations.md)
Goal: Merge SVO-typed edges with fuzzy mention detection under safe chunking; PMI dataframe for diagnostics; bar plot for relation type freq; merged JSON writer. 
Task D — Dashboards (from visualizaiton.md)
Goal: Panel apps:
PCA Explorer (cluster/parent toggle, silhouette & suggestions, counts)
PMI Chord (threshold slider, counts table; exclude p==e).
Export .html artifacts. 

10) Quickstart (env + commands)
# 1) Create env (macOS/Apple Silicon & Linux)
conda create -n ner_web python=3.10 -y
conda activate ner_web
pip install -U pip wheel

# 2) Install deps (pin minimal set)
pip install spacy==3.7.4 scikit-learn==1.4.2 pandas==2.2.2 numpy==1.26.4 \
            tqdm==4.66.4 rapidfuzz==3.9.6 networkx==3.3 matplotlib==3.8.4 \
            pdfplumber==0.11.0 wordninja==2.0.0 nltk==3.8.1 hdbscan==0.8.33 \
            transformers==4.41.2 torch --index-url https://download.pytorch.org/whl/cpu \
            holoviews==1.18.3 panel==1.4.2 bokeh==3.4.1
python -m spacy download en_core_web_sm

# 3) Run E2E
python scripts/run_all.py --pdf_dir ./data/inputs/pdfs --percentile 50 --entitiesbysize 90 \
  --save_dir ./data/outputs/$(date +%Y%m%d)

# 4) Launch dashboards
panel serve apps/entities_explorer.ipynb apps/pmi_chord_dashboard.ipynb --autoreload


11) Evaluation & Metrics (produce markdown table in RUN_SUMMARY)
Entities: % verbs removed; silhouette of clusters; top-k coverage vs. TF-IDF seeds.
Relations: #edges, PMI distribution summary; % edges with ≥1 evidence window.
Viz: #nodes/#edges plotted; #pairs passing PMI threshold.

12) Safety, Determinism & Repro
Determinism: set global seeds; document bounded nondeterminism (HDBSCAN).
Security: honor mode.local_only; never exfiltrate data; redact emails/URLs during cleaning (already implemented). 
Compatibility: no hidden constants; thresholds in config.toml; artifact directories predictable.

13) Roadmap (next 2 sprints)
API service + UI wiring: FastAPI endpoints + Panel buttons calling the CLIs.
Graph export: Neo4j CSVs + RDF/Turtle with provenance (doc_id, offsets).
Local LLM RAG (llama.cpp): /answer with citations; faithfulness checks.
Contract tests: JSON schema validation; fuzz tests for cleaning & relations windows.
Scalability: batch corpus streaming; memory footprints & p95 latency snapshots.

14) Example Task Card (template Claude must use)
goal: Add relations & PMI to entities JSON
inputs:
  entities_json: ./data/outputs/2025-09-16/hierachy_entity_50_90.json
  corpus: ./data/outputs/2025-09-16/clean_corpus.txt
deliverables:
  - code_files: [pkgs/ner_re/relations.py, scripts/run_all.py]
  - tests: [tests/test_relations.py]
  - demo: apps/pmi_chord_dashboard.ipynb
  - artifacts: [with_relations.json, pmi_pairs.csv, relation_freq.png]
acceptance:
  - [ ] runs in fresh env
  - [ ] one-cell demo works
  - [ ] unit tests pass
  - [ ] PMI pairs > 0 and p==e excluded
params:
  fuzzy_thresh: 90
  pmi_threshold: 11
notes: >
  Keep chunking guards; attach evidence windows; cite config hash.


15) Deliver Protocol (Claude must always reply with)
PLAN – concise steps + risks
PATCH – unified diff(s) creating/updating files
RUN – exact commands + one-cell demo; show tqdm and print key metrics

Appendix: File-specific anchors
pdf_ner.md — TF-IDF percentile, SciBERT embedding, HDBSCAN, blacklist & OCR fixes, emits hierachy_entity_50_90.json. 
PCA_engaged.md — loads HIERARCHY_PATH, adds coords|cluster|parent, handles sklearn param drift, saves cleaned JSON. 
visualizaiton.md — Panel/HoloViews PCA explorer & PMI chord dashboard; PMI_THRESHOLD, parent≠entity, .servable(). 
with_relations.md — CHAR_CHUNK_SIZE 80k, FUZZY_THRESH 90, safe SVO over spaCy, PMI compute, merge relations back to JSON. 

