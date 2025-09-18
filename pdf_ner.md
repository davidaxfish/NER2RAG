```python
import os
import re
import json
import unicodedata
import numpy as np
import nltk
import spacy
import wordninja
import pdfplumber
from tqdm import tqdm
from collections import Counter, defaultdict
from nltk.corpus import stopwords, wordnet as wn
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.phrases import Phrases, Phraser
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoTokenizer, AutoModel
import hdbscan
 

# ===== PARAMETERS (tunable) =====
PDF_DIRECTORY = r"C:\RAGDATA\David_Kuo\David_Kuo"
NUM_PDFS = 50
NGRAM_RANGE = (1, 3)
TFIDF_TOP_PERCENTILE = 90
SEED_TOKEN_LIMIT = 7000
EMBEDDING_MODEL_NAME = r'C:\Users\e182868\OneDrive - Applied Materials\script_cloud\model\scibert_scivocab_uncased\scibert_scivocab_uncased'
EMBED_BATCH_SIZE = 64
W2V_MIN_COUNT = 1
W2V_PHRASE_THRESHOLD = 2
HDBSCAN_MIN_CLUSTER_SIZE = 2
HDBSCAN_MIN_SAMPLES = 1
MIN_SYNONYM_COSINE_SIM = 0.8
 

# abbreviation mapping
ABBR_MAP_ENABLED = True
ABBR_MAPPING = {
    'mf': 'mainframe',
    'll': 'load lock'
}
 

# patterns to exclude
EXCLUDE_SERIAL_PATTERNS = [
    r"\b\d{4,}\b",
    r"\b\d+[A-Za-z]+\b",
    r"\b[A-Za-z]+\d+\b",
    r"\b\d{2,}[-_]\d{2,}\b"
]
 

OUTPUT_JSON_PATH = "./result/hierachy_entity_50_90.json"
SEED = 42
 

# init
np.random.seed(SEED)
torch.manual_seed(SEED)
nltk.download('stopwords')
nltk.download('wordnet')
STOPWORDS = set(stopwords.words("english"))
BLACKLIST = {"applied","materials","confidential","internal","figure","page","table","**","th","TM"}
# spaCy for lemmatization / POS
nlp = spacy.load("en_core_web_sm", disable=["parser","ner"])
 

# ===== TEXT CLEANING =====
def is_serial_pattern(tok):
    return any(re.fullmatch(p, tok) for p in EXCLUDE_SERIAL_PATTERNS)
 

def clean_text_advanced(text):
    # normalize unicode & remove control chars
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', text)
    # remove non-ascii
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    # remove headers, sections, urls, emails
    text = re.sub(r'(Section|Chapter|Table)\s*\d+(\.\d+)*', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'Page\s*\d+', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\b\S+@\S+\.\S+\b', ' ', text)
    text = re.sub(r'http[s]?://\S+', ' ', text)
    # break attached numbers except value-unit pairs
    text = re.sub(r'\b(\d+)([a-zA-Z]+)\b', r'\1 \2', text)
    # remove long serial patterns
    text = re.sub(r'\b\d{3,}[-_]\d+\b', ' ', text)
    # remove pure standalone dates: e.g. 2024-12-09, 12/09/2024, Dec 9, 2024
    text = re.sub(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4})\b", ' ', text)
    # fix OCR/connected words
    text = _fix_ocr_and_split(text)
    # remove unwanted patterns
    patterns = [r'Applied Materials [®™]?', r'Sym3™ Etch Chamber Manual', r'Confidential',
                r'Revision\s*\d+', r'\bPart No\.\d+', r'(Traditional|Simplified) Chinese',
                r'(Korean|Japanese)']
    for p in patterns:
        text = re.sub(p, ' ', text, flags=re.IGNORECASE)
    # collapse spaces
    text = re.sub(r'\s{2,}', ' ', text).strip()
    return text
 

# helper: OCR and sticky-word split via wordninja + wordnet check
 

def _fix_ocr_and_split(text):
    tokens = []
    for word in text.split():
        # if numeric+unit, keep
        m = re.match(r'^(\d+)([a-zA-Z]+)$', word)
        if m:
            tokens.extend([m.group(1), m.group(2)])
            continue
        parts = wordninja.split(word)
        # accept split only if all parts are valid English (wordnet)
        if len(parts)>1 and all(wn.synsets(p) for p in parts):
            tokens.extend(parts)
        else:
            tokens.append(word)
    return ' '.join(tokens)
 

# valid entity filter
 

def valid_entity(term):
    t = term.lower().strip()
    if len(t)<3 or t in STOPWORDS or t in BLACKLIST: return False
    if any(is_serial_pattern(tok) for tok in [t]): return False
    if t.isnumeric(): return False
    if sum(c.isalpha() for c in t)<2: return False
    if re.fullmatch(r"[\d\-]+", t): return False
    if re.fullmatch(r"[^\w]+", t): return False
    # remove verbs
    doc = nlp(t)
    if any(tok.pos_=='VERB' for tok in doc): return False
    return True
 

# === entity extraction ===
 

def extract_text_from_pdf(path):
    text=""
    try:
        with pdfplumber.open(path) as pdf:
            for p in pdf.pages:
                raw = p.extract_text() or ""
                text += clean_text_advanced(raw) + ' '
    except:
        from PyPDF2 import PdfReader
        r=PdfReader(path)
        for p in r.pages:
            raw = p.extract_text() or ""
            text += clean_text_advanced(raw) + ' '
    # remove stray newlines
    text = text.replace("\n"," ")
    return re.sub(r'\s{2,}',' ', text).strip()
 

def extract_texts(pdf_dir, n):
    files = sorted([os.path.join(pdf_dir,f) for f in os.listdir(pdf_dir) if f.endswith('.pdf')],
                   key=lambda x: os.path.getsize(x), reverse=True)[:n]
    out={}
    for f in tqdm(files, desc="PDFs"):
        out[os.path.basename(f)] = extract_text_from_pdf(f)
    return out
 

# === tokenization & phrases ===
 

def tokenize_and_phrase(texts):
    sents=[]
    for t in texts:
        toks=[w for w in word_tokenize(t.lower()) if w not in STOPWORDS and w not in BLACKLIST]
        sents.append(toks)
    bigram=Phrases(sents, min_count=W2V_MIN_COUNT, threshold=W2V_PHRASE_THRESHOLD)
    bg=Phraser(bigram)
    trigram=Phrases(bg[sents], min_count=W2V_MIN_COUNT, threshold=W2V_PHRASE_THRESHOLD)
    tg=Phraser(trigram)
    return [[tok.replace('_',' ') for tok in tg[bg[s]]] for s in sents]
 

# === seed extraction (ICI/TF-IDF) ===
 

def extract_seed_entities(sents, counts):
    docs=[" ".join(s) for s in sents]
    vect=TfidfVectorizer(ngram_range=NGRAM_RANGE, stop_words='english')
    mat=vect.fit_transform(docs)
    names=vect.get_feature_names_out()
    scores=mat.sum(axis=0).A1
    ici={f:s for f,s in zip(names,scores)}
    thr=np.percentile(scores, TFIDF_TOP_PERCENTILE)
    filtered={term:counts.get(term,0) for term,sc in zip(names,scores)
              if sc>=thr and valid_entity(term)}
    top=sorted(filtered.items(), key=lambda x:-x[1])[:SEED_TOKEN_LIMIT]
    return dict(top), ici
 

# === word2vec ===
 

def train_w2v(sents):
    m=Word2Vec(sents, vector_size=100, window=5, min_count=W2V_MIN_COUNT, workers=4)
    flat=[w for s in sents for w in s]
    return m, Counter(flat)
 

# === embedding/clustering ===
class Embedder:
    def __init__(self,mn):
        dev='cuda' if torch.cuda.is_available() else 'cpu'
        self.tok=AutoTokenizer.from_pretrained(mn)
        self.mod=AutoModel.from_pretrained(mn).to(dev).eval()
        self.dev=dev
    def encode(self, texts):
        ems=[]
        for i in tqdm(range(0,len(texts), EMBED_BATCH_SIZE), desc="Embedding"):
            batch=texts[i:i+EMBED_BATCH_SIZE]
            enc=self.tok(batch, padding=True, truncation=True, max_length=32, return_tensors='pt').to(self.dev)
            with torch.no_grad():
                out=self.mod(**enc)
                ems.append(out.last_hidden_state[:,0,:].cpu().numpy())
        return np.vstack(ems)
 

# === synonyms strict ===
 

def get_synonyms(entity_list, labels, embs, ici, counts):
    d=defaultdict(list)
    for ent,lab in zip(entity_list,labels):
        if lab==-1: continue
        d[lab].append(ent)
    syn={}
    for lab,ents in d.items():
        idxs=[entity_list.index(e) for e in ents]
        mats=cosine_similarity(embs[idxs])
        minsim=np.min(mats[np.triu_indices_from(mats,1)]) if len(ents)>1 else 1
        if minsim<MIN_SYNONYM_COSINE_SIM or len(ents)>30: continue
        # choose root: shortest phrase
        canon=sorted(ents, key=lambda x:(len(x.split()),x))[0]
        group=[]
        for e in sorted(set(ents), key=lambda x:(len(x.split()),x)):
            group.append({
                'entity': e,
                'ici': round(float(ici.get(e,0)),6),
                'count': int(counts.get(e,0))
            })
        syn[canon]=group
    # apply abbreviation mapping
    if ABBR_MAP_ENABLED:
        for ab,full in ABBR_MAPPING.items():
            if ab in syn:
                grp=syn.pop(ab)
                if full in syn:
                    syn[full].extend(grp)
                else:
                    syn[full]=grp
    return syn
 

# ===== MAIN =====
if __name__=='__main__':
    # extract & preprocess
    pdfs=extract_texts(PDF_DIRECTORY, NUM_PDFS)
    sents=tokenize_and_phrase(list(pdfs.values()))
    # train w2v
    w2v,counts=train_w2v(sents)
    # seed entities
    seeds, ici=extract_seed_entities(sents,counts)
    ent_list=list(seeds.keys())
    # embed
    embder=Embedder(EMBEDDING_MODEL_NAME)
    embs=embder.encode(ent_list)
    # cluster
    lbls=hdbscan.HDBSCAN(min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
                         min_samples=HDBSCAN_MIN_SAMPLES, metric='euclidean')
    labs=lbls.fit_predict(embs)
    # synonyms
    result=get_synonyms(ent_list, labs, embs, ici, counts)
    # save
    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    with open(OUTPUT_JSON_PATH,'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Saved → {OUTPUT_JSON_PATH}")






```
 
    [nltk_data] Downloading package stopwords to
    [nltk_data]     C:\Users\e182868\AppData\Roaming\nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    [nltk_data] Downloading package wordnet to
    [nltk_data]     C:\Users\e182868\AppData\Roaming\nltk_data...
    [nltk_data]   Package wordnet is already up-to-date!
    C:\Users\e182868\AppData\Local\miniconda3\envs\Seedproject\Lib\site-packages\spacy\util.py:922: UserWarning: [W095] Model 'en_core_web_sm' (3.7.1) was trained with spaCy v3.7.2 and may not be 100% compatible with the current version (3.8.7). If you see errors or degraded performance, download a newer compatible model or retrain your custom model with the current spaCy version. For more details and available updates, run: python -m spacy validate
      warnings.warn(warn_msg)
    PDFs: 100%|█████████████████████████████████████████████████████████████████████████| 50/50 [1:48:42<00:00, 130.44s/it]
    Embedding: 100%|█████████████████████████████████████████████████████████████████████| 110/110 [01:14<00:00,  1.47it/s]
   
 
    Saved → ./result/synonym_dict_nin2_50_90.json
   
 

```python
import os
 
# Define the directory and filename
directory = r"C:\Users\e182868\OneDrive - Applied Materials\script_cloud\result"
filename = "my_text_file_50_90_0713_nin_v2.txt"
content = str(pdf_texts.values())
 
# Create the directory if it doesn't exist
# os.makedirs creates all necessary intermediate directories
# exist_ok=True prevents an error if the directory already exists
os.makedirs(directory, exist_ok=True)
 
# Construct the full file path
file_path = os.path.join(directory, filename)
 
# Write content to the file
# 'w' mode opens the file for writing. If the file exists, its content is truncated.
# If the file does not exist, a new one is created.
try:
    with open(file_path, 'w') as file:
        file.write(content)
    print(f"File '{filename}' successfully written to '{os.path.abspath(directory)}'")
except IOError as e:
    print(f"Error writing file: {e}")
 
```
 
    File 'my_text_file_50_90_0713_nin_v2.txt' successfully written to 'C:\Users\e182868\OneDrive - Applied Materials\script_cloud\result'
   
 

```python
 
```
 