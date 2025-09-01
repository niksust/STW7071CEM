# app.py
import json, re, math, time
from collections import Counter, defaultdict
from urllib.parse import quote

import streamlit as st

JSON_PATH = "index.json"

# ---------- Utils ----------
def norm(s: str) -> str:
    if not s: return ""
    s = s.lower()
    s = re.sub(r"[^\w\s-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize(s: str):
    return [t for t in norm(s).split() if t]

def extract_year(date_str: str):
    if not date_str:
        return -1
    # try YYYY
    m = re.search(r"\b(20\d{2}|19\d{2})\b", date_str)
    if m:
        return int(m.group(1))
    return -1

def highlight(text: str, query_terms):
    """very simple term highlighter in markdown"""
    if not text: return ""
    out = text
    # sort longest first to avoid nested replacements
    for q in sorted(set(query_terms), key=len, reverse=True):
        if not q: continue
        out = re.sub(fr'(?i)\b({re.escape(q)})\b', r'**\1**', out)
    return out

# ---------- Load data ----------
@st.cache_data(show_spinner=False)
def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # If we're loading the new inverted index structure
    if isinstance(data, dict) and "docs" in data:
        docs = data["docs"]
    else:
        # Backward-compat: old res.json (a list of docs)
        docs = data

    # dedupe by pub_url (kept exactly as before)
    seen = set(); clean = []
    for d in docs:
        key = (d.get("pub_url") or "").strip()
        if key and key not in seen:
            seen.add(key)
            clean.append(d)

    # write back cleaned docs if we're in index mode
    if isinstance(data, dict) and "docs" in data:
        data["docs"] = clean
        return data

    # else (legacy list mode)
    return clean

# ---------- Build TF-IDF ----------
@st.cache_data(show_spinner=False)
def build_index(docs):
    """
    Create a tiny TF-IDF index over title + abstract + author fields.
    """
    N = len(docs)
    fields = ("title", "abstract", "cu_author", "category")
    tok_docs = []
    df = Counter()
    for i, d in enumerate(docs):
        text_bits = []
        for f in fields:
            v = d.get(f,"")
            if isinstance(v, list):
                v = " ".join(v)
            text_bits.append(v or "")
        # also index co-author names
        for ca in d.get("co_authors", []):
            text_bits.append(ca.get("name",""))
        full = " ".join(text_bits)
        toks = tokenize(full)
        tok_docs.append(Counter(toks))
        df.update(set(toks))
    # idf
    idf = {t: math.log((N + 1) / (df[t] + 0.5)) + 1 for t in df}
    return tok_docs, idf

def score_doc(q_terms, tok_doc, idf, doc):
    """
    simple TF-IDF with a tiny title boost
    """
    if not q_terms: return 0.0
    score = 0.0
    title = tokenize(doc.get("title",""))
    title_terms = set(title)

    for t in q_terms:
        tf = tok_doc.get(t, 0)
        if tf == 0: 
            continue
        score += (1 + math.log(tf)) * idf.get(t, 0.5)
        # bonus if the term appears in title
        if t in title_terms:
            score *= 1.15
    return score

def search(query, docs, tok_docs, idf):
    q_terms = tokenize(query)
    scored = []
    for i, d in enumerate(docs):
        s = score_doc(q_terms, tok_docs[i], idf, d)
        if s > 0:
            scored.append((s, i))
    scored.sort(reverse=True)
    return q_terms, [docs[i] for _, i in scored]

# ---------- UI ----------
st.set_page_config(page_title="Coventry EFA Scholar", page_icon="🔍", layout="wide")

st.title("🔍 Coventry EFA Scholar")
st.caption("&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Author: Nitesh Khadka Student ID: 250207 CUID: 16542697")
st.caption("Search publications where at least one co-author is a member of Coventry’s School of Economics, Finance & Accounting.")

# Sidebar
with st.sidebar:
    st.subheader("Settings")
    sort_mode = st.radio("Sort by", ["Relevance", "Year (new→old)"])
    per_page = st.slider("Results per page", 10, 100, 25, 5)
    show_abstract = st.checkbox("Show abstracts by default", value=False)
    st.markdown("---")
    st.caption("Data source: `index.json` (prebuilt inverted index)")
    reload = st.button("Reload dataset")

# Load / rebuild
if reload:
    st.cache_data.clear()

loaded = load_data(JSON_PATH)
if isinstance(loaded, dict) and "docs" in loaded:
    # Using prebuilt inverted index
    docs = loaded["docs"]
    tok_docs = loaded.get("tok_docs", [])
    idf = {k: float(v) for k, v in loaded.get("idf", {}).items()}
else:
    # Fallback for legacy res.json
    docs = loaded
    tok_docs, idf = build_index(docs)

# Search box
q = st.text_input("Search", placeholder="e.g. corporate governance, microfinance, Piotr Lis", label_visibility="collapsed")
if not q:
    st.info("Type a query to start searching. Example: **finance innovation**")
    st.stop()

q_terms, results = search(q, docs, tok_docs, idf)

# Optional sort by year
if sort_mode.startswith("Year"):
    results.sort(key=lambda d: extract_year(d.get("date","")), reverse=True)

st.write(f"**{len(results)}** results for _{q}_")

# Pagination (simple)
page = st.session_state.get("page", 1)
max_page = max(1, (len(results) + per_page - 1)//per_page)
col_a, col_b, col_c = st.columns([1,2,1])
with col_a:
    if st.button("◀ Prev", disabled=(page<=1)):
        page = max(1, page-1)
with col_c:
    if st.button("Next ▶", disabled=(page>=max_page)):
        page = min(max_page, page+1)
st.session_state["page"] = page
st.caption(f"Page {page} / {max_page}")

start = (page-1)*per_page
chunk = results[start:start+per_page]

# Render results
for d in chunk:
    title = d.get("title","(untitled)")
    url = d.get("pub_url","#")
    date = d.get("date","")
    cu = (d.get("cu_author") or "").strip()
    cu_url = (d.get("cu_author_url") or "").strip()

    # clickable title
    st.markdown(f"### [{highlight(title, q_terms)}]({url})", unsafe_allow_html=True)

    # meta line: date • Dept author (linked if we have cu_author_url)
    meta_bits = []
    if date:
        meta_bits.append(date)
    if cu:
        if cu_url:
            meta_bits.append(f"Dept author: [{cu}]({cu_url})")
        else:
            meta_bits.append(f"Dept author: {cu}")
    if meta_bits:
        st.markdown(" · ".join(meta_bits))

    # co-authors: names only, no links, de-dup, exclude the Coventry author
    raw_co = d.get("co_authors") or []
    seen = set()
    co_names = []
    for a in raw_co:
        nm = (a.get("name") or "").strip()
        if not nm:
            continue
        if cu and nm.lower() == cu.lower():
            continue  # don't duplicate the main author in co-authors
        key = nm.lower()
        if key in seen:
            continue
        seen.add(key)
        co_names.append(nm)
    if co_names:
        st.write("Co-authors: " + " · ".join(co_names))

    # abstract
    abs_text = d.get("abstract") or ""
    # Apply the custom class to the expander
    with st.expander("Abstract", expanded=show_abstract):
        st.markdown(f"""
            <div style="font-size: 14px; font-weight: bold; font-family: Courier New">
                {abs_text if abs_text else "_No abstract available._"}
            </div>
            """, unsafe_allow_html=True)

    
    
    # with st.expander("Abstract", expanded=show_abstract):
    #     st.write(abs_text if abs_text else "_No abstract available._")

    st.divider()
