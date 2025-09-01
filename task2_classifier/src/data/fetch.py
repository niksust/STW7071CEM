# src/data/fetch.py
import re, time
from typing import Dict, List
import requests, feedparser, pandas as pd

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X) AppleWebKit/537.36 (KHTML, like Gecko) Python-requests feed fetcher"

def clean_html(s: str) -> str:
    if not s: return ""
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def _get_feed_bytes(url: str, timeout: int = 10) -> bytes:
    # try https, then http fallback if caller passed https
    urls = [url]
    if url.startswith("https://"):
        urls.append("http://" + url[len("https://"):])
    last_err = None
    for u in urls:
        try:
            r = requests.get(u, headers={"User-Agent": UA}, timeout=timeout, allow_redirects=True)
            r.raise_for_status()
            return r.content
        except Exception as e:
            last_err = e
            print(f"[fetch] GET {u} -> ERROR: {e}", flush=True)
            time.sleep(0.5)
    raise last_err

def fetch_feed(label: str, url: str, timeout: int = 10, verbose: bool = True) -> List[dict]:
    if verbose: print(f"[fetch] {label}: {url}", flush=True)
    raw = _get_feed_bytes(url, timeout=timeout)
    d = feedparser.parse(raw)
    n = len(getattr(d, "entries", []))
    if verbose: print(f"[fetch] {label}: entries={n}", flush=True)
    docs = []
    for e in d.entries:
        title = clean_html(getattr(e, "title", ""))
        summary = clean_html(getattr(e, "summary", getattr(e, "description", "")))
        link = getattr(e, "link", "")
        text = (title + ". " + summary).strip()
        if len(text.split()) >= 8:
            docs.append({"label": label, "text": text, "title": title, "link": link, "source": url})
    return docs

def collect_corpus(feeds: Dict[str, str], timeout: int = 10, retries: int = 1, verbose: bool = True) -> pd.DataFrame:
    all_docs = []
    for label, url in feeds.items():
        last_err = None
        for attempt in range(1, retries + 2):
            try:
                if verbose: print(f"[collect] {label} attempt {attempt}/{retries+1}", flush=True)
                all_docs.extend(fetch_feed(label, url, timeout=timeout, verbose=verbose))
                break
            except Exception as e:
                last_err = e
                print(f"[collect] {label} failed: {e}", flush=True)
                if attempt <= retries:
                    time.sleep(1.0)
                else:
                    print(f"[collect] giving up on {label}", flush=True)
        time.sleep(0.3)
    # dedupe
    seen, deduped = set(), []
    for d in all_docs:
        key = d["link"] or d["text"][:120]
        if key not in seen:
            seen.add(key); deduped.append(d)
    df = pd.DataFrame(deduped)
    print(f"[collect] total={len(all_docs)}  unique={len(df)}", flush=True)
    if df.empty:
        raise RuntimeError("No documents fetched. Check network/SSL or try http feeds.")
    return df
