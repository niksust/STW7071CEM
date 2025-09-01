import time, json, traceback
import re,math
from collections import defaultdict
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

from webdriver_manager.chrome import ChromeDriverManager
from urllib import robotparser

# =====================
# CONFIG
# =====================
SEED_PROFILES_URL = "https://pureportal.coventry.ac.uk/en/organisations/fbl-school-of-economics-finance-and-accounting/persons/"
REQUEST_DELAY = 2  # polite delay in seconds
pub_data = []

# =====================
# WebDriver factory
# =====================
def make_driver():
    opts = Options()
    opts.add_argument("--headless=new")   # comment if you want a visible browser
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1280,2000")
    opts.add_argument("--log-level=3")
    opts.add_argument("--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/119.0.0.0 Safari/537.36")
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=opts)

# =====================
# Robots.txt loader via Selenium
# =====================
def load_robots_with_selenium(driver, url="https://pureportal.coventry.ac.uk/robots.txt"):
    driver.get(url)
    time.sleep(1)
    try:
        return driver.find_element(By.TAG_NAME, "pre").text
    except Exception:
        try:
            return driver.find_element(By.TAG_NAME, "body").text
        except Exception:
            return ""

def get_robot_parser(driver, url="https://pureportal.coventry.ac.uk/robots.txt"):
    robots_text = load_robots_with_selenium(driver, url)
    rp = robotparser.RobotFileParser()
    if robots_text.strip():
        rp.parse(robots_text.splitlines())
    else:
        # fallback allow-all
        rp.parse(["User-agent: *", "Disallow:"])
        print("[Warning] Could not fetch robots.txt, assuming allow-all.")
    return rp

def polite_get(driver, url, rp=None):
    if rp and not rp.can_fetch("*", url):
        print("Blocked by robots.txt:", url)
        return False
    driver.get(url)
    time.sleep(REQUEST_DELAY)
    return True

# =====================
# Extract publication detail
# =====================
def crawl_detail(driver, pub_url, rp):
    abstract, topics, authors = "", [], []
    if not polite_get(driver, pub_url, rp):
        return abstract, topics, authors

    # --- Abstract ---
    try:
        el = WebDriverWait(driver, 4).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.content-content.publication-content > div"))
        )
        abstract = el.text.strip()
    except TimeoutException:
        abstract = ""

    # --- Authors (names only; no profile links for co-authors) ---
    try:
        soup = BeautifulSoup(driver.page_source, "lxml")
        author_block = soup.select_one("ul.relations.persons, div.relations.persons")
        if author_block:
            for li in author_block.select("li, span, div"):
                name = li.get_text(strip=True)
                if not name:
                    continue
                # store only the name for co-authors
                authors.append({"name": name})

        # fallback: citation meta tags (names only)
        if not authors:
            for meta in soup.select("meta[name='citation_author']"):
                nm = meta.get("content")
                if nm:
                    authors.append({"name": nm.strip()})

    except Exception as e:
        print("Author parse failed:", e)

    # --- Categories (fingerprints) ---
    try:
        polite_get(driver, pub_url.rstrip("/") + "/fingerprints/", rp)
        for h3 in driver.find_elements(By.CSS_SELECTOR, "div.publication-fingerprint-thesauri > h3"):
            topics.append(h3.text.strip())
    except Exception:
        pass

    return abstract, topics, authors

# =====================
# Scrape one author
# =====================
def scrape_author(driver, author_url, rp):
    global pub_data
    if not polite_get(driver, author_url, rp):
        return
    try:
        name = WebDriverWait(driver, 4).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.header.person-details > h1"))
        ).text
    except TimeoutException:
        name = "Unknown Author"

    soup = BeautifulSoup(driver.page_source, "lxml")
    for card in soup.select("div.result-container"):
        a = card.find("a")
        if not a:
            continue
        title = a.get_text(strip=True)
        pub_url = a.get("href")
        date_el = card.find("span", class_="date")
        date = date_el.get_text(strip=True) if date_el else ""
        abstract, topics, authors = crawl_detail(driver, pub_url, rp)
        pub_data.append({
            "title": title,
            "pub_url": pub_url,
            "date": date,
            "cu_author": name,
            "cu_author_url": author_url,   
            "co_authors": authors,         # names only
            "abstract": abstract,
            "category": topics
        })

# =====================
# Collect School authors
# =====================
def collect_school_authors(driver, profiles_url, rp):
    polite_get(driver, profiles_url, rp)
    soup = BeautifulSoup(driver.page_source, "lxml")
    links = set()
    for a in soup.select("a[href*='/en/persons/']"):
        href = a.get("href")
        if href and href.startswith("https://pureportal.coventry.ac.uk/en/persons/"):
            links.add(href)
    return list(links)

# ---------- Inverted Index helpers ----------
def _norm(s: str) -> str:
    if not s: return ""
    s = s.lower()
    s = re.sub(r"[^\w\s-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _tokenize(text: str):
    return [t for t in _norm(text).split() if t]

def _build_inverted_index(docs):
    """
    Produce a compact on-disk index the app can load directly.
    index.json schema:
    {
      "docs": [... original docs ...],
      "tok_docs": [ {term: tf, ...}, ... ],
      "idf": {term: idf, ...},
      "built_at": <unix_ts>
    }
    """
    fields = ("title", "abstract", "cu_author", "category")
    tok_docs = []
    df = {}

    for d in docs:
        text_bits = []
        for f in fields:
            v = d.get(f, "")
            if isinstance(v, list):
                v = " ".join(v)
            text_bits.append(v or "")
        # include co-author names
        for ca in d.get("co_authors", []):
            text_bits.append((ca or {}).get("name", "") or "")
        full = " ".join(text_bits)
        toks = _tokenize(full)

        # per-doc term frequencies
        td = {}
        for t in toks:
            td[t] = td.get(t, 0) + 1
        tok_docs.append(td)

        # document frequency
        for t in set(toks):
            df[t] = df.get(t, 0) + 1

    N = max(1, len(docs))
    idf = {t: (math.log((N + 1) / (df[t] + 0.5)) + 1.0) for t in df}
    return {
        "docs": docs,
        "tok_docs": tok_docs,
        "idf": idf,
        "built_at": int(time.time()),
    }


# =====================
# Orchestrator
# =====================
def initCrawlerScraper(profiles_url=SEED_PROFILES_URL, max_authors=7):
    global pub_data
    pub_data = []
    driver = make_driver()
    try:
        # load robots.txt with Selenium
        rp = get_robot_parser(driver)

        author_links = collect_school_authors(driver, profiles_url, rp)
        print(f"Found {len(author_links)} author profiles at the School page.")
        for url in author_links[:max_authors]:
            print("Scraping author:", url)
            try:
                scrape_author(driver, url, rp)
            except Exception:
                traceback.print_exc()
        print("Publications collected:", len(pub_data))
    finally:
        driver.quit()

    # NEW: build + persist inverted index for the app
    index = _build_inverted_index(pub_data)
    with open("index.json", "w", encoding="utf-8") as f:
        json.dump(index, f)

    return pub_data

# =====================
# CLI + Weekly scheduler (schedule lib)
# =====================
if __name__ == "__main__":
    import schedule

    def run_crawl():
        try:
            initCrawlerScraper(SEED_PROFILES_URL, max_authors=7)
            print("[OK] index.json updated.")
        except Exception as e:
            traceback.print_exc()

    # Schedule: run once a week (e.g. every Monday at 02:00 AM)
    schedule.every().monday.at("02:00").do(run_crawl)

    print("Scheduler started. Will crawl every Monday at 02:00 AM.")
    run_crawl()  # run immediately once

    while True:
        schedule.run_pending()
        time.sleep(60)  # check once per minute
