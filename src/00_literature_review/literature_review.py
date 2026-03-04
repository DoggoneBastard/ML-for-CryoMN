#!/usr/bin/env python3
"""
Literature Review MCP Server
=============================
An MCP (Model Context Protocol) server that automates literature review by
searching PubMed, Semantic Scholar, OpenAlex, and Scopus for relevant
publications given a natural language prompt.  Retrieves **full text**
whenever possible (via PubMed Central XML or open-access PDFs), falling
back to abstracts.

Usage (stdio transport):
    python src/literature_review.py

Tools exposed:
    - search_literature: Search PubMed, Semantic Scholar, OpenAlex, and
      Scopus for papers matching a natural language query, deduplicate, and
      return structured results with full text when available.
"""

import argparse
import re
import io
import itertools
import logging
import os
import sys
import threading
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Optional

import requests
from Bio import Entrez
from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEMANTIC_SCHOLAR_API_KEY = "ZAkdjJsC8aoWP19Dj1ei50QZ7iOsDzF1iDui3251"
PUBMED_API_KEY = "6a40f099c7dd4ab902a9915d7993226cd008"
PUBMED_EMAIL = "lit-review-mcp@example.com"  # Required by NCBI policy
OPENALEX_API_KEY = "xZaLLUwerMPyNYBcgS0cnz"
ELSEVIER_API_KEY = "af8ecbeae8e99e8f7539e4d8a0380f18"

SEMANTIC_SCHOLAR_SEARCH_URL = (
    "https://api.semanticscholar.org/graph/v1/paper/search"
)
SEMANTIC_SCHOLAR_FIELDS = (
    "title,authors,year,abstract,url,externalIds,citationCount,openAccessPdf"
)

OPENALEX_SEARCH_URL = "https://api.openalex.org/works"
SCOPUS_SEARCH_URL = "https://api.elsevier.com/content/search/scopus"

DEFAULT_MAX_RESULTS = 20  # per source

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Spinner Utility for CLI
# ---------------------------------------------------------------------------
class Spinner:
    def __init__(self, message: str = "Loading..."):
        self.spinner = itertools.cycle(['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'])
        self.message = message
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._spin, daemon=True)
        self.thread.start()

    def _spin(self):
        while self.running:
            with self._lock:
                msg = self.message
            sys.stderr.write(f'\r\033[K{next(self.spinner)} {msg}')
            sys.stderr.flush()
            time.sleep(0.1)

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join()
        sys.stderr.write('\r\033[K')
        sys.stderr.flush()

    def update(self, message: str):
        with self._lock:
            self.message = message

_cli_spinner: Optional[Spinner] = None

def set_spinner_message(msg: str):
    if _cli_spinner and _cli_spinner.running:
        _cli_spinner.update(msg)

# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------
mcp = FastMCP(
    "Literature Review",
    instructions=(
        "Search PubMed, Semantic Scholar, OpenAlex, and Scopus for "
        "relevant publications and return structured results with full "
        "text when available."
    ),
)


# ---------------------------------------------------------------------------
# Natural Language Query Parser
# ---------------------------------------------------------------------------

# Keywords that signal an institutional affiliation
_INST_KEYWORDS_RE = re.compile(
    r'\b(?:university|universit[àáa]|institute|college|laboratory|'
    r'center|centre|school|hospital|polytechnic|academy|clinic)\b',
    re.IGNORECASE,
)
# Connector words commonly found inside institution names
_INST_CONNECTORS = {'of', 'for', 'the', 'and', 'in', 'at', 'de', 'di', 'du', 'des'}

# Context cues that the preceding/following words are an author name
_AUTHOR_CUES_AFTER = {
    'publications', 'papers', 'articles', 'work', 'works',
    'research', 'studies', 'study',
}


def _parse_query(raw_query: str) -> dict:
    """Parse a natural language query into structured components.

    Detects:
      * Date ranges  — "2023-2026", "since 2023", "from 2020 to 2024"
      * Institutions — phrases containing university / institute / …
      * Author names — capitalised name-like words near contextual cues
      * Topic        — everything remaining

    Returns a dict with keys: author, institution, year_start, year_end, topic.
    """
    current_year = datetime.now().year
    text = raw_query.strip()
    parsed: dict = {
        'author': '',
        'institution': '',
        'year_start': None,
        'year_end': None,
        'topic': '',
    }

    # --- 1. Date ranges ---------------------------------------------------
    date_patterns: list[tuple] = [
        (r'(?:from\s+)?(\d{4})\s*[-–]\s*(\d{4})',
         lambda m: (int(m.group(1)), int(m.group(2)))),
        (r'(?:from\s+)?(\d{4})\s+to\s+(\d{4})',
         lambda m: (int(m.group(1)), int(m.group(2)))),
        (r'since\s+(\d{4})',
         lambda m: (int(m.group(1)), current_year)),
        (r'after\s+(\d{4})',
         lambda m: (int(m.group(1)) + 1, current_year)),
        (r'before\s+(\d{4})',
         lambda m: (None, int(m.group(1)) - 1)),
    ]
    for pat, extractor in date_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            parsed['year_start'], parsed['year_end'] = extractor(m)
            text = (text[:m.start()] + ' ' + text[m.end():]).strip()
            break

    # --- 2. Institution ---------------------------------------------------
    inst_match = _INST_KEYWORDS_RE.search(text)
    if inst_match:
        # Walk backwards from keyword to grab leading capitalised words
        words_before = text[:inst_match.start()].split()
        inst_prefix: list[str] = []
        non_connector_count = 0
        for w in reversed(words_before):
            clean = w.strip('.,;:')
            if not clean:
                break
            is_connector = clean.lower() in _INST_CONNECTORS
            if is_connector:
                inst_prefix.insert(0, clean)
            elif clean[0].isupper():
                # Stop at all-caps short words (likely surnames: HU, LI, WU)
                if clean.isupper() and len(clean) <= 3 and inst_prefix:
                    break
                non_connector_count += 1
                if non_connector_count > 2:
                    break
                inst_prefix.insert(0, clean)
            else:
                break

        # Walk forwards from keyword to grab trailing words
        words_from_kw = text[inst_match.start():].split()
        inst_suffix: list[str] = []
        for j, w in enumerate(words_from_kw):
            clean = w.strip('.,;:')
            if j == 0:                       # the keyword itself
                inst_suffix.append(clean)
            elif clean and (clean[0].isupper() or clean.lower() in _INST_CONNECTORS):
                inst_suffix.append(clean)
            else:
                break

        institution = ' '.join(inst_prefix + inst_suffix)
        parsed['institution'] = institution

        # Remove the institution span from text
        full_span = ' '.join(inst_prefix + inst_suffix)
        # Build a regex that matches the institution words with flexible spacing
        escaped_words = [re.escape(w.strip('.,;:')) for w in (inst_prefix + inst_suffix)]
        span_re = re.compile(r'\s*' + r'\s+'.join(escaped_words) + r'\s*', re.IGNORECASE)
        text = span_re.sub(' ', text, count=1).strip()

    # --- 3. Author name ---------------------------------------------------
    # Pattern A: "by <Name>"
    by_match = re.search(
        r'\bby\s+([A-Z][a-zA-Z]+(?:\s+[A-Za-z]+){0,3})', text
    )
    if by_match:
        parsed['author'] = by_match.group(1).strip()
        text = (text[:by_match.start()] + ' ' + text[by_match.end():]).strip()
    else:
        # Pattern B: "<Name> publications/papers/…"
        pub_pat = (
            r'([A-Z][a-zA-Z]+(?:\s+[A-Za-z]+){0,3})'
            r'\s+(?:' + '|'.join(_AUTHOR_CUES_AFTER) + r')'
        )
        pub_match = re.search(pub_pat, text, re.IGNORECASE)
        if pub_match:
            parsed['author'] = pub_match.group(1).strip()
            text = (text[:pub_match.start()] + ' ' + text[pub_match.end():]).strip()
        elif parsed['institution']:
            # If there is an institution, remaining capitalised multi‐word is
            # likely an author name
            name_match = re.search(
                r'([A-Z][a-zA-Z]+(?:\s+[A-Za-z]+){1,2})', text
            )
            if name_match:
                candidate = name_match.group(1).strip()
                # Simple sanity: must be 2–3 words
                if 2 <= len(candidate.split()) <= 3:
                    parsed['author'] = candidate
                    text = (
                        text[:name_match.start()] + ' ' + text[name_match.end():]
                    ).strip()

    # --- 4. Remaining text → topic ----------------------------------------
    noise = {
        'publications', 'papers', 'articles', 'works', 'research',
        'studies', 'from', 'by', 'at', 'on', 'about',
    }
    topic_words = [
        w for w in text.split()
        if w.lower().strip('.,;:') not in noise
    ]
    parsed['topic'] = ' '.join(topic_words).strip()

    return parsed


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------

def _extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes using PyMuPDF (fitz).

    Returns the concatenated text of all pages, or an empty string on failure.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.warning(
            "PyMuPDF (fitz) not installed – cannot extract text from PDFs. "
            "Install with: pip install PyMuPDF"
        )
        return ""

    text_parts: list[str] = []
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for page in doc:
                text_parts.append(page.get_text())
    except Exception as exc:
        logger.error("Failed to extract text from PDF: %s", exc)
        return ""
    return "\n".join(text_parts).strip()


_PDF_DOWNLOAD_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,*/*",
}


def _download_pdf_text(url: str) -> str:
    """Download a PDF from *url* and return its extracted text."""
    try:
        resp = requests.get(url, headers=_PDF_DOWNLOAD_HEADERS, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("PDF download failed (%s): %s", url, exc)
        return ""
    return _extract_text_from_pdf(resp.content)


# ---------------------------------------------------------------------------
# Semantic Scholar helpers
# ---------------------------------------------------------------------------

def _search_semantic_scholar(
    parsed: dict, max_results: int = DEFAULT_MAX_RESULTS
) -> list[dict]:
    """Query Semantic Scholar Graph API and return a list of paper dicts."""
    set_spinner_message("Semantic Scholar: Querying API...")

    # Build search string – Semantic Scholar is keyword-only,
    # so we combine author + topic into the query string.
    search_parts: list[str] = []
    if parsed['author']:
        search_parts.append(parsed['author'])
    if parsed['topic']:
        search_parts.append(parsed['topic'])
    ss_query = ' '.join(search_parts) if search_parts else ''
    if not ss_query:
        return []

    headers = {"x-api-key": SEMANTIC_SCHOLAR_API_KEY}
    params: dict = {
        "query": ss_query,
        "limit": max_results,
        "fields": SEMANTIC_SCHOLAR_FIELDS,
    }
    # Year range filter
    if parsed['year_start'] and parsed['year_end']:
        params["year"] = f"{parsed['year_start']}-{parsed['year_end']}"
    elif parsed['year_start']:
        params["year"] = f"{parsed['year_start']}-"
    elif parsed['year_end']:
        params["year"] = f"-{parsed['year_end']}"

    try:
        resp = requests.get(
            SEMANTIC_SCHOLAR_SEARCH_URL,
            headers=headers,
            params=params,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        logger.error("Semantic Scholar request failed: %s", exc)
        return []

    papers: list[dict] = []
    items = data.get("data", [])
    total = len(items)
    for i, item in enumerate(items, 1):
        authors = [a.get("name", "") for a in item.get("authors", [])]
        doi = (item.get("externalIds") or {}).get("DOI", "")

        # Attempt to get full text from open-access PDF
        full_text = ""
        oa_pdf = item.get("openAccessPdf") or {}
        pdf_url = oa_pdf.get("url", "")
        if pdf_url:
            title_short = item.get("title", "")[:30]
            logger.info("Downloading open-access PDF for: %s", item.get("title", ""))
            set_spinner_message(f"Semantic Scholar: PDF {i}/{total} ({title_short}...)")
            full_text = _download_pdf_text(pdf_url)

        papers.append(
            {
                "title": item.get("title", ""),
                "authors": authors,
                "year": item.get("year"),
                "abstract": item.get("abstract", ""),
                "full_text": full_text,
                "url": item.get("url", ""),
                "doi": doi,
                "citation_count": item.get("citationCount"),
                "source": "Semantic Scholar",
            }
        )
    return papers


# ---------------------------------------------------------------------------
# PubMed / PMC helpers
# ---------------------------------------------------------------------------

def _fetch_pmc_full_text(pmcid: str) -> str:
    """Fetch the full text of an article from PubMed Central as plain text.

    Uses Entrez.efetch(db='pmc') to retrieve the JATS/XML and then
    recursively extracts the <body> text content.
    """
    Entrez.email = PUBMED_EMAIL
    Entrez.api_key = PUBMED_API_KEY

    try:
        handle = Entrez.efetch(
            db="pmc", id=pmcid, rettype="xml", retmode="xml"
        )
        xml_data = handle.read()
        handle.close()
    except Exception as exc:
        logger.warning("PMC efetch failed for %s: %s", pmcid, exc)
        return ""

    return _parse_pmc_body(xml_data)


def _parse_pmc_body(xml_data: bytes | str) -> str:
    """Extract the <body> text from PMC JATS XML."""
    try:
        root = ET.fromstring(xml_data)
    except ET.ParseError as exc:
        logger.warning("Failed to parse PMC XML: %s", exc)
        return ""

    # The body is typically at /article/body or /pmc-articleset/article/body
    body = root.find(".//body")
    if body is None:
        return ""

    # Recursively extract all text
    return "".join(body.itertext()).strip()


def _search_pubmed(
    parsed: dict, max_results: int = DEFAULT_MAX_RESULTS
) -> list[dict]:
    """Query PubMed via NCBI E-Utilities and return a list of paper dicts.

    For articles available in PubMed Central, the full text body is fetched.
    """
    set_spinner_message("PubMed: Querying esearch API...")
    Entrez.email = PUBMED_EMAIL
    Entrez.api_key = PUBMED_API_KEY

    # Build structured PubMed query
    pm_parts: list[str] = []
    if parsed['author']:
        pm_parts.append(f"{parsed['author']}[Author]")
    if parsed['institution']:
        pm_parts.append(f"{parsed['institution']}[Affiliation]")
    if parsed['topic']:
        pm_parts.append(parsed['topic'])
    if parsed['year_start'] or parsed['year_end']:
        ys = parsed['year_start'] or 1900
        ye = parsed['year_end'] or datetime.now().year
        pm_parts.append(f"{ys}:{ye}[dp]")
    pm_query = ' AND '.join(pm_parts) if pm_parts else ''
    if not pm_query:
        return []
    logger.info("PubMed query: %s", pm_query)

    # Step 1: search for IDs
    try:
        search_handle = Entrez.esearch(
            db="pubmed", term=pm_query, retmax=max_results, sort="relevance"
        )
        search_results = Entrez.read(search_handle)
        search_handle.close()
    except Exception as exc:
        logger.error("PubMed esearch failed: %s", exc)
        return []

    id_list = search_results.get("IdList", [])
    if not id_list:
        return []

    set_spinner_message(f"PubMed: Fetching {len(id_list)} XML summaries...")
    # Step 2: fetch article details as XML
    try:
        fetch_handle = Entrez.efetch(
            db="pubmed", id=",".join(id_list), rettype="xml", retmode="xml"
        )
        xml_data = fetch_handle.read()
        fetch_handle.close()
    except Exception as exc:
        logger.error("PubMed efetch failed: %s", exc)
        return []

    return _parse_pubmed_xml(xml_data)


def _parse_pubmed_xml(xml_data: bytes | str) -> list[dict]:
    """Parse PubMed XML response into a list of paper dicts.

    If a PMCID is present for an article, we also fetch its full text from
    PubMed Central.
    """
    papers: list[dict] = []
    try:
        root = ET.fromstring(xml_data)
    except ET.ParseError as exc:
        logger.error("Failed to parse PubMed XML: %s", exc)
        return []

    articles = root.findall(".//PubmedArticle")
    total = len(articles)
    for i, article_elem in enumerate(articles, 1):
        medline = article_elem.find("MedlineCitation")
        if medline is None:
            continue
        article = medline.find("Article")
        if article is None:
            continue

        # Title
        title_elem = article.find("ArticleTitle")
        title = (
            "".join(title_elem.itertext()) if title_elem is not None else ""
        )

        # Abstract
        abstract_parts: list[str] = []
        abstract_elem = article.find("Abstract")
        if abstract_elem is not None:
            for text_elem in abstract_elem.findall("AbstractText"):
                label = text_elem.get("Label", "")
                body = "".join(text_elem.itertext())
                if label:
                    abstract_parts.append(f"{label}: {body}")
                else:
                    abstract_parts.append(body)
        abstract = "\n".join(abstract_parts)

        # Authors
        authors: list[str] = []
        author_list = article.find("AuthorList")
        if author_list is not None:
            for author_elem in author_list.findall("Author"):
                last = author_elem.findtext("LastName", default="")
                fore = author_elem.findtext("ForeName", default="")
                name = f"{fore} {last}".strip()
                if name:
                    authors.append(name)

        # Year
        journal = article.find("Journal")
        year: Optional[int] = None
        if journal is not None:
            pub_date = journal.find("JournalIssue/PubDate")
            if pub_date is not None:
                year_text = pub_date.findtext("Year", default="")
                if year_text.isdigit():
                    year = int(year_text)

        # DOI & PMCID
        doi = ""
        pmcid = ""
        article_id_list = article_elem.find("PubmedData/ArticleIdList")
        if article_id_list is not None:
            for aid in article_id_list.findall("ArticleId"):
                id_type = aid.get("IdType", "")
                if id_type == "doi":
                    doi = (aid.text or "").strip()
                elif id_type == "pmc":
                    pmcid = (aid.text or "").strip()

        # PMID
        pmid = medline.findtext("PMID", default="")
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""

        # Fetch full text from PMC if available
        full_text = ""
        if pmcid:
            title_short = title[:30]
            logger.info("Fetching PMC full text for %s (%s)", title[:60], pmcid)
            set_spinner_message(f"PubMed: Fetching PMC {i}/{total} ({title_short}...)")
            full_text = _fetch_pmc_full_text(pmcid)

        papers.append(
            {
                "title": title,
                "authors": authors,
                "year": year,
                "abstract": abstract,
                "full_text": full_text,
                "url": url,
                "doi": doi,
                "citation_count": None,
                "source": "PubMed",
            }
        )
    return papers


# ---------------------------------------------------------------------------
# OpenAlex helpers
# ---------------------------------------------------------------------------

def _reconstruct_abstract(inverted_index: dict) -> str:
    """Reconstruct abstract text from OpenAlex inverted abstract index.

    OpenAlex stores abstracts as {word: [positions]} mappings.  We rebuild
    the plain-text abstract by placing each word at its recorded position(s).
    """
    if not inverted_index:
        return ""
    word_positions: list[tuple[int, str]] = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    word_positions.sort(key=lambda x: x[0])
    return " ".join(word for _, word in word_positions)


def _search_openalex(
    parsed: dict, max_results: int = DEFAULT_MAX_RESULTS
) -> list[dict]:
    """Query the OpenAlex Works API and return a list of paper dicts."""
    set_spinner_message("OpenAlex: Querying API...")

    params: dict = {
        "per_page": min(max_results, 200),  # API max is 200
        "api_key": OPENALEX_API_KEY,
        "select": (
            "id,title,authorships,publication_year,"
            "abstract_inverted_index,doi,cited_by_count,"
            "open_access,best_oa_location"
        ),
    }

    # Use search for topic keywords
    if parsed['topic']:
        params["search"] = parsed['topic']

    # Build filter string for structured fields
    filters: list[str] = []

    # Author: two-step lookup — resolve name → OpenAlex author ID first
    if parsed['author']:
        set_spinner_message("OpenAlex: Resolving author ID...")
        try:
            author_resp = requests.get(
                "https://api.openalex.org/authors",
                params={"search": parsed['author'], "per_page": 1,
                        "api_key": OPENALEX_API_KEY},
                timeout=15,
            )
            author_resp.raise_for_status()
            author_data = author_resp.json()
            author_results = author_data.get("results", [])
            if author_results:
                oa_id = author_results[0]["id"].replace(
                    "https://openalex.org/", ""
                )
                filters.append(f"author.id:{oa_id}")
                logger.info(
                    "OpenAlex author resolved: %s → %s",
                    parsed['author'], oa_id,
                )
            else:
                # Fall back to putting author name in the search param
                search_parts = [parsed['author']]
                if parsed['topic']:
                    search_parts.append(parsed['topic'])
                params["search"] = ' '.join(search_parts)
        except requests.RequestException as exc:
            logger.warning("OpenAlex author lookup failed: %s", exc)
            # Fall back to keyword search
            search_parts = [parsed['author']]
            if parsed['topic']:
                search_parts.append(parsed['topic'])
            params["search"] = ' '.join(search_parts)

    if parsed['year_start'] and parsed['year_end']:
        filters.append(
            f"publication_year:{parsed['year_start']}-{parsed['year_end']}"
        )
    elif parsed['year_start']:
        filters.append(f"publication_year:{parsed['year_start']}-")
    elif parsed['year_end']:
        filters.append(f"publication_year:-{parsed['year_end']}")
    if filters:
        params["filter"] = ','.join(filters)

    # Need at least search or filter
    if "search" not in params and "filter" not in params:
        return []

    logger.info("OpenAlex params: %s", params)

    try:
        resp = requests.get(
            OPENALEX_SEARCH_URL, params=params, timeout=30
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        logger.error("OpenAlex request failed: %s", exc)
        return []

    papers: list[dict] = []
    items = data.get("results", [])
    total = len(items)
    for i, item in enumerate(items, 1):
        # Authors
        authors = []
        for authorship in item.get("authorships", []):
            name = (authorship.get("author") or {}).get("display_name", "")
            if name:
                authors.append(name)

        # DOI (strip leading https://doi.org/)
        raw_doi = item.get("doi", "") or ""
        doi = raw_doi.replace("https://doi.org/", "")

        # Abstract
        abstract = _reconstruct_abstract(
            item.get("abstract_inverted_index") or {}
        )

        # Attempt to get full text from open-access PDF
        full_text = ""
        best_oa = item.get("best_oa_location") or {}
        pdf_url = best_oa.get("pdf_url", "")
        if pdf_url:
            title_short = (item.get("title") or "")[:30]
            logger.info(
                "Downloading open-access PDF for: %s",
                item.get("title", ""),
            )
            set_spinner_message(
                f"OpenAlex: PDF {i}/{total} ({title_short}...)"
            )
            full_text = _download_pdf_text(pdf_url)

        openalex_id = item.get("id", "")
        url = openalex_id if openalex_id else ""

        papers.append(
            {
                "title": item.get("title", ""),
                "authors": authors,
                "year": item.get("publication_year"),
                "abstract": abstract,
                "full_text": full_text,
                "url": url,
                "doi": doi,
                "citation_count": item.get("cited_by_count"),
                "source": "OpenAlex",
            }
        )
    return papers


# ---------------------------------------------------------------------------
# Scopus (Elsevier) helpers
# ---------------------------------------------------------------------------

def _search_scopus(
    parsed: dict, max_results: int = DEFAULT_MAX_RESULTS
) -> list[dict]:
    """Query the Elsevier Scopus Search API and return a list of paper dicts."""
    set_spinner_message("Scopus: Querying API...")

    # Build Scopus structured query
    sq_parts: list[str] = []
    if parsed['author']:
        sq_parts.append(f'AUTH({parsed["author"]})')
    if parsed['institution'] and not parsed['author']:
        sq_parts.append(f'AFFIL("{parsed["institution"]}")')
    if parsed['topic']:
        sq_parts.append(f'TITLE-ABS-KEY({parsed["topic"]})')
    if parsed['year_start']:
        sq_parts.append(f'PUBYEAR > {parsed["year_start"] - 1}')
    if parsed['year_end']:
        sq_parts.append(f'PUBYEAR < {parsed["year_end"] + 1}')
    scopus_query = ' AND '.join(sq_parts) if sq_parts else ''
    if not scopus_query:
        return []
    logger.info("Scopus query: %s", scopus_query)

    headers = {
        "X-ELS-APIKey": ELSEVIER_API_KEY,
        "Accept": "application/json",
    }
    params = {
        "query": scopus_query,
        "count": min(max_results, 25),  # Scopus default max per page
        "sort": "relevancy",
        "field": (
            "dc:title,dc:creator,author,prism:coverDate,"
            "prism:doi,citedby-count,dc:description,"
            "prism:url,prism:publicationName,dc:identifier"
        ),
    }

    try:
        resp = requests.get(
            SCOPUS_SEARCH_URL, headers=headers, params=params, timeout=30
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        logger.error("Scopus request failed: %s", exc)
        return []

    search_results = data.get("search-results", {})
    entries = search_results.get("entry", [])

    # Scopus returns an error entry when no results are found
    if len(entries) == 1 and "error" in entries[0]:
        logger.info("Scopus returned no results for query.")
        return []

    papers: list[dict] = []
    total = len(entries)
    for i, entry in enumerate(entries, 1):
        title = entry.get("dc:title", "")

        # Authors – try 'author' list first, fall back to dc:creator
        authors: list[str] = []
        author_list = entry.get("author", [])
        if isinstance(author_list, list):
            for a in author_list:
                name = a.get("authname", "") or a.get("given-name", "")
                if name:
                    authors.append(name)
        if not authors:
            creator = entry.get("dc:creator", "")
            if creator:
                authors.append(creator)

        # Year from prism:coverDate (YYYY-MM-DD)
        cover_date = entry.get("prism:coverDate", "")
        year: Optional[int] = None
        if cover_date and len(cover_date) >= 4:
            try:
                year = int(cover_date[:4])
            except ValueError:
                pass

        doi = entry.get("prism:doi", "")

        # Abstract / description
        abstract = entry.get("dc:description", "") or ""

        # Citation count
        cite_count: Optional[int] = None
        raw_cite = entry.get("citedby-count", None)
        if raw_cite is not None:
            try:
                cite_count = int(raw_cite)
            except (ValueError, TypeError):
                pass

        # URL – Scopus abstract link
        url = ""
        links = entry.get("link", [])
        if isinstance(links, list):
            for link in links:
                if link.get("@ref") == "scopus":
                    url = link.get("@href", "")
                    break
        if not url:
            url = entry.get("prism:url", "")

        title_short = title[:30]
        set_spinner_message(f"Scopus: Processing {i}/{total} ({title_short}...)")

        papers.append(
            {
                "title": title,
                "authors": authors,
                "year": year,
                "abstract": abstract,
                "full_text": "",  # Scopus doesn't provide full text via search
                "url": url,
                "doi": doi,
                "citation_count": cite_count,
                "source": "Scopus",
            }
        )
    return papers


# ---------------------------------------------------------------------------
# Deduplication & formatting
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Lowercase, strip whitespace and punctuation for comparison."""
    return "".join(ch for ch in text.lower() if ch.isalnum())


def _deduplicate(papers: list[dict]) -> list[dict]:
    """Remove duplicate papers preferring the entry with more metadata."""
    seen_titles: dict[str, int] = {}
    seen_dois: dict[str, int] = {}
    unique: list[dict] = []

    def _richness(p: dict) -> int:
        """Score how much content a paper entry contains."""
        return len(p.get("full_text", "")) + len(p.get("abstract", ""))

    for paper in papers:
        norm_title = _normalize(paper.get("title", ""))
        doi = (paper.get("doi") or "").strip().lower()

        # Check if already seen by DOI
        if doi and doi in seen_dois:
            idx = seen_dois[doi]
            if _richness(paper) > _richness(unique[idx]):
                unique[idx] = paper
            continue

        # Check if already seen by title
        if norm_title and norm_title in seen_titles:
            idx = seen_titles[norm_title]
            if _richness(paper) > _richness(unique[idx]):
                unique[idx] = paper
            continue

        idx = len(unique)
        unique.append(paper)
        if norm_title:
            seen_titles[norm_title] = idx
        if doi:
            seen_dois[doi] = idx

    return unique


def _format_results(papers: list[dict]) -> str:
    """Format paper list into a readable markdown string."""
    if not papers:
        return "No papers found for the given query."

    lines: list[str] = []
    for i, p in enumerate(papers, 1):
        authors_str = ", ".join(p["authors"]) if p["authors"] else "N/A"
        year_str = str(p["year"]) if p["year"] else "N/A"
        citation_str = (
            str(p["citation_count"])
            if p["citation_count"] is not None
            else "N/A"
        )

        lines.append(f"## {i}. {p['title']}")
        lines.append(f"**Authors:** {authors_str}")
        lines.append(f"**Year:** {year_str}  ")
        lines.append(f"**Citations:** {citation_str}  ")
        lines.append(f"**Source:** {p['source']}  ")
        if p.get("doi"):
            lines.append(f"**DOI:** {p['doi']}  ")
        if p.get("url"):
            lines.append(f"**URL:** {p['url']}  ")
        lines.append("")

        # Full text or abstract
        full_text = (p.get("full_text") or "").strip()
        abstract = (p.get("abstract") or "").strip()

        if full_text:
            lines.append("### Full Text")
            lines.append(full_text)
        elif abstract:
            lines.append(f"**Abstract:** {abstract}")
        else:
            lines.append("*No text available.*")

        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core search logic (shared by CLI and MCP)
# ---------------------------------------------------------------------------

def _run_search(query: str, max_results_per_source: int = DEFAULT_MAX_RESULTS) -> str:
    """Run the literature search pipeline and return formatted markdown."""
    parsed = _parse_query(query)
    logger.info(
        "Searching for: %s (parsed: %s, max %d per source)",
        query, parsed, max_results_per_source,
    )

    ss_papers = _search_semantic_scholar(parsed, max_results_per_source)
    pm_papers = _search_pubmed(parsed, max_results_per_source)
    oa_papers = _search_openalex(parsed, max_results_per_source)
    sc_papers = _search_scopus(parsed, max_results_per_source)

    logger.info(
        "Found %d from Semantic Scholar, %d from PubMed, "
        "%d from OpenAlex, %d from Scopus",
        len(ss_papers),
        len(pm_papers),
        len(oa_papers),
        len(sc_papers),
    )

    set_spinner_message("Deduplicating papers...")
    all_papers = ss_papers + pm_papers + oa_papers + sc_papers
    unique_papers = _deduplicate(all_papers)

    logger.info("After deduplication: %d unique papers", len(unique_papers))

    set_spinner_message("Formatting results...")
    return _format_results(unique_papers)


# ---------------------------------------------------------------------------
# MCP Tool (wraps the shared search logic)
# ---------------------------------------------------------------------------

@mcp.tool()
def search_literature(
    query: str,
    max_results_per_source: int = DEFAULT_MAX_RESULTS,
) -> str:
    """Search PubMed, Semantic Scholar, OpenAlex, and Scopus for papers
    matching a natural language query.  Returns a deduplicated, formatted
    list of publications with title, authors, year, citation count, URL,
    and **full text** when available (via PubMed Central or open-access
    PDFs), falling back to abstracts.

    Args:
        query: Natural language search query describing the topic of interest
               (e.g. "machine learning for cryo-electron microscopy").
        max_results_per_source: Maximum number of results to fetch from each
                                 source (default 20).
    """
    return _run_search(query, max_results_per_source)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli_search(args: argparse.Namespace) -> None:
    """Handle the 'search' subcommand."""
    global _cli_spinner
    query = args.query
    if not query:
        query = input("Enter your search query: ").strip()
        if not query:
            print("Error: empty query. Exiting.")
            return

    # Suppress logger info so it doesn't mess with the spinner
    logger.setLevel(logging.WARNING)
    _cli_spinner = Spinner("Starting search...")
    _cli_spinner.start()
    try:
        result = _run_search(query, args.max_results)
    finally:
        _cli_spinner.stop()
        logger.setLevel(logging.INFO)

    # Build output path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, os.pardir, "results")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(results_dir, f"literature_review_{timestamp}.md")

    # Write header + results
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# Literature Review\n")
        f.write(f"**Query:** {query}  \n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write(f"**Max results per source:** {args.max_results}\n\n")
        f.write(result)

    print(f"\n✅ Results saved to: {os.path.abspath(out_path)}")


def _cli_serve(_args: argparse.Namespace) -> None:
    """Handle the 'serve' subcommand."""
    print("Starting Literature Review MCP server (stdio transport)...")
    mcp.run(transport="stdio")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Literature Review – search PubMed, Semantic Scholar, OpenAlex & Scopus"
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- search ---
    search_parser = subparsers.add_parser(
        "search", help="Run a standalone literature search"
    )
    search_parser.add_argument(
        "--query", "-q", type=str, default=None,
        help="Natural language search query (interactive prompt if omitted)",
    )
    search_parser.add_argument(
        "--max-results", "-n", type=int, default=DEFAULT_MAX_RESULTS,
        help=f"Max results per source (default {DEFAULT_MAX_RESULTS})",
    )

    # --- serve ---
    subparsers.add_parser(
        "serve", help="Start the MCP server (stdio transport)"
    )

    args = parser.parse_args()

    if args.command == "serve":
        _cli_serve(args)
    else:
        # Default to search (interactive if no --query)
        if args.command is None:
            args = search_parser.parse_args([])  # defaults
        _cli_search(args)


if __name__ == "__main__":
    main()
