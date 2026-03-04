#!/usr/bin/env python3
"""
Literature Review MCP Server
=============================
An MCP (Model Context Protocol) server that automates literature review by
searching PubMed and Semantic Scholar for relevant publications given a
natural language prompt.  Retrieves **full text** whenever possible (via
PubMed Central XML or open-access PDFs), falling back to abstracts.

Usage (stdio transport):
    python src/literature_review.py

Tools exposed:
    - search_literature: Search both PubMed and Semantic Scholar for papers
      matching a natural language query, deduplicate, and return structured
      results with full text when available.
"""

import argparse
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

SEMANTIC_SCHOLAR_SEARCH_URL = (
    "https://api.semanticscholar.org/graph/v1/paper/search"
)
SEMANTIC_SCHOLAR_FIELDS = (
    "title,authors,year,abstract,url,externalIds,citationCount,openAccessPdf"
)

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
        "Search PubMed and Semantic Scholar for relevant publications "
        "and return structured results with full text when available."
    ),
)


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


def _download_pdf_text(url: str) -> str:
    """Download a PDF from *url* and return its extracted text."""
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("PDF download failed (%s): %s", url, exc)
        return ""
    return _extract_text_from_pdf(resp.content)


# ---------------------------------------------------------------------------
# Semantic Scholar helpers
# ---------------------------------------------------------------------------

def _search_semantic_scholar(
    query: str, max_results: int = DEFAULT_MAX_RESULTS
) -> list[dict]:
    """Query Semantic Scholar Graph API and return a list of paper dicts."""
    set_spinner_message("Semantic Scholar: Querying API...")
    headers = {"x-api-key": SEMANTIC_SCHOLAR_API_KEY}
    params = {
        "query": query,
        "limit": max_results,
        "fields": SEMANTIC_SCHOLAR_FIELDS,
    }

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
    query: str, max_results: int = DEFAULT_MAX_RESULTS
) -> list[dict]:
    """Query PubMed via NCBI E-Utilities and return a list of paper dicts.

    For articles available in PubMed Central, the full text body is fetched.
    """
    set_spinner_message("PubMed: Querying esearch API...")
    Entrez.email = PUBMED_EMAIL
    Entrez.api_key = PUBMED_API_KEY

    # Step 1: search for IDs
    try:
        search_handle = Entrez.esearch(
            db="pubmed", term=query, retmax=max_results, sort="relevance"
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
    logger.info(
        "Searching for: %s (max %d per source)", query, max_results_per_source
    )

    ss_papers = _search_semantic_scholar(query, max_results_per_source)
    pm_papers = _search_pubmed(query, max_results_per_source)

    logger.info(
        "Found %d from Semantic Scholar, %d from PubMed",
        len(ss_papers),
        len(pm_papers),
    )

    set_spinner_message("Deduplicating papers...")
    all_papers = ss_papers + pm_papers
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
    """Search PubMed and Semantic Scholar for papers matching a natural
    language query.  Returns a deduplicated, formatted list of publications
    with title, authors, year, citation count, URL, and **full text** when
    available (via PubMed Central or open-access PDFs), falling back to
    abstracts.

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
        description="Literature Review – search PubMed & Semantic Scholar"
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
