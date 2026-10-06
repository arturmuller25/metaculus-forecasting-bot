"""
Fetches the pages and data files named in a question's resolution criteria
and fine print, without any LLM, so the forecasters see the resolution
source itself and not only what the research models chose to quote.

For each URL (Metaculus pages excluded, at most MAX_URLS):
  1. a direct download: a CSV or JSON file is summarized as data (columns,
     row count, the latest rows, filtered to rows that mention the
     question's key words when the file is large); an HTML page becomes
     plain text, plus the CSV file it links to, if any (data portals); a
     Humanitarian Data Exchange dataset is read through its CKAN API;
  2. when the direct download is blocked or the page is built by
     JavaScript, the r.jina.ai reader, which renders the page;
  3. as a last resort, the latest Internet Archive snapshot, labeled with
     its date.
A URL that cannot be read is reported as such, so the forecaster knows the
source was not seen.
"""

from __future__ import annotations

import csv
import html
import io
import json
import re
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime, timezone
from html.parser import HTMLParser
from urllib.parse import urljoin

import requests

MAX_URLS = 4
MAX_DOWNLOAD = 25_000_000  # bytes read from one response
MAX_TEXT = 6_000  # characters kept per URL
TIMEOUT = 25
DEADLINE = 100  # seconds for all URLs of a question together
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/json,text/csv,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

_URL_RE = re.compile(r"https?://[^\s<>\"'\)\]\}]+")
# Bot checks and error pages that come back with a normal status.
_BLOCKED = re.compile(
    r"Just a moment|Enable JavaScript and cookies|Attention Required|Checking your browser"
    r"|returned error 40\d|Access Denied|are you a robot|captcha",
    re.IGNORECASE,
)
_CSV_LINK = re.compile(r"""(?:href=["']|\()([^"'\s()<>]+\.csv(?:\?[^"'\s()<>]*)?)""", re.IGNORECASE)
_STOP = {
    "will", "what", "which", "before", "after", "between", "there", "their",
    "about", "under", "least", "more", "than", "this", "that", "with", "from",
    "into", "have", "been", "when", "where", "much", "many", "total", "number",
    "value", "according", "official", "report", "reported", "question",
}


def urls_in(*texts: str | None) -> list[str]:
    """Distinct URLs in the texts, in order, without Metaculus pages."""
    found: list[str] = []
    for text in texts:
        for url in _URL_RE.findall(text or ""):
            url = url.rstrip(".,;:!?*_")
            if "metaculus.com" in url or url in found:
                continue
            found.append(url)
    return found[:MAX_URLS]


def keywords(text: str, criteria: str | None = None) -> list[str]:
    """
    Distinctive words used to pick the relevant rows or passages: the words
    of the question title, then the proper names in the resolution criteria
    (which often name the ports, series or places the title leaves out).
    """
    words = re.findall(r"[A-Za-z][A-Za-z\-]{3,}", text or "")
    words += re.findall(r"\b[A-Z][a-z][A-Za-z\-]{2,}", re.sub(r"https?://\S+", "", criteria or ""))
    seen: list[str] = []
    for w in words:
        lw = w.lower()
        if lw not in _STOP and lw not in seen:
            seen.append(lw)
    return seen[:20]


class _Text(HTMLParser):
    """Visible text of an HTML page, without scripts, styles and navigation."""

    _SKIP = {"script", "style", "noscript", "svg", "nav", "footer", "header", "form"}
    _BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "table", "section", "article"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip = 0
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self.skip += 1
        if tag == "title":
            self._in_title = True
        if tag in self._BLOCK:
            self.parts.append("\n")
        if tag in ("td", "th"):
            self.parts.append(" | ")

    def handle_endtag(self, tag):
        if tag in self._SKIP and self.skip:
            self.skip -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self.skip:
            self.parts.append(data)

    def text(self) -> str:
        raw = "".join(self.parts)
        lines = [re.sub(r"[ \t\r\f\v]+", " ", line).strip() for line in raw.split("\n")]
        return "\n".join(line for line in lines if line)


def _focus(text: str, words: list[str]) -> str:
    """The start of a long text plus the passages around the question's key words."""
    if len(text) <= MAX_TEXT:
        return text
    head = text[:2000]
    spans: list[tuple[int, int]] = []
    low = text.lower()
    for w in words:
        for m in re.finditer(re.escape(w), low):
            spans.append((max(0, m.start() - 300), min(len(text), m.end() + 300)))
    spans.sort()
    merged: list[list[int]] = []
    for a, b in spans:
        if a < 2000:
            a = 2000
        if a >= b:
            continue
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    out, budget = [head], MAX_TEXT - len(head)
    for a, b in merged:
        piece = text[a:b]
        if budget <= 0:
            break
        out.append("[...] " + piece[:budget])
        budget -= len(piece)
    if len(out) == 1:
        out.append("[...] " + text[2000:MAX_TEXT])
    return "\n".join(out)


def _summarize_csv(raw: str, words: list[str], truncated: bool) -> str:
    try:
        dialect = csv.Sniffer().sniff(raw[:5000], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    rows = list(csv.reader(io.StringIO(raw), dialect))
    if truncated and rows:
        rows = rows[:-1]  # the last row may be cut
    # The header is the first row with three filled cells: files from
    # statistics offices often open with title lines.
    start = next((i for i, r in enumerate(rows[:20]) if sum(1 for c in r if c.strip()) >= 3), 0)
    rows = [r for r in rows[start:] if any(c.strip() for c in r)]
    if not rows:
        return "Empty file."
    header, body = rows[0], rows[1:]
    keep = [i for i, h in enumerate(header) if h.strip() or any(i < len(r) and r[i].strip() for r in body[-50:])]
    header = [header[i] for i in keep]
    body = [[r[i] if i < len(r) else "" for i in keep] for r in body]
    # Sorted by an ISO date column when there is one, so the latest rows come
    # last whatever the file's own order.
    iso = re.compile(r"\d{4}-\d{2}")
    date_col = next(
        (i for i in range(len(header)) if body and all(iso.match(r[i] or "") for r in body[:5] + body[-5:])),
        None,
    )
    if date_col is not None:
        body.sort(key=lambda r: r[date_col])
    # Rows naming the question's distinctive words, ignoring words that
    # appear in most rows (such as the country in a one-country file).
    lowered = [" ".join(r).lower() for r in body]
    useful = [w for w in words if 0 < sum(w in t for t in lowered) < 0.5 * len(body)] if len(body) > 60 else []
    chosen = [r for r, t in zip(body, lowered) if any(w in t for w in useful)] if useful else body
    lines = [f"CSV with {len(body)} rows{' (download truncated)' if truncated else ''}; columns: {', '.join(header)}"]
    label = f"rows mentioning {', '.join(useful)}" if useful else "rows"
    lines.append(f"Last {min(40, len(chosen))} {label}{' by date' if date_col is not None else ''}:")
    lines.append(",".join(header))
    lines += [",".join(r) for r in chosen[-40:]]
    return "\n".join(lines)[: MAX_TEXT * 2]


def _summarize_json(raw: str) -> str:
    data = json.loads(raw)
    if isinstance(data, list):
        return f"JSON list with {len(data)} items; last 20:\n" + json.dumps(data[-20:], ensure_ascii=False)[:MAX_TEXT]
    return json.dumps(data, ensure_ascii=False)[:MAX_TEXT]


def _download(url: str) -> tuple[int, str, str, bool]:
    """Status, content type, text and whether the body was truncated."""
    with requests.get(url, headers=HEADERS, timeout=TIMEOUT, stream=True, allow_redirects=True) as r:
        chunks, size, truncated = [], 0, False
        for chunk in r.iter_content(65536):
            chunks.append(chunk)
            size += len(chunk)
            if size >= MAX_DOWNLOAD:
                truncated = True
                break
        body = b"".join(chunks)
        encoding = r.encoding or "utf-8"
        return r.status_code, r.headers.get("Content-Type", ""), body.decode(encoding, errors="replace"), truncated


def _linked_data(page_url: str, raw: str, words: list[str]) -> str:
    """Summary of the CSV file a data portal page links to, preferring one that mentions the question's words."""
    links = [urljoin(page_url, html.unescape(link)) for link in _CSV_LINK.findall(raw)]
    if not links:
        return ""
    link = next((x for x in links if any(w in x.lower() for w in words)), links[0])
    try:
        status, _, text, truncated = _download(link)
    except Exception:
        return ""
    if status >= 400:
        return ""
    return f"\n\nLinked data file {link}:\n{_summarize_csv(text, words, truncated)}"


def _read(url: str, words: list[str]) -> str | None:
    """Summary of a directly downloaded URL, or None when it was blocked or empty."""
    status, ctype, text, truncated = _download(url)
    if status >= 400:
        return None
    path = url.lower().split("?")[0]
    if "csv" in ctype or path.endswith((".csv", ".tsv")):
        return _summarize_csv(text, words, truncated)
    if "json" in ctype or path.endswith(".json"):
        try:
            return _summarize_json(text)
        except ValueError:
            pass
    if "pdf" in ctype or path.endswith(".pdf"):
        return None
    parser = _Text()
    parser.feed(text)
    body = parser.text()
    if len(body) < 500 or _BLOCKED.search(body[:3000]):  # built by JavaScript, or a bot check
        return None
    title = parser.title.strip()
    return (f"Title: {title}\n" if title else "") + _focus(body, words) + _linked_data(url, text, words)


def _read_rendered(url: str, words: list[str]) -> str | None:
    r = requests.get(f"https://r.jina.ai/{url}", headers={"Accept": "text/plain"}, timeout=40)
    if r.status_code >= 400 or len(r.text) < 300 or _BLOCKED.search(r.text[:3000]):
        return None
    return _focus(r.text, words) + _linked_data(url, r.text, words)


def _read_hdx(url: str, words: list[str]) -> str | None:
    """
    A Humanitarian Data Exchange dataset page: its landing page is mostly
    navigation and sits behind a bot check, but the CKAN API lists the data
    files, which download directly.
    """
    m = re.match(r"https?://data\.humdata\.org/dataset/([^/?#]+)", url)
    if not m:
        return None
    r = requests.get(
        "https://data.humdata.org/api/3/action/package_show",
        params={"id": m.group(1)}, headers=HEADERS, timeout=TIMEOUT,
    )
    if not r.ok:
        return None
    files = [x for x in r.json()["result"].get("resources", []) if (x.get("format") or "").upper() == "CSV"]
    if not files:
        return None
    pick = next(
        (x for x in files if any(w in f"{x.get('name', '')} {x.get('description', '')}".lower() for w in words)),
        files[0],
    )
    summary = _read(pick["url"], words)
    return f"HDX data file {pick.get('name', '')} ({pick['url']}):\n{summary}" if summary else None


def _read_archived(url: str, words: list[str]) -> tuple[str, str] | None:
    r = requests.get("https://archive.org/wayback/available", params={"url": url}, timeout=TIMEOUT)
    snap = (r.json().get("archived_snapshots") or {}).get("closest") or {}
    if not snap.get("available"):
        return None
    content = _read(snap["url"], words)
    return (snap.get("timestamp", ""), content) if content else None


def fetch_one(url: str, words: list[str]) -> tuple[str, str]:
    """(how it was read, text) for one URL; never raises."""
    errors = []
    readers = (("HDX data API", _read_hdx), ("direct download", _read), ("rendered by r.jina.ai", _read_rendered))
    for how, reader in readers:
        if how == "HDX data API" and "data.humdata.org/dataset/" not in url:
            continue
        try:
            text = reader(url, words)
            if text:
                return how, text
            errors.append(f"{how}: blocked or empty")
        except Exception as exc:  # network errors, timeouts, bad encodings
            errors.append(f"{how}: {type(exc).__name__}")
    try:
        archived = _read_archived(url, words)
        if archived:
            stamp, text = archived
            return f"Internet Archive snapshot {stamp[:8]}", text
        errors.append("archive: no snapshot")
    except Exception as exc:
        errors.append(f"archive: {type(exc).__name__}")
    return "not read", "; ".join(errors)


def fetch_resolution_sources(question) -> str:
    """Research block with the resolution sources read directly, or an empty string when there are no URLs."""
    urls = urls_in(question.resolution_criteria, question.fine_print)
    if not urls:
        return ""
    words = keywords(question.question_text or "", question.resolution_criteria)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    parts = [
        f"Pages and files named in the resolution criteria, downloaded by code at {now}. "
        "This is the source as it stands now; it may lag the events it records."
    ]
    # In parallel and under one deadline, so a slow site never holds up the
    # question; the LLM research runs alongside and takes longer anyway.
    pool = ThreadPoolExecutor(max_workers=len(urls))
    futures = {url: pool.submit(fetch_one, url, words) for url in urls}
    done, _ = wait(futures.values(), timeout=DEADLINE)
    pool.shutdown(wait=False, cancel_futures=True)
    for url, future in futures.items():
        how, text = future.result() if future in done else ("not read", f"no answer within {DEADLINE} s")
        if how == "not read":
            parts.append(f"### {url}\nNOT READ ({text}). The forecaster has not seen this source.")
        else:
            parts.append(f"### {url}\nRead by {how}.\n{text}")
    return "\n\n".join(parts)
