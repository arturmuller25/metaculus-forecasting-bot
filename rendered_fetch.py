"""
Reads a web page through a rendering reader, for pages whose direct download
comes back empty, blocked, or as a shell that JavaScript fills in.

The reader is the public r.jina.ai service: it loads the page in a browser
and returns its text as Markdown. It needs no key and no new dependency, and
the request carries the page URL and nothing else. PDF files are read too.

What this module adds to a plain call to the reader:
  - menus, headers, footers, sidebars and the consent dialogs of the common
    vendors are removed on the reader's side, link targets and images are
    dropped, and consent text and blank lines are removed here, so the text
    starts at the content. A plain call spent the text budget on navigation
    links in 33 of the 92 pages read this way in MiniBench rounds 3 and 4
    and the Fall 2026 tournament;
  - a URL with a fragment ("#/caseinfo/2026/1353", the route of a
    single-page app) is sent by POST, because a GET drops the fragment;
  - a page that comes back almost empty, still loading, or as a copy the
    reader had stored is read once more, fresh and with a wait; one that is
    still almost empty is then read with nothing removed (some sites keep
    their content in a header or sidebar element). A stored copy that could
    not be refreshed is delivered with a note saying so;
  - a bot check is told from a page by its size, so a long page that only
    mentions a CAPTCHA (a consent dialog listing a reCAPTCHA cookie) is kept;
  - one answer per URL is kept in memory for the life of the process;
  - a time limit, a size limit and no exception: any failure returns None.

RENDERED_FETCH=0 turns the reader off.
"""

from __future__ import annotations

import os
import re
import threading
import time
from typing import NamedTuple

import requests

READER = "https://r.jina.ai/"
TIMEOUT = 40  # seconds for one call to the reader
BUDGET = 60  # seconds for every call made for one URL
MAX_BYTES = 2_000_000  # bytes read from one answer
MAX_CHARS = 200_000  # characters of page text kept
MIN_CONTENT = 100  # letters and digits below which a page counts as unread
ALMOST_EMPTY = 200  # letters and digits below which the page is read again
BOT_CHECK = 1_000  # letters and digits above which a page is not a bot check

# Elements removed by the reader. A form is not among them: an ASP.NET page
# is one form around all its content. The ids and classes are the consent
# dialogs of OneTrust, Cookiebot, CookieYes, Usercentrics, Didomi, TrustArc,
# Quantcast, Osano and Complianz.
_FURNITURE = (
    "header, footer, nav, aside, #onetrust-consent-sdk, #CybotCookiebotDialog, .cky-consent-container, "
    ".cky-modal, #usercentrics-root, #didomi-host, #truste-consent-track, .qc-cmp2-container, "
    ".osano-cm-window, #cmplz-cookiebanner-container"
)
_HEADERS = {
    "Accept": "text/plain",
    "X-Retain-Images": "none",
    "X-Md-Link-Style": "discarded",
    "X-With-Links-Summary": "true",
    "X-Remove-Selector": _FURNITURE,
}
# The second reading is fresh (the reader keeps pages for an hour and at
# times serves an older, even empty, copy) and waits for the page to settle,
# which is the reader's own advice when it warns that a page was still
# loading. The third also removes nothing.
_HEADERS_FRESH = _HEADERS | {"X-No-Cache": "true", "X-Timeout": "15", "X-With-Shadow-Dom": "true"}
_HEADERS_WHOLE = {k: v for k, v in _HEADERS_FRESH.items() if k != "X-Remove-Selector"}

_TARGET_ERROR = re.compile(r"(?m)^Warning: Target URL returned error \d{3}")
_STILL_LOADING = re.compile(r"(?m)^Warning: .*(not yet fully loaded|shadow DOM)")
_CACHED = re.compile(r"(?m)^Warning: This is a cached snapshot")
_SUSPECTED = re.compile(r"(?m)^Warning: .*requiring CAPTCHA")
_BOT_CHECK = re.compile(
    r"Just a moment|Enable JavaScript and cookies|Attention Required|Checking your browser"
    r"|Access Denied|are you a robot|are you a human|verify you are (a )?human|captcha"
    r"|security verification|request (was|has been) blocked|unusual traffic",
    re.IGNORECASE,
)
# Lines of a consent notice that survive the reader's cleaning. A line about
# cookies is dropped only when it also uses the vocabulary of a notice and
# carries no number, so "cookie sales rose 4%" stays.
_COOKIE = re.compile(r"cookies?\b", re.IGNORECASE)
_NOTICE = re.compile(
    r"\b(we use|uses?|using|accept|consent|polic(y|ies)|settings?|preferences?|necessary|experience"
    r"|browser|analytics|advertising|third.part(y|ies)|website|site|personal)\b",
    re.IGNORECASE,
)
_CONSENT_LINE = re.compile(
    r"^skip to [a-z ]+$"
    r"|^(i )?(accept|reject|allow|decline|deny)( all)?( cookies)?$"
    r"|\b(accept all|reject all|only necessary|manage (consent )?preferences"
    r"|customi[sz]e (your choice|consent preferences)|save and (exit|close)|we value your privacy"
    r"|protected by recaptcha)\b",
    re.IGNORECASE,
)
# The path of a login, consent or challenge page a site redirects to.
_WALL = re.compile(r"/(log-?in|sign-?in|sso|auth|consent|captcha|challenge|checkpoint)(\.[a-z]+)?(/|$)", re.IGNORECASE)
_SPACES = re.compile(r"[ \t\r\f\v\xa0]+")  # \xa0 is the no-break space
# A line of Markdown markup alone: rules, table separators, empty bullets and
# check boxes. The last two characters are the middle dot and the bullet.
_MARKUP_ONLY = re.compile(r"^[\s*_#>|\-=+.:\[\]x\xb7" + chr(0x2022) + r"]*$")


class Page(NamedTuple):
    """A page as the reader returned it."""

    text: str  # "Title: ..." and the cleaned content
    links: str  # the reader's list of the page's links, one "- [label](url)" per line
    size: int  # letters and digits in the content, without the title lines


_cache: dict[str, Page | None] = {}
_lock = threading.Lock()


def enabled() -> bool:
    """False when RENDERED_FETCH is "0"; on by default."""
    return os.getenv("RENDERED_FETCH", "").strip() != "0"


def clear_cache() -> None:
    with _lock:
        _cache.clear()


def content_size(text: str) -> int:
    """Letters and digits in a text: its size without markup, spaces and punctuation."""
    return sum(c.isalnum() for c in text)


def tidy(markdown: str) -> str:
    """The reader's Markdown without blank lines, consent notices, skip links and lines of markup alone."""
    out: list[str] = []
    for line in markdown.split("\n"):
        line = _SPACES.sub(" ", line).strip()
        if not line or (line != "-" and _MARKUP_ONLY.match(line)):  # a lone dash can be an empty table cell
            continue
        if len(line) <= 1200:  # a longer line is a paragraph of the page, whatever it mentions
            if _CONSENT_LINE.search(line):
                continue
            if _COOKIE.search(line) and _NOTICE.search(line) and not re.search(r"\d", line):
                continue
        out.append(line)
    return "\n".join(out)


def _parse(raw: str) -> Page | None:
    """The reader's answer as a Page, or None when it carries an error page or a bot check."""
    head, sep, body = raw.partition("Markdown Content:\n")
    if not sep:  # an answer without the reader's preamble: all of it is content
        head, body = "", raw
    if _TARGET_ERROR.search(head):
        return None
    links = ""
    cut = body.rfind("\nLinks/Buttons:\n")
    if cut >= 0:
        body, links = body[:cut], body[cut + len("\nLinks/Buttons:\n"):]
    content = tidy(body[:MAX_CHARS])
    size = content_size(content)
    # The date is kept for a PDF only, where it is the file's own. On a web
    # page it is often the day the page was created, which reads as "stale"
    # above figures updated today.
    labels = ("Title", "Final URL", "Published Time", "Number of Pages") if "\nNumber of Pages: " in head else ("Title", "Final URL")
    lines = []
    for label in labels:
        m = re.search(rf"(?m)^{label}: (.+)$", head)
        if m:
            lines.append(f"{label}: {m.group(1).strip()}")
    # A bot check is a short page that says so in its title or text. The
    # reader's own "maybe requiring CAPTCHA" warning counts only on a page
    # that is almost empty: it also comes with ordinary pages that load a
    # reCAPTCHA script.
    if size < BOT_CHECK and _BOT_CHECK.search("\n".join(lines[:1] + [content])):
        return None
    if size < ALMOST_EMPTY and _SUSPECTED.search(head):
        return None
    final = re.search(r"(?m)^Final URL: (.+)$", head)
    if final and size < BOT_CHECK and _WALL.search(final.group(1).split("?")[0]):
        return None
    if _CACHED.search(head):
        lines.append("Note: the reader served a stored copy of this page, which may be out of date.")
    return Page("\n".join(lines + [content]), links.strip(), size)


def _call(url: str, headers: dict[str, str], deadline: float) -> str | None:
    """One answer of the reader as text, or None; waits once when the reader says it is rate limited."""
    for attempt in (1, 2):
        left = deadline - time.monotonic()
        if left < 5:
            return None
        timeout = min(TIMEOUT, left)
        if "#" in url:  # a GET would drop the fragment
            request = requests.post(READER, data={"url": url}, headers=headers, timeout=timeout, stream=True)
        else:
            request = requests.get(READER + url, headers=headers, timeout=timeout, stream=True)
        with request as r:
            if r.status_code == 429 and attempt == 1:
                pause = str(r.headers.get("Retry-After", ""))
                pause = min(float(pause), 10.0) if pause.replace(".", "", 1).isdigit() else 5.0
                if deadline - time.monotonic() < pause + 5:
                    return None
                time.sleep(pause)
                continue
            if r.status_code >= 400:
                return None
            chunks, size = [], 0
            for chunk in r.iter_content(65536):
                chunks.append(chunk)
                size += len(chunk)
                if size >= MAX_BYTES or time.monotonic() > deadline:
                    break
            return b"".join(chunks)[:MAX_BYTES].decode("utf-8", errors="replace")  # the reader answers in UTF-8
    return None


def _fetch(url: str) -> tuple[Page | None, bool]:
    """The page, and whether the reader answered at all (an answer is kept, a failed call is tried again)."""
    deadline = time.monotonic() + BUDGET
    page, answered = None, False
    try:
        raw = _call(url, _HEADERS, deadline)
        if raw is None:
            return None, False
        answered = True
        page = _parse(raw)
        if page is None:
            return None, True  # the site answered with an error or a bot check: reading again gives the same
        stored = bool(_CACHED.search(raw[:2000]))
        if page.size < ALMOST_EMPTY or stored or _STILL_LOADING.search(raw[:2000]):
            fresh = _call(url, _HEADERS_FRESH, deadline)
            fresh = _parse(fresh) if fresh else None
            # A fresh page replaces a stored copy even when it is shorter.
            if fresh and (fresh.size > page.size or (stored and fresh.size >= ALMOST_EMPTY)):
                page = fresh
            if page.size < ALMOST_EMPTY:
                whole = _call(url, _HEADERS_WHOLE, deadline)
                whole = _parse(whole) if whole else None
                if whole and whole.size > page.size:
                    page = whole
    except Exception:  # network errors, timeouts, bad encodings
        pass
    if page is not None and page.size < MIN_CONTENT:
        page = None
    return page, answered


def fetch_rendered(url: str) -> Page | None:
    """
    The page at the URL as the reader renders it, or None when the reader is
    off, fails, times out, or returns an error page, a bot check or nothing.
    Never raises.
    """
    if not enabled():
        return None
    with _lock:
        if url in _cache:
            return _cache[url]
    page, answered = _fetch(url)
    if answered:
        with _lock:
            _cache[url] = page
    return page
