"""
Notes that Metaculus staff left on a question's page, for the forecast prompt.

Staff sometimes clarify or amend how a question resolves in a comment (an
edge case, a replaced source, a pushed-back date, an annulment). The question
text the bot reads does not carry those comments. build() asks the comments
API for the staff comments of one post and returns them as a short dated
block, newest first.

What goes in:
  - comments whose author is Metaculus staff (author.is_staff is True); a
    project admin or curator who is not staff (the owner of a community
    project, for example) stays out;
  - any clarification field the question's own JSON carries. The API has no
    such field as of 2026-11; the lookup is by key name ("clarif").
What stays out:
  - every comment by anyone else, and every comment by a bot account;
  - a comment or clarification field that carries a forecast
    (included_forecast) or talks about forecast numbers, the community
    prediction or the crowd's view, even from staff;
  - private and deleted comments.

One GET per post every CACHE_S seconds (/api/comments/?post=<id>&author_is_staff=true),
shared by every build of the process (the shadows of one question and the
subquestions of one group post), at most MAX_CALLS calls in flight at once,
at most TIMEOUT seconds per build, never raises, None when there is nothing
to show. A failed call is remembered for CACHE_S too. The server filter
keeps the answer small on posts with hundreds of bot comments, and a bot
account must send it: the API refuses (403) a comment listing without
author_is_staff or the account's own author id. The same test is applied
again here, so a server that ignored the filter would not let other
forecasters' comments through. Per the API documentation that filter returns
root comments only: a staff reply inside another comment's thread is not
read. On a group post (a series of subquestions) the notes of the whole post
come in, for example the resolved value of an earlier period.

The block reaches a prompt only through the "+admin" variant of
prompt_variants.py; the caller decides where it goes. build() reads the
comments as they are now: a replay must call render() on saved comments with
`before` set to the time of the original forecast.
"""

from __future__ import annotations

import html
import logging
import os
import re
import threading
import time
from datetime import datetime, timezone

import requests

logger = logging.getLogger(__name__)

API = "https://www.metaculus.com/api"
HEADER = "## Notes from Metaculus staff on this question"
MAX_CHARS = 1_500  # the whole block, header included
MIN_NOTE = 250  # characters a note is given before older ones are left out
TIMEOUT = 10  # seconds a build waits for the staff comments of its post
LIMIT = 50  # staff comments asked for; a post rarely has more than three
STAFF_PERMISSIONS = {"admin", "curator"}  # shown as the staff member's role; does not admit anyone
CACHE_S = 600  # a post's answer (or failed call) is reused for this many seconds
MAX_CALLS = 2  # calls in flight at once, across all builds of the process

_cache: dict[int, tuple[float, list | None]] = {}  # post id -> (time.monotonic() of the call, rows or None)
_cache_lock = threading.Lock()
_post_locks: dict[int, threading.Lock] = {}  # one call per post at a time; the others wait for its answer
_net = threading.BoundedSemaphore(MAX_CALLS)

# A staff member can also comment as a forecaster. Those comments stay out.
_FORECAST_TALK = re.compile(
    r"(?i)\bcommunity (?:prediction|forecast|median|estimate|aggregate)\b"
    r"|\bcommunity\b[^.\n]{0,40}?\b(?:assign\w*|probabilit\w*|forecast\w*|predict\w*|odds|median)\b"
    # "It got all the way down to 0.1%": the question's own forecast, without the word.
    # A resolution note's "It was 3.4%." is data and stays in.
    r"|\bit\s+(?:has\s+|had\s+)?(?:got|gotten|went|gone|fell|fallen|dropped|rose|risen|climbed|jumped|moved)\b"
    r"[^.\n]{0,20}?(?:\b(?:down|up) to|\bas (?:low|high) as) \d+(?:\.\d+)?\s?%"
    r"|\b(?:this|the) question\s+(?:has\s+|had\s+)?(?:got|gotten|went|gone|fell|fallen|dropped|rose|risen|climbed|jumped"
    r"|moved|sat|stood|was|is|been)\b[^.\n]{0,30}?\d+(?:\.\d+)?\s?%"
    r"|\bmetaculus (?:prediction|pros?)\b|\bpro forecasters?\b"
    r"|\bmy (?:own )?(?:forecast|prediction|estimate|median|credence)\b"
    r"|\bI(?:'m| am| was| would be)? (?:at|around|near|below|above) \d"
    r"|\b(?:forecast\w*|predict\w*|probabilit\w*|odds|chances?|credence)\b[^.\n]{0,60}?\d+(?:\.\d+)?\s?%"
    r"|\d+(?:\.\d+)?\s?%[^.\n]{0,60}?\b(?:forecast\w*|predict\w*|probabilit\w*|odds|chances?|credence|too (?:low|high))\b"
)
_CP = re.compile(r"\bCP\b")  # the site's shorthand for the community prediction
# Forecast numbers and the crowd's view without the words above ("the crowd is
# at 40%", "the aggregate sits at 0.4", "most forecasters think"). Measured on
# 1,000 real staff notes on 2026-11-23: it leaves out one more, a staff note
# that cites the community's value. "Median: 35%." still passes: a resolution
# note can say the same ("The median was 28.5%.").
_CROWD = re.compile(
    r"(?i)\bcommunity\b[^.\n]{0,40}?\d"
    r"|\b(?:crowd|forecasters|predictors|users|recency[- ]weighted|cp)\b[^.\n]{0,40}?\d"
    r"|\baggregate (?:forecast|prediction|median|mean|sits|is at|was at)\b"
    r"|\bbots? (?:median|average|mean|forecast\w*|are at|is at|was at)\b"
    r"|\bmetaculus\b[^.\n]{0,20}?\b(?:puts?|gives?|has|is at|sits)\b[^.\n]{0,20}?\d"
    r"|\b(?:forecast\w*|predict\w*|probabilit\w*|odds|chances?|credence)\b[^.\n]{0,60}?\d+(?:\.\d+)?\s?percent\b"
    r"|\b(?:most|many|few) (?:forecasters|predictors|users|bots)\b"
)


def talks_forecast(text: str) -> bool:
    """True when a cleaned text talks about forecast numbers, the community prediction or the crowd's view."""
    return bool(_FORECAST_TALK.search(text) or _CP.search(text) or _CROWD.search(text))


def _when(value) -> datetime | None:
    """An API timestamp as an aware datetime, or None."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def clean(text) -> str:
    """Comment Markdown as one line of plain text: no images, tags or escapes."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(text).replace("\xa0", " ").replace("\x00", "")
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)  # Markdown images, alt text included
    text = re.sub(r"</?[A-Za-z][^<>]*>", " ", text)  # HTML tags such as <img ...>
    text = re.sub(r"\\([\\$&\[\]()#*_<>.!~+-])", r"\1", text)  # Markdown escapes
    text = re.sub(r"(?m)^\s*>\s?", "", text)  # quote markers
    text = re.sub(r"\s+", " ", text).strip()
    return "" if text.lower() in ("[deleted]", "deleted") else text


def is_staff_comment(comment) -> bool:
    """True for a public, live comment written by staff that carries no forecast."""
    if not isinstance(comment, dict):
        return False
    author = comment.get("author")
    if not isinstance(author, dict) or author.get("is_bot"):
        return False
    if author.get("is_staff") is not True:
        return False
    if comment.get("is_private") or comment.get("is_soft_deleted"):
        return False
    # The API sends the forecast as an object, or null; its schema says boolean.
    if comment.get("included_forecast") not in (None, False):
        return False
    # The filter reads the text the block would show ("40&#37;" is "40%" there).
    text = clean(comment.get("text"))
    if not text or talks_forecast(text):
        return False
    return True


def _field_notes(api_json) -> list[tuple[str, str]]:
    """(key, text) for each clarification field in the post JSON or its question."""
    found: list[tuple[str, str]] = []
    if not isinstance(api_json, dict):
        return found
    question = api_json.get("question")
    for source in (api_json, question if isinstance(question, dict) else {}):
        for key, value in source.items():
            if "clarif" not in str(key).lower():
                continue
            items = value if isinstance(value, list) else [value]
            for item in items:
                if isinstance(item, dict):
                    item = item.get("text") or item.get("content") or item.get("description")
                text = clean(item)
                if text and not talks_forecast(text) and (str(key), text) not in found:
                    found.append((str(key), text))
    return found


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def _cut(text: str, room: int) -> str:
    """The text within `room` characters, cut at a word and marked."""
    if len(text) <= room:
        return text
    mark = " [...]"
    head = text[: max(0, room - len(mark))]
    space = head.rfind(" ")
    if space > len(head) * 0.6:
        head = head[:space]
    return head.rstrip() + mark


def render(
    comments,
    api_json=None,
    research_text: str | None = None,
    *,
    post_id: int | None = None,
    before: datetime | None = None,
    now: datetime | None = None,
) -> str | None:
    """
    The block for a list of comment objects as the API returns them, or None.
    `before` leaves out comments created at or after that time (replays);
    `post_id` leaves out comments of any other post.
    """
    known = _squash(research_text or "")
    notes: list[tuple[datetime | None, str, str]] = []  # (created, label, text)
    for key, text in _field_notes(api_json):
        if _squash(text)[:120] not in known or not known:
            notes.append((None, f"question field {key}", text))
    seen: set[str] = set()
    for comment in comments if isinstance(comments, list) else []:
        if not is_staff_comment(comment):
            continue
        if post_id is not None and comment.get("on_post") not in (None, post_id):
            continue
        created = _when(comment.get("created_at"))
        if before is not None and (created is None or created >= _when(before)):
            continue
        text = clean(comment.get("text"))
        fingerprint = _squash(text)[:120]
        if not text or fingerprint in seen or (known and fingerprint in known):
            continue
        seen.add(fingerprint)
        author = comment["author"]
        role = comment.get("author_staff_permission") if comment.get("author_staff_permission") in STAFF_PERMISSIONS else "staff"
        name = clean(str(author.get("username") or "")) or "Metaculus"
        label = f"{created:%Y-%m-%d %H:%M} UTC, {name} ({role})" if created else f"undated, {name} ({role})"
        edited = _when(comment.get("text_edited_at"))
        if edited and created and edited.date() != created.date():
            label += f", edited {edited:%Y-%m-%d}"
        if comment.get("parent_id") is not None:
            label += ", in reply to another comment"
        notes.append((created, label, text))
    if not notes:
        return None
    # Question fields first (undated), then comments from the newest to the oldest.
    floor = datetime.min.replace(tzinfo=timezone.utc)
    notes.sort(key=lambda n: (n[0] is None, n[0].astimezone(timezone.utc) if n[0] else floor), reverse=True)

    stamp = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        HEADER,
        f"Read by code from the question page at {stamp}, newest first. A staff note can clarify or amend "
        "the resolution criteria. Comments and forecasts by other forecasters are not included.",
    ]
    used = sum(len(line) + 1 for line in lines)
    for i, (_, label, text) in enumerate(notes):
        left = len(notes) - i
        tail = f"({left} older note{'s' if left > 1 else ''} not shown.)"
        prefix = f"- {label}: "
        budget = MAX_CHARS - used - len(prefix)
        # Room for this note: an even share of what is left, at least MIN_NOTE,
        # keeping space for the "not shown" line when notes remain after it.
        share = max(MIN_NOTE, budget // left) if left > 1 else budget
        room = min(share, budget - (len(tail) + 1 if left > 1 else 0))
        if room < min(MIN_NOTE, len(text)):
            if used + len(tail) <= MAX_CHARS:
                lines.append(tail)
            break
        line = prefix + _cut(text, room)
        lines.append(line)
        used += len(line) + 1
    if len(lines) == 2:
        return None
    return "\n".join(lines)[:MAX_CHARS]


def _request(post_id: int):
    """The one API call: the decoded answer of the staff comments list of a post."""
    headers = {"User-Agent": "metaculus-forecasting-bot"}
    token = (os.getenv("METACULUS_TOKEN") or "").strip()
    if token:
        headers["Authorization"] = f"Token {token}"
    base = (os.getenv("METACULUS_API_BASE_URL") or "").strip().rstrip("/") or API
    response = requests.get(
        f"{base}/comments/",
        params={"post": post_id, "author_is_staff": "true", "limit": LIMIT, "sort": "-created_at"},
        headers=headers,
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def _fetch_once(post_id: int, wait: float) -> tuple[list | None, bool]:
    """
    (rows, asked): one call for the staff comments of a post, given at most
    `wait` seconds. rows is None when the call failed or did not answer in
    time; asked is False when no call went out (MAX_CALLS calls stayed in
    flight the whole time), so the failure is not the post's.
    """
    if wait <= 0:
        return None, False
    box: dict = {}
    gone = threading.Event()  # the caller stopped waiting

    def work() -> None:
        if not _net.acquire(timeout=wait):
            box["error"] = f"{MAX_CALLS} calls already in flight for {wait:.1f} s"
            return
        try:
            if gone.is_set():
                return
            box["asked"] = True
            box["payload"] = _request(post_id)
        except Exception as exc:  # noqa: BLE001 - network, HTTP status, bad JSON
            box["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
        finally:
            _net.release()

    # requests' timeout bounds each wait on the socket, not the whole call;
    # the thread join is what holds the total to `wait`.
    thread = threading.Thread(target=work, name="admin-notes", daemon=True)
    thread.start()
    thread.join(wait)
    gone.set()
    if "payload" not in box:
        logger.warning(f"Staff notes not read for post {post_id}: {box.get('error') or f'no answer within {wait:.1f} s'}")
        return None, bool(box.get("asked"))
    payload = box["payload"]
    rows = payload.get("results") if isinstance(payload, dict) else payload
    return (rows if isinstance(rows, list) else []), True


def fetch(post_id: int) -> list | None:
    """
    The staff comments of a post as the API returns them, or None when the
    call fails or has not answered within TIMEOUT seconds. Never raises.
    The answer, or the failure, of a call is reused for CACHE_S seconds;
    builds that ask for the same post at once wait for the one call.
    """
    deadline = time.monotonic() + TIMEOUT
    with _cache_lock:
        lock = _post_locks.setdefault(post_id, threading.Lock())
    if not lock.acquire(timeout=TIMEOUT):
        logger.warning(f"Staff notes not read for post {post_id}: another build's call did not end within {TIMEOUT} s")
        return None
    try:
        with _cache_lock:
            hit = _cache.get(post_id)
        if hit is not None and time.monotonic() - hit[0] < CACHE_S:
            return None if hit[1] is None else list(hit[1])
        rows, asked = _fetch_once(post_id, deadline - time.monotonic())
        if asked:
            with _cache_lock:
                now = time.monotonic()
                for key in [k for k, (at, _) in _cache.items() if now - at >= CACHE_S]:
                    del _cache[key]
                _cache[post_id] = (now, rows)
        return None if rows is None else list(rows)
    finally:
        lock.release()


def build(question, research_text: str | None = None, *, before: datetime | None = None) -> str | None:
    """
    The "Notes from Metaculus staff on this question" block for a
    MetaculusQuestion, or None when staff left nothing (or the call failed).
    A note whose text already appears in `research_text` is not repeated.
    """
    try:
        post_id = getattr(question, "id_of_post", None)
        api_json = getattr(question, "api_json", None)
        comments = fetch(post_id) if isinstance(post_id, int) and not isinstance(post_id, bool) else None
        block = render(comments or [], api_json, research_text, post_id=post_id, before=before)
        if block:
            logger.info(f"Staff notes for post {post_id}: {block.count(chr(10) + '- ')} note(s), {len(block)} chars")
        return block
    except Exception as exc:  # noqa: BLE001 - a missing note never stops a forecast
        logger.warning(f"Staff notes failed: {type(exc).__name__}: {str(exc)[:160]}")
        return None
