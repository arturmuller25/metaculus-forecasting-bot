"""
Evidence packet: one extra call that reorganizes the research before the
forecasters read it.

Several of the most consistent bots in MiniBench show their forecasters a
structured digest instead of raw research: AXE-theomachado (6th in round 1) a
packet of about 12 dated items read by five forecasters; mmBot (top 10 in 10
of 12 rounds) a table of what each piece of evidence establishes and what it
does not, after a section that pins down exactly what resolves the question.
The packet below combines the two. It never adds facts: it only sorts, dates
and grades what the research already says, and names what is missing.
"""

from __future__ import annotations

PROMPT = """You are preparing an evidence packet for forecasters. Below are a forecasting question and research notes from several sources. Do not forecast and do not give a probability.

Question:
{question}

Research notes:
{research}

Write the packet in exactly four sections:

1. Resolution mechanics (2 to 4 lines): what exactly makes this question resolve (or sets its value), from which source, at which date and time. If the notes suggest that the question's background is outdated (a rescheduled event, a newer data release, a value that already changed), say so here.

2. Evidence, newest first, at most 12 items, one line each:
YYYY-MM-DD | source (domain or document) | the fact, with its numbers | what it establishes for the resolution | what it does not establish
Use the date the fact was published or observed, or "undated". Prefer primary sources (the resolution source, official documents, filings, data tables) over commentary, merge duplicates, and keep only items that bear on the resolution. Never add a fact that is not in the notes.

3. Conflicts: facts in the notes that contradict each other, and which one is newer or closer to the primary source. Write "None" if there are none.

4. Missing: the one or two facts that would most change the forecast and that the notes do not contain."""


def question_text(question) -> str:
    parts = [question.question_text]
    for label, attr in (("Resolution criteria", "resolution_criteria"), ("Fine print", "fine_print"),
                        ("Background", "background_info")):
        value = (getattr(question, attr, None) or "").strip()
        if value:
            parts.append(f"{label}: {value}")
    return "\n\n".join(parts)


async def build(llm, question, research: str) -> str:
    """The packet text, to append after the research. Raises if the call fails."""
    text = await llm.invoke(PROMPT.format(question=question_text(question), research=research))
    return "## Evidence packet (organized from the research above; no new facts)\n\n" + text.strip()
