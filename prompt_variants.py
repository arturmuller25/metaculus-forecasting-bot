"""
Prompt variants for the shadow forecasters on the Claude Code CLI.

SHADOW_MODELS takes "<claudecode spec>+<variant>", for example
"claudecode/claude-sonnet-5-5@high+escape": the forecaster of that spec,
asked with the forecast prompt changed by the variant named after the plus
sign, and recorded under the whole name. The same spec without a variant is
its control.

A variant is a function (prompt, question, research) -> str | None: the
changed prompt, or None when it has nothing to change on this question. The
shadow is then skipped on that question, which is not a failure
(bot.run_claude_code_shadows). `research` is the research text the prompt
was built from. A variant that raises counts as a failed answer.

Variants:

- "escape": numeric and discrete questions only. Before the percentiles the
  forecaster states its central range, names the surprise scenarios that
  would take the outcome outside it, and gives the probability mass below
  and above that range, which its percentiles must then match.
- "market": adds the block that market_block.build(question, research)
  returns, at the end of the research.
- "admin": adds the block that admin_notes.build(question, research) returns,
  in the same place.

market_block.py and admin_notes.py are optional modules with the signature
build(question, research_text) -> str | None. Without the module, when it
fails to import, or when it returns no text, the variant changes nothing.
"""

from __future__ import annotations

import importlib
import logging
import re
from typing import Any, Callable

logger = logging.getLogger(__name__)

# The line of the numeric prompt (bot._run_forecast_on_numeric) that comes
# right before the percentile block.
_ANSWER_LINE = "The last thing you write is your final answer as:"
# A prompt that ends with a percentile line: the numeric answer template.
_ENDS_WITH_PERCENTILE = re.compile(r"Percentile \d+:[^\n]*[\s\"]*$")

# The text E1 of the replay test of prompt variants, frozen on 2026-11-16
# before its calls ran.
ESCAPE = """\
Before the percentiles you also write:
(g) The central range (the low and the high value between which (b), (c) and (d) put the outcome when nothing surprising happens) and, by name, the surprise scenarios that would take the outcome outside it (for example a revision or restatement, a one-off event, a delayed release, a rule of the resolution source, a break in the trend), each with the side it pushes to.
(h) The probability mass you give those surprises: one number for the outcome landing below the central range and one for the outcome landing above it. If a bound of the question is open ("likely not lower than" or "likely not higher than"), also the mass beyond that bound. A side with less than 2% needs a stated reason.
Your percentiles must then agree with (g) and (h): the mass they leave below and above the central range is the mass you stated, and the percentiles in between describe the central range itself, no wider and no narrower than the evidence supports."""

# Optional modules already looked for: the module, or None when it is absent.
_MODULES: dict[str, Any] = {}


def escape(prompt: str, question: Any, research: str) -> str | None:
    """
    The numeric prompt with the escape items right before its answer block;
    None for the other question types. The type is read from the answer
    block at the end of the prompt, not from the whole text: the research
    or the question's text can quote percentiles.
    """
    at = prompt.rfind(_ANSWER_LINE)
    if at < 0:
        # A multiple choice prompt has another answer line. A numeric one
        # without this line means the prompt's wording changed.
        if _ENDS_WITH_PERCENTILE.search(prompt):
            raise ValueError("no answer line in the numeric prompt")
        return None
    if "Percentile 10:" not in prompt[at:]:
        return None  # a binary prompt
    return prompt[:at] + ESCAPE + "\n\n" + prompt[at:]


def _after_research(prompt: str, block: str) -> str:
    """The prompt with a block at the end of the research, right before the prompt's own "Today is" line (as gap_fill.with_findings)."""
    at = prompt.rfind("\nToday is ")
    if at < 0:
        raise ValueError("no 'Today is' line in the forecast prompt")
    return prompt[:at] + "\n\n" + block + "\n" + prompt[at:]


def _module_block(module_name: str) -> Callable[[str, Any, str], str | None]:
    """A variant that adds what <module_name>.build(question, research) returns to the research."""

    def variant(prompt: str, question: Any, research: str) -> str | None:
        if module_name not in _MODULES:
            try:
                _MODULES[module_name] = importlib.import_module(module_name)
            except Exception as exc:  # noqa: BLE001 - absent or broken, the variant changes nothing
                _MODULES[module_name] = None
                logger.warning(f"Prompt variant: {module_name}.py is not available ({type(exc).__name__}: {exc}); "
                               "its variant changes nothing")
        module = _MODULES[module_name]
        if module is None:
            return None
        block = (module.build(question, research) or "").strip()
        return _after_research(prompt, block) if block else None

    return variant


VARIANTS: dict[str, Callable[[str, Any, str], str | None]] = {
    "escape": escape,
    "market": _module_block("market_block"),
    "admin": _module_block("admin_notes"),
}
