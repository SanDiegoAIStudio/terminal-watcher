#!/usr/bin/env python3
"""
Terminal Watcher AI - Contextual Development Assistant

This script watches your terminal activity and provides:
1. On-demand explanations ("what's happening?")
2. Proactive research (finds docs, GitHub repos, solutions)
3. Error detection and automatic help
4. Context-aware suggestions

Usage:
  # Start watching in background
  tw watch &

  # Get explanation of recent activity
  tw explain

  # Get explanation with research (searches docs/GitHub)
  tw research

  # Watch with auto-research on errors
  tw watch --auto

Requirements:
  uv tool install .   (or: uv run tw ...)
"""

import os
import sys
import re
import json
import time
import unicodedata
import argparse
import subprocess
import urllib.parse
from datetime import datetime, date
from pathlib import Path
from collections import deque
from typing import Callable, NoReturn, Tuple

try:
    import anthropic
except ImportError:
    print("Error: anthropic package not installed. Install tw with: uv tool install .")
    sys.exit(1)

try:
    import requests
except ImportError:
    print("Error: requests package not installed. Install tw with: uv tool install .")
    sys.exit(1)


# Configuration
TERMINAL_LOGS_DIR = Path.home() / "Documents" / "terminal-logs" / "raw"
SUGGESTIONS_FILE = Path.home() / ".claude" / "data" / "watcher_suggestions.md"

# How much recent log to analyze (characters)
RECENT_LOG_SIZE = 15000

DEFAULT_MODEL = "claude-sonnet-5-5"
MAX_TOKENS = 8000
DEFAULT_EFFORT = "low"
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")

# One-shot commands wrap all stdout in one pair. Watch mode wraps each message.
# A log line counts as a marker when it contains either string.
TW_BEGIN = "[tw-output-begin]"
TW_END = "[tw-output-end]"
# A begin marker that never gets an end marker hides at most this many lines.
UNMATCHED_BEGIN_LINES = 200

# API calls --auto may attempt in a rolling hour. TW_MAX_AUTO_PER_HOUR overrides this.
MAX_AUTO_PER_HOUR = 6
_AUTO_WINDOW_SECONDS = 3600

# Error patterns to detect
ERROR_PATTERNS = [
    r'error[:\s]',
    r'Error[:\s]',
    r'ERROR[:\s]',
    r'failed',
    r'FAILED',
    r'exception',
    r'Exception',
    r'traceback',
    r'Traceback',
    r'cannot find',
    r'not found',
    r'permission denied',
    r'command not found',
    r'Module not found',
    r'ImportError',
    r'TypeError',
    r'SyntaxError',
    r'ReferenceError',
]

# npm package names, including an optional @scope/. Reject anything else before a request.
_NPM_PACKAGE_NAME = re.compile(
    r"(?:@[A-Za-z0-9][A-Za-z0-9._~-]*/)?[A-Za-z0-9][A-Za-z0-9._~-]*"
)


def get_today_log_file() -> Path | None:
    """Get today's terminal log file."""
    today = date.today().strftime("%Y-%m-%d")
    log_file = TERMINAL_LOGS_DIR / f"{today}.log"
    return log_file if log_file.exists() else None


def read_recent_log(num_chars: int = RECENT_LOG_SIZE) -> str:
    """Read the most recent portion of today's log.

    Opened in binary mode so the seek offset is a byte count.
    """
    log_file = get_today_log_file()
    if not log_file:
        return ""

    try:
        with open(log_file, "rb") as f:
            # Seek to near end. Byte offsets are defined for a binary file.
            f.seek(0, 2)  # End of file
            file_size = f.tell()

            if file_size <= num_chars:
                f.seek(0)
            else:
                f.seek(file_size - num_chars)
                f.readline()  # Skip partial line

            raw = f.read()
        content = raw.decode("utf-8", errors="replace")
    except OSError as exc:
        raise TWError(f"could not read the terminal log: {exc}") from None

    # Clean ANSI codes, then drop any report this process printed into the log.
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    content = ansi_escape.sub('', content)
    content = content.replace('\r', '')
    return strip_own_output(content)


def detect_errors(log_content: str) -> list[str]:
    """Return the last 20 distinct matching lines, in the order they appear."""
    errors = []
    seen = set()
    for line in log_content.splitlines():
        if line in seen:
            continue
        for pattern in ERROR_PATTERNS:
            if re.search(pattern, line, re.IGNORECASE):
                seen.add(line)
                errors.append(line)
                break
    if len(errors) > 20:
        return errors[-20:]
    return errors


def _tail_for_errors(redacted: str) -> str:
    """Last 2,000 characters, dropping a first line the cut split in half."""
    limit = 2000
    if len(redacted) <= limit:
        return redacted
    tail = redacted[-limit:]
    if redacted[-limit - 1] != "\n":
        newline = tail.find("\n")
        if newline == -1:
            return ""
        return tail[newline + 1:]
    return tail


def search_github(query: str) -> list[dict]:
    """Search GitHub for relevant repositories or issues."""
    try:
        # Use GitHub search API (no auth needed for basic searches)
        url = "https://api.github.com/search/repositories"
        params = {
            "q": query,
            "sort": "stars",
            "per_page": 5
        }
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if not isinstance(data, dict):
                return []
            items = data.get("items", [])
            if not isinstance(items, list):
                return []
            results = []
            for repo in items[:5]:
                if not isinstance(repo, dict):
                    continue
                name = repo.get("full_name")
                html_url = repo.get("html_url")
                if name is None or html_url is None:
                    continue
                description = repo.get("description")
                if description:
                    description = description[:200]
                else:
                    description = ""
                stars = repo.get("stargazers_count", 0)
                if stars is None:
                    stars = 0
                if isinstance(name, str):
                    name = _strip_controls(name)
                if isinstance(html_url, str):
                    html_url = _strip_controls(html_url)
                if isinstance(description, str):
                    description = _strip_controls(description)
                results.append({
                    "name": name,
                    "description": description,
                    "url": html_url,
                    "stars": stars,
                })
            return results
    except (requests.RequestException, ValueError) as e:
        _out(f"GitHub search error: {e}")
    return []


def search_npm(package_name: str) -> dict | None:
    """Search npm for package info."""
    if len(package_name) > 214 or _NPM_PACKAGE_NAME.fullmatch(package_name) is None:
        return None
    try:
        quoted = urllib.parse.quote(package_name, safe="@")
        url = f"https://registry.npmjs.org/{quoted}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if not isinstance(data, dict):
                return None
            dist_tags = data.get("dist-tags")
            if not isinstance(dist_tags, dict):
                return None
            latest = dist_tags.get("latest", "")
            description = data.get("description") or ""
            if not isinstance(description, str):
                description = str(description)
            description = _strip_controls(description)
            name = data.get("name")
            if isinstance(name, str):
                name = _strip_controls(name)
            if isinstance(latest, str):
                latest = _strip_controls(latest)
            else:
                latest = ""
            homepage = data.get("homepage", "")
            if isinstance(homepage, str):
                homepage = _strip_controls(homepage)
            return {
                "name": name,
                "description": description[:200],
                "version": latest,
                "homepage": homepage,
            }
    except (requests.RequestException, ValueError):
        pass
    return None


def model_name() -> str:
    return os.environ.get("TW_MODEL") or DEFAULT_MODEL


class TWError(Exception):
    """A terminal-watcher failure the caller can print or keep running through."""


def effort_setting(model: str) -> str | None:
    """Effort to send for this model, or None when the request should omit it."""
    raw = os.environ.get("TW_EFFORT")
    if raw is None:
        if model == DEFAULT_MODEL:
            return DEFAULT_EFFORT
        return None
    if raw in EFFORT_LEVELS:
        return raw
    if raw == "":
        return None
    raise TWError("TW_EFFORT must be one of low, medium, high, xhigh, max.")


def _strip_controls(text: str) -> str:
    """Drop ESC, other C0 controls except tab and newline, DEL, C1, format characters, U+2028, and U+2029."""
    kept = []
    for char in text:
        code = ord(char)
        if code == 9 or code == 10:
            kept.append(char)
            continue
        if code < 32 or code == 127 or 128 <= code <= 159:
            continue
        if code in (0x2028, 0x2029) or unicodedata.category(char) == "Cf":
            continue
        kept.append(char)
    return "".join(kept)


def _status_message(message: str) -> str:
    """Drop control characters, then keep the first 200 characters."""
    kept = []
    for char in message:
        code = ord(char)
        if code < 32 or code == 127 or 128 <= code <= 159:
            continue
        kept.append(char)
    return "".join(kept)[:200]


def fail(message: str) -> NoReturn:
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(1)


def emit(text: str) -> None:
    """Print one stdout block between marker lines and flush."""
    print(TW_BEGIN)
    print(text)
    print(TW_END)
    sys.stdout.flush()


def _out(text: str) -> None:
    """Print one-shot stdout. The caller owns the single marker pair."""
    print(text)
    sys.stdout.flush()


def strip_own_output(text: str) -> str:
    """Drop every marked tw block so detection never sees tw's own report.

    A begin marker with no end marker hides at most the next 200 lines.
    """
    lines = text.splitlines(keepends=True)
    kept = []
    index = 0
    total = len(lines)
    seen_begin = False
    while index < total:
        marker = lines[index].rstrip("\r\n")
        if TW_BEGIN in marker:
            seen_begin = True
            end_at = None
            look = index + 1
            while look < total:
                if TW_END in lines[look].rstrip("\r\n"):
                    end_at = look
                    break
                look += 1
            if end_at is None:
                index = min(index + 1 + UNMATCHED_BEGIN_LINES, total)
            else:
                index = end_at + 1
            continue
        if TW_END in marker and not seen_begin:
            kept = []
            index += 1
            continue
        kept.append(lines[index])
        index += 1
    return "".join(kept)


def _replace_token(match: re.Match) -> str:
    return "[redacted]"


def _replace_bearer(match: re.Match) -> str:
    return "Bearer" + match.group(1) + "[redacted]"


def _replace_authorization(match: re.Match) -> str:
    return match.group(1) + match.group(2) + "[redacted]"


def _replace_flag(match: re.Match) -> str:
    return match.group(1) + match.group(2) + match.group(3) + "[redacted]"


def _replace_labeled_bearer(match: re.Match) -> str:
    return match.group(1) + match.group(2) + "[redacted]"


def _replace_url_password(match: re.Match) -> str:
    return match.group(1) + "[redacted]@"


def _replace_slack(match: re.Match) -> str:
    return match.group(1) + "[redacted]"


def _replace_quoted(match: re.Match) -> str:
    return (
        match.group(1)
        + match.group(2)
        + match.group(1)
        + match.group(3)
        + match.group(4)
        + "[redacted]"
        + match.group(4)
    )


def _replace_unquoted(match: re.Match) -> str:
    return match.group(1) + match.group(2) + "[redacted]"


# Specific shapes first. Labeled values run after, so a value already replaced
# with [redacted] is not counted again.
_LABEL_SUFFIX = r"password|passwd|pwd|key|token|secret|passphrase"

_REDACTORS: Tuple[Tuple[re.Pattern, Callable[[re.Match], str]], ...] = (
    (
        re.compile(
            r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----"
            r".*?"
            r"-----END [A-Z0-9 ]*PRIVATE KEY-----",
            re.DOTALL,
        ),
        _replace_token,
    ),
    (re.compile(r"([A-Za-z][A-Za-z0-9+.-]*://[^/\s:@]+:)([^@\s/]+)@"), _replace_url_password),
    (
        re.compile(
            r"(?i)(?<![A-Za-z0-9_])"
            r"(--)"
            r"([A-Za-z0-9_-]*?(?:" + _LABEL_SUFFIX + r"))"
            r"([ \t]*=[ \t]*|[ \t]+)"
            r"(?!\[redacted\])"
            r"(\S{4,})"
        ),
        _replace_flag,
    ),
    (re.compile(r"(hooks\.slack\.com/services/)(\S+)"), _replace_slack),
    (re.compile(r"eyJ[A-Za-z0-9_-]*\.eyJ[A-Za-z0-9_-]*\.[A-Za-z0-9_-]+"), _replace_token),
    (re.compile(r"sk-[A-Za-z0-9_-]{16,}"), _replace_token),
    (re.compile(r"sk_live_[A-Za-z0-9]{16,}"), _replace_token),
    (re.compile(r"sk_test_[A-Za-z0-9]{16,}"), _replace_token),
    (re.compile(r"rk_live_[A-Za-z0-9]{16,}"), _replace_token),
    (re.compile(r"(?:ghp_|gho_|ghs_|ghr_|ghu_)[A-Za-z0-9]{20,}"), _replace_token),
    (re.compile(r"github_pat_[A-Za-z0-9_]{20,}"), _replace_token),
    (re.compile(r"glpat-[A-Za-z0-9]{20,}"), _replace_token),
    (re.compile(r"AIza[A-Za-z0-9_-]{35}(?![A-Za-z0-9_-])"), _replace_token),
    (re.compile(r"npm_[A-Za-z0-9]{36}(?![A-Za-z0-9])"), _replace_token),
    (re.compile(r"AKIA[0-9A-Z]{16}(?![0-9A-Z])"), _replace_token),
    (re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"), _replace_token),
    (
        re.compile(
            r"(?<![A-Za-z0-9_])"
            r"((?i:[A-Za-z0-9_-]*?(?:" + _LABEL_SUFFIX + r"))"
            r"[ \t]*[=:][ \t]*"
            r"Bearer)"
            r"(\s+)"
            r"([-A-Za-z0-9._~+/=]+)"
        ),
        _replace_labeled_bearer,
    ),
    (re.compile(r"Bearer(\s+)([-A-Za-z0-9._~+/=]{16,})"), _replace_bearer),
    (
        re.compile(
            r"(?i)(?<![A-Za-z0-9_])"
            r"(Authorization)"
            r"([ \t]*[=:][ \t]*)"
            r"[^\r\n]*"
        ),
        _replace_authorization,
    ),
)

def _quoted_label(quote: str) -> re.Pattern:
    """Label, separator, and a quoted value. The value may contain spaces."""
    return re.compile(
        r"(?i)(?<![A-Za-z0-9_])"
        r"(%s?)([A-Za-z0-9_-]*?(?:%s))\1"
        r"([ \t]*[=:][ \t]*)"
        r"(%s)(?!\[redacted\]%s)([^%s\r\n]+)%s"
        % (quote, _LABEL_SUFFIX, quote, quote, quote, quote)
    )


_LABEL_REDACTORS: Tuple[Tuple[re.Pattern, Callable[[re.Match], str]], ...] = (
    (_quoted_label('"'), _replace_quoted),
    (_quoted_label("'"), _replace_quoted),
    (
        re.compile(
            r"(?i)(?<![A-Za-z0-9_])"
            r"([A-Za-z0-9_-]*?(?:" + _LABEL_SUFFIX + r"))"
            r"([ \t]*[=:][ \t]*)"
            r"(?!\[redacted\])"
            r"(?!Bearer\s+)"
            r"(?![\"'])"
            r"(\S{4,})"
        ),
        _replace_unquoted,
    ),
)


def _apply_redactors(
    text: str,
    redactors: Tuple[Tuple[re.Pattern, Callable[[re.Match], str]], ...],
) -> Tuple[str, int]:
    count = 0
    pieces = []
    pos = 0
    length = len(text)
    while pos < length:
        best = None
        best_replace = None
        for regex, replace in redactors:
            match = regex.search(text, pos)
            if match is None:
                continue
            if best is None or match.start() < best.start() or (
                match.start() == best.start() and match.end() > best.end()
            ):
                best = match
                best_replace = replace
        if best is None or best_replace is None:
            pieces.append(text[pos:])
            break
        pieces.append(text[pos:best.start()])
        pieces.append(best_replace(best))
        count += 1
        next_pos = best.end()
        if next_pos <= pos:
            next_pos = pos + 1
        pos = next_pos
    return "".join(pieces), count


def redact(text: str) -> Tuple[str, int]:
    """Replace token-shaped strings with [redacted] and return how many changed."""
    text, count = _apply_redactors(text, _REDACTORS)
    text, extra = _apply_redactors(text, _LABEL_REDACTORS)
    return text, count + extra


def _auto_limit() -> int:
    """Positive TW_MAX_AUTO_PER_HOUR, or MAX_AUTO_PER_HOUR when unset or invalid."""
    raw = os.environ.get("TW_MAX_AUTO_PER_HOUR")
    if raw is None:
        return MAX_AUTO_PER_HOUR
    try:
        limit = int(raw)
    except ValueError:
        return MAX_AUTO_PER_HOUR
    if limit < 1:
        return MAX_AUTO_PER_HOUR
    return limit


def _error_alert(new_errors: list) -> str:
    lines = [
        f"\n⚠️  Detected {len(new_errors)} new error(s)!",
        "   Run `tw explain` or `tw research` for help.",
    ]
    for err in new_errors[:3]:
        lines.append(f"   • {err[:80]}...")
    return "\n".join(lines)


def build_prompt(log_content: str, mode: str = "explain", errors: list = None) -> str:
    """Return the prompt analyze_with_ai sends for this mode."""
    if mode == "explain":
        return f"""You are a friendly development assistant watching someone's terminal session.
Analyze this recent terminal activity and explain what's happening in simple, layman's terms.
Be concise but helpful. If you see errors, explain what they mean and suggest fixes.

Recent Terminal Activity:
```
{log_content}
```

Respond with a JSON object:
{{
    "summary": "One sentence summary of what they're doing",
    "explanation": "2-3 paragraph explanation in simple terms",
    "current_task": "What they seem to be working on",
    "potential_issues": ["Any problems you notice"],
    "suggestions": ["Helpful suggestions"],
    "technologies_detected": ["tech", "stack", "items"]
}}"""

    elif mode == "research":
        return f"""You are a research assistant watching someone's terminal session.
Analyze what they're working on and suggest helpful resources, documentation, or tools.

Recent Terminal Activity:
```
{log_content}
```

{f"Errors detected: {errors}" if errors else ""}

Respond with a JSON object:
{{
    "summary": "What they're trying to accomplish",
    "technologies": ["detected", "technologies"],
    "search_queries": ["3-5 specific search queries to find helpful docs/repos"],
    "likely_documentation_needed": ["specific docs that would help"],
    "github_search_terms": ["terms to search GitHub for helpful repos"],
    "npm_packages_to_check": ["relevant npm packages if applicable"],
    "stackoverflow_keywords": ["keywords for SO search"],
    "advice": "Brief advice on their current approach"
}}"""

    elif mode == "error":
        return f"""You are a debugging assistant. Someone encountered errors in their terminal.
Analyze the errors and provide clear, actionable help.

Recent Terminal Activity:
```
{log_content}
```

Errors Detected:
{json.dumps(errors, indent=2)}

Respond with a JSON object:
{{
    "error_summary": "What went wrong in simple terms",
    "root_cause": "Most likely cause",
    "fix_steps": ["Step 1", "Step 2", "..."],
    "commands_to_try": ["specific commands that might fix it"],
    "documentation_links": ["where to find more info"],
    "search_query": "Best query to search for a solution"
}}"""

    else:
        return f"Analyze this terminal output: {log_content}"


def analyze_with_ai(log_content: str, mode: str = "explain", errors: list = None) -> dict:
    """Use Claude to analyze terminal activity."""
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        raise TWError('ANTHROPIC_API_KEY is not set. Create a key at https://console.anthropic.com and run: export ANTHROPIC_API_KEY="sk-ant-..."')

    prompt = build_prompt(log_content, mode, errors)
    model = model_name()
    effort = effort_setting(model)
    create_kwargs = {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "messages": [{"role": "user", "content": prompt}],
    }
    if effort is not None:
        create_kwargs["output_config"] = {"effort": effort}
    try:
        client = anthropic.Anthropic()
        try:
            response = client.messages.create(**create_kwargs)
        except TypeError as exc:
            if "output_config" in str(exc):
                raise TWError(
                    "this anthropic package is too old for the effort setting. "
                    "Reinstall with: uv tool install . --reinstall"
                ) from None
            raise
    except anthropic.AuthenticationError:
        raise TWError("the API key was refused (401). Check ANTHROPIC_API_KEY.") from None
    except anthropic.NotFoundError:
        raise TWError(f"model {model} was not found. Set TW_MODEL to a current model id.") from None
    except anthropic.RateLimitError:
        raise TWError("the API is rate limiting this key (429). Try again in a minute.") from None
    except anthropic.APIStatusError as exc:
        if exc.status_code == 400:
            raise TWError(
                "the API refused the request (400): " + _status_message(exc.message)
            ) from None
        raise TWError(f"the API returned status {exc.status_code}.") from None
    except anthropic.APIConnectionError:
        raise TWError("could not reach the Anthropic API. Check your network.") from None
    except anthropic.APIError as exc:
        raise TWError(f"the API call failed ({type(exc).__name__}).") from None

    if response.stop_reason == "refusal":
        return {"raw_response": "The model declined to analyze this log."}

    response_text = "".join(
        block.text for block in response.content if block.type == "text"
    )
    if response.stop_reason == "max_tokens":
        return {
            "raw_response": response_text
            + "\n\n(The answer was cut off at the "
            + str(MAX_TOKENS)
            + " token limit.)"
        }
    response_text = response_text.strip()

    # Parse JSON from response
    if response_text.startswith("```"):
        lines = response_text.split("\n")
        response_text = "\n".join(lines[1:-1])

    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError:
        return {"raw_response": response_text}
    if not isinstance(parsed, dict):
        return {"raw_response": response_text}
    return parsed


def _text(value):
    """Coerce a model field used as text. None stays None."""
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    return _strip_controls(value)


def _string_list(value) -> list[str]:
    """Coerce a model field used as a list of strings.

    A string is one item. A list keeps its string items. Anything else is empty.
    """
    if isinstance(value, str):
        return [_strip_controls(value)]
    if isinstance(value, list):
        return [_strip_controls(item) for item in value if isinstance(item, str)]
    return []


def do_research(analysis: dict) -> dict:
    """Perform actual research based on AI analysis."""
    research_results = {
        "github_repos": [],
        "npm_packages": [],
        "documentation_hints": []
    }

    # Search GitHub
    for term in _string_list(analysis.get("github_search_terms"))[:3]:
        repos = search_github(term)
        research_results["github_repos"].extend(repos)

    # Check npm packages
    for pkg in _string_list(analysis.get("npm_packages_to_check"))[:5]:
        pkg_info = search_npm(pkg)
        if pkg_info:
            research_results["npm_packages"].append(pkg_info)

    return research_results


def format_output(analysis: dict, research: dict = None, mode: str = "explain") -> str:
    """Format the analysis and research into readable output."""
    output = []
    output.append("\n" + "=" * 60)
    output.append("🤖 TERMINAL WATCHER AI")
    output.append("=" * 60)

    raw = analysis.get("raw_response")
    if raw:
        output.append(f"\n{_text(raw)}")
        output.append("\n" + "=" * 60)
        return "\n".join(output)

    if mode == "explain":
        output.append(f"\n📋 SUMMARY: {_text(analysis.get('summary', 'N/A'))}")
        output.append(f"\n🎯 CURRENT TASK: {_text(analysis.get('current_task', 'N/A'))}")
        output.append(f"\n📖 EXPLANATION:\n{_text(analysis.get('explanation', 'N/A'))}")

        issues = _string_list(analysis.get("potential_issues"))
        if issues:
            output.append("\n⚠️  POTENTIAL ISSUES:")
            for issue in issues:
                output.append(f"   • {issue}")

        suggestions = _string_list(analysis.get("suggestions"))
        if suggestions:
            output.append("\n💡 SUGGESTIONS:")
            for suggestion in suggestions:
                output.append(f"   • {suggestion}")

        detected = _string_list(analysis.get("technologies_detected"))
        if detected:
            output.append(f"\n🔧 TECH DETECTED: {', '.join(detected)}")

    elif mode == "research":
        output.append(f"\n📋 SUMMARY: {_text(analysis.get('summary', 'N/A'))}")
        output.append(f"\n💡 ADVICE: {_text(analysis.get('advice', 'N/A'))}")

        technologies = _string_list(analysis.get("technologies"))
        if technologies:
            output.append(f"\n🔧 TECHNOLOGIES: {', '.join(technologies)}")

        queries = _string_list(analysis.get("search_queries"))
        if queries:
            output.append("\n🔍 RECOMMENDED SEARCHES:")
            for query in queries:
                output.append(f"   • {query}")

        if research and research.get("github_repos"):
            output.append("\n📦 RELEVANT GITHUB REPOS:")
            for repo in research["github_repos"][:5]:
                name = repo["name"]
                description = repo["description"]
                url = repo["url"]
                if isinstance(name, str):
                    name = _strip_controls(name)
                if isinstance(description, str):
                    description = _strip_controls(description)
                if isinstance(url, str):
                    url = _strip_controls(url)
                output.append(f"   ⭐ {repo['stars']:,} | {name}")
                output.append(f"      {description[:80]}...")
                output.append(f"      {url}")

        if research and research.get("npm_packages"):
            output.append("\n📦 NPM PACKAGES:")
            for pkg in research["npm_packages"]:
                name = pkg["name"]
                description = pkg["description"]
                if isinstance(name, str):
                    name = _strip_controls(name)
                if isinstance(description, str):
                    description = _strip_controls(description)
                output.append(f"   • {name}@{pkg['version']}")
                output.append(f"     {description[:60]}...")

    elif mode == "error":
        output.append(f"\n❌ ERROR: {_text(analysis.get('error_summary', 'Unknown error'))}")
        output.append(f"\n🔍 ROOT CAUSE: {_text(analysis.get('root_cause', 'Unknown'))}")

        steps = _string_list(analysis.get("fix_steps"))
        if steps:
            output.append("\n🔧 FIX STEPS:")
            for i, step in enumerate(steps, 1):
                output.append(f"   {i}. {step}")

        commands = _string_list(analysis.get("commands_to_try"))
        if commands:
            output.append("\n💻 COMMANDS TO TRY:")
            for cmd in commands:
                output.append(f"   $ {cmd}")

        search_query = _text(analysis.get("search_query"))
        if search_query:
            output.append(f"\n🔍 SEARCH: {search_query}")

    output.append("\n" + "=" * 60)
    return "\n".join(output)


def save_suggestions(content: str):
    """Save suggestions to a file for easy reference."""
    SUGGESTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(SUGGESTIONS_FILE, flags, 0o600)
    handle = None
    try:
        handle = os.fdopen(descriptor, "w")
        handle.write("# Terminal Watcher Suggestions\n")
        handle.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        handle.write(content)
    finally:
        if handle is None:
            os.close(descriptor)
        else:
            handle.close()
        os.chmod(SUGGESTIONS_FILE, 0o600)


def watch_step(state: dict, auto_research: bool, now: float) -> None:
    """Run one watch pass. Reading and analysis failures are reported and do not exit."""
    try:
        log_content = read_recent_log(5000)  # Check last 5k chars
    except TWError as exc:
        message = f"Error: {exc} (watch continues)"
        if state.get("read_error") != message:
            emit(message)
            state["read_error"] = message
        return

    state["read_error"] = None

    if log_content == state["last_content"]:
        return

    redacted, masked = redact(log_content)

    errors = detect_errors(_tail_for_errors(redacted))
    new_errors = [err for err in errors if err not in state["reported"]]
    state["reported"] = set(errors)

    if new_errors:
        if auto_research:
            limit = _auto_limit()
            recent = [stamp for stamp in state["auto_times"] if now - stamp < _AUTO_WINDOW_SECONDS]
            state["auto_times"] = recent
            if len(recent) >= limit:
                alert = _error_alert(new_errors)
                if not state.get("auto_pause_announced"):
                    alert += (
                        "\nAuto-analysis paused: "
                        f"{len(recent)} API calls in the last hour "
                        f"(limit {limit}). Run `tw error` to analyze now."
                    )
                    state["auto_pause_announced"] = True
                emit(alert)
            else:
                state["auto_pause_announced"] = False
                emit(
                    f"\n⚠️  Detected {len(new_errors)} new error(s)!\n"
                    "🔍 Analyzing and researching..."
                )
                if masked > 0:
                    emit(f"Masked {masked} token-shaped string(s) before sending.")
                state["auto_times"].append(now)
                try:
                    analysis = analyze_with_ai(redacted, mode="error", errors=new_errors)
                except TWError as exc:
                    emit(f"Error: {exc} (watch continues)")
                else:
                    output = format_output(analysis, mode="error")
                    emit(output)
                    try:
                        save_suggestions(output)
                    except OSError as exc:
                        emit(f"Could not save suggestions: {exc} (watch continues)")
        else:
            emit(_error_alert(new_errors))

    state["last_content"] = log_content


def safe_watch_step(state: dict, auto_research: bool, now: float) -> None:
    """Run one watch pass. An unexpected error is reported once per class and the loop goes on."""
    try:
        watch_step(state, auto_research, now)
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        kind = type(exc)
        if state.get("unexpected_error") is not kind:
            detail = redact(str(exc))[0]
            message = f"Error: unexpected {kind.__name__}: {detail} (watch continues)"
            emit(message)
            state["unexpected_error"] = kind
        return
    state["unexpected_error"] = None


def watch_mode(auto_research: bool = False, interval: int = 60):
    """Continuously watch terminal and detect issues."""
    lines = [
        "🤖 Terminal Watcher started. Monitoring for errors...",
        f"   Auto-research: {'ON' if auto_research else 'OFF'}",
        f"   Check interval: {interval}s",
        "   Press Ctrl+C to stop.",
    ]
    today_log = TERMINAL_LOGS_DIR / f"{date.today().strftime('%Y-%m-%d')}.log"
    if not today_log.exists():
        lines.append(f"   No log for today yet. Waiting for: {today_log}")
    emit("\n".join(lines) + "\n")

    state = {
        "last_content": "",
        "reported": set(),
        "auto_times": [],
        "auto_pause_announced": False,
        "read_error": None,
        "unexpected_error": None,
    }

    try:
        while True:
            safe_watch_step(state, auto_research, time.time())
            time.sleep(interval)
    except KeyboardInterrupt:
        emit("\n\n🤖 Terminal Watcher stopped.")


def main():
    parser = argparse.ArgumentParser(
        description="Terminal Watcher AI - Contextual Development Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  tw explain              Explain what's happening in your terminal
  tw research             Explain + search for helpful resources
  tw error                Analyze recent errors and suggest fixes
  tw watch                Watch for errors in background
  tw watch --auto         Watch + auto-research on errors
        """
    )

    parser.add_argument("command", nargs="?", default="explain",
                       choices=["explain", "research", "error", "watch", "e", "r", "w"],
                       help="Command to run")
    parser.add_argument("--auto", "--auto-research", action="store_true",
                       help="Auto-research when errors detected (watch mode)")
    parser.add_argument("--interval", type=int, default=30,
                       help="Check interval in seconds (watch mode)")
    parser.add_argument("--chars", type=int, default=RECENT_LOG_SIZE,
                       help="Number of recent characters to analyze")
    parser.add_argument("--dry-run", action="store_true",
                       help="Print the prompt tw would send and send nothing (no API key needed)")

    args = parser.parse_args()
    if args.interval < 1:
        parser.error("--interval must be at least 1")
    if args.chars < 100:
        parser.error("--chars must be at least 100")
    if args.chars > 200000:
        parser.error("--chars must be at most 200000")

    # Normalize command shortcuts
    cmd_map = {"e": "explain", "r": "research", "w": "watch"}
    command = cmd_map.get(args.command, args.command)

    if command == "watch":
        if args.dry_run:
            fail("--dry-run works with explain, research and error.")
        watch_mode(auto_research=args.auto, interval=args.interval)
        return

    print(TW_BEGIN)
    sys.stdout.flush()
    try:
        log_content = read_recent_log(args.chars)
        lines = ["📖 Reading recent terminal activity..."]
        if not log_content or len(log_content) < 100:
            lines.append("❌ No recent terminal activity found.")
            lines.append("   Make sure logging is enabled in your terminal config.")
            lines.append(f"   Looking for: {TERMINAL_LOGS_DIR}/{date.today()}.log")
            _out("\n".join(lines))
            return

        redacted, masked = redact(log_content)
        lines.append(f"   Found {len(log_content):,} characters of activity")

        # Detect errors in the text that would actually be sent.
        errors = detect_errors(redacted)
        if errors:
            lines.append(f"⚠️  Detected {len(errors)} potential error(s)")
        _out("\n".join(lines))

        if masked > 0:
            verb = "showing" if args.dry_run else "sending"
            _out(f"Masked {masked} token-shaped string(s) before {verb}.")

        if args.dry_run and command in ("explain", "research", "error"):
            prompt = build_prompt(redacted, command, errors)
            model = model_name()
            effort = effort_setting(model)
            named = model
            if effort is not None:
                named = model + " (effort " + effort + ")"
            _out(
                "Dry run: nothing was sent. tw would send this prompt to "
                + named
                + ":\n\n"
                + prompt
                + "\n\nEnd of prompt."
            )
            return

        _out(f"🤖 Analyzing with AI ({command} mode)...")

        if command == "error":
            analysis = analyze_with_ai(redacted, mode="error", errors=errors)
            output = format_output(analysis, mode="error")
        elif command == "research":
            analysis = analyze_with_ai(redacted, mode="research", errors=errors)
            _out("🔍 Performing research...")
            research = do_research(analysis)
            output = format_output(analysis, research, mode="research")
        else:  # explain
            analysis = analyze_with_ai(redacted, mode="explain", errors=errors)
            output = format_output(analysis, mode="explain")

        _out(output)
        try:
            save_suggestions(output)
        except OSError as exc:
            _out(f"Could not save suggestions: {exc}")
        else:
            _out(f"\n📁 Saved to: {SUGGESTIONS_FILE}")
    except KeyboardInterrupt:
        print("Stopped.", file=sys.stderr)
        sys.exit(130)
    except TWError as exc:
        fail(str(exc))
    finally:
        print(TW_END)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
