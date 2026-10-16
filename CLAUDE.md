# Terminal Watcher

AI-powered terminal activity analyzer using Claude Sonnet.

## Tech Stack

- **Language**: Python 3.10+
- **AI**: Anthropic Claude Sonnet (via `anthropic` SDK)
- **Research**: GitHub Search API, npm registry
- **Build**: setuptools via pyproject.toml

## Commands

```bash
# Run from the clone (uv creates the environment)
uv run tw explain
uv run tw research
uv run tw error
uv run tw watch --auto --interval 30

# Install the tw command
uv tool install .
```

## Key Files

```
terminal_watcher.py    # Single-file application (all logic)
pyproject.toml         # Package config, deps, entry points
README.md              # User documentation
LICENSE                # MIT
```

## Architecture

Single-file Python script with these components:
- `read_recent_log()` -- reads tail of today's terminal log file
- `detect_errors()` -- regex-based error pattern matching (16 patterns)
- `analyze_with_ai()` -- Claude API call with mode-specific prompts
- `do_research()` -- GitHub and npm API searches
- `format_output()` -- structured terminal output formatting
- `watch_mode()` -- continuous monitoring loop with error cooldown
- `main()` -- argparse CLI entry point

## Environment Variables

- `ANTHROPIC_API_KEY` -- required for all AI analysis

## Data Paths

- Log source: `~/Documents/terminal-logs/raw/YYYY-MM-DD.log`
- State file: `~/.claude/data/watcher_state.json`
- Suggestions: `~/.claude/data/watcher_suggestions.md`

## Conventions

- Use uv (`uv run`, `uv tool install`) for Python, never raw `pip`
- Single-file architecture -- keep everything in `terminal_watcher.py`
- Entry points defined in `pyproject.toml` (`tw` and `terminal-watcher`)
