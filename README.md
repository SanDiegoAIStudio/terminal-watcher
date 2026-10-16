# Terminal Watcher AI

An AI-powered assistant that watches your terminal activity and provides contextual help.

## Features

- **On-demand explanations**: "What just happened in my terminal?"
- **Proactive research**: Finds relevant docs, GitHub repos, packages
- **Error detection**: Automatically detects errors and suggests fixes
- **Context-aware help**: Understands what you're working on
- **Watch mode**: Continuous background monitoring with configurable intervals
- **Multiple output modes**: Explain, research, and error-specific analysis

## Quick Start

```bash
# Install the tw command from a clone
git clone https://github.com/SanDiegoAIStudio/terminal-watcher.git
cd terminal-watcher
uv tool install .

# Or run it from the clone without installing
uv run tw --help

# Set your Anthropic API key
export ANTHROPIC_API_KEY="sk-ant-..."
```

Set `TW_MODEL` to use another Claude model; the default is `claude-sonnet-5-5`.

tw asks the default model for low effort, which keeps answers quick. Set `TW_EFFORT` to `low`, `medium`, `high`, `xhigh` or `max` to change it. With another model, effort is sent only when `TW_EFFORT` is set.

```bash
# Use it
tw explain    # Explain recent terminal activity
tw research   # Explain + search for helpful resources
tw error      # Analyze errors and suggest fixes
tw watch      # Monitor for errors in background
```

## See what tw sends, without a key

```bash
tw explain --dry-run
```

It prints the prompt and the model tw would use and sends nothing.

## What tw sends

`explain`, `research` and `error` send the last 15,000 characters of today's terminal log to the Anthropic API (`--chars` changes that). Each `watch --auto` analysis sends the last 5,000. Before anything is sent, tw removes its own output and masks strings that look like tokens or passwords.

One-shot commands print `[tw-output-begin]` once before the first line and `[tw-output-end]` once after the last. Watch mode keeps one pair around each message. A log line counts as a marker when it contains either string, so a timestamp prefix still works. A begin marker that never gets its end marker hides at most the next 200 lines.

Masking is best effort and is not a guarantee, so keep secrets out of the terminal you log.

`--dry-run` shows the exact text.

`watch --auto` makes at most 6 API calls an hour (`TW_MAX_AUTO_PER_HOUR` changes that) and prints one line when it pauses.

## Prerequisites

1. **Terminal Logging**: tw reads `~/Documents/terminal-logs/raw/YYYY-MM-DD.log`. Create the folder first, because `script` cannot start without it:

   ```bash
   mkdir -p ~/Documents/terminal-logs/raw
   ```

   Then start your shell through `script`. These lines use the macOS `script` arguments. `-a` appends, so a second window does not wipe the first one's log.

   **Ghostty** (`~/.config/ghostty/config`):
   ```
   command = /bin/zsh -c 'exec script -a -q ~/Documents/terminal-logs/raw/$(date +%Y-%m-%d).log /bin/zsh -l'
   ```

   **WezTerm** (`~/.config/wezterm/wezterm.lua`):
   ```lua
   config.default_prog = {"/bin/zsh", "-c", "exec script -a -q ~/Documents/terminal-logs/raw/$(date +%Y-%m-%d).log /bin/zsh -l"}
   ```

   **Any other terminal**: make the same `script` line its startup command.

   The file is named for the day the window was opened, and tw reads today's file. After midnight, open a new window.

2. **Anthropic API Key**: Get from [console.anthropic.com](https://console.anthropic.com)

## Usage

### Explain Mode (default)

```bash
tw explain
# or just
tw
```

Explains what's happening in your terminal in simple, layman's terms. Output includes a summary, current task detection, explanation, potential issues, suggestions, and detected technologies.

### Research Mode

```bash
tw research
# or
tw r
```

Like explain, but also searches GitHub and npm for helpful resources. Returns recommended search queries, relevant GitHub repos with star counts, npm packages, and documentation suggestions.

### Error Mode

```bash
tw error
```

Specifically analyzes errors and provides actionable fix steps, commands to try, and documentation links.

### Watch Mode

```bash
tw watch               # Monitor silently, alert on errors
tw watch --auto        # Auto-research when errors detected
tw watch --interval 60 # Check every 60 seconds (default: 30s)
```

Continuously monitors your terminal and alerts you to errors. An error line is reported once for as long as it stays in the last 2,000 characters of the log.

### Command Shortcuts

| Full | Short |
|------|-------|
| `tw explain` | `tw e` |
| `tw research` | `tw r` |
| `tw watch` | `tw w` |

### Options

```
--auto              Auto-research when errors detected (watch mode)
--interval SECONDS  Check interval in seconds (watch mode, default: 30)
--chars NUM         Number of recent characters to analyze (default: 15000)
--dry-run           Print the prompt tw would send and send nothing
```

## Configuration

Log directory: `~/Documents/terminal-logs/raw/YYYY-MM-DD.log`

To customize, edit `terminal_watcher.py`:
```python
TERMINAL_LOGS_DIR = Path.home() / "Documents" / "terminal-logs" / "raw"
RECENT_LOG_SIZE = 15000  # Characters to analyze
```

Suggestions are saved to `~/.claude/data/watcher_suggestions.md`.

The suggestions file is created with mode 0600, because it can hold text from the terminal log.

## Shell Aliases

`uv tool install .` puts both `tw` and `terminal-watcher` on your PATH. Optional extras for `~/.zshrc`:
```bash
alias tw-research='tw research'
alias tw-watch='tw watch --auto'
```

## How It Works

1. **Reads** recent terminal output from log files
2. **Cleans** ANSI escape codes and formats content
3. **Detects** error lines with a list of regex patterns
4. **Analyzes** with Claude Sonnet for contextual explanations
5. **Researches** (optional) GitHub repos and npm packages
6. **Outputs** formatted, actionable information
7. **Saves** suggestions to `~/.claude/data/watcher_suggestions.md`

## API Usage

Uses the Anthropic API. Each one-shot request sends up to 15,000 characters of log text and asks for at most 8,000 tokens back, which covers the model's thinking as well as the answer. `watch --auto` sends up to 5,000 characters per analysis and makes at most 6 calls an hour.

## Requirements

- Python 3.10+
- `anthropic` package (1.0 or newer; `uv tool install .` installs a current one)
- `requests` package (>=2.28.0)
- Anthropic API key
- Terminal logging enabled

## The story (then and now)

**Then.** On December 4, 2025, I set up my terminal (Ghostty) to write every session to a dated log file, with a script to process the logs at the end of the day. The next morning I was going through a to-do list and said yes to item F: "Package Terminal Watcher as open-source GitHub repo". The first release was committed at 08:42 that morning: four commands (`tw explain`, `tw research`, `tw error`, `tw watch`) that read today's log and ask Claude what just happened. Then it sat in a private repo.

**Now.** I stopped logging my terminal in May 2026, so tw doesn't run in my setup today. Before this release it got a cleanup: `--dry-run` prints exactly what would be sent, strings that look like tokens or passwords are masked first, tw's own output is fenced so it never explains itself, `watch --auto` is capped at 6 API calls an hour, and 87 tests pass.

State: retired from my setup; it works, and it's early. You need terminal logging on (see Prerequisites). Try `tw explain --dry-run` first: it needs no key.

## License

MIT License - see [LICENSE](LICENSE)

## Contributing

PRs welcome! Areas for improvement:
- Support for more terminal emulators
- Additional search providers (Stack Overflow, MDN)
- Local LLM support (Ollama)
- VS Code extension
- Real-time streaming analysis
