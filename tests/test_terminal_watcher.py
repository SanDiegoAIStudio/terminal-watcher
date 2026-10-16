import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import anthropic
try:
    import httpx2 as httpx
except ImportError:
    import httpx

import terminal_watcher


_ENV_KEYS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "TW_MODEL", "TW_EFFORT")


class TestAnalyzeWithAi(unittest.TestCase):
    def setUp(self):
        self._env_backup = {key: os.environ.get(key) for key in _ENV_KEYS}
        os.environ.pop("TW_EFFORT", None)

    def tearDown(self):
        for key, value in self._env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_missing_key_exits_with_one_line(self):
        """source: a log with no API key ended in a Python traceback"""
        os.environ.pop("ANTHROPIC_API_KEY", None)
        os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)
        with patch("terminal_watcher.anthropic.Anthropic") as mock_anthropic:
            with self.assertRaises(terminal_watcher.TWError) as caught:
                terminal_watcher.analyze_with_ai("x" * 200)
            self.assertEqual(
                str(caught.exception),
                'ANTHROPIC_API_KEY is not set. Create a key at '
                'https://console.anthropic.com and run: export ANTHROPIC_API_KEY="sk-ant-..."',
            )
            mock_anthropic.assert_not_called()

        stderr = io.StringIO()
        stdout = io.StringIO()
        with patch("terminal_watcher.read_recent_log", return_value="x" * 200), \
             patch("terminal_watcher.anthropic.Anthropic") as mock_anthropic, \
             patch("sys.argv", ["tw", "explain"]):
            with redirect_stderr(stderr), redirect_stdout(stdout):
                with self.assertRaises(SystemExit) as caught:
                    terminal_watcher.main()
        self.assertEqual(caught.exception.code, 1)
        self.assertTrue(
            stderr.getvalue().startswith("Error: ANTHROPIC_API_KEY is not set")
        )
        self.assertEqual(stderr.getvalue().count("\n"), 1)
        mock_anthropic.assert_not_called()

    def test_model_defaults_to_current_and_can_be_overridden(self):
        """source: the model id was pinned to claude-sonnet-4-20250514, which Anthropic deprecated"""
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test"
        os.environ.pop("TW_MODEL", None)
        fake_response = SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text='{"summary": "ok"}')],
        )
        client = MagicMock()
        client.messages.create.return_value = fake_response
        with patch("terminal_watcher.anthropic.Anthropic", return_value=client):
            terminal_watcher.analyze_with_ai("x" * 200)
            kwargs = client.messages.create.call_args.kwargs
            self.assertEqual(kwargs["model"], "claude-sonnet-5-5")
            self.assertEqual(kwargs["max_tokens"], 8000)

            client.messages.create.reset_mock()
            os.environ["TW_MODEL"] = "claude-haiku-4-5"
            terminal_watcher.analyze_with_ai("x" * 200)
            kwargs = client.messages.create.call_args.kwargs
            self.assertEqual(kwargs["model"], "claude-haiku-4-5")
            self.assertEqual(kwargs["max_tokens"], 8000)

    def test_connection_error_exits_with_one_line(self):
        """source: an API failure ended in a traceback"""
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test"
        client = MagicMock()
        client.messages.create.side_effect = anthropic.APIConnectionError(
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        )
        with patch("terminal_watcher.anthropic.Anthropic", return_value=client):
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                with self.assertRaises(terminal_watcher.TWError) as caught:
                    try:
                        terminal_watcher.analyze_with_ai("x" * 200)
                    except anthropic.APIConnectionError as exc:
                        self.fail(
                            "the connection error escaped analyze_with_ai: %r" % (exc,)
                        )
            self.assertNotIn("Traceback", stderr.getvalue())
            self.assertEqual(
                str(caught.exception),
                "could not reach the Anthropic API. Check your network.",
            )

    def test_unknown_model_exits_and_names_tw_model(self):
        """source: a retired model id ended in a traceback"""
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test"
        client = MagicMock()
        client.messages.create.side_effect = anthropic.NotFoundError(
            "model not found",
            response=httpx.Response(
                404,
                request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
            ),
            body=None,
        )
        with patch("terminal_watcher.anthropic.Anthropic", return_value=client):
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                with self.assertRaises(terminal_watcher.TWError) as caught:
                    terminal_watcher.analyze_with_ai("x" * 200)
            self.assertNotIn("Traceback", stderr.getvalue())
            err = str(caught.exception)
            self.assertIn("was not found", err)
            self.assertIn("TW_MODEL", err)

    def test_reply_reads_text_blocks_only(self):
        """source: the reply was read from content[0] only"""
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test"
        fake_response = SimpleNamespace(
            stop_reason="end_turn",
            content=[
                SimpleNamespace(type="thinking", thinking="..."),
                SimpleNamespace(type="text", text='```json\n{"summary": "ok"}\n```'),
            ],
        )
        client = MagicMock()
        client.messages.create.return_value = fake_response
        with patch("terminal_watcher.anthropic.Anthropic", return_value=client):
            try:
                result = terminal_watcher.analyze_with_ai("x" * 200)
            except AttributeError as exc:
                self.fail("analyze_with_ai read a block that has no text: %s" % exc)
        self.assertEqual(result, {"summary": "ok"})

    def test_refusal_returns_a_message(self):
        """source: a refusal with no text block raised IndexError"""
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test"
        fake_response = SimpleNamespace(stop_reason="refusal", content=[])
        client = MagicMock()
        client.messages.create.return_value = fake_response
        with patch("terminal_watcher.anthropic.Anthropic", return_value=client):
            result = terminal_watcher.analyze_with_ai("x" * 200)
        self.assertEqual(
            result,
            {"raw_response": "The model declined to analyze this log."},
        )

    def test_refused_key_exits_with_one_line(self):
        """source: a refused API key ended in a traceback"""
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test"
        client = MagicMock()
        client.messages.create.side_effect = anthropic.AuthenticationError(
            "message",
            response=httpx.Response(
                401,
                request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
            ),
            body=None,
        )
        with patch("terminal_watcher.anthropic.Anthropic", return_value=client):
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                with self.assertRaises(terminal_watcher.TWError) as caught:
                    try:
                        terminal_watcher.analyze_with_ai("x" * 200)
                    except anthropic.APIError as exc:
                        self.fail(
                            "the API error escaped analyze_with_ai: %r" % (exc,)
                        )
            self.assertNotIn("Traceback", stderr.getvalue())
            self.assertEqual(
                str(caught.exception),
                "the API key was refused (401). Check ANTHROPIC_API_KEY.",
            )

    def test_rate_limit_exits_with_one_line(self):
        """source: a rate limit ended in a traceback"""
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test"
        client = MagicMock()
        client.messages.create.side_effect = anthropic.RateLimitError(
            "message",
            response=httpx.Response(
                429,
                request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
            ),
            body=None,
        )
        with patch("terminal_watcher.anthropic.Anthropic", return_value=client):
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                with self.assertRaises(terminal_watcher.TWError) as caught:
                    try:
                        terminal_watcher.analyze_with_ai("x" * 200)
                    except anthropic.APIError as exc:
                        self.fail(
                            "the API error escaped analyze_with_ai: %r" % (exc,)
                        )
            self.assertNotIn("Traceback", stderr.getvalue())
            self.assertEqual(
                str(caught.exception),
                "the API is rate limiting this key (429). Try again in a minute.",
            )

    def test_other_status_exits_with_one_line(self):
        """source: a server error ended in a traceback"""
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test"
        client = MagicMock()
        client.messages.create.side_effect = anthropic.InternalServerError(
            "message",
            response=httpx.Response(
                500,
                request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
            ),
            body=None,
        )
        with patch("terminal_watcher.anthropic.Anthropic", return_value=client):
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                with self.assertRaises(terminal_watcher.TWError) as caught:
                    try:
                        terminal_watcher.analyze_with_ai("x" * 200)
                    except anthropic.APIError as exc:
                        self.fail(
                            "the API error escaped analyze_with_ai: %r" % (exc,)
                        )
            self.assertNotIn("Traceback", stderr.getvalue())
            self.assertEqual(
                str(caught.exception),
                "the API returned status 500.",
            )

    def test_raw_response_is_printed_in_every_mode(self):
        """source: a reply that was not JSON printed only N/A fields"""
        for mode in ("explain", "research", "error"):
            kwargs = {"mode": mode}
            if mode == "research":
                kwargs["research"] = {}
            output = terminal_watcher.format_output(
                {"raw_response": "plain text reply"}, **kwargs
            )
            self.assertIn("plain text reply", output)
            self.assertNotIn("N/A", output)

    def test_auth_token_is_accepted(self):
        """source: the key check refused ANTHROPIC_AUTH_TOKEN, which the SDK accepts"""
        os.environ.pop("ANTHROPIC_API_KEY", None)
        os.environ["ANTHROPIC_AUTH_TOKEN"] = "test-token"
        fake_response = SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text='{"summary": "ok"}')],
        )
        client = MagicMock()
        client.messages.create.return_value = fake_response
        with patch("terminal_watcher.anthropic.Anthropic", return_value=client):
            try:
                result = terminal_watcher.analyze_with_ai("x" * 200)
            except (SystemExit, terminal_watcher.TWError):
                self.fail("the key check refused ANTHROPIC_AUTH_TOKEN")
        self.assertEqual(result, {"summary": "ok"})

    def test_explain_output_unchanged_for_json(self):
        """source: printing raw_response must not change the normal explain output"""
        output = terminal_watcher.format_output(
            {"summary": "S", "current_task": "T", "explanation": "E"},
            mode="explain",
        )
        self.assertIn("SUMMARY: S", output)
        self.assertIn("CURRENT TASK: T", output)
        self.assertIn("EXPLANATION:\nE", output)

    def test_build_prompt_matches_each_mode(self):
        """source: the dry run must print the same prompt analyze_with_ai sends"""
        log = "recent activity: cargo build"
        errors = ["error: build failed"]
        phrases = {
            "explain": "friendly development assistant",
            "research": "research assistant",
            "error": "debugging assistant",
        }
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test"
        fake_response = SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text='{"summary": "ok"}')],
        )
        for mode, phrase in phrases.items():
            prompt = terminal_watcher.build_prompt(log, mode, errors)
            self.assertIn(log, prompt)
            self.assertIn(phrase, prompt)
            if mode == "error":
                self.assertIn(json.dumps(errors, indent=2), prompt)
            client = MagicMock()
            client.messages.create.return_value = fake_response
            with patch("terminal_watcher.anthropic.Anthropic", return_value=client):
                terminal_watcher.analyze_with_ai(log, mode=mode, errors=errors)
            sent = client.messages.create.call_args.kwargs["messages"][0]["content"]
            self.assertEqual(sent, prompt)

    def test_dry_run_sends_nothing_without_a_key(self):
        """source: without a key, tw showed only an error line; the dry run must work with no key and send nothing"""
        os.environ.pop("ANTHROPIC_API_KEY", None)
        os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)
        marker = "error: build failed"
        sample = marker + ("x" * (300 - len(marker)))
        stdout = io.StringIO()
        with patch("terminal_watcher.anthropic.Anthropic", new=Mock()) as mock_anthropic, \
             patch("terminal_watcher.read_recent_log", return_value=sample), \
             patch("terminal_watcher.save_suggestions", new=Mock()) as mock_save, \
             patch("sys.argv", ["tw", "explain", "--dry-run"]):
            with redirect_stdout(stdout):
                try:
                    terminal_watcher.main()
                except SystemExit as exc:
                    self.fail(
                        "the dry run exited with %r; it must not need a key"
                        % (exc.code,)
                    )
        output = stdout.getvalue()
        self.assertIn("Dry run: nothing was sent.", output)
        self.assertIn("error: build failed", output)
        self.assertIn("End of prompt.", output)
        mock_anthropic.assert_not_called()
        mock_save.assert_not_called()

    def test_dry_run_research_skips_research_calls(self):
        """source: a dry run must make no network call, research included"""
        os.environ.pop("ANTHROPIC_API_KEY", None)
        os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)
        marker = "error: build failed"
        sample = marker + ("x" * (300 - len(marker)))
        stdout = io.StringIO()
        with patch("terminal_watcher.anthropic.Anthropic", new=Mock()) as mock_anthropic, \
             patch("terminal_watcher.read_recent_log", return_value=sample), \
             patch("terminal_watcher.save_suggestions", new=Mock()) as mock_save, \
             patch("terminal_watcher.do_research", new=Mock()) as mock_research, \
             patch("sys.argv", ["tw", "research", "--dry-run"]):
            with redirect_stdout(stdout):
                try:
                    terminal_watcher.main()
                except SystemExit as exc:
                    self.fail(
                        "the dry run exited with %r; it must not need a key"
                        % (exc.code,)
                    )
        output = stdout.getvalue()
        self.assertIn("Dry run: nothing was sent.", output)
        self.assertIn("error: build failed", output)
        self.assertIn("End of prompt.", output)
        self.assertIn("research assistant", output)
        self.assertNotIn("friendly development assistant", output)
        mock_anthropic.assert_not_called()
        mock_save.assert_not_called()
        mock_research.assert_not_called()

    def test_watch_rejects_dry_run(self):
        """source: watch mode has no single prompt to show"""
        stderr = io.StringIO()
        with patch("terminal_watcher.watch_mode", new=Mock()) as mock_watch, \
             patch("sys.argv", ["tw", "watch", "--dry-run"]):
            with redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as caught:
                    terminal_watcher.main()
        self.assertEqual(caught.exception.code, 1)
        self.assertIn(
            "--dry-run works with explain, research and error",
            stderr.getvalue(),
        )
        mock_watch.assert_not_called()

    def test_unnamed_api_error_becomes_twerror(self):
        """source: an API error class the code does not name escapes analyze_with_ai"""
        os.environ["ANTHROPIC_API_KEY"] = "sk" + "-ant-test"
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        response = httpx.Response(200, request=request)
        client = MagicMock()
        client.messages.create.side_effect = anthropic.APIResponseValidationError(
            response,
            body=None,
        )
        with patch("terminal_watcher.anthropic.Anthropic", return_value=client):
            with self.assertRaises(terminal_watcher.TWError) as caught:
                terminal_watcher.analyze_with_ai("x" * 200)
        self.assertEqual(
            str(caught.exception),
            "the API call failed (APIResponseValidationError).",
        )

    def _client(self, response):
        client = MagicMock()
        client.messages.create.return_value = response
        return client

    def test_default_model_sends_low_effort_and_8000_tokens(self):
        """source: the default model was called with no effort and max_tokens of 2000, so thinking cut the answer off"""
        os.environ["ANTHROPIC_API_KEY"] = "sk" + "-ant-test"
        os.environ.pop("TW_MODEL", None)
        os.environ.pop("TW_EFFORT", None)
        fake_response = SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text='{"summary": "ok"}')],
        )
        client = self._client(fake_response)
        escaped = None
        with patch("terminal_watcher.anthropic.Anthropic", return_value=client):
            try:
                terminal_watcher.analyze_with_ai("x" * 200)
            except Exception as exc:
                escaped = exc
        self.assertIsNone(escaped)
        kwargs = client.messages.create.call_args.kwargs
        self.assertEqual(terminal_watcher.MAX_TOKENS, 8000)
        self.assertEqual(terminal_watcher.DEFAULT_EFFORT, "low")
        self.assertEqual(
            terminal_watcher.EFFORT_LEVELS,
            ("low", "medium", "high", "xhigh", "max"),
        )
        self.assertEqual(kwargs["model"], "claude-sonnet-5-5")
        self.assertEqual(kwargs["max_tokens"], 8000)
        self.assertIn("output_config", kwargs)
        self.assertEqual(kwargs["output_config"], {"effort": "low"})

    def test_other_model_omits_effort_unless_tw_effort_is_set(self):
        """source: effort was sent to a model that was not asked for it, and TW_EFFORT was ignored on that model"""
        os.environ["ANTHROPIC_API_KEY"] = "sk" + "-ant-test"
        os.environ["TW_MODEL"] = "claude-haiku-4-5"
        os.environ.pop("TW_EFFORT", None)
        fake_response = SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text='{"summary": "ok"}')],
        )
        client = self._client(fake_response)
        with patch("terminal_watcher.anthropic.Anthropic", return_value=client):
            terminal_watcher.analyze_with_ai("x" * 200)
            kwargs = client.messages.create.call_args.kwargs
            self.assertEqual(kwargs["model"], "claude-haiku-4-5")
            self.assertNotIn("output_config", kwargs)

            client.messages.create.reset_mock()
            os.environ["TW_EFFORT"] = "medium"
            terminal_watcher.analyze_with_ai("x" * 200)
            kwargs = client.messages.create.call_args.kwargs
            self.assertEqual(kwargs["output_config"], {"effort": "medium"})

    def test_bad_tw_effort_raises_before_the_client_is_created(self):
        """source: a bad TW_EFFORT was sent to the API"""
        os.environ["ANTHROPIC_API_KEY"] = "sk" + "-ant-test"
        os.environ["TW_EFFORT"] = "bogus"
        with patch("terminal_watcher.anthropic.Anthropic") as mock_anthropic:
            with self.assertRaises(terminal_watcher.TWError) as caught:
                terminal_watcher.analyze_with_ai("x" * 200)
        self.assertEqual(
            str(caught.exception),
            "TW_EFFORT must be one of low, medium, high, xhigh, max.",
        )
        mock_anthropic.assert_not_called()

    def test_empty_tw_effort_omits_output_config(self):
        """source: an empty TW_EFFORT still sent low effort for the default model"""
        os.environ["ANTHROPIC_API_KEY"] = "sk" + "-ant-test"
        os.environ.pop("TW_MODEL", None)
        os.environ["TW_EFFORT"] = ""
        fake_response = SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text='{"summary": "ok"}')],
        )
        client = self._client(fake_response)
        with patch("terminal_watcher.anthropic.Anthropic", return_value=client):
            terminal_watcher.analyze_with_ai("x" * 200)
        kwargs = client.messages.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "claude-sonnet-5-5")
        self.assertNotIn("output_config", kwargs)

    def test_max_tokens_stop_is_reported_and_not_parsed(self):
        """source: an answer cut off at max_tokens was parsed as JSON and the cutoff was silent"""
        os.environ["ANTHROPIC_API_KEY"] = "sk" + "-ant-test"
        os.environ.pop("TW_EFFORT", None)
        half = '{"summary": "unfin'
        fake_response = SimpleNamespace(
            stop_reason="max_tokens",
            content=[SimpleNamespace(type="text", text=half)],
        )
        client = self._client(fake_response)
        sentence = (
            "(The answer was cut off at the "
            + str(terminal_watcher.MAX_TOKENS)
            + " token limit.)"
        )
        with patch("terminal_watcher.anthropic.Anthropic", return_value=client):
            result = terminal_watcher.analyze_with_ai("x" * 200)
        self.assertEqual(list(result.keys()), ["raw_response"])
        self.assertEqual(result["raw_response"], half + "\n\n" + sentence)
        self.assertTrue(result["raw_response"].endswith(sentence))
        output = terminal_watcher.format_output(result, mode="explain")
        self.assertIn(sentence, output)

    def test_status_400_shows_a_cleaned_api_message_and_500_stays(self):
        """source: a 400 response hid the API message, and a 500 no longer said status 500"""
        os.environ["ANTHROPIC_API_KEY"] = "sk" + "-ant-test"
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        letters = "m" * 300
        message = "\x1b" + "\x7f" + "\x80" + "\x9f" + "\n" + letters
        client = MagicMock()
        client.messages.create.side_effect = anthropic.APIStatusError(
            message,
            response=httpx.Response(400, request=request),
            body=None,
        )
        with patch("terminal_watcher.anthropic.Anthropic", return_value=client):
            with self.assertRaises(terminal_watcher.TWError) as caught:
                terminal_watcher.analyze_with_ai("x" * 200)
        prefix = "the API refused the request (400): "
        self.assertEqual(str(caught.exception), prefix + letters[:200])
        self.assertEqual(len(str(caught.exception)) - len(prefix), 200)
        for char in str(caught.exception):
            code = ord(char)
            self.assertGreaterEqual(code, 32)
            self.assertNotEqual(code, 127)
            self.assertFalse(128 <= code <= 159)

        client.messages.create.side_effect = anthropic.APIStatusError(
            "\x1b" + ("e" * 50),
            response=httpx.Response(500, request=request),
            body=None,
        )
        with patch("terminal_watcher.anthropic.Anthropic", return_value=client):
            with self.assertRaises(terminal_watcher.TWError) as caught:
                terminal_watcher.analyze_with_ai("x" * 200)
        self.assertEqual(str(caught.exception), "the API returned status 500.")

    def test_output_config_typeerror_names_reinstall_and_other_typeerrors_propagate(self):
        """source: an old anthropic package raised TypeError for output_config, and every TypeError was replaced"""
        os.environ["ANTHROPIC_API_KEY"] = "sk" + "-ant-test"
        os.environ.pop("TW_MODEL", None)
        os.environ.pop("TW_EFFORT", None)
        client = MagicMock()
        client.messages.create.side_effect = TypeError(
            "unexpected keyword argument output_config"
        )
        escaped = None
        caught = None
        with patch("terminal_watcher.anthropic.Anthropic", return_value=client):
            try:
                terminal_watcher.analyze_with_ai("x" * 200)
            except TypeError as exc:
                if "output_config" in str(exc):
                    escaped = exc
                else:
                    raise
            except terminal_watcher.TWError as exc:
                caught = exc
        self.assertIsNone(escaped)
        self.assertIsInstance(caught, terminal_watcher.TWError)
        self.assertEqual(
            str(caught),
            "this anthropic package is too old for the effort setting. "
            "Reinstall with: uv tool install . --reinstall",
        )

        client.messages.create.side_effect = TypeError("boom")
        with patch("terminal_watcher.anthropic.Anthropic", return_value=client):
            with self.assertRaises(TypeError) as caught_type:
                terminal_watcher.analyze_with_ai("x" * 200)
        self.assertEqual(str(caught_type.exception), "boom")

    def test_dry_run_names_effort_only_when_one_would_be_sent(self):
        """source: the dry run hid the effort, and named an effort for a model that would not be sent one"""
        os.environ.pop("ANTHROPIC_API_KEY", None)
        os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)
        os.environ.pop("TW_MODEL", None)
        os.environ.pop("TW_EFFORT", None)
        sample = "x" * 200
        stdout = io.StringIO()
        with patch("terminal_watcher.read_recent_log", return_value=sample), \
             patch("terminal_watcher.anthropic.Anthropic") as mock_anthropic, \
             patch("sys.argv", ["tw", "explain", "--dry-run"]):
            with redirect_stdout(stdout):
                terminal_watcher.main()
        output = stdout.getvalue()
        self.assertIn(
            "tw would send this prompt to claude-sonnet-5-5 (effort low):",
            output,
        )
        mock_anthropic.assert_not_called()

        os.environ["TW_MODEL"] = "claude-haiku-4-5"
        stdout = io.StringIO()
        with patch("terminal_watcher.read_recent_log", return_value=sample), \
             patch("terminal_watcher.anthropic.Anthropic") as mock_anthropic, \
             patch("sys.argv", ["tw", "explain", "--dry-run"]):
            with redirect_stdout(stdout):
                terminal_watcher.main()
        output = stdout.getvalue()
        self.assertIn(
            "tw would send this prompt to claude-haiku-4-5:",
            output,
        )
        self.assertNotIn("(effort", output)
        mock_anthropic.assert_not_called()

        os.environ["TW_EFFORT"] = "bogus"
        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch("terminal_watcher.read_recent_log", return_value=sample), \
             patch("terminal_watcher.anthropic.Anthropic") as mock_anthropic, \
             patch("sys.argv", ["tw", "explain", "--dry-run"]):
            with redirect_stdout(stdout), redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as caught:
                    terminal_watcher.main()
        self.assertEqual(caught.exception.code, 1)
        self.assertEqual(
            stderr.getvalue(),
            "Error: TW_EFFORT must be one of low, medium, high, xhigh, max.\n",
        )
        mock_anthropic.assert_not_called()


def _watch_state():
    return {
        "last_content": "",
        "reported": set(),
        "auto_times": [],
        "auto_pause_announced": False,
    }


def _clear_auto_limit(case):
    """Drop TW_MAX_AUTO_PER_HOUR for one test and restore it afterwards."""
    previous = os.environ.pop("TW_MAX_AUTO_PER_HOUR", None)

    def restore():
        if previous is None:
            os.environ.pop("TW_MAX_AUTO_PER_HOUR", None)
        else:
            os.environ["TW_MAX_AUTO_PER_HOUR"] = previous

    case.addCleanup(restore)


def _today_log(directory):
    return Path(directory) / (date.today().strftime("%Y-%m-%d") + ".log")


class TestWatchLoopSafety(unittest.TestCase):
    def test_strip_own_output_removes_marked_blocks(self):
        """source: tw watch --auto prints its report into the log it reads, so the next pass analyzes that report again"""
        begin = terminal_watcher.TW_BEGIN
        end = terminal_watcher.TW_END
        full = "before\n" + begin + "\nsecret\n" + end + "\nafter\n"
        self.assertEqual(terminal_watcher.strip_own_output(full), "before\nafter\n")
        unterminated = "before\n" + begin + "\nsecret without end\n"
        self.assertEqual(terminal_watcher.strip_own_output(unterminated), "before\n")
        leading_end = "head noise\n" + end + "\nkeep\n"
        self.assertEqual(terminal_watcher.strip_own_output(leading_end), "keep\n")

    def test_watch_step_does_not_reanalyze_its_own_output(self):
        """source: tw watch --auto prints its report into the log it reads, so the next pass analyzes that report again"""
        _clear_auto_limit(self)
        mock_ai = Mock(return_value={"error_summary": "x"})
        with tempfile.TemporaryDirectory() as tmp:
            log_path = _today_log(tmp)
            log_path.write_text("Error: build failed\n")
            state = _watch_state()
            with patch("terminal_watcher.TERMINAL_LOGS_DIR", Path(tmp)), \
                 patch("terminal_watcher.analyze_with_ai", mock_ai), \
                 patch("terminal_watcher.save_suggestions"):
                first = io.StringIO()
                with redirect_stdout(first):
                    terminal_watcher.watch_step(state, True, 1_000_000.0)
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(first.getvalue())
                second = io.StringIO()
                with redirect_stdout(second):
                    terminal_watcher.watch_step(state, True, 1_000_010.0)
                third = io.StringIO()
                with redirect_stdout(third):
                    terminal_watcher.watch_step(state, True, 1_000_410.0)
        self.assertEqual(mock_ai.call_count, 1)
        self.assertIn("Detected", first.getvalue())
        self.assertNotIn("Detected", second.getvalue())
        self.assertNotIn("Detected", third.getvalue())

    def test_watch_step_without_auto_alerts_once(self):
        """source: tw watch prints its alert into the log it reads, so the next pass treats that alert as a new error"""
        _clear_auto_limit(self)
        mock_ai = Mock(return_value={"error_summary": "x"})
        with tempfile.TemporaryDirectory() as tmp:
            log_path = _today_log(tmp)
            log_path.write_text("Error: build failed\n")
            state = _watch_state()
            with patch("terminal_watcher.TERMINAL_LOGS_DIR", Path(tmp)), \
                 patch("terminal_watcher.analyze_with_ai", mock_ai), \
                 patch("terminal_watcher.save_suggestions"):
                first = io.StringIO()
                with redirect_stdout(first):
                    terminal_watcher.watch_step(state, False, 1_000_000.0)
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(first.getvalue())
                second = io.StringIO()
                with redirect_stdout(second):
                    terminal_watcher.watch_step(state, False, 1_000_010.0)
        self.assertEqual(mock_ai.call_count, 0)
        self.assertEqual(first.getvalue().count("Detected"), 1)
        self.assertNotIn("Detected", second.getvalue())

    def test_auto_analysis_pauses_after_six_per_hour(self):
        """source: nothing limits how often --auto analyzes"""
        previous = os.environ.pop("TW_MAX_AUTO_PER_HOUR", None)
        mock_ai = Mock(return_value={"error_summary": "x"})
        try:
            with tempfile.TemporaryDirectory() as tmp:
                log_path = _today_log(tmp)
                log_path.write_text("")
                state = _watch_state()
                start = 1_700_000_000.0
                stdout = io.StringIO()
                with patch("terminal_watcher.TERMINAL_LOGS_DIR", Path(tmp)), \
                     patch("terminal_watcher.analyze_with_ai", mock_ai), \
                     patch("terminal_watcher.save_suggestions"):
                    with redirect_stdout(stdout):
                        for index in range(8):
                            with log_path.open("a", encoding="utf-8") as handle:
                                handle.write("Error: build failed number %s\n" % index)
                            terminal_watcher.watch_step(state, True, start + index * 10)
                    self.assertEqual(mock_ai.call_count, 6)
                    self.assertIn(
                        "Auto-analysis paused: 6 API calls in the last hour "
                        "(limit 6). Run `tw error` to analyze now.",
                        stdout.getvalue(),
                    )
                    self.assertEqual(stdout.getvalue().count("Auto-analysis paused"), 1)
                    with log_path.open("a", encoding="utf-8") as handle:
                        handle.write("Error: build failed number 8\n")
                    later = io.StringIO()
                    with redirect_stdout(later):
                        terminal_watcher.watch_step(state, True, start + 4000)
                    self.assertEqual(mock_ai.call_count, 7)
                    self.assertNotIn("Auto-analysis paused", later.getvalue())
        finally:
            if previous is None:
                os.environ.pop("TW_MAX_AUTO_PER_HOUR", None)
            else:
                os.environ["TW_MAX_AUTO_PER_HOUR"] = previous

    def test_api_failure_in_watch_step_continues(self):
        """source: one API failure exits the long-running watcher"""
        message = "the API is rate limiting this key (429). Try again in a minute."
        with tempfile.TemporaryDirectory() as tmp:
            log_path = _today_log(tmp)
            log_path.write_text("Error: build failed\n")
            state = _watch_state()
            stdout = io.StringIO()
            with patch("terminal_watcher.TERMINAL_LOGS_DIR", Path(tmp)), \
                 patch(
                     "terminal_watcher.analyze_with_ai",
                     side_effect=terminal_watcher.TWError(message),
                 ), \
                 patch("terminal_watcher.save_suggestions"):
                escaped = None
                with redirect_stdout(stdout):
                    try:
                        terminal_watcher.watch_step(state, True, 2_000.0)
                    except Exception as exc:
                        escaped = exc
        self.assertIsNone(escaped)
        self.assertIn(
            "Error: " + message + " (watch continues)",
            stdout.getvalue(),
        )

    def test_read_failure_exits_main_and_watch_continues(self):
        """source: a failed log read is returned as if it were log text and can be sent to the API"""
        with tempfile.TemporaryDirectory() as tmp:
            log_path = _today_log(tmp)
            log_path.write_text("Error: build failed\n" + ("x" * 200))
            real_open = open

            def open_denied(file, *args, **kwargs):
                if Path(file) == log_path:
                    raise OSError("denied")
                return real_open(file, *args, **kwargs)

            stderr = io.StringIO()
            with patch("terminal_watcher.TERMINAL_LOGS_DIR", Path(tmp)), \
                 patch("builtins.open", open_denied), \
                 patch("sys.argv", ["tw", "explain"]):
                with redirect_stderr(stderr), redirect_stdout(io.StringIO()):
                    with self.assertRaises(SystemExit) as caught:
                        terminal_watcher.main()
            self.assertEqual(caught.exception.code, 1)
            self.assertIn("Error: could not read the terminal log", stderr.getvalue())

            state = _watch_state()
            stdout = io.StringIO()
            with patch("terminal_watcher.TERMINAL_LOGS_DIR", Path(tmp)), \
                 patch("builtins.open", open_denied):
                with redirect_stdout(stdout):
                    terminal_watcher.watch_step(state, True, 3_000.0)
            self.assertIn(
                "Error: could not read the terminal log: denied (watch continues)",
                stdout.getvalue(),
            )

            empty = Path(tmp) / "empty"
            empty.mkdir()
            with patch("terminal_watcher.TERMINAL_LOGS_DIR", empty):
                self.assertEqual(terminal_watcher.read_recent_log(), "")

    def test_redact_masks_each_token_shape(self):
        """source: raw terminal text, which can hold tokens, is sent with no masking"""
        cases = {
            "sk": "sk" + "-" + ("b" * 16),
            "ghp": "ghp" + "_" + ("a" * 20),
            "gho": "gho" + "_" + ("a" * 20),
            "ghs": "ghs" + "_" + ("a" * 20),
            "ghr": "ghr" + "_" + ("a" * 20),
            "github_pat": "github" + "_" + "pat" + "_" + ("d" * 20),
            "akia": "AK" + "IA" + ("Z" * 16),
            "xoxb": "xox" + "b" + "-" + ("c" * 10),
            "xoxa": "xox" + "a" + "-" + ("c" * 10),
            "xoxp": "xox" + "p" + "-" + ("c" * 10),
            "xoxr": "xox" + "r" + "-" + ("c" * 10),
            "xoxs": "xox" + "s" + "-" + ("c" * 10),
        }
        for name, token in cases.items():
            with self.subTest(shape=name):
                redacted, count = terminal_watcher.redact("pre " + token + " post")
                self.assertEqual(count, 1)
                self.assertEqual(redacted, "pre [redacted] post")
                self.assertNotIn(token, redacted)

        bearer_secret = "e" * 20
        redacted, count = terminal_watcher.redact("Bearer " + bearer_secret)
        self.assertEqual(count, 1)
        self.assertEqual(redacted, "Bearer [redacted]")
        self.assertNotIn(bearer_secret, redacted)

        begin = "-----" + "BEGIN " + "RSA " + "PRIVATE KEY" + "-----"
        end = "-----" + "END " + "RSA " + "PRIVATE KEY" + "-----"
        pem = begin + "\n" + "abc" + "\n" + end
        redacted, count = terminal_watcher.redact("note\n" + pem + "\ndone")
        self.assertEqual(count, 1)
        self.assertNotIn(begin, redacted)
        self.assertNotIn(end, redacted)
        self.assertNotIn("abc", redacted)
        self.assertIn("note", redacted)
        self.assertIn("done", redacted)
        self.assertIn("[redacted]", redacted)

        value = "m" * 8
        labeled = "API" + "_KEY=" + value
        redacted, count = terminal_watcher.redact("pre " + labeled + " post")
        self.assertEqual(count, 1)
        self.assertIn("API" + "_KEY=", redacted)
        self.assertNotIn(value, redacted)
        self.assertEqual(redacted, "pre " + "API" + "_KEY=" + "[redacted]" + " post")

        hyphen_name = "api" + "-key=" + ("n" * 8)
        redacted, count = terminal_watcher.redact(hyphen_name)
        self.assertEqual((redacted, count), ("api" + "-key=" + "[redacted]", 1))

        ordinary = "task-runner finished in 12s"
        self.assertEqual(terminal_watcher.redact(ordinary), (ordinary, 0))

        short = "sk" + "-" + ("a" * 15)
        self.assertEqual(terminal_watcher.redact(short), (short, 0))

    def test_dry_run_masks_token_before_showing(self):
        """source: raw terminal text, which can hold tokens, is sent with no masking"""
        token = "sk" + "-" + ("a" * 24)
        body = token + "\n" + "error: build failed\n" + ("x" * 200)
        with tempfile.TemporaryDirectory() as tmp:
            log_path = _today_log(tmp)
            log_path.write_text(body)
            stdout = io.StringIO()
            with patch("terminal_watcher.TERMINAL_LOGS_DIR", Path(tmp)), \
                 patch("terminal_watcher.anthropic.Anthropic", new=Mock()) as mock_anthropic, \
                 patch("sys.argv", ["tw", "explain", "--dry-run"]):
                with redirect_stdout(stdout):
                    terminal_watcher.main()
        output = stdout.getvalue()
        self.assertNotIn(token, output)
        self.assertIn("[redacted]", output)
        self.assertIn("Masked 1 token-shaped string(s) before showing.", output)
        mock_anthropic.assert_not_called()


class TestReviewFindings(unittest.TestCase):
    def setUp(self):
        self._env_backup = {key: os.environ.get(key) for key in _ENV_KEYS}
        os.environ.pop("TW_EFFORT", None)

    def tearDown(self):
        for key, value in self._env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_oneshot_wraps_stdout_in_one_marker_pair(self):
        """source: every block tw prints is wrapped in its own pair of marker lines, which clutters a one-shot command's output"""
        os.environ.pop("ANTHROPIC_API_KEY", None)
        os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)
        begin = terminal_watcher.TW_BEGIN
        end = terminal_watcher.TW_END
        marker = "error: build failed"
        sample = marker + ("x" * (300 - len(marker)))
        stdout = io.StringIO()
        with patch("terminal_watcher.anthropic.Anthropic", new=Mock()) as mock_anthropic, \
             patch("terminal_watcher.read_recent_log", return_value=sample), \
             patch("terminal_watcher.save_suggestions", new=Mock()) as mock_save, \
             patch("sys.argv", ["tw", "explain", "--dry-run"]):
            with redirect_stdout(stdout):
                terminal_watcher.main()
        lines = stdout.getvalue().splitlines()
        begin_at = [index for index, line in enumerate(lines) if line == begin]
        end_at = [index for index, line in enumerate(lines) if line == end]
        self.assertEqual(begin_at, [0])
        self.assertEqual(end_at, [len(lines) - 1])
        between = "\n".join(lines[1:-1])
        self.assertIn("Dry run: nothing was sent.", between)
        self.assertIn("End of prompt.", between)
        mock_anthropic.assert_not_called()
        mock_save.assert_not_called()

        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch("terminal_watcher.read_recent_log", return_value="x" * 200), \
             patch("terminal_watcher.anthropic.Anthropic", new=Mock()) as mock_anthropic, \
             patch("sys.argv", ["tw", "explain"]):
            with redirect_stdout(stdout), redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as caught:
                    terminal_watcher.main()
        self.assertEqual(caught.exception.code, 1)
        failed = stdout.getvalue().splitlines()
        self.assertEqual([index for index, line in enumerate(failed) if line == begin], [0])
        self.assertEqual(
            [index for index, line in enumerate(failed) if line == end],
            [len(failed) - 1],
        )
        self.assertTrue(stderr.getvalue().startswith("Error: ANTHROPIC_API_KEY is not set"))
        self.assertEqual(stderr.getvalue().count("\n"), 1)
        self.assertNotIn(end, stderr.getvalue())
        mock_anthropic.assert_not_called()

    def test_strip_own_output_accepts_prefixed_markers(self):
        """source: markers are matched only as whole lines, so a terminal logger that prefixes lines defeats them"""
        prefix = "2026-10-21 10:00:00 "
        begin = prefix + terminal_watcher.TW_BEGIN
        end = prefix + terminal_watcher.TW_END
        text = "before\n" + begin + "\nsecret\n" + end + "\nafter\n"
        self.assertEqual(terminal_watcher.strip_own_output(text), "before\nafter\n")
        self.assertNotIn("secret", terminal_watcher.strip_own_output(text))

    def test_watch_prints_masked_only_when_analysis_is_sent(self):
        """source: the Masked N line says before sending in watch mode even when nothing is sent"""
        _clear_auto_limit(self)
        token = "sk" + "-" + ("a" * 24)
        mock_ai = Mock(return_value={"error_summary": "x"})
        with tempfile.TemporaryDirectory() as tmp:
            log_path = _today_log(tmp)
            log_path.write_text(token + "\nError: build failed\n")
            with patch("terminal_watcher.TERMINAL_LOGS_DIR", Path(tmp)), \
                 patch("terminal_watcher.analyze_with_ai", mock_ai), \
                 patch("terminal_watcher.save_suggestions"):
                quiet = io.StringIO()
                with redirect_stdout(quiet):
                    terminal_watcher.watch_step(_watch_state(), False, 1_000_000.0)
                self.assertNotIn("Masked", quiet.getvalue())
                self.assertEqual(mock_ai.call_count, 0)

                mock_ai.reset_mock()
                log_path.write_text(token + "\nError: build failed\n")
                sent = io.StringIO()
                state = _watch_state()
                with redirect_stdout(sent):
                    terminal_watcher.watch_step(state, True, 2_000_000.0)
                    with log_path.open("a", encoding="utf-8") as handle:
                        handle.write("Error: linker failed\n")
                    terminal_watcher.watch_step(state, True, 2_000_010.0)
                self.assertEqual(mock_ai.call_count, 2)
                self.assertEqual(sent.getvalue().count("Masked "), 2)
                self.assertIn("before sending.", sent.getvalue())
                self.assertNotIn("before showing.", sent.getvalue())

    def test_unreadable_log_is_reported_once_until_a_successful_read(self):
        """source: a log that stays unreadable prints the same error on every pass"""
        with tempfile.TemporaryDirectory() as tmp:
            log_path = _today_log(tmp)
            log_path.write_text("all quiet\n")
            real_open = open
            denied = {"on": True}
            reason = {"text": "denied"}

            def open_maybe(file, *args, **kwargs):
                if denied["on"] and Path(file) == log_path:
                    raise OSError(reason["text"])
                return real_open(file, *args, **kwargs)

            state = _watch_state()
            denied_message = "Error: could not read the terminal log: denied (watch continues)"
            gone_message = "Error: could not read the terminal log: gone (watch continues)"
            with patch("terminal_watcher.TERMINAL_LOGS_DIR", Path(tmp)), \
                 patch("builtins.open", open_maybe):
                first = io.StringIO()
                with redirect_stdout(first):
                    for _ in range(3):
                        terminal_watcher.watch_step(state, False, 3_000.0)
                self.assertEqual(first.getvalue().count(denied_message), 1)

                reason["text"] = "gone"
                changed = io.StringIO()
                with redirect_stdout(changed):
                    terminal_watcher.watch_step(state, False, 3_010.0)
                self.assertEqual(changed.getvalue().count(gone_message), 1)
                self.assertNotIn(denied_message, changed.getvalue())

                denied["on"] = False
                with redirect_stdout(io.StringIO()):
                    terminal_watcher.watch_step(state, False, 3_020.0)

                denied["on"] = True
                reason["text"] = "denied"
                again = io.StringIO()
                with redirect_stdout(again):
                    terminal_watcher.watch_step(state, False, 3_030.0)
                self.assertEqual(again.getvalue().count(denied_message), 1)

    def test_non_dict_json_reply_becomes_raw_response(self):
        """source: the model's reply may parse as a list or a string"""
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test"
        for reply in ("[1, 2]", '"just a string"'):
            with self.subTest(reply=reply):
                fake_response = SimpleNamespace(
                    stop_reason="end_turn",
                    content=[SimpleNamespace(type="text", text=reply)],
                )
                client = MagicMock()
                client.messages.create.return_value = fake_response
                with patch("terminal_watcher.anthropic.Anthropic", return_value=client):
                    result = terminal_watcher.analyze_with_ai("x" * 200)
                self.assertEqual(result, {"raw_response": reply})

    def test_string_and_scalar_fields_are_not_iterated_as_characters(self):
        """source: a field such as github_search_terms may be a string, and format_output or do_research then fail with AttributeError or iterate over characters"""
        analysis = {
            "github_search_terms": "one term",
            "npm_packages_to_check": "left-pad",
            "technologies": "python",
            "search_queries": 7,
        }
        with patch("terminal_watcher.search_github", return_value=[]) as mock_gh, \
             patch("terminal_watcher.search_npm", return_value=None) as mock_npm:
            terminal_watcher.do_research(analysis)
        self.assertEqual([call.args[0] for call in mock_gh.call_args_list], ["one term"])
        self.assertEqual([call.args[0] for call in mock_npm.call_args_list], ["left-pad"])
        rendered = terminal_watcher.format_output(analysis, mode="research")
        self.assertIn("TECHNOLOGIES: python", rendered)
        self.assertNotIn("p, y, t, h, o, n", rendered)
        self.assertNotIn("\n   • 7", rendered)

    def test_search_npm_rejects_unsafe_names_and_quotes_scope(self):
        """source: a package name suggested by the model is put into the npm registry URL unquoted"""
        with patch("terminal_watcher.requests.get", new=Mock()) as mock_get:
            self.assertIsNone(terminal_watcher.search_npm("../../etc"))
            self.assertIsNone(terminal_watcher.search_npm("pkg?x=1"))
            self.assertIsNone(terminal_watcher.search_npm("a" * 215))
            mock_get.assert_not_called()

        response = Mock()
        response.status_code = 200
        response.json.return_value = {
            "name": "@scope/pkg",
            "description": "scoped",
            "dist-tags": {"latest": "1.2.3"},
            "homepage": "https://example.com",
        }
        with patch("terminal_watcher.requests.get", return_value=response) as mock_get:
            info = terminal_watcher.search_npm("@scope/pkg")
        self.assertEqual(info["name"], "@scope/pkg")
        self.assertTrue(mock_get.call_args.args[0].endswith("@scope%2Fpkg"))

    def test_detect_errors_keeps_the_last_twenty_in_order(self):
        """source: detect_errors keeps the first 20 matching lines, so the newest errors are dropped"""
        lines = ["error: failure %s" % index for index in range(30)]
        log = "\n".join(lines)
        first = terminal_watcher.detect_errors(log)
        second = terminal_watcher.detect_errors(log)
        self.assertEqual(first, lines[-20:])
        self.assertEqual(second, first)

    def test_save_suggestions_uses_mode_0600(self):
        """source: the suggestions file, which holds text from the terminal log, is written with default permissions"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "notes" / "watcher_suggestions.md"
            with patch("terminal_watcher.SUGGESTIONS_FILE", path):
                terminal_watcher.save_suggestions("hello from the log")
            self.assertTrue(path.is_file())
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            path.chmod(0o644)
            self.assertEqual(path.stat().st_mode & 0o777, 0o644)
            with patch("terminal_watcher.SUGGESTIONS_FILE", path):
                terminal_watcher.save_suggestions("again")
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertIn("again", path.read_text())

    def test_keyboard_interrupt_propagates_from_search_github(self):
        """source: bare except clauses swallow KeyboardInterrupt"""
        with patch("terminal_watcher.requests.get", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                terminal_watcher.search_github("terminal watcher")


class TestThirdPass(unittest.TestCase):
    def test_failed_calls_count_toward_the_hourly_cap(self):
        """source: the hourly cap counts successful analyses only, so failed calls are unlimited"""
        previous = os.environ.pop("TW_MAX_AUTO_PER_HOUR", None)
        mock_ai = Mock(side_effect=terminal_watcher.TWError("down"))
        try:
            with tempfile.TemporaryDirectory() as tmp:
                log_path = _today_log(tmp)
                log_path.write_text("")
                suggestions = Path(tmp) / "watcher_suggestions.md"
                state = _watch_state()
                start = 1_700_000_000.0
                stdout = io.StringIO()
                with patch("terminal_watcher.TERMINAL_LOGS_DIR", Path(tmp)), \
                     patch("terminal_watcher.SUGGESTIONS_FILE", suggestions), \
                     patch("terminal_watcher.analyze_with_ai", mock_ai), \
                     patch("terminal_watcher.save_suggestions"):
                    escaped = None
                    with redirect_stdout(stdout):
                        try:
                            for index in range(8):
                                with log_path.open("a", encoding="utf-8") as handle:
                                    handle.write("Error: build failed number %s\n" % index)
                                terminal_watcher.watch_step(state, True, start + index * 10)
                        except Exception as exc:
                            escaped = exc
                self.assertIsNone(escaped)
                self.assertEqual(mock_ai.call_count, 6)
                self.assertIn(
                    "Auto-analysis paused: 6 API calls in the last hour "
                    "(limit 6). Run `tw error` to analyze now.",
                    stdout.getvalue(),
                )
                self.assertEqual(stdout.getvalue().count("Auto-analysis paused"), 1)
        finally:
            if previous is None:
                os.environ.pop("TW_MAX_AUTO_PER_HOUR", None)
            else:
                os.environ["TW_MAX_AUTO_PER_HOUR"] = previous

    def test_stale_error_line_is_not_reported_twice(self):
        """source: an error line is new again every 5 minutes while it stays near the end of the log"""
        _clear_auto_limit(self)
        mock_ai = Mock(return_value={"error_summary": "x"})
        with tempfile.TemporaryDirectory() as tmp:
            log_path = _today_log(tmp)
            log_path.write_text("Error: build failed\n")
            suggestions = Path(tmp) / "watcher_suggestions.md"
            state = _watch_state()
            with patch("terminal_watcher.TERMINAL_LOGS_DIR", Path(tmp)), \
                 patch("terminal_watcher.SUGGESTIONS_FILE", suggestions), \
                 patch("terminal_watcher.analyze_with_ai", mock_ai), \
                 patch("terminal_watcher.save_suggestions"):
                first = io.StringIO()
                with redirect_stdout(first):
                    terminal_watcher.watch_step(state, True, 1_000_000.0)
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write("task finished quietly\n")
                second = io.StringIO()
                with redirect_stdout(second):
                    terminal_watcher.watch_step(state, True, 1_000_400.0)
        self.assertEqual(mock_ai.call_count, 1)
        self.assertIn("Detected", first.getvalue())
        self.assertEqual(second.getvalue(), "")

    def test_error_line_is_reported_again_after_it_returns(self):
        """source: an error line that left the tail and came back is never reported again"""
        _clear_auto_limit(self)
        line = "Error: build failed\n"
        padding = "quiet ok\n" * 250
        self.assertGreater(len(padding), 2000)
        mock_ai = Mock(return_value={"error_summary": "x"})
        with tempfile.TemporaryDirectory() as tmp:
            log_path = _today_log(tmp)
            log_path.write_text(line)
            suggestions = Path(tmp) / "watcher_suggestions.md"
            state = _watch_state()
            with patch("terminal_watcher.TERMINAL_LOGS_DIR", Path(tmp)), \
                 patch("terminal_watcher.SUGGESTIONS_FILE", suggestions), \
                 patch("terminal_watcher.analyze_with_ai", mock_ai), \
                 patch("terminal_watcher.save_suggestions"):
                first = io.StringIO()
                with redirect_stdout(first):
                    terminal_watcher.watch_step(state, True, 5_000.0)
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(padding)
                second = io.StringIO()
                with redirect_stdout(second):
                    terminal_watcher.watch_step(state, True, 5_010.0)
                self.assertEqual(mock_ai.call_count, 1)
                self.assertEqual(state["reported"], set())
                self.assertNotIn("Detected", second.getvalue())
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(line)
                third = io.StringIO()
                with redirect_stdout(third):
                    terminal_watcher.watch_step(state, True, 5_020.0)
        self.assertEqual(mock_ai.call_count, 2)
        self.assertIn("Detected", first.getvalue())
        self.assertIn("Detected", third.getvalue())

    def test_partial_first_line_is_not_treated_as_new(self):
        """source: a line cut in half at the start of the 2,000 character tail looks new on every pass"""
        _clear_auto_limit(self)
        long_line = ("n" * 1500) + "failed" + ("n" * 1500)
        complete = "Error: kept"
        text = "start\n" + long_line + "\n" + complete + "\n"
        cut = len(text) - 2000
        failed_at = text.find("failed")
        self.assertGreater(cut, len("start\n"))
        self.assertLess(cut, len("start\n") + len(long_line))
        self.assertNotEqual(text[cut - 1], "\n")
        self.assertGreater(failed_at, cut)
        self.assertLess(failed_at, text.find("\n", cut))
        added = "abcd"
        text2 = text + added
        cut2 = len(text2) - 2000
        self.assertGreater(failed_at, cut2)
        self.assertLess(cut2, len("start\n") + len(long_line))
        self.assertNotEqual(text2[cut2 - 1], "\n")
        mock_ai = Mock(return_value={"error_summary": "x"})
        with tempfile.TemporaryDirectory() as tmp:
            log_path = _today_log(tmp)
            log_path.write_text(text)
            suggestions = Path(tmp) / "watcher_suggestions.md"
            state = _watch_state()
            with patch("terminal_watcher.TERMINAL_LOGS_DIR", Path(tmp)), \
                 patch("terminal_watcher.SUGGESTIONS_FILE", suggestions), \
                 patch("terminal_watcher.analyze_with_ai", mock_ai), \
                 patch("terminal_watcher.save_suggestions"):
                first = io.StringIO()
                with redirect_stdout(first):
                    terminal_watcher.watch_step(state, True, 1_000_000.0)
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(added)
                second = io.StringIO()
                with redirect_stdout(second):
                    terminal_watcher.watch_step(state, True, 1_000_010.0)
        self.assertEqual(mock_ai.call_count, 1)
        self.assertEqual(mock_ai.call_args.kwargs["errors"], [complete])
        self.assertEqual(
            first.getvalue().count("Detected") + second.getvalue().count("Detected"),
            1,
        )
        self.assertNotIn("Detected", second.getvalue())

    def test_save_failure_does_not_stop_watch_or_explain(self):
        """source: a failure to save the suggestions file kills tw watch and the one-shot commands"""
        mock_ai = Mock(return_value={"error_summary": "x"})
        with tempfile.TemporaryDirectory() as tmp:
            log_path = _today_log(tmp)
            log_path.write_text("Error: build failed\n")
            suggestions = Path(tmp) / "watcher_suggestions.md"
            state = _watch_state()
            stdout = io.StringIO()
            with patch("terminal_watcher.TERMINAL_LOGS_DIR", Path(tmp)), \
                 patch("terminal_watcher.SUGGESTIONS_FILE", suggestions), \
                 patch("terminal_watcher.analyze_with_ai", mock_ai), \
                 patch("terminal_watcher.save_suggestions", side_effect=OSError("disk full")):
                escaped = None
                with redirect_stdout(stdout):
                    try:
                        terminal_watcher.watch_step(state, True, 8_000.0)
                    except Exception as exc:
                        escaped = exc
            self.assertIsNone(escaped)
            self.assertIn(
                "Could not save suggestions: disk full (watch continues)",
                stdout.getvalue(),
            )
            self.assertIn("ERROR: x", stdout.getvalue())

            previous = os.environ.get("ANTHROPIC_API_KEY")
            os.environ["ANTHROPIC_API_KEY"] = "sk" + "-ant-test"
            try:
                log_path.write_text("Error: build failed\n" + ("x" * 200))
                fake = SimpleNamespace(
                    stop_reason="end_turn",
                    content=[SimpleNamespace(
                        type="text",
                        text='{"summary": "ok", "current_task": "t", "explanation": "e"}',
                    )],
                )
                client = MagicMock()
                client.messages.create.return_value = fake
                out = io.StringIO()
                err = io.StringIO()
                escaped = None
                with patch("terminal_watcher.TERMINAL_LOGS_DIR", Path(tmp)), \
                     patch("terminal_watcher.SUGGESTIONS_FILE", suggestions), \
                     patch("terminal_watcher.anthropic.Anthropic", return_value=client), \
                     patch(
                         "terminal_watcher.save_suggestions",
                         side_effect=OSError("disk full"),
                     ), \
                     patch("sys.argv", ["tw", "explain"]):
                    with redirect_stdout(out), redirect_stderr(err):
                        try:
                            terminal_watcher.main()
                        except SystemExit as exc:
                            escaped = exc
                        except Exception as exc:
                            escaped = exc
                self.assertIsNone(escaped)
                self.assertIn("Could not save suggestions: disk full", out.getvalue())
                self.assertNotIn("(watch continues)", out.getvalue())
                self.assertNotIn("Saved to:", out.getvalue())
                self.assertIn("SUMMARY: ok", out.getvalue())
                self.assertEqual(err.getvalue(), "")
            finally:
                if previous is None:
                    os.environ.pop("ANTHROPIC_API_KEY", None)
                else:
                    os.environ["ANTHROPIC_API_KEY"] = previous

    def test_read_recent_log_only_catches_oserror(self):
        """source: read_recent_log catches Exception, so a ValueError from open becomes a log-read failure"""
        with tempfile.TemporaryDirectory() as tmp:
            log_path = _today_log(tmp)
            log_path.write_text("hello\n")
            suggestions = Path(tmp) / "watcher_suggestions.md"
            real_open = open

            def open_value(file, *args, **kwargs):
                if Path(file) == log_path:
                    raise ValueError("bad mode")
                return real_open(file, *args, **kwargs)

            with patch("terminal_watcher.TERMINAL_LOGS_DIR", Path(tmp)), \
                 patch("terminal_watcher.SUGGESTIONS_FILE", suggestions), \
                 patch("builtins.open", open_value):
                with self.assertRaises(ValueError) as caught:
                    terminal_watcher.read_recent_log()
            self.assertEqual(str(caught.exception), "bad mode")

            def open_denied(file, *args, **kwargs):
                if Path(file) == log_path:
                    raise OSError("denied")
                return real_open(file, *args, **kwargs)

            with patch("terminal_watcher.TERMINAL_LOGS_DIR", Path(tmp)), \
                 patch("terminal_watcher.SUGGESTIONS_FILE", suggestions), \
                 patch("builtins.open", open_denied):
                with self.assertRaises(terminal_watcher.TWError) as caught:
                    terminal_watcher.read_recent_log()
            self.assertEqual(
                str(caught.exception),
                "could not read the terminal log: denied",
            )

    def test_redact_masks_password_inside_url(self):
        """source: a password inside a URL is sent unmasked"""
        secret = "s" + "3cret" + "V4lue"
        text = "postgres://app_user:" + secret + "@db.example.com:5432/app"
        redacted, count = terminal_watcher.redact(text)
        self.assertEqual(
            redacted,
            "postgres://app_user:[redacted]@db.example.com:5432/app",
        )
        self.assertEqual(count, 1)
        self.assertNotIn(secret, redacted)

        other = "db" + "pass" + "99"
        text = "mysql://app_user:" + other + "@db.example.com/app"
        redacted, count = terminal_watcher.redact(text)
        self.assertEqual(redacted, "mysql://app_user:[redacted]@db.example.com/app")
        self.assertEqual(count, 1)
        self.assertNotIn(other, redacted)

    def test_redact_masks_quoted_labeled_values(self):
        """source: a quoted JSON password is sent unmasked"""
        secret = "hello" + " " + "world"
        text = '"password": "' + secret + '"'
        redacted, count = terminal_watcher.redact(text)
        self.assertEqual(redacted, '"password": "[redacted]"')
        self.assertEqual(count, 1)
        self.assertNotIn(secret, redacted)

        secret2 = "k" + "eyvalue" + "99"
        text = "'api_key': '" + secret2 + "'"
        redacted, count = terminal_watcher.redact(text)
        self.assertEqual(redacted, "'api_key': '[redacted]'")
        self.assertEqual(count, 1)
        self.assertNotIn(secret2, redacted)

    def test_redact_masks_labeled_values_of_four_characters(self):
        """source: a labeled password shorter than 8 characters is sent unmasked"""
        secret = "w" * 4
        labels = ("password", "passwd", "pwd", "api_key", "token", "secret")
        for label in labels:
            for sep in ("=", ": "):
                with self.subTest(label=label, sep=sep):
                    text = label + sep + secret
                    redacted, count = terminal_watcher.redact(text)
                    self.assertEqual(redacted, label + sep + "[redacted]")
                    self.assertEqual(count, 1)
                    self.assertNotIn(secret, redacted)
        short = "password=" + ("w" * 3)
        self.assertEqual(terminal_watcher.redact(short), (short, 0))

    def test_redact_masks_bearer_after_a_label(self):
        """source: a token that follows the word Bearer after a label is sent unmasked"""
        opaque = "z" * 12
        text = "token: Bearer " + opaque
        redacted, count = terminal_watcher.redact(text)
        self.assertEqual(redacted, "token: Bearer [redacted]")
        self.assertEqual(count, 1)
        self.assertNotIn(opaque, redacted)

        opaque2 = "p" * 12
        text = "Authorization: Bearer " + opaque2
        redacted, count = terminal_watcher.redact(text)
        self.assertEqual(redacted, "Authorization: [redacted]")
        self.assertEqual(count, 1)
        self.assertNotIn(opaque2, redacted)

    def test_redact_masks_basic_authorization(self):
        """source: an Authorization Basic value is sent unmasked"""
        blob = "QW" + "xhZGRp" + "bnNlc2FtZQ"
        text = "Authorization: Basic " + blob
        redacted, count = terminal_watcher.redact(text)
        self.assertEqual(redacted, "Authorization: [redacted]")
        self.assertEqual(count, 1)
        self.assertNotIn(blob, redacted)

    def test_redact_masks_jwt(self):
        """source: a JWT is sent unmasked"""
        part = "ey" + "J"
        header = part + "hbGciOiJIUzI1NiJ9"
        payload = part + "c3ViIjoidXNlciJ9"
        signature = "SflKxw" + "RJSMe"
        token = header + "." + payload + "." + signature
        redacted, count = terminal_watcher.redact("pre " + token + " post")
        self.assertEqual(redacted, "pre [redacted] post")
        self.assertEqual(count, 1)
        self.assertNotIn(token, redacted)
        self.assertNotIn(header, redacted)
        self.assertNotIn(payload, redacted)
        self.assertNotIn(signature, redacted)

    def test_redact_masks_more_token_prefixes(self):
        """source: github, gitlab, google, stripe and npm token prefixes are sent unmasked"""
        shapes = {
            "ghu": "ghu" + "_" + ("a" * 20),
            "glpat": "glpat" + "-" + ("a" * 20),
            "aiza": "AI" + "za" + ("b" * 35),
            "sk_live": "sk" + "_" + "live" + "_" + ("c" * 16),
            "sk_test": "sk" + "_" + "test" + "_" + ("c" * 16),
            "rk_live": "rk" + "_" + "live" + "_" + ("c" * 16),
            "npm": "npm" + "_" + ("d" * 36),
        }
        for name, token in shapes.items():
            with self.subTest(shape=name):
                redacted, count = terminal_watcher.redact("pre " + token + " post")
                self.assertEqual(count, 1)
                self.assertEqual(redacted, "pre [redacted] post")
                self.assertNotIn(token, redacted)

    def test_redact_masks_slack_webhook(self):
        """source: a Slack webhook address is sent unmasked"""
        tail = "T" + "000/" + "B" + "000/" + "secretTail99"
        prefix = "https://" + "hooks.slack.com/" + "services/"
        text = "see " + prefix + tail + " now"
        redacted, count = terminal_watcher.redact(text)
        self.assertEqual(redacted, "see " + prefix + "[redacted] now")
        self.assertEqual(count, 1)
        self.assertNotIn(tail, redacted)
        self.assertNotIn("secretTail99", redacted)

    def test_redact_leaves_ordinary_text_and_commit_ids(self):
        """source: ordinary words and a commit id are masked"""
        samples = [
            "task-runner finished in 12s",
            "key not found",
            "Enter password:",
            "token: ",
            "monkey business",
            "ab" * 20,
        ]
        for sample in samples:
            with self.subTest(sample=sample):
                self.assertEqual(terminal_watcher.redact(sample), (sample, 0))

    def test_redact_counts_a_labeled_token_once(self):
        """source: a labeled value that is already redacted is counted a second time"""
        token = "sk" + "-" + ("a" * 16)
        text = "API" + "_KEY=" + token
        redacted, count = terminal_watcher.redact(text)
        self.assertEqual(redacted, "API" + "_KEY=" + "[redacted]")
        self.assertEqual(count, 1)
        self.assertNotIn(token, redacted)

    def test_import_errors_name_uv_tool_install(self):
        """source: the import errors told the user to pip install"""
        source = Path(terminal_watcher.__file__).read_text(encoding="utf-8")
        self.assertIn(
            "anthropic package not installed. Install tw with: uv tool install .",
            source,
        )
        self.assertIn(
            "requests package not installed. Install tw with: uv tool install .",
            source,
        )
        self.assertEqual(source.count("Install tw with: uv tool install ."), 2)
        self.assertNotIn("pip", source)

    def test_interval_below_one_is_refused(self):
        """source: an interval below 1 is accepted"""
        stderr = io.StringIO()
        stdout = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            suggestions = Path(tmp) / "watcher_suggestions.md"
            with patch("terminal_watcher.TERMINAL_LOGS_DIR", Path(tmp)), \
                 patch("terminal_watcher.SUGGESTIONS_FILE", suggestions), \
                 patch("terminal_watcher.time.sleep", side_effect=KeyboardInterrupt), \
                 patch("sys.argv", ["tw", "watch", "--interval", "0"]):
                with redirect_stderr(stderr), redirect_stdout(stdout):
                    with self.assertRaises(SystemExit) as caught:
                        terminal_watcher.main()
        self.assertEqual(caught.exception.code, 2)
        self.assertIn("--interval must be at least 1", stderr.getvalue())

    def test_watch_start_names_a_missing_log(self):
        """source: tw watch starts without naming a missing log file"""
        with tempfile.TemporaryDirectory() as tmp:
            expected = _today_log(tmp)
            suggestions = Path(tmp) / "watcher_suggestions.md"
            stdout = io.StringIO()
            with patch("terminal_watcher.TERMINAL_LOGS_DIR", Path(tmp)), \
                 patch("terminal_watcher.SUGGESTIONS_FILE", suggestions), \
                 patch("terminal_watcher.time.sleep", side_effect=KeyboardInterrupt), \
                 patch("sys.argv", ["tw", "watch"]):
                with redirect_stdout(stdout):
                    terminal_watcher.main()
            output = stdout.getvalue()
            message = "   No log for today yet. Waiting for: " + str(expected)
            begin = output.index(terminal_watcher.TW_BEGIN)
            end = output.index(terminal_watcher.TW_END)
            block = output[begin:end]
            self.assertIn(message, block)
            self.assertIn("Press Ctrl+C to stop.", block)

            expected.write_text("ok\n")
            stdout = io.StringIO()
            with patch("terminal_watcher.TERMINAL_LOGS_DIR", Path(tmp)), \
                 patch("terminal_watcher.SUGGESTIONS_FILE", suggestions), \
                 patch("terminal_watcher.time.sleep", side_effect=KeyboardInterrupt), \
                 patch("sys.argv", ["tw", "watch"]):
                with redirect_stdout(stdout):
                    terminal_watcher.main()
            self.assertNotIn("No log for today yet", stdout.getvalue())


class TestMaskingCorrections(unittest.TestCase):
    def test_redact_leaves_ports_in_urls(self):
        """source: a port in a URL was masked as a password"""
        first = "curl http://localhost:3000/path?email=a@b.com ok"
        redacted, count = terminal_watcher.redact(first)
        self.assertEqual(redacted, first)
        self.assertEqual(count, 0)

        second = "GET https://registry.npmjs.org:443/@scope/pkg 200"
        redacted, count = terminal_watcher.redact(second)
        self.assertEqual(redacted, second)
        self.assertEqual(count, 0)

    def test_redact_leaves_plain_words_after_basic_and_bearer(self):
        """source: plain words after Basic and Bearer were masked, and a bearer value shorter than 16 characters was masked"""
        basic = "Basic authentication failed for user"
        redacted, count = terminal_watcher.redact(basic)
        self.assertEqual(redacted, basic)
        self.assertEqual(count, 0)

        bearer = "Bearer token is invalid"
        redacted, count = terminal_watcher.redact(bearer)
        self.assertEqual(redacted, bearer)
        self.assertEqual(count, 0)

        short = "ccccc" + "ccccc" + "ccccc"
        text = "Bearer " + short
        redacted, count = terminal_watcher.redact(text)
        self.assertEqual(redacted, text)
        self.assertEqual(count, 0)

        value = "ab" + "cd" + "ef" + "gh" + "ij" + "kl" + "mn" + "op"
        text = "Bearer " + value
        redacted, count = terminal_watcher.redact(text)
        self.assertEqual(redacted, "Bearer [redacted]")
        self.assertEqual(count, 1)
        self.assertNotIn(value, redacted)

        blob = "YQ" + "li"
        text = "authorization=Basic " + blob
        redacted, count = terminal_watcher.redact(text)
        self.assertEqual(redacted, "authorization=[redacted]")
        self.assertEqual(count, 1)
        self.assertNotIn(blob, redacted)

        text = "AUTHORIZATION: BASIC " + blob
        redacted, count = terminal_watcher.redact(text)
        self.assertEqual(redacted, "AUTHORIZATION: [redacted]")
        self.assertEqual(count, 1)
        self.assertNotIn(blob, redacted)

    def test_redact_masks_flag_values(self):
        """source: a flag value such as --password was sent unmasked"""
        cases = (
            ("--password ", "--password [redacted]"),
            ("--token ", "--token [redacted]"),
            ("--api-key ", "--api-key [redacted]"),
            ("--secret=", "--secret=[redacted]"),
        )
        value = "ab" + "cd"
        for prefix, expected in cases:
            with self.subTest(prefix=prefix):
                text = prefix + value
                redacted, count = terminal_watcher.redact(text)
                self.assertEqual(redacted, expected)
                self.assertEqual(count, 1)
                self.assertNotIn(value, redacted)

    def test_redact_leaves_empty_quoted_value(self):
        """source: an empty quoted token was masked"""
        text = 'cfg "token": "" end'
        redacted, count = terminal_watcher.redact(text)
        self.assertEqual(redacted, text)
        self.assertEqual(count, 0)

    def test_redact_quoted_value_stops_at_line_end(self):
        """source: a quoted value ran past the end of its line"""
        second = 'next line with "quote" later'
        text = 'password: "open quote\n' + second
        redacted, count = terminal_watcher.redact(text)
        self.assertEqual(count, 0)
        self.assertEqual(redacted, text)
        self.assertEqual(redacted.splitlines()[1], second)


class TestFourthPass(unittest.TestCase):
    def setUp(self):
        self._env_backup = {key: os.environ.get(key) for key in _ENV_KEYS}
        os.environ.pop("TW_EFFORT", None)
        os.environ.pop("TW_MODEL", None)

    def tearDown(self):
        for key, value in self._env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_watch_sends_the_masked_text(self):
        """source: watch mode sends the raw log to the API"""
        _clear_auto_limit(self)
        token = "sk" + "-" + ("c" * 20)
        with tempfile.TemporaryDirectory() as tmp:
            log_path = _today_log(tmp)
            log_path.write_text("Error: build failed\n" + token + "\n")
            state = _watch_state()
            mock_ai = Mock(return_value={"error_summary": "x"})
            with patch("terminal_watcher.TERMINAL_LOGS_DIR", Path(tmp)), \
                 patch("terminal_watcher.SUGGESTIONS_FILE", Path(tmp) / "notes.md"), \
                 patch("terminal_watcher.analyze_with_ai", mock_ai), \
                 patch("terminal_watcher.save_suggestions"):
                with redirect_stdout(io.StringIO()):
                    terminal_watcher.watch_step(state, True, 9_000.0)
        self.assertEqual(mock_ai.call_count, 1)
        sent = mock_ai.call_args.args[0]
        self.assertNotIn(token, sent)
        self.assertIn("[redacted]", sent)

    def test_explain_sends_the_masked_text(self):
        """source: explain, error and research send the raw log to the API"""
        token = "sk" + "-" + ("b" * 20)
        body = "Error: build failed\n" + token + "\n" + ("x" * 200)
        os.environ["ANTHROPIC_API_KEY"] = "sk" + "-ant-test"
        fake = SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text='{"summary": "ok"}')],
        )
        with tempfile.TemporaryDirectory() as tmp:
            log_path = _today_log(tmp)
            log_path.write_text(body)
            suggestions = Path(tmp) / "watcher_suggestions.md"
            for command in ("explain", "error", "research"):
                with self.subTest(command=command):
                    client = MagicMock()
                    client.messages.create.return_value = fake
                    with patch("terminal_watcher.TERMINAL_LOGS_DIR", Path(tmp)), \
                         patch("terminal_watcher.SUGGESTIONS_FILE", suggestions), \
                         patch("terminal_watcher.anthropic.Anthropic", return_value=client), \
                         patch("terminal_watcher.search_github", return_value=[]), \
                         patch("terminal_watcher.search_npm", return_value=None), \
                         patch("terminal_watcher.requests.get") as mock_get, \
                         patch("sys.argv", ["tw", command]):
                        with redirect_stdout(io.StringIO()):
                            terminal_watcher.main()
                    self.assertEqual(client.messages.create.call_count, 1)
                    prompt = client.messages.create.call_args.kwargs["messages"][0]["content"]
                    self.assertNotIn(token, prompt)
                    self.assertIn("[redacted]", prompt)
                    self.assertEqual(mock_get.call_count, 0)

    def test_search_github_tolerates_missing_fields(self):
        """source: search_github raises KeyError when a result lacks a field"""
        items = [
            {
                "html_url": "https://github.com/octo/skip",
                "description": "should not appear",
                "stargazers_count": 3,
            },
            {
                "full_name": "octo/no-url",
                "description": "absent url",
                "stargazers_count": 8,
            },
            {
                "full_name": "octo/null-desc",
                "html_url": "https://github.com/octo/null-desc",
                "description": None,
                "stargazers_count": 4,
            },
            {
                "full_name": "octo/no-stars",
                "html_url": "https://github.com/octo/no-stars",
                "description": "kept",
            },
        ]
        expected = [
            {
                "name": "octo/null-desc",
                "description": "",
                "url": "https://github.com/octo/null-desc",
                "stars": 4,
            },
            {
                "name": "octo/no-stars",
                "description": "kept",
                "url": "https://github.com/octo/no-stars",
                "stars": 0,
            },
        ]
        shaped = {
            "full_name": "octo/from-list",
            "html_url": "https://github.com/octo/from-list",
            "description": "nope",
            "stargazers_count": 9,
        }

        def response(payload):
            reply = Mock()
            reply.status_code = 200
            reply.json.return_value = payload
            return reply

        with patch(
            "terminal_watcher.requests.get",
            side_effect=[
                response({"items": items}),
                response([shaped]),
                response({"items": {"full_name": "octo/not-a-list"}}),
            ],
        ):
            found = terminal_watcher.search_github("terminal watcher")
            self.assertEqual(found, expected)
            self.assertEqual(terminal_watcher.search_github("terminal watcher"), [])
            self.assertEqual(terminal_watcher.search_github("terminal watcher"), [])

    def test_ctrl_c_in_a_one_shot_command_exits_130(self):
        """source: Ctrl+C during a one-shot command prints a traceback"""
        with tempfile.TemporaryDirectory() as tmp:
            log_path = _today_log(tmp)
            log_path.write_text("Error: build failed\n" + ("x" * 200))
            suggestions = Path(tmp) / "watcher_suggestions.md"
            stdout = io.StringIO()
            stderr = io.StringIO()
            with patch("terminal_watcher.TERMINAL_LOGS_DIR", Path(tmp)), \
                 patch("terminal_watcher.SUGGESTIONS_FILE", suggestions), \
                 patch(
                     "terminal_watcher.analyze_with_ai",
                     side_effect=KeyboardInterrupt,
                 ), \
                 patch("sys.argv", ["tw", "explain"]):
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    with self.assertRaises(SystemExit) as caught:
                        terminal_watcher.main()
        self.assertEqual(caught.exception.code, 130)
        self.assertIn("Stopped.", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())
        lines = stdout.getvalue().splitlines()
        self.assertIn(terminal_watcher.TW_END, lines)
        self.assertNotIn("Traceback", stdout.getvalue())

    def test_chars_below_100_is_refused(self):
        """source: a chars value below 100 is accepted"""
        for raw in ("0", "-1", "99"):
            with self.subTest(chars=raw):
                stderr = io.StringIO()
                stdout = io.StringIO()
                with tempfile.TemporaryDirectory() as tmp:
                    suggestions = Path(tmp) / "watcher_suggestions.md"
                    with patch("terminal_watcher.TERMINAL_LOGS_DIR", Path(tmp)), \
                         patch("terminal_watcher.SUGGESTIONS_FILE", suggestions), \
                         patch("terminal_watcher.analyze_with_ai") as mock_ai, \
                         patch("sys.argv", ["tw", "explain", "--chars", raw]):
                        with redirect_stderr(stderr), redirect_stdout(stdout):
                            with self.assertRaises(SystemExit) as caught:
                                terminal_watcher.main()
                self.assertEqual(caught.exception.code, 2)
                self.assertIn("--chars must be at least 100", stderr.getvalue())
                self.assertEqual(mock_ai.call_count, 0)


class TestSixthPass(unittest.TestCase):
    def test_detect_errors_returns_the_newest_twenty(self):
        """source: detect_errors keeps the first 20 matching lines, so the newest errors are the ones dropped"""
        _clear_auto_limit(self)
        lines = ["error: failure %s" % index for index in range(30)]
        log = "\n".join(lines)
        self.assertLess(len(log), 2000)
        found = terminal_watcher.detect_errors(log)
        self.assertEqual(found, lines[-20:])
        self.assertEqual(found[-1], lines[-1])
        self.assertNotIn(lines[0], found)

        mock_ai = Mock(return_value={"error_summary": "x"})
        with tempfile.TemporaryDirectory() as tmp:
            log_path = _today_log(tmp)
            log_path.write_text(log + "\n")
            state = _watch_state()
            with patch("terminal_watcher.TERMINAL_LOGS_DIR", Path(tmp)), \
                 patch("terminal_watcher.analyze_with_ai", mock_ai), \
                 patch("terminal_watcher.save_suggestions"):
                with redirect_stdout(io.StringIO()):
                    terminal_watcher.watch_step(state, True, 10_000.0)
        self.assertEqual(mock_ai.call_count, 1)
        reported = mock_ai.call_args.kwargs["errors"]
        self.assertEqual(reported, lines[-20:])
        self.assertIn(lines[-1], reported)
        self.assertIn(lines[-1], state["reported"])
        self.assertNotIn(lines[0], state["reported"])

    def test_safe_watch_step_reports_unexpected_error_once(self):
        """source: an exception in a watch pass that is not a TWError or an OSError ends tw watch"""
        state = _watch_state()
        line = "Error: unexpected RuntimeError: boom (watch continues)"
        stdout = io.StringIO()
        caught = None
        try:
            with patch("terminal_watcher.watch_step", side_effect=RuntimeError("boom")):
                with redirect_stdout(stdout):
                    for _ in range(3):
                        terminal_watcher.safe_watch_step(state, False, 1.0)
        except Exception as exc:
            caught = exc
        self.assertIsNone(caught)
        self.assertEqual(stdout.getvalue().count(line), 1)
        self.assertIs(state.get("unexpected_error"), RuntimeError)

        with patch("terminal_watcher.watch_step", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                terminal_watcher.safe_watch_step(_watch_state(), False, 2.0)

    def test_watch_mode_uses_the_guarded_step(self):
        """source: the watch loop could call the unguarded step and one unexpected error would end tw watch"""
        calls = {"count": 0}

        def fake_watch_step(state, auto_research, now):
            calls["count"] += 1
            if calls["count"] == 1:
                raise RuntimeError("boom")
            return None

        def fake_sleep(interval):
            fake_sleep.calls += 1
            if fake_sleep.calls == 1:
                return None
            raise KeyboardInterrupt

        fake_sleep.calls = 0
        line = "Error: unexpected RuntimeError: boom (watch continues)"
        stdout = io.StringIO()
        caught = None
        try:
            with tempfile.TemporaryDirectory() as tmp:
                with patch("terminal_watcher.TERMINAL_LOGS_DIR", Path(tmp)), \
                     patch("terminal_watcher.watch_step", side_effect=fake_watch_step), \
                     patch("terminal_watcher.time.sleep", side_effect=fake_sleep):
                    with redirect_stdout(stdout):
                        terminal_watcher.watch_mode(auto_research=False, interval=1)
        except BaseException as exc:
            caught = exc
        self.assertIsNone(caught)
        self.assertEqual(calls["count"], 2)
        self.assertEqual(stdout.getvalue().count(line), 1)
        self.assertIn("Terminal Watcher stopped.", stdout.getvalue())

    def test_read_recent_log_window_may_start_mid_character(self):
        """source: read_recent_log seeks to a byte offset in a file opened as text, which Python does not define"""
        drawing = "\u2500" * 5000
        emoji = "\U0001F600"
        last = "last line intact " + emoji
        raw = (drawing + emoji + "\n" + last + "\n").encode("utf-8")
        window = len(raw) - 1
        self.assertEqual(raw[1] & 0xC0, 0x80)
        self.assertGreater(len(raw), window)
        with tempfile.TemporaryDirectory() as tmp:
            log_path = _today_log(tmp)
            log_path.write_bytes(raw)
            caught = None
            result = None
            try:
                with patch("terminal_watcher.TERMINAL_LOGS_DIR", Path(tmp)):
                    result = terminal_watcher.read_recent_log(window)
            except Exception as exc:
                caught = exc
        self.assertIsNone(caught)
        self.assertIsNotNone(result)
        self.assertEqual(result.splitlines()[-1], last)
        self.assertNotIn("\ufffd", result.splitlines()[-1])

    def test_unmatched_begin_hides_at_most_200_lines(self):
        """source: a begin marker with no end marker hides everything after it until it leaves the window"""
        begin = terminal_watcher.TW_BEGIN
        end = terminal_watcher.TW_END
        ordinary = ["row %03d" % index for index in range(300)]
        unmatched = begin + "\n" + "\n".join(ordinary) + "\n"
        result = terminal_watcher.strip_own_output(unmatched)
        self.assertEqual(result.splitlines(), ordinary[200:])
        self.assertEqual(len(result.splitlines()), 100)
        self.assertNotIn(begin, result)
        self.assertNotIn(ordinary[0], result.splitlines())
        self.assertIn(ordinary[-1], result.splitlines())

        blocked = ["blocked %03d" % index for index in range(300)]
        complete = (
            "before\n" + begin + "\n" + "\n".join(blocked) + "\n" + end + "\nafter\n"
        )
        result = terminal_watcher.strip_own_output(complete)
        self.assertEqual(result, "before\nafter\n")
        self.assertNotIn(blocked[0], result)
        self.assertNotIn(blocked[-1], result)

    def test_outside_text_is_printed_without_controls(self):
        """source: text from the model and from GitHub search is printed as received, so it can carry terminal escape sequences"""
        esc = "\x1b[31m"
        c1 = "\u009b"
        summary = "hello " + "\x00" + esc + "world" + c1 + "\x7f"
        output = terminal_watcher.format_output(
            {
                "summary": summary,
                "current_task": "task",
                "explanation": "keep\n\tthis",
            },
            mode="explain",
        )
        self.assertNotIn("\x1b", output)
        self.assertNotIn("\x00", output)
        self.assertNotIn(c1, output)
        self.assertNotIn("\x7f", output)
        self.assertIn("SUMMARY: hello [31mworld", output)
        self.assertIn("keep\n\tthis", output)

        raw = terminal_watcher.format_output(
            {"raw_response": "see " + esc + "this" + c1},
            mode="explain",
        )
        self.assertNotIn("\x1b", raw)
        self.assertNotIn(c1, raw)
        self.assertIn("see [31mthis", raw)

        dirty = "alpha " + "\x1b[32m" + "beta"
        research = {
            "github_repos": [
                {
                    "name": "octo/repo",
                    "description": dirty,
                    "url": "https://github.com/octo/repo",
                    "stars": 3,
                }
            ],
            "npm_packages": [],
        }
        rendered = terminal_watcher.format_output(
            {"summary": "s", "advice": "a"},
            research=research,
            mode="research",
        )
        self.assertNotIn("\x1b", rendered)
        self.assertIn("alpha [32mbeta", rendered)

    def test_search_npm_non_object_returns_none(self):
        """source: a registry answer that is not an object, or whose dist-tags is not an object, raises"""
        payloads = (
            ["left-pad"],
            {"name": "left-pad", "dist-tags": None},
        )
        for payload in payloads:
            with self.subTest(payload=payload):
                response = Mock()
                response.status_code = 200
                response.json.return_value = payload
                caught = None
                result = "unset"
                try:
                    with patch("terminal_watcher.requests.get", return_value=response):
                        result = terminal_watcher.search_npm("left-pad")
                except Exception as exc:
                    caught = exc
                self.assertIsNone(caught)
                self.assertIsNone(result)

    def test_chars_above_200000_is_refused(self):
        """source: --chars above 200000 is accepted"""
        stderr = io.StringIO()
        stdout = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            suggestions = Path(tmp) / "watcher_suggestions.md"
            with patch("terminal_watcher.TERMINAL_LOGS_DIR", Path(tmp)), \
                 patch("terminal_watcher.SUGGESTIONS_FILE", suggestions), \
                 patch("terminal_watcher.analyze_with_ai") as mock_ai, \
                 patch("sys.argv", ["tw", "explain", "--chars", "200001"]):
                with redirect_stderr(stderr), redirect_stdout(stdout):
                    with self.assertRaises(SystemExit) as caught:
                        terminal_watcher.main()
        self.assertEqual(caught.exception.code, 2)
        self.assertIn("--chars must be at most 200000", stderr.getvalue())
        self.assertEqual(mock_ai.call_count, 0)

    def test_redact_masks_authorization_token_and_passphrase(self):
        """source: an Authorization scheme other than Bearer or Basic, and a passphrase label, are sent unmasked"""
        token = "t" * 12
        text = "Authorization: Token " + token
        redacted, count = terminal_watcher.redact(text)
        self.assertEqual(redacted, "Authorization: [redacted]")
        self.assertEqual(count, 1)
        self.assertNotIn(token, redacted)

        api_key = "k" * 12
        text = "Authorization: ApiKey " + api_key
        redacted, count = terminal_watcher.redact(text)
        self.assertEqual(redacted, "Authorization: [redacted]")
        self.assertEqual(count, 1)
        self.assertNotIn(api_key, redacted)

        secret = "p" * 8
        text = "my_passphrase=" + secret
        redacted, count = terminal_watcher.redact(text)
        self.assertEqual(redacted, "my_passphrase=[redacted]")
        self.assertEqual(count, 1)
        self.assertNotIn(secret, redacted)

    def test_save_suggestions_does_not_follow_a_symlink(self):
        """source: save_suggestions follows a symbolic link and writes the suggestions into its target"""
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "target.md"
            target.write_text("original contents")
            link = Path(tmp) / "watcher_suggestions.md"
            link.symlink_to(target)
            with patch("terminal_watcher.SUGGESTIONS_FILE", link):
                with self.assertRaises(OSError):
                    terminal_watcher.save_suggestions("new suggestions")
            self.assertEqual(target.read_text(), "original contents")
            self.assertTrue(link.is_symlink())


class TestReviewFixes(unittest.TestCase):
    def test_search_npm_cleans_version(self):
        """source: the npm version string is printed without being cleaned of control characters, though the name and description are"""
        response = Mock()
        response.status_code = 200
        response.json.return_value = {
            "name": "left-pad",
            "description": "pad",
            "dist-tags": {"latest": "1.2.3" + "\x1b" + "[31m"},
            "homepage": "https://example.com",
        }
        with patch("terminal_watcher.requests.get", return_value=response):
            info = terminal_watcher.search_npm("left-pad")
        self.assertEqual(info["version"], "1.2.3[31m")
        self.assertNotIn("\x1b", info["version"])

        response.json.return_value = {
            "name": "left-pad",
            "description": "pad",
            "dist-tags": {"latest": ["1.2.3"]},
            "homepage": "https://example.com",
        }
        with patch("terminal_watcher.requests.get", return_value=response):
            info = terminal_watcher.search_npm("left-pad")
        self.assertEqual(info["version"], "")

    def test_save_suggestions_without_o_nofollow(self):
        """source: os.O_NOFOLLOW does not exist on every platform, so save_suggestions can raise AttributeError"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "watcher_suggestions.md"
            with patch.dict(os.__dict__):
                del os.O_NOFOLLOW
                self.assertFalse(hasattr(os, "O_NOFOLLOW"))
                with patch("terminal_watcher.SUGGESTIONS_FILE", path):
                    caught = None
                    try:
                        terminal_watcher.save_suggestions("kept text")
                    except Exception as exc:
                        caught = exc
                    self.assertIsNone(caught)
            self.assertTrue(path.is_file())
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertIn("kept text", path.read_text())
        self.assertTrue(hasattr(os, "O_NOFOLLOW"))

    def test_redact_masks_the_rest_of_an_authorization_line(self):
        """source: the Authorization masking has three rules and still lets a bare token through when words follow it"""
        before = "line before stays"
        after = "line after stays"
        opaque = "q" + "7opaque" + "zz"
        credential = "c" + "redValue" + "42"
        token = "t" + "okenValue" + "42"
        bearer = "b" + ("e" * 16)
        cases = (
            (
                "Authorization: " + opaque + " trailing words",
                "Authorization: [redacted]",
                opaque,
            ),
            (
                "Authorization: AWS4-HMAC-SHA256 Credential=" + credential,
                "Authorization: [redacted]",
                credential,
            ),
            (
                "authorization = Token " + token,
                "authorization = [redacted]",
                token,
            ),
            (
                "Authorization: Bearer " + bearer,
                "Authorization: [redacted]",
                bearer,
            ),
        )
        for line, expected, secret in cases:
            with self.subTest(line=expected):
                text = before + "\n" + line + "\n" + after
                redacted, count = terminal_watcher.redact(text)
                self.assertEqual(count, 1)
                self.assertEqual(redacted, before + "\n" + expected + "\n" + after)
                self.assertNotIn(secret, redacted)
                self.assertNotIn("trailing words", redacted)
                self.assertEqual(redacted.splitlines()[0], before)
                self.assertEqual(redacted.splitlines()[-1], after)

    def test_format_output_drops_format_characters(self):
        """source: direction-override and zero-width characters survive the cleaning of printed text"""
        summary = "ab" + "\u202e" + "cd" + "\u200b" + "ef" + "\u2028" + "gh" + "\u2029" + "ij"
        output = terminal_watcher.format_output(
            {
                "summary": summary,
                "current_task": "task",
                "explanation": "ok",
            },
            mode="explain",
        )
        self.assertNotIn("\u202e", output)
        self.assertNotIn("\u200b", output)
        self.assertNotIn("\u2028", output)
        self.assertNotIn("\u2029", output)
        self.assertIn("SUMMARY: abcdefghij", output)

    def test_safe_watch_step_reports_once_per_exception_class(self):
        """source: an unexpected error whose text changes on each pass is printed on every pass, and its text is not masked"""
        state = _watch_state()
        errors = [
            RuntimeError("pass 1"),
            RuntimeError("pass 2"),
            RuntimeError("pass 3"),
            ValueError("other"),
        ]
        stdout = io.StringIO()
        with patch("terminal_watcher.watch_step", side_effect=errors):
            with redirect_stdout(stdout):
                for _ in range(4):
                    terminal_watcher.safe_watch_step(state, False, 1.0)
        output = stdout.getvalue()
        self.assertEqual(output.count("Error: unexpected RuntimeError:"), 1)
        self.assertEqual(output.count("Error: unexpected ValueError:"), 1)
        self.assertIn("pass 1", output)
        self.assertNotIn("pass 2", output)
        self.assertNotIn("pass 3", output)
        self.assertIn("other", output)
        self.assertIs(state.get("unexpected_error"), ValueError)

        token = "sk" + "-" + ("b" * 16)
        state = _watch_state()
        stdout = io.StringIO()
        with patch("terminal_watcher.watch_step", side_effect=RuntimeError("saw " + token)):
            with redirect_stdout(stdout):
                terminal_watcher.safe_watch_step(state, False, 2.0)
        output = stdout.getvalue()
        self.assertIn("[redacted]", output)
        self.assertNotIn(token, output)
        self.assertIn(
            "Error: unexpected RuntimeError: saw [redacted] (watch continues)",
            output,
        )


if __name__ == "__main__":
    unittest.main()

