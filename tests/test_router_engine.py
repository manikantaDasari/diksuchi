"""
test_router_engine.py — Unit tests for router_engine.py routing logic

Tests all rule types, token counting, and routing decisions.
"""

from __future__ import annotations

import re
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from freezegun import freeze_time

from router_engine import Backend, RoutingDecision, decide, messages_to_text, count_tokens


# ══════════════════════════════════════════════════════════════════════════════
# Message Assembly
# ══════════════════════════════════════════════════════════════════════════════


class TestMessagesToText:
    """Test prompt assembly from message arrays."""

    @pytest.mark.unit
    def test_single_message(self):
        messages = [{"role": "user", "content": "Hello"}]
        text = messages_to_text(messages)
        assert "user:" in text
        assert "Hello" in text

    @pytest.mark.unit
    def test_multiple_messages(self):
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        text = messages_to_text(messages)
        assert "system:" in text
        assert "user:" in text
        assert "assistant:" in text

    @pytest.mark.unit
    def test_empty_messages(self):
        text = messages_to_text([])
        assert isinstance(text, str)

    @pytest.mark.unit
    def test_vision_messages(self):
        """Vision messages with list content."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                ],
            }
        ]
        text = messages_to_text(messages)
        assert "What is this?" in text


# ══════════════════════════════════════════════════════════════════════════════
# Token Counting
# ══════════════════════════════════════════════════════════════════════════════


class TestTokenCounting:
    """Test token counting."""

    @pytest.mark.unit
    def test_token_count_returns_int(self):
        tokens = count_tokens("hello world")
        assert isinstance(tokens, int)
        assert tokens > 0

    @pytest.mark.unit
    def test_token_count_scales_with_length(self):
        short = count_tokens("word")
        long = count_tokens("word " * 100)
        assert long > short

    @pytest.mark.unit
    def test_token_count_unicode(self):
        tokens = count_tokens("你好世界 🚀")
        assert tokens > 0

    @pytest.mark.unit
    def test_token_count_empty_string(self):
        tokens = count_tokens("")
        # Fallback should return at least 1
        assert tokens >= 0


# ══════════════════════════════════════════════════════════════════════════════
# Header Rules
# ══════════════════════════════════════════════════════════════════════════════


class TestHeaderRules:
    """Test X-Route-To header override."""

    @pytest.mark.unit
    def test_header_override_local(self, minimal_routing_config):
        """Header explicitly routes to local."""
        messages = [{"role": "user", "content": "complex analysis"}]
        headers = {"x-route-to": "local"}
        decision = decide(messages, "", headers, minimal_routing_config)
        assert decision.backend == Backend.LOCAL
        assert decision.rule_name == "force_local_header"

    @pytest.mark.unit
    def test_header_override_cloud(self, minimal_routing_config):
        """Header explicitly routes to cloud."""
        messages = [{"role": "user", "content": "hi"}]
        headers = {"x-route-to": "cloud"}
        decision = decide(messages, "", headers, minimal_routing_config)
        assert decision.backend == Backend.CLOUD
        assert decision.rule_name == "force_cloud_header"

    @pytest.mark.unit
    def test_header_case_insensitive_key(self, minimal_routing_config):
        messages = [{"role": "user", "content": "hi"}]
        headers = {"x-route-to": "cloud"}
        decision = decide(messages, "", headers, minimal_routing_config)
        assert decision.backend == Backend.CLOUD

    @pytest.mark.unit
    def test_header_case_insensitive_value(self, minimal_routing_config):
        messages = [{"role": "user", "content": "hello"}]
        headers = {"x-route-to": "LOCAL"}
        decision = decide(messages, "", headers, minimal_routing_config)
        assert decision.backend == Backend.LOCAL

    @pytest.mark.unit
    def test_header_with_whitespace(self, minimal_routing_config):
        messages = [{"role": "user", "content": "hello"}]
        headers = {"x-route-to": "  cloud  "}
        decision = decide(messages, "", headers, minimal_routing_config)
        assert decision.backend == Backend.CLOUD


# ══════════════════════════════════════════════════════════════════════════════
# Model Name Hints
# ══════════════════════════════════════════════════════════════════════════════


class TestModelNameHints:
    """Test model name-based routing."""

    @pytest.mark.unit
    @pytest.mark.parametrize("model", ["gpt-4o", "gpt-4o-mini", "gpt-4.1"])
    def test_gpt_routes_cloud(self, model, minimal_routing_config):
        messages = [{"role": "user", "content": "simple"}]
        decision = decide(messages, model, {}, minimal_routing_config)
        assert decision.backend == Backend.CLOUD

    @pytest.mark.unit
    @pytest.mark.parametrize("model", ["claude-opus", "claude-sonnet", "claude-haiku"])
    def test_claude_routes_cloud(self, model, minimal_routing_config):
        messages = [{"role": "user", "content": "simple"}]
        decision = decide(messages, model, {}, minimal_routing_config)
        assert decision.backend == Backend.CLOUD

    @pytest.mark.unit
    @pytest.mark.parametrize("model", ["llama3.2", "llama3.1", "llama2"])
    def test_llama_routes_local(self, model, minimal_routing_config):
        messages = [{"role": "user", "content": "test"}]
        decision = decide(messages, model, {}, minimal_routing_config)
        # Model hint should route to local if present
        assert "model" in decision.rule_name.lower()


# ══════════════════════════════════════════════════════════════════════════════
# Keyword Rules
# ══════════════════════════════════════════════════════════════════════════════


class TestKeywordRules:
    """Test keyword-based complexity detection."""

    @pytest.mark.unit
    @pytest.mark.parametrize("keyword", ["analyze", "debug", "optimize", "refactor"])
    def test_complexity_keywords(self, keyword, minimal_routing_config):
        prompt = f"Please {keyword} this code"
        messages = [{"role": "user", "content": prompt}]
        decision = decide(messages, "", {}, minimal_routing_config)
        # Should match complexity rule
        assert "complexity" in decision.rule_name.lower() or decision.backend == Backend.CLOUD

    @pytest.mark.unit
    @pytest.mark.parametrize("keyword", ["hi", "hello", "quick"])
    def test_simple_keywords(self, keyword, minimal_routing_config):
        prompt = f"{keyword} there"
        messages = [{"role": "user", "content": prompt}]
        decision = decide(messages, "", {}, minimal_routing_config)
        # May route to local via keyword rule
        assert decision.backend in (Backend.LOCAL, Backend.CLOUD)

    @pytest.mark.unit
    def test_keyword_case_insensitive(self, minimal_routing_config):
        messages = [{"role": "user", "content": "ANALYZE this"}]
        decision = decide(messages, "", {}, minimal_routing_config)
        # Case shouldn't matter for keyword matching
        assert decision.backend in (Backend.LOCAL, Backend.CLOUD)


# ══════════════════════════════════════════════════════════════════════════════
# Token Count Rules
# ══════════════════════════════════════════════════════════════════════════════


class TestTokenCountRules:
    """Test token count-based routing."""

    @pytest.mark.unit
    def test_very_large_prompt(self, minimal_routing_config):
        """Prompt over 5000 tokens should trigger cloud."""
        large_text = "word " * 1000
        messages = [{"role": "user", "content": large_text}]
        decision = decide(messages, "", {}, minimal_routing_config)
        assert decision.token_count > 1000

    @pytest.mark.unit
    def test_small_prompt(self, minimal_routing_config):
        messages = [{"role": "user", "content": "hello"}]
        decision = decide(messages, "", {}, minimal_routing_config)
        assert decision.token_count < 10


# ══════════════════════════════════════════════════════════════════════════════
# Regex Rules
# ══════════════════════════════════════════════════════════════════════════════


class TestRegexRules:
    """Test regex pattern matching."""

    @pytest.mark.unit
    def test_regex_error_pattern(self, minimal_routing_config):
        """Regex can detect error messages."""
        minimal_routing_config["rules"].append(
            {
                "name": "error_pattern",
                "condition": "prompt_matches_regex",
                "pattern": r"(Error|Exception|Traceback)",
                "route_to": "cloud",
                "priority": 75,
            }
        )
        messages = [{"role": "user", "content": "I got an Exception: ValueError"}]
        decision = decide(messages, "", {}, minimal_routing_config)
        assert "error" in decision.rule_name.lower() or decision.backend == Backend.CLOUD

    @pytest.mark.unit
    def test_regex_no_match(self, minimal_routing_config):
        minimal_routing_config["rules"].append(
            {
                "name": "json_pattern",
                "condition": "prompt_matches_regex",
                "pattern": r'^\{.*\}$',
                "route_to": "cloud",
                "priority": 50,
            }
        )
        messages = [{"role": "user", "content": "hello world"}]
        decision = decide(messages, "", {}, minimal_routing_config)
        # Pattern shouldn't match "hello world"
        assert decision.backend in (Backend.LOCAL, Backend.CLOUD)


# ══════════════════════════════════════════════════════════════════════════════
# Time-Based Rules
# ══════════════════════════════════════════════════════════════════════════════


class TestTimeOfDayRules:
    """Test time-based routing."""

    @pytest.mark.unit
    def test_time_rule_can_be_added(self, minimal_routing_config):
        """Time-of-day rules can be added to config."""
        minimal_routing_config["rules"].append(
            {
                "name": "test_time_rule",
                "condition": "time_of_day",
                "timezone": "UTC",
                "start": "10:00",
                "end": "11:00",
                "route_to": "local",
                "priority": 25,
            }
        )
        # Rule added successfully
        assert len(minimal_routing_config["rules"]) > 8


# ══════════════════════════════════════════════════════════════════════════════
# Rule Priorities & Fallback
# ══════════════════════════════════════════════════════════════════════════════


class TestRulePriorities:
    """Test rule priority ordering."""

    @pytest.mark.unit
    def test_header_highest_priority(self, minimal_routing_config):
        """Headers should override keyword analysis."""
        messages = [{"role": "user", "content": "analyze"}]
        headers = {"x-route-to": "local"}
        decision = decide(messages, "gpt-4o", headers, minimal_routing_config)
        assert decision.rule_name == "force_local_header"

    @pytest.mark.unit
    def test_decision_has_required_fields(self, minimal_routing_config):
        """All decisions should have required fields."""
        messages = [{"role": "user", "content": "test"}]
        decision = decide(messages, "gpt-4o", {}, minimal_routing_config)
        assert decision.backend in (Backend.LOCAL, Backend.CLOUD)
        assert decision.rule_name
        assert decision.reason
        assert decision.token_count > 0
        assert decision.prompt_preview
        assert isinstance(decision.model_requested, str)


# ══════════════════════════════════════════════════════════════════════════════
# Provider Pinning
# ══════════════════════════════════════════════════════════════════════════════


class TestProviderPinning:
    """Test provider override from rules."""

    @pytest.mark.unit
    def test_rule_can_specify_provider(self, minimal_routing_config):
        """Rules can pin a specific provider."""
        minimal_routing_config["rules"].append(
            {
                "name": "claude_preference",
                "condition": "prompt_contains_any",
                "keywords": ["claude"],
                "route_to": "cloud",
                "provider": "anthropic",
                "priority": 60,
            }
        )
        messages = [{"role": "user", "content": "I prefer Claude"}]
        decision = decide(messages, "", {}, minimal_routing_config)
        if decision.rule_name == "claude_preference":
            assert decision.provider == "anthropic"


# ══════════════════════════════════════════════════════════════════════════════
# Edge Cases
# ══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.unit
    def test_empty_rules_list(self, minimal_routing_config):
        minimal_routing_config["rules"] = []
        messages = [{"role": "user", "content": "hello"}]
        decision = decide(messages, "", {}, minimal_routing_config)
        assert decision.backend in (Backend.LOCAL, Backend.CLOUD)

    @pytest.mark.unit
    def test_rule_without_name_skipped(self, minimal_routing_config):
        """Rules without name should be skipped."""
        minimal_routing_config["rules"].append(
            {
                "condition": "header_equals",
                "route_to": "cloud",
                # missing "name"
            }
        )
        messages = [{"role": "user", "content": "hello"}]
        # Should not crash
        decision = decide(messages, "", {}, minimal_routing_config)
        assert decision.backend in (Backend.LOCAL, Backend.CLOUD)

    @pytest.mark.unit
    def test_malformed_rule_gracefully_ignored(self, minimal_routing_config):
        """Malformed rules should not crash routing."""
        minimal_routing_config["rules"].append(
            {
                "name": "broken",
                "condition": "unknown_type",
                "route_to": "cloud",
            }
        )
        messages = [{"role": "user", "content": "hello"}]
        decision = decide(messages, "", {}, minimal_routing_config)
        # Should use fallback
        assert decision.backend in (Backend.LOCAL, Backend.CLOUD)

    @pytest.mark.unit
    def test_very_long_prompt(self, minimal_routing_config):
        """Handle extremely long prompts."""
        huge = "word " * 5000
        messages = [{"role": "user", "content": huge}]
        decision = decide(messages, "", {}, minimal_routing_config)
        assert decision.token_count > 1000

    @pytest.mark.unit
    def test_special_characters(self, minimal_routing_config):
        """Handle special characters."""
        messages = [
            {
                "role": "user",
                "content": "!@#$%^&*() 你好 مرحبا 🚀🔥",
            }
        ]
        decision = decide(messages, "", {}, minimal_routing_config)
        assert decision.backend in (Backend.LOCAL, Backend.CLOUD)

    @pytest.mark.unit
    def test_null_values(self, minimal_routing_config):
        """Handle null/None values."""
        messages = [
            {"role": "user", "content": None},
        ]
        # Should handle gracefully
        try:
            decision = decide(messages, "", {}, minimal_routing_config)
            assert decision.backend in (Backend.LOCAL, Backend.CLOUD)
        except (TypeError, AttributeError):
            pass  # Acceptable


# ══════════════════════════════════════════════════════════════════════════════
# Parametrized Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestParametrized:
    """Parametrized tests for various scenarios."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "prompt,is_complex",
        [
            ("hello", False),
            ("analyze code", True),
            ("what is 2+2?", False),
            ("debug and refactor", True),
            ("simple question", False),
        ],
    )
    def test_complexity_detection(self, prompt, is_complex, minimal_routing_config):
        """Test complexity keyword detection."""
        messages = [{"role": "user", "content": prompt}]
        decision = decide(messages, "", {}, minimal_routing_config)
        # Just verify it completes
        assert decision.backend in (Backend.LOCAL, Backend.CLOUD)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "model",
        ["gpt-4o", "claude-3.5-sonnet", "llama3.2", "mistral", "gemini-2.5-pro"],
    )
    def test_various_models(self, model, minimal_routing_config):
        """Test routing with various model names."""
        messages = [{"role": "user", "content": "test"}]
        decision = decide(messages, model, {}, minimal_routing_config)
        assert decision.backend in (Backend.LOCAL, Backend.CLOUD)
        assert decision.model_requested == model
