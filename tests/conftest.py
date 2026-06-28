"""
conftest.py — Shared fixtures and configuration for all tests.

Provides:
  - FastAPI TestClient with real config
  - Mock httpx clients
  - Routing config fixtures
  - Provider config fixtures
  - Common test data
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from fastapi.testclient import TestClient
from httpx import AsyncClient

# Ensure the main module can be imported
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from router_engine import Backend, RoutingDecision


# ─── Config Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def minimal_routing_config() -> dict:
    """Minimal routing configuration for tests."""
    return {
        "default_backend": "local",
        "fallback_chain": ["local", "cloud"],
        "classifier_confidence_threshold": 0.35,
        "rules": [
            {
                "name": "force_local_header",
                "condition": "header_equals",
                "header": "X-Route-To",
                "value": "local",
                "route_to": "local",
                "reason": "Forced local via header",
                "priority": 100,
            },
            {
                "name": "force_cloud_header",
                "condition": "header_equals",
                "header": "X-Route-To",
                "value": "cloud",
                "route_to": "cloud",
                "reason": "Forced cloud via header",
                "priority": 100,
            },
            {
                "name": "gpt_model_hint",
                "condition": "model_name_contains",
                "values": ["gpt-"],
                "route_to": "cloud",
                "reason": "GPT model specified",
                "priority": 90,
            },
            {
                "name": "claude_model_hint",
                "condition": "model_name_contains",
                "values": ["claude-"],
                "route_to": "cloud",
                "reason": "Claude model specified",
                "priority": 90,
            },
            {
                "name": "llama_model_hint",
                "condition": "model_name_contains",
                "values": ["llama"],
                "route_to": "local",
                "reason": "Llama model specified",
                "priority": 90,
            },
            {
                "name": "complexity_keywords",
                "condition": "prompt_contains_any",
                "keywords": ["analyze", "debug", "optimize", "refactor", "architecture"],
                "route_to": "cloud",
                "reason": "High complexity keywords detected",
                "priority": 50,
            },
            {
                "name": "simple_keywords",
                "condition": "prompt_contains_any",
                "keywords": ["hi", "hello", "quick", "simple", "what"],
                "route_to": "local",
                "reason": "Simple keywords detected",
                "priority": 40,
            },
            {
                "name": "large_prompt",
                "condition": "token_count_gte",
                "threshold": 2000,
                "route_to": "cloud",
                "reason": "Prompt exceeds 2000 tokens",
                "priority": 30,
            },
        ],
        "centroid_buckets": {
            "simple": {"tier": 0},
            "coding": {"tier": 1},
            "reasoning": {"tier": 1},
            "creative": {"tier": 0},
            "critical": {"tier": 2},
        },
    }


@pytest.fixture
def minimal_providers_config() -> dict:
    """Minimal providers configuration for tests."""
    return {
        "local": {
            "active": "ollama",
            "list": [
                {
                    "name": "ollama",
                    "type": "ollama",
                    "base_url": "http://localhost:11434",
                    "default_model": "llama3.2",
                    "timeout_seconds": 120,
                    "enabled": True,
                    "models": ["llama3.2", "mistral", "phi4"],
                }
            ],
        },
        "cloud": {
            "active": "openai",
            "list": [
                {
                    "name": "openai",
                    "type": "openai",
                    "base_url": "https://api.openai.com/v1",
                    "default_model": "gpt-4o-mini",
                    "api_key_env": "OPENAI_API_KEY",
                    "timeout_seconds": 60,
                    "enabled": True,
                    "models": ["gpt-4o", "gpt-4o-mini", "gpt-4.1"],
                },
                {
                    "name": "anthropic",
                    "type": "anthropic",
                    "base_url": "https://api.anthropic.com",
                    "default_model": "claude-haiku-4-5",
                    "api_key_env": "ANTHROPIC_API_KEY",
                    "timeout_seconds": 60,
                    "enabled": False,
                    "models": ["claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5"],
                },
            ],
        },
    }


@pytest.fixture
def minimal_config(minimal_routing_config, minimal_providers_config) -> dict:
    """Complete minimal config for tests."""
    return {
        "server": {
            "host": "0.0.0.0",
            "port": 8080,
            "log_level": "info",
        },
        "observability": {
            "log_routing_decisions": True,
            "request_log_size": 200,
            "stats_enabled": True,
        },
        "routing": minimal_routing_config,
        "providers": minimal_providers_config,
    }


# ─── Mock HTTP Fixtures ───────────────────────────────────────────────────────

@pytest.fixture
def mock_ollama_response() -> dict:
    """Typical Ollama /api/chat response."""
    return {
        "model": "llama3.2",
        "message": {
            "role": "assistant",
            "content": "This is a local response from Ollama.",
        },
        "done": True,
        "prompt_eval_count": 15,
        "eval_count": 12,
        "total_duration": 2000000000,
        "load_duration": 500000000,
        "prompt_eval_duration": 1000000000,
        "eval_duration": 500000000,
    }


@pytest.fixture
def mock_openai_response() -> dict:
    """Typical OpenAI /v1/chat/completions response."""
    return {
        "id": "chatcmpl-mock123456",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a cloud response from OpenAI.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 25,
            "completion_tokens": 12,
            "total_tokens": 37,
        },
    }


@pytest.fixture
def mock_anthropic_response() -> dict:
    """Typical Anthropic /v1/messages response."""
    return {
        "id": "msg_mock123456",
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "This is a cloud response from Anthropic.",
            }
        ],
        "model": "claude-haiku-4-5",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 25,
            "output_tokens": 12,
        },
    }


# ─── Test Message Fixtures ────────────────────────────────────────────────────

@pytest.fixture
def simple_message() -> list[dict]:
    """Simple test message."""
    return [{"role": "user", "content": "Hi, what is 2+2?"}]


@pytest.fixture
def complex_message() -> list[dict]:
    """Complex test message."""
    return [
        {
            "role": "user",
            "content": "Analyze and debug the following architecture: we have a microservices system with 50+ services. Optimize our deployment pipeline.",
        }
    ]


@pytest.fixture
def large_message() -> list[dict]:
    """Large prompt (>2000 tokens)."""
    large_text = "word " * 500  # ~2500 tokens
    return [{"role": "user", "content": large_text}]


@pytest.fixture
def empty_message() -> list[dict]:
    """Empty message."""
    return [{"role": "user", "content": ""}]


@pytest.fixture
def multipart_message() -> list[dict]:
    """Multi-turn conversation."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a programming language."},
        {"role": "user", "content": "Tell me more about it."},
    ]


# ─── Routing Decision Fixtures ────────────────────────────────────────────────

@pytest.fixture
def local_decision() -> RoutingDecision:
    """Local routing decision."""
    return RoutingDecision(
        backend=Backend.LOCAL,
        rule_name="test_rule",
        reason="Test reason",
        token_count=10,
        prompt_preview="test prompt",
        model_requested="",
    )


@pytest.fixture
def cloud_decision() -> RoutingDecision:
    """Cloud routing decision."""
    return RoutingDecision(
        backend=Backend.CLOUD,
        rule_name="test_rule",
        reason="Test reason",
        token_count=2500,
        prompt_preview="test prompt",
        model_requested="gpt-4o",
        provider="openai",
    )


# ─── Environment Fixtures ─────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clean_env():
    """Clean and isolated environment for each test."""
    old_env = os.environ.copy()
    # Set required env vars for tests
    os.environ["OPENAI_API_KEY"] = "test-key-openai"
    os.environ["ANTHROPIC_API_KEY"] = "test-key-anthropic"
    os.environ["GROQ_API_KEY"] = "test-key-groq"
    os.environ["OPENROUTER_API_KEY"] = "test-key-openrouter"

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(old_env)


@pytest.fixture
def temp_config_file(minimal_config):
    """Create a temporary config.yaml file for testing."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
        yaml.dump(minimal_config, f)
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_prefs_file():
    """Create a temporary model_preferences.json file."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump({}, f)
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


# ─── Parametrization Fixtures ────────────────────────────────────────────────

@pytest.fixture(params=["gpt-4o", "gpt-4o-mini", "claude-opus-4-6", "llama3.2"])
def various_model_names(request) -> str:
    """Various model names for parametrized testing."""
    return request.param


@pytest.fixture(
    params=[
        ("local", Backend.LOCAL),
        ("cloud", Backend.CLOUD),
    ]
)
def backend_pair(request) -> tuple[str, Backend]:
    """Pair of (backend_string, Backend.enum) for parametrized tests."""
    return request.param
