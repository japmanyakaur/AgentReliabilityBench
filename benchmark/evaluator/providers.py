"""
LLM provider abstraction layer.
Supports OpenAI, Anthropic, Google, and Together AI APIs.
"""

import os
import json
import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


class DailyQuotaExhausted(Exception):
    """Raised when Google's daily free-tier quota (500 req/day) is exhausted."""
    pass


@dataclass
class LLMResponse:
    """Standardized response from any provider."""
    content: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    tool_calls: list[dict] | None = None
    finish_reason: str = "stop"
    raw: dict | None = None


class BaseProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def complete(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        ...


class OpenAIProvider(BaseProvider):
    """OpenAI provider using raw HTTP to avoid SDK/pydantic version conflicts."""

    def __init__(self, model_id: str):
        import httpx
        self.api_key = os.environ["OPENAI_API_KEY"]
        self.model_id = model_id
        self.http_client = httpx.Client(
            base_url="https://api.openai.com/v1",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=120.0,
        )

    def complete(self, messages, temperature=0.7, max_tokens=2048, tools=None):
        t0 = time.time()
        payload = {
            "model": self.model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        resp = self.http_client.post("/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        latency = (time.time() - t0) * 1000

        choice = data["choices"][0]
        msg = choice["message"]

        tool_calls = None
        if msg.get("tool_calls"):
            tool_calls = [
                {
                    "name": tc["function"]["name"],
                    "arguments": json.loads(tc["function"]["arguments"]),
                }
                for tc in msg["tool_calls"]
            ]

        usage = data.get("usage", {})
        return LLMResponse(
            content=msg.get("content") or "",
            model=data.get("model", self.model_id),
            input_tokens=usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0) or 0,
            output_tokens=usage.get("completion_tokens", 0) or usage.get("output_tokens", 0) or 0,
            latency_ms=latency,
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason", "stop"),
            raw=data,
        )


class AnthropicProvider(BaseProvider):
    def __init__(self, model_id: str):
        from anthropic import Anthropic
        self.client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model_id = model_id

    def complete(self, messages, temperature=0.7, max_tokens=2048, tools=None):
        system = None
        user_messages = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                user_messages.append(m)

        t0 = time.time()
        kwargs = dict(
            model=self.model_id,
            messages=user_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = [
                {
                    "name": t["function"]["name"],
                    "description": t["function"].get("description", ""),
                    "input_schema": t["function"]["parameters"],
                }
                for t in tools
            ]

        resp = self.client.messages.create(**kwargs)
        latency = (time.time() - t0) * 1000

        content_parts = []
        tool_calls = []
        for block in resp.content:
            if block.type == "text":
                content_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append({"name": block.name, "arguments": block.input})

        raw = None
        try:
            raw = resp.model_dump()
        except Exception:
            raw = None

        return LLMResponse(
            content="\n".join(content_parts),
            model=resp.model,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            latency_ms=latency,
            tool_calls=tool_calls if tool_calls else None,
            finish_reason=resp.stop_reason,
            raw=raw,
        )


class GoogleProvider(BaseProvider):
    """Google Gemini provider using the new google.genai SDK."""

    _last_call_time = 0.0
    _min_interval = 4.5

    def __init__(self, model_id: str):
        from google import genai
        self.client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        self.model_id = model_id

    def _rate_limit_wait(self):
        now = time.time()
        elapsed = now - GoogleProvider._last_call_time
        if elapsed < self._min_interval:
            wait_time = self._min_interval - elapsed
            logger.debug(f"Rate limit: waiting {wait_time:.1f}s")
            time.sleep(wait_time)
        GoogleProvider._last_call_time = time.time()

    def complete(self, messages, temperature=0.7, max_tokens=2048, tools=None):
        from google.genai import types

        contents = []
        for m in messages:
            role = "user" if m["role"] in ("user", "system") else "model"
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=m["content"])],
                )
            )

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        max_retries = 3
        for attempt in range(max_retries):
            self._rate_limit_wait()

            try:
                t0 = time.time()
                resp = self.client.models.generate_content(
                    model=self.model_id,
                    contents=contents,
                    config=config,
                )
                latency = (time.time() - t0) * 1000

                text = ""
                if resp.candidates and resp.candidates[0].content and resp.candidates[0].content.parts:
                    text = resp.candidates[0].content.parts[0].text or ""

                input_tokens = 0
                output_tokens = 0
                if hasattr(resp, 'usage_metadata') and resp.usage_metadata:
                    input_tokens = getattr(resp.usage_metadata, 'prompt_token_count', 0) or 0
                    output_tokens = getattr(resp.usage_metadata, 'candidates_token_count', 0) or 0

                return LLMResponse(
                    content=text,
                    model=self.model_id,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_ms=latency,
                    finish_reason="stop",
                )
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "quota" in err_str.lower():
                    is_daily = "PerDay" in err_str
                    if not is_daily:
                        import re
                        delay_match = re.search(r'seconds:\s*(\d+)', err_str)
                        if delay_match and int(delay_match.group(1)) > 600:
                            is_daily = True

                    if is_daily:
                        logger.error("DAILY QUOTA EXHAUSTED (500 req/day). Stopping.")
                        raise DailyQuotaExhausted(
                            "Daily quota exhausted (500 requests/day). "
                            "Saving partial results and stopping."
                        )

                    if attempt < max_retries - 1:
                        logger.warning(f"Rate limited (attempt {attempt+1}/{max_retries}), waiting 60s...")
                        time.sleep(60)
                        GoogleProvider._last_call_time = time.time()
                        continue
                raise


class TogetherProvider(BaseProvider):
    """Together AI provider using raw HTTP (same pattern as OpenAI)."""

    def __init__(self, model_id: str):
        import httpx
        self.model_id = model_id
        self.http_client = httpx.Client(
            base_url="https://api.together.xyz/v1",
            headers={
                "Authorization": f"Bearer {os.environ['TOGETHER_API_KEY']}",
                "Content-Type": "application/json",
            },
            timeout=120.0,
        )

    def complete(self, messages, temperature=0.7, max_tokens=2048, tools=None):
        t0 = time.time()
        payload = {
            "model": self.model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        resp = self.http_client.post("/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        latency = (time.time() - t0) * 1000

        choice = data["choices"][0]
        msg = choice["message"]

        tool_calls = None
        if msg.get("tool_calls"):
            tool_calls = [
                {
                    "name": tc["function"]["name"],
                    "arguments": json.loads(tc["function"]["arguments"]),
                }
                for tc in msg["tool_calls"]
            ]

        usage = data.get("usage", {})
        return LLMResponse(
            content=msg.get("content") or "",
            model=data.get("model", self.model_id),
            input_tokens=usage.get("prompt_tokens", 0) or 0,
            output_tokens=usage.get("completion_tokens", 0) or 0,
            latency_ms=latency,
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason", "stop"),
        )


class GroqProvider(BaseProvider):
    """Groq provider using raw HTTP with rate limiting."""

    _last_call_time = 0.0
    _min_interval = 2.5

    def __init__(self, model_id: str):
        import httpx
        self.model_id = model_id
        self.http_client = httpx.Client(
            base_url="https://api.groq.com/openai/v1",
            headers={
                "Authorization": f"Bearer {os.environ['GROQ_API_KEY']}",
                "Content-Type": "application/json",
            },
            timeout=120.0,
        )

    def _rate_limit_wait(self):
        now = time.time()
        elapsed = now - GroqProvider._last_call_time
        if elapsed < self._min_interval:
            wait_time = self._min_interval - elapsed
            logger.debug(f"Groq rate limit: waiting {wait_time:.1f}s")
            time.sleep(wait_time)
        GroqProvider._last_call_time = time.time()

    def complete(self, messages, temperature=0.7, max_tokens=2048, tools=None):
        self._rate_limit_wait()

        t0 = time.time()
        payload = {
            "model": self.model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = self.http_client.post("/chat/completions", json=payload)
                resp.raise_for_status()
                data = resp.json()
                latency = (time.time() - t0) * 1000

                choice = data["choices"][0]
                msg = choice["message"]

                tool_calls = None
                if msg.get("tool_calls"):
                    tool_calls = [
                        {
                            "name": tc["function"]["name"],
                            "arguments": json.loads(tc["function"]["arguments"]),
                        }
                        for tc in msg["tool_calls"]
                    ]

                usage = data.get("usage", {})
                return LLMResponse(
                    content=msg.get("content") or "",
                    model=data.get("model", self.model_id),
                    input_tokens=usage.get("prompt_tokens", 0) or 0,
                    output_tokens=usage.get("completion_tokens", 0) or 0,
                    latency_ms=latency,
                    tool_calls=tool_calls,
                    finish_reason=choice.get("finish_reason", "stop"),
                )
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "rate" in err_str.lower():
                    if attempt < max_retries - 1:
                        logger.warning(f"Groq rate limited (attempt {attempt+1}/{max_retries}), waiting 30s...")
                        time.sleep(30)
                        GroqProvider._last_call_time = time.time()
                        continue
                raise


def get_provider(provider_name: str, model_id: str) -> BaseProvider:
    """Factory function to create a provider instance."""
    providers = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "google": GoogleProvider,
        "together": TogetherProvider,
        "groq": GroqProvider,
    }
    if provider_name not in providers:
        raise ValueError(f"Unknown provider: {provider_name}. Available: {list(providers.keys())}")
    return providers[provider_name](model_id)
