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
    def __init__(self, model_id: str):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model_id = model_id

    def complete(self, messages, temperature=0.7, max_tokens=2048, tools=None):
        t0 = time.time()
        kwargs = dict(
            model=self.model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        resp = self.client.chat.completions.create(**kwargs)
        latency = (time.time() - t0) * 1000

        choice = resp.choices[0]
        tool_calls = None
        if choice.message.tool_calls:
            tool_calls = [
                {
                    "name": tc.function.name,
                    "arguments": json.loads(tc.function.arguments),
                }
                for tc in choice.message.tool_calls
            ]

        return LLMResponse(
            content=choice.message.content or "",
            model=resp.model,
            input_tokens=resp.usage.prompt_tokens,
            output_tokens=resp.usage.completion_tokens,
            latency_ms=latency,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason,
            raw=resp.model_dump(),
        )


class AnthropicProvider(BaseProvider):
    def __init__(self, model_id: str):
        from anthropic import Anthropic
        self.client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model_id = model_id

    def complete(self, messages, temperature=0.7, max_tokens=2048, tools=None):
        # Separate system message if present
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

        return LLMResponse(
            content="\n".join(content_parts),
            model=resp.model,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            latency_ms=latency,
            tool_calls=tool_calls if tool_calls else None,
            finish_reason=resp.stop_reason,
            raw=resp.model_dump(),
        )


class GoogleProvider(BaseProvider):
    """Google Gemini provider with built-in rate limiting for free tier."""

    # Class-level rate limiter shared across all GoogleProvider instances
    _last_call_time = 0.0
    _min_interval = 4.5  # seconds between calls (15 RPM ≈ 4s, with safety margin)

    def __init__(self, model_id: str):
        import google.generativeai as genai
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        self.model = genai.GenerativeModel(model_id)
        self.model_id = model_id

    def _rate_limit_wait(self):
        """Wait if needed to stay under 15 RPM free-tier limit."""
        now = time.time()
        elapsed = now - GoogleProvider._last_call_time
        if elapsed < self._min_interval:
            wait_time = self._min_interval - elapsed
            logger.debug(f"Rate limit: waiting {wait_time:.1f}s")
            time.sleep(wait_time)
        GoogleProvider._last_call_time = time.time()

    def complete(self, messages, temperature=0.7, max_tokens=2048, tools=None):
        # Convert OpenAI message format → Gemini format
        parts = []
        for m in messages:
            role = "user" if m["role"] in ("user", "system") else "model"
            parts.append({"role": role, "parts": [m["content"]]})

        max_retries = 3
        for attempt in range(max_retries):
            self._rate_limit_wait()

            try:
                t0 = time.time()
                resp = self.model.generate_content(
                    parts,
                    generation_config={"temperature": temperature, "max_output_tokens": max_tokens},
                )
                latency = (time.time() - t0) * 1000

                return LLMResponse(
                    content=resp.text if resp.text else "",
                    model=self.model_id,
                    input_tokens=resp.usage_metadata.prompt_token_count if resp.usage_metadata else 0,
                    output_tokens=resp.usage_metadata.candidates_token_count if resp.usage_metadata else 0,
                    latency_ms=latency,
                    finish_reason="stop",
                )
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "quota" in err_str.lower():
                    # Check if it's a DAILY quota — look for "PerDay" in quota ID
                    # or retry delay > 600 seconds (10 minutes)
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

                    # Per-minute rate limit — wait and retry
                    if attempt < max_retries - 1:
                        logger.warning(f"Rate limited (attempt {attempt+1}/{max_retries}), waiting 60s...")
                        time.sleep(60)
                        GoogleProvider._last_call_time = time.time()
                        continue
                raise


class TogetherProvider(BaseProvider):
    def __init__(self, model_id: str):
        from openai import OpenAI
        self.client = OpenAI(
            api_key=os.environ["TOGETHER_API_KEY"],
            base_url="https://api.together.xyz/v1",
        )
        self.model_id = model_id

    def complete(self, messages, temperature=0.7, max_tokens=2048, tools=None):
        t0 = time.time()
        kwargs = dict(
            model=self.model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        resp = self.client.chat.completions.create(**kwargs)
        latency = (time.time() - t0) * 1000

        choice = resp.choices[0]
        tool_calls = None
        if choice.message.tool_calls:
            tool_calls = [
                {
                    "name": tc.function.name,
                    "arguments": json.loads(tc.function.arguments),
                }
                for tc in choice.message.tool_calls
            ]

        return LLMResponse(
            content=choice.message.content or "",
            model=resp.model,
            input_tokens=resp.usage.prompt_tokens if resp.usage else 0,
            output_tokens=resp.usage.completion_tokens if resp.usage else 0,
            latency_ms=latency,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason,
        )


def get_provider(provider_name: str, model_id: str) -> BaseProvider:
    """Factory function to create a provider instance."""
    providers = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "google": GoogleProvider,
        "together": TogetherProvider,
    }
    if provider_name not in providers:
        raise ValueError(f"Unknown provider: {provider_name}. Available: {list(providers.keys())}")
    return providers[provider_name](model_id)
