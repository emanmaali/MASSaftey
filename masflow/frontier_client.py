"""
frontier_client.py -- a thin wrapper around real, closed-weight frontier
model APIs (OpenAI, Anthropic, Google Gemini), for transfer-attack
experiments (Section~sec:setup, "frontier models" plan): does an
already-optimized local suffix, replayed verbatim against a frontier
model the attacker never had gradient access to, still hijack it?

Deliberately NOT a drop-in replacement for the (model, tokenizer) pair the
rest of this project's eval code expects -- frontier chat-completion APIs
don't expose per-token logprobs for arbitrary hypothetical continuations
(what get_tool_logprobs needs for the ft/fu/top diagnostics), only for
tokens actually sampled. So frontier_eval.py's eval_a_content_frontier()
supports the primary hd (generation-based hit) check only, and reports the
logprob-based diagnostics as not-applicable rather than faking them.
"""
import time

from masflow.config import get_openai_client, get_anthropic_client, get_google_client

# Model-name prefixes routed to each non-OpenAI provider; anything else
# (including gpt-*) goes to OpenAI.
_ANTHROPIC_PREFIXES = ("claude-",)
_GOOGLE_PREFIXES = ("gemini-",)


class FrontierClient:
    def __init__(self, model_name: str, max_retries: int = 3, retry_delay: float = 2.0):
        self.model_name = model_name
        if model_name.startswith(_ANTHROPIC_PREFIXES):
            self.provider = "anthropic"
            self.client = get_anthropic_client()
        elif model_name.startswith(_GOOGLE_PREFIXES):
            self.provider = "google"
            self.client = get_google_client()
        else:
            self.provider = "openai"
            self.client = get_openai_client()
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def generate_chat(self, system: str, user: str, max_tokens: int = 64) -> str:
        """One chat-completion call; returns the assistant's text content.
        Retries on transient errors (rate limits, timeouts) with backoff."""
        last_err = None
        for attempt in range(self.max_retries):
            try:
                if self.provider == "anthropic":
                    resp = self.client.messages.create(
                        model=self.model_name,
                        system=system,
                        messages=[{"role": "user", "content": user}],
                        max_tokens=max_tokens,
                        temperature=0,
                    )
                    return "".join(b.text for b in resp.content if b.type == "text")
                if self.provider == "google":
                    from google.genai import types
                    resp = self.client.models.generate_content(
                        model=self.model_name,
                        contents=user,
                        config=types.GenerateContentConfig(
                            system_instruction=system,
                            max_output_tokens=max_tokens,
                            temperature=0,
                            # Disabled: thinking tokens otherwise eat the
                            # max_output_tokens budget before any visible
                            # text is produced (finish_reason=MAX_TOKENS,
                            # empty .text) -- confirmed via a live smoke
                            # test. Not needed for a short hit-detection
                            # response anyway.
                            thinking_config=types.ThinkingConfig(thinking_budget=0),
                        ),
                    )
                    return resp.text or ""
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    max_tokens=max_tokens,
                    temperature=0,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001 -- broad on purpose: retry any transient API error
                last_err = e
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
        raise RuntimeError(f"FrontierClient({self.model_name}) failed after {self.max_retries} attempts") from last_err
