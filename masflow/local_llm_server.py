"""
local_llm_server.py -- a minimal OpenAI-compatible chat-completions server
backed directly by our own HF model loader (masflow.config.load_local_model),
so third-party MAS frameworks (TradingAgents, MetaGPT) that only speak HTTP via
langchain_openai/openai-python can be pointed at one of this project's own
attack-target models (qwen2.5-0.5b/1.5b/3b-instruct, phi-3.5-mini-instruct,
gemma-2-2b-it) instead of a hosted API.

Tool-calling uses each tokenizer's own chat template (passing `tools=`) plus
Hermes-style <tool_call>{...}</tool_call> parsing of the raw generation --
this matches Qwen2.5's native tool-calling format exactly (confirmed via
apply_chat_template) and is what vLLM/TGI's own "hermes" tool-call parser
does. Not guaranteed to work for models whose template doesn't support
`tools=` (phi-3.5-mini, gemma-2 do not; use qwen2.5-* for tool-calling MAS
frameworks).

Usage:
  python3 -u -m masflow.local_llm_server --model-name Qwen/Qwen2.5-1.5B-Instruct --port 8011
"""
import argparse
import json
import os
import re
import time
import uuid

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(os.path.dirname(__file__)), ".hf_cache"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from transformers import LogitsProcessor, LogitsProcessorList

from masflow.config import load_local_model

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

app = FastAPI()


class GenerationOnlyNoRepeatNGram(LogitsProcessor):
    """Like HF's built-in no_repeat_ngram_size, but scoped to only the
    newly-generated continuation, not the prompt. The built-in version scans
    the full sequence (prompt + generation) for repeats, which hard-bans the
    model from ever quoting back a short phrase that appeared in its own
    input -- exactly the marker-survival behavior this server exists to
    measure. This variant only prevents the model from looping on its own
    output, leaving verbatim quoting of prompt content unaffected."""

    def __init__(self, ngram_size: int, prompt_len: int):
        self.ngram_size = ngram_size
        self.prompt_len = prompt_len

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        for hypo_idx in range(input_ids.shape[0]):
            gen_tokens = input_ids[hypo_idx, self.prompt_len:].tolist()
            if len(gen_tokens) < self.ngram_size:
                continue
            seen = {}
            for i in range(len(gen_tokens) - self.ngram_size + 1):
                prefix = tuple(gen_tokens[i:i + self.ngram_size - 1])
                seen.setdefault(prefix, set()).add(gen_tokens[i + self.ngram_size - 1])
            banned = seen.get(tuple(gen_tokens[-(self.ngram_size - 1):]), set())
            if banned:
                scores[hypo_idx, list(banned)] = -float("inf")
        return scores
STATE = {}


CONTENT_BLOCK_RE = re.compile(r"\[CONTENT\](.*?)\[/CONTENT\]", re.DOTALL)


def _repair_json_regions(text: str) -> str:
    """Fix syntactically-malformed JSON in structural regions only (MetaGPT's
    own [CONTENT]...[/CONTENT] convention, or a response that is entirely a
    JSON blob) -- never touches ordinary prose, so this can't alter the
    substance of a marker-survival check, only the punctuation of an
    unrelated structured-output field. Uses json_repair (handles missing
    quotes, wrong bracket types, trailing commas, etc.), which recovers from
    the specific malformations observed here (Python-set-literal-style pairs,
    unquoted keys) that MetaGPT's own repair_llm_raw_output could not."""
    from json_repair import repair_json

    def _fix_block(m):
        return "[CONTENT]" + repair_json(m.group(1)) + "[/CONTENT]"

    if CONTENT_BLOCK_RE.search(text):
        return CONTENT_BLOCK_RE.sub(_fix_block, text)
    stripped = text.strip()
    if stripped[:1] in "{[":
        return repair_json(stripped)
    return text


def _parse_tool_calls(text: str):
    calls = []
    for m in TOOL_CALL_RE.finditer(text):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        args = obj.get("arguments", {})
        # The model sometimes emits "arguments" as an already-JSON-encoded
        # string (valid per Qwen's own template docs) rather than a nested
        # object; blindly re-serializing a string double-encodes it, which
        # breaks callers expecting OpenAI's single-encoded convention.
        args_str = args if isinstance(args, str) else json.dumps(args)
        calls.append({
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {
                "name": obj.get("name", ""),
                "arguments": args_str,
            },
        })
    return calls


@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{"id": STATE["model_name"], "object": "model"}]}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    tools = body.get("tools")
    max_tokens = body.get("max_tokens") or 512
    temperature = body.get("temperature", 0.0) or 0.0
    stream = bool(body.get("stream", False))

    tok = STATE["tok"]
    model = STATE["model"]

    template_kwargs = {"add_generation_prompt": True, "tokenize": False}
    if tools:
        template_kwargs["tools"] = tools
    try:
        prompt_text = tok.apply_chat_template(messages, **template_kwargs)
    except Exception:
        template_kwargs.pop("tools", None)
        prompt_text = tok.apply_chat_template(messages, **template_kwargs)

    inputs = tok(prompt_text, return_tensors="pt", add_special_tokens=False).to(model.device)
    prompt_len = inputs["input_ids"].shape[1]
    logits_processor = LogitsProcessorList([GenerationOnlyNoRepeatNGram(4, prompt_len)])
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0.0,
            temperature=temperature if temperature > 0.0 else None,
            pad_token_id=tok.pad_token_id,
            # Greedy decoding (temperature=0, the common case for these MAS
            # frameworks) is prone to degenerate repetition loops on small
            # models generating long structured output (observed: MetaGPT's
            # WriteDesign step looping on a mermaid diagram field until the
            # JSON parser choked). Scoped to the generated continuation only
            # (see GenerationOnlyNoRepeatNGram) so it doesn't suppress
            # verbatim quoting of injected prompt content.
            logits_processor=logits_processor,
        )
    gen_ids = out_ids[0][inputs["input_ids"].shape[1]:]
    gen_text = tok.decode(gen_ids, skip_special_tokens=True)
    # Small models sometimes hallucinate JS-style line comments inside JSON
    # they're asked to emit strictly (observed: MetaGPT's WritePRD inserting
    # "// Requirement Analysis: ..." on its own line), which breaks strict
    # JSON parsing and isn't recoverable by the caller's own repair logic.
    # Stripping a line that is only a `//` comment (optionally indented) is
    # safe for prose, since normal sentences don't start a line with `//`.
    gen_text = re.sub(r"(?m)^[ \t]*//.*$\n?", "", gen_text)
    gen_text = _repair_json_regions(gen_text)

    tool_calls = _parse_tool_calls(gen_text) if tools else []
    content = TOOL_CALL_RE.sub("", gen_text).strip() if tool_calls else gen_text
    if os.environ.get("LOCAL_LLM_DEBUG_TOOLCALLS") and tools:
        with open(os.environ["LOCAL_LLM_DEBUG_TOOLCALLS"], "a") as f:
            f.write(f"=== raw gen_text ===\n{gen_text!r}\n=== tool_calls ===\n{tool_calls!r}\n\n")

    message = {"role": "assistant", "content": content or None}
    finish_reason = "stop"
    if tool_calls:
        message["tool_calls"] = tool_calls
        finish_reason = "tool_calls"

    chat_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    if stream:
        # Not real token-by-token streaming (generation above already ran to
        # completion) -- this is a wire-protocol shim: MetaGPT's aask() (and
        # other clients) default to stream=True and silently get empty
        # content back from a plain JSON response, since the openai SDK's
        # streaming code path expects SSE `data: {...}` chunks. Emitting the
        # whole message as a single delta chunk satisfies that parser.
        def _sse():
            first_delta = {"role": "assistant"}
            if tool_calls:
                first_delta["tool_calls"] = tool_calls
            else:
                first_delta["content"] = content
            chunk = {
                "id": chat_id, "object": "chat.completion.chunk", "created": created,
                "model": STATE["model_name"],
                "choices": [{"index": 0, "delta": first_delta, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            final_chunk = {
                "id": chat_id, "object": "chat.completion.chunk", "created": created,
                "model": STATE["model_name"],
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(_sse(), media_type="text/event-stream")

    return JSONResponse({
        "id": chat_id,
        "object": "chat.completion",
        "created": created,
        "model": STATE["model_name"],
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": int(inputs["input_ids"].shape[1]),
            "completion_tokens": int(gen_ids.shape[0]),
            "total_tokens": int(inputs["input_ids"].shape[1] + gen_ids.shape[0]),
        },
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--port", type=int, default=8011)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    model, tok = load_local_model(args.model_name)
    STATE["model"] = model
    STATE["tok"] = tok
    STATE["model_name"] = args.model_name

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
