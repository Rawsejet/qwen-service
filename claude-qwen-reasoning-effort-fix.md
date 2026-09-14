# claude-qwen ↔ Qwen3.8-27B reasoning-effort fix

**Date:** 2026-09-14
**Symptom:** `claude-qwen` connects to the local vLLM server but every turn hangs /
times out — Claude Code never gets a response.

## Root cause

Claude Code (Opus profile) sends `output_config.effort: "high"` on every
`POST /v1/messages`. vLLM's Anthropic adapter maps that to `reasoning_effort="high"`
and passes it into the Qwen3.5 chat template. The stock template only accepts
`xhigh` / `medium` / `low`, so it raises:

```
Error in create_messages: Unexpected reasoning effort high.
Supported types are xhigh (default), medium, and low.
-> HTTP 500 on every request
```

Claude Code retries the 500s, so the harness appears to "hang / not connect".
(The port/URL resolution in `claude-qwen` was fine all along — server on 8085,
`.port` file correct.)

## Fix

Patch the model's chat template to remap the standard Anthropic/OpenAI effort
tiers onto Qwen's, so `high`/`max` → `xhigh` and `minimal`/`none` → `low`.

File: `~/models/qwen3/Qwen3.8-27B/chat_template.jinja` (a standalone
`chat_template.jinja` overrides the copy embedded in `tokenizer_config.json`
under transformers ≥ 4.44, so only this file needs editing).

Inserted right after `set resolved_reasoning_effort = reasoning_effort|default('xhigh')`:

```jinja
{%- if resolved_reasoning_effort in ('high', 'max') %}
    {%- set resolved_reasoning_effort = 'xhigh' %}
{%- elif resolved_reasoning_effort in ('minimal', 'none') %}
    {%- set resolved_reasoning_effort = 'low' %}
{%- endif %}
```

A backup of the patched template is kept alongside this doc as
`chat_template.qwen38-27b.patched.jinja`. **The change only takes effect after a
server restart** (the template is baked into the tokenizer at load).

## Verify

```bash
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://127.0.0.1:8085/v1/messages \
  -H 'content-type: application/json' \
  -d '{"model":"Qwen3.8-27B","max_tokens":30,"output_config":{"effort":"high"},
       "messages":[{"role":"user","content":"Reply: PONG"}]}'
# expect 200 (was 500)
```

## Related

Same restart bumped the dual-GPU (TP2) context from 131072 → **262144** (model's
native max; `rope_scaling: none`). Measured KV budget at 0.85 util / TP2 =
1,519,664 tokens → 5.8× concurrency at 262k. Reflected in `start-qwen.sh` model 1
(single-GPU 65536→131072, dual-solo 131072→262144).
