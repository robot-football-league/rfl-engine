"""LLM decision agents: provider adapters behind the common Agent interface.

Providers: anthropic, openai, google, mock (offline, for plumbing tests).
Every request/response is logged verbatim to llm.jsonl per episode. Decisions
are synchronous — sim time pauses while the API call is in flight, so latency
never affects outcomes.

Temperature 0 is requested where supported; models that reject sampling or
thinking parameters (400) get those parameters dropped automatically and the
call is retried, so one code path serves all model generations.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from string import Template

import numpy as np

from . import paths
from .envelope import load_envelope
from .util import wrap_angle

PROMPT_VERSION = "v1"
MAX_RETRIES = 5
RETRY_BASE_S = 2.0
REQUEST_TIMEOUT_S = 120.0
# Params to drop-and-retry when a 400 mentions any of their aliases (newer
# models reject temperature; some reject thinking config; smaller Anthropic
# models reject effort, whose error text says "effort", not "output_config").
DROPPABLE_PARAMS = {
    "temperature": ("temperature",),
    "thinking": ("thinking",),
    "output_config": ("output_config", "effort"),
    "reasoning_effort": ("reasoning_effort",),
    # AIMLAPI fans out to every vendor, and they disagree about which of
    # these two spellings of "stop after N tokens" they accept. Both are
    # droppable so a 400 from either resolves itself on the retry.
    "max_tokens": ("max_tokens",),
    "max_completion_tokens": ("max_completion_tokens",),
    # The Responses endpoint's reasoning block ({effort, summary}); a model
    # that rejects a summary request gets called without one.
    "reasoning": ("reasoning",),
    # Prompt-prefix caching. Sent as a block-shaped `system`; an aggregator
    # that will not take that shape gets the plain string instead.
    "cache_control": ("cache_control", "cache control"),
}


def _extract_json(text: str, required=frozenset(("vx", "vy", "wz"))):
    """First JSON object in the text containing the required keys.

    required=None accepts the first (outermost) object — used by agents with
    free-form reply schemas like the football manager, whose nested "move"
    sub-object would otherwise be matched instead of the full reply."""
    if not text:
        return None
    dec = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                obj, _ = dec.raw_decode(text[i:])
            except ValueError:
                continue
            if isinstance(obj, dict) and (required is None or required <= set(obj)):
                return obj
    return None


# Season-3 gaffer roster, priced by their FULL AIMLAPI id so the substring
# match in estimate_cost() cannot collide with the bare-name rows below.
# These are vendor list prices, $/1M tokens, and they are a STARTING POINT:
# an aggregator may mark up. `scripts/calibrate_aiml_prices.py` measures the
# real rate against the dashboard and this table is corrected from it before
# season budgets are trusted. Every roster model MUST have a row — a model
# with no row silently disables the budget cap, so the gaffer harness
# refuses to run one.
# MEASURED, not assumed. On 2026-09-01 a billing export gave ground
# truth for 13 gemini-3.7-flash calls: AIMLAPI charged 1.94x the vendor
# list price, tightly clustered (1.80-2.05), which is a flat markup rather
# than noise. Every row below is list price times this multiplier.
#
# This matters more than tidiness: the season purse meters against these
# numbers, so an under-count does not just misreport spend, it lets a club
# keep buying turns after its budget is gone. Before this was measured the
# station's own ledger read well under half what the dashboard did.
#
# Re-measure with scripts/calibrate_aiml_prices.py whenever the roster or
# the aggregator's pricing changes; a markup is a business decision and
# will move without warning.
AIML_MARKUP = 1.94

AIML_PRICES_LIST = [
    ("anthropic/claude-fable-5", 10.0, 50.0),
    ("openai/gpt-5.6-sol", 5.0, 30.0),
    ("google/gemini-3.7-flash", 0.5, 3.0),
    ("meta/muse-spark-1.2", 0.6, 2.4),
    ("zhipu/glm-5.3", 0.6, 2.2),
    ("deepseek/deepseek-v4-pro", 0.55, 2.2),
    # alternates on the shelf (STATE names these as swap-ins)
    ("moonshot/kimi-k3", 0.6, 2.5),
    ("alibaba/qwen3.8-max", 1.2, 6.0),
    ("x-ai/grok-4-6", 3.0, 15.0),
    ("minimax/minimax-m3", 0.3, 1.2),
]

AIML_PRICES = [(k, i * AIML_MARKUP, o * AIML_MARKUP)
               for k, i, o in AIML_PRICES_LIST]

PRICES = AIML_PRICES + [
    ("claude-fable-5", 10.0, 50.0),
    ("gpt-5.6-sol", 5.0, 30.0),
    ("gpt-5.6-terra", 2.0, 12.0),
    ("gpt-5.6-luna", 0.20, 1.20),
    ("gpt-5.4-mini", 0.75, 4.50),
    ("gpt-5.4-nano", 0.20, 1.25),
    ("gemini-3.7-flash", 0.5, 3.0),  # placeholder until confirmed
    ("claude-opus-5", 5.0, 25.0),
    ("claude-sonnet-5", 2.0, 10.0),
    ("claude-haiku-4-5", 1.0, 5.0),
    ("gemini-pro", 1.25, 10.0),
    ("gemini-robotics-er", 0.30, 2.50),  # preview pricing unpublished - flash-class guess
    ("gemini-flash-lite", 0.10, 0.40),  # check before "gemini-flash"
    ("gemini-flash", 0.30, 2.50),
]


def estimate_cost(agent_name: str, tin: int, tout: int, tcr: int, tcw: int) -> float | None:
    """Rough $ for a model call. None if the model isn't in PRICES."""
    for key, pi, po in PRICES:
        if key in agent_name:
            return (tin * pi + tcr * 0.1 * pi + tcw * 1.25 * pi + tout * po) / 1e6
    return None


class LLMAgent:
    def __init__(self, provider: str, model: str, history_n: int = 10,
                 prompt: str = PROMPT_VERSION, static_block: str | None = None):
        # Set for every provider so a caller can read them unconditionally;
        # only the providers that surface reasoning ever populate them.
        self.last_reasoning = None
        self.last_reasoning_tokens = 0
        # Set by callers that PUBLISH the model's thinking (the gaffer). Off
        # by default: a player brain has a 2 s decision interval and no
        # reader, so it is never sent through a thinking endpoint.
        self.want_reasoning = False
        self.reasoning_budget = 2048     # Anthropic thinking budget, tokens
        self._aclient = None             # anthropic SDK client (AIMLAPI native)
        self.provider = provider
        self.model = model
        self.name = f"llm:{provider}:{model}"
        self.history_n = history_n
        self.prompt_name = prompt
        env = load_envelope()
        tpl = Template((paths.PROMPTS / f"system_{prompt}.md").read_text())
        self.system_prompt = tpl.substitute(
            vx0=env["vx"][0], vx1=env["vx"][1],
            vy0=env["vy"][0], vy1=env["vy"][1],
            wz0=env["wz"][0], wz1=env["wz"][1],
        )
        # Cost control: bake the fixed course layout into the system prompt
        # (a stable, cacheable prefix) and send only MOVING hazards per tick.
        self.static_block = static_block
        if static_block:
            self.system_prompt += (
                "\n\n# Static course layout (fixed for the whole race)\n"
                "These never move and are OMITTED from the per-tick 'obstacles' "
                "list, which contains only MOVING hazards:\n" + static_block)
        self._client = None
        self._dropped_params: set[str] = set()
        self._fatal: str | None = None
        self._mock_count = 0
        self.history: list[dict] = []
        self.episode_usage: dict = {}
        self.log_path: Path | None = None

    # ------------------------------------------------------------ harness hooks

    def begin_episode(self, log_dir=None):
        self.history = []
        self._mock_count = 0
        self._mock_last_cmd = None
        self.episode_usage = {}
        self.log_path = Path(log_dir) / "llm.jsonl" if log_dir else None

    def decide(self, obs: dict):
        # egocentric camera frames ride outside the JSON (np arrays; set by
        # the race runner when camera mode is on). Providers that can see,
        # see them: _frames is an [older, current] motion pair.
        self._frame = obs.get("_frame")
        self._frames = obs.get("_frames")
        user_text = self._build_user(obs)
        t0 = time.time()
        text, usage, error = self._call_with_retries(user_text)
        self._log({
            "provider": self.provider, "model": self.model,
            "prompt_version": self.prompt_name,
            "request_user_text": user_text,
            "response_text": text, "usage": usage, "error": error,
            "latency_s": round(time.time() - t0, 2),
        })
        if usage:
            for k, v in usage.items():
                if isinstance(v, int):
                    self.episode_usage[k] = self.episode_usage.get(k, 0) + v
        cmd = (_extract_json(text, getattr(self, "reply_keys",
                                          frozenset(("vx", "vy", "wz"))))
               if text else None)
        # Compact history: keep what informs future decisions (own state,
        # hazard rhythm, action taken) and drop the bulky static/derived parts.
        hist = {
            "t": obs.get("time_remaining_s"),
            "self": obs.get("self"),
            "action": cmd if cmd is not None else {"invalid_reply": (text or error or "")[:200]},
        }
        if "next_checkpoint" in obs:
            hist["gate"] = obs["next_checkpoint"]["index"]
        if "obstacles" in obs:  # absent in sensors mode: movers are camera-only
            hist["movers"] = [o for o in obs["obstacles"] if o["type"] == "moving"]
        self.history.append(hist)
        del self.history[: -self.history_n]
        return cmd  # None -> harness counts ignored_invalid and holds last command

    # ------------------------------------------------------------ prompt build

    def _build_user(self, obs: dict) -> str:
        if any(k.startswith("_") for k in obs):
            obs = {k: v for k, v in obs.items() if not k.startswith("_")}
        if self.static_block and "obstacles" in obs:
            # movers always; statics only when nearby (live wall awareness —
            # the full fixed layout is in the system prompt)
            px, py = obs["self"]["position"]

            def keep(o):
                if o["type"] == "moving":
                    return True
                x0, y0, x1, y1 = o["aabb"]
                return (x0 - 5.5 < px < x1 + 5.5) and (y0 - 5.5 < py < y1 + 5.5)

            obs = {**obs, "obstacles": [o for o in obs["obstacles"] if keep(o)]}
        lines = []
        if self.history:
            lines.append("Recent history (your state, moving hazards, your action), oldest first:")
            for h in self.history:
                lines.append(json.dumps(h, separators=(",", ":")))
        lines.append("Current observation:")
        lines.append(json.dumps(obs, separators=(",", ":")))
        lines.append('Reply with exactly one JSON object: {"vx": <m/s>, "vy": <m/s>, "wz": <rad/s>}')
        return "\n".join(lines)

    # ------------------------------------------------------------ transport

    def _log(self, record: dict):
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(record) + "\n")

    def _call_with_retries(self, user_text: str):
        if self._fatal:
            return None, None, self._fatal
        delay = RETRY_BASE_S
        last_err = None
        attempt = 0
        while attempt < MAX_RETRIES:
            try:
                text, usage = self._call(user_text)
                return text, usage, None
            except Exception as e:  # classified below; SDKs differ per provider
                status = getattr(e, "status_code", None) or getattr(e, "code", None)
                msg = f"{type(e).__name__}: {e}"
                if self.provider == "mock":
                    return None, None, msg  # local bug: retrying can't fix it
                # A TIMEOUT IS NOT A FAILED REQUEST. It is this client giving
                # up; the server may already have generated the whole reply,
                # and it bills for what it generated whether or not we ever
                # receive it. Re-sending the same prompt therefore buys a
                # second full-price generation and, if it also runs long, a
                # third. On 2026-09-02 that went on re-sending two prompts,
                # one four times and one three; it accounted for more than
                # half of that day's spend and emptied the daily limit before
                # four of the six clubs had made a single call.
                #
                # So a timeout is returned, never retried. This is deliberately
                # conservative: a connect-timeout that never reached the server
                # would have been safe to resend, but the SDKs do not reliably
                # distinguish that from a read-timeout, and the downside is
                # asymmetric -- an unnecessary retry costs real money, while a
                # missed one costs one of MAX_MODEL_ERRORS attempts.
                if "timeout" in type(e).__name__.lower() or isinstance(
                        e, TimeoutError):
                    return None, None, (
                        msg + "  [not retried: the generation may have "
                        "completed and been billed]")
                if status == 400:
                    dropped = [p for p, aliases in DROPPABLE_PARAMS.items()
                               if p not in self._dropped_params
                               and any(a in str(e) for a in aliases)]
                    if dropped:
                        self._dropped_params.update(dropped)
                        continue  # retry immediately without the param
                    return None, None, msg  # other 400s won't fix themselves
                if status in (401, 403, 404) or "api key" in str(e).lower():
                    # bad/missing key or nonexistent model: neither fixes
                    # itself mid-episode — fail every future tick instantly
                    # (the genai SDK raises key errors as plain ValueError,
                    # which otherwise masquerades as a hung call via backoff)
                    self._fatal = msg
                    return None, None, msg
                last_err = msg  # rate limit / 5xx / network: back off
                attempt += 1
                time.sleep(delay)
                delay = min(delay * 2, 30.0)
        return None, None, last_err

    def _call(self, user_text: str):
        return {
            "anthropic": self._call_anthropic,
            "openai": self._call_openai,
            "google": self._call_google,
            "ollama": self._call_ollama,
            "zhipu": self._call_zhipu,
            "aiml": self._call_aiml,
            "mock": self._call_mock,
        }[self.provider](user_text)

    def _call_anthropic(self, user_text: str):
        import anthropic
        if self._client is None:
            # our retry loop owns backoff; SDK-internal retries would hide
            # rate limiting as silent multi-second latency
            self._client = anthropic.Anthropic(timeout=REQUEST_TIMEOUT_S,
                                               max_retries=0)
        # Pre-4.6 models (Haiku 4.5) accept an assistant prefill: force the
        # reply to be bare JSON by prefilling "{" and stopping at "}" — the
        # model physically cannot narrate first. Newer models reject prefill;
        # they keep thinking ON at low effort instead (disabling it makes
        # them write reasoning into the visible reply).
        # Probed 2026-08-16: haiku WITH thinking (1024 budget) takes 10-19 s
        # per decision (~1.3k output tokens) and often buries the JSON — a
        # non-starter under a 2.5 s shot clock. Reflex prefill is haiku's
        # only deadline-compatible mode; the resulting zero-reasoning play is
        # a measured property of the model class, not a harness choice.
        prefill = "haiku" in self.model
        messages = [{"role": "user", "content": user_text}]
        kwargs = dict(
            model=self.model,
            max_tokens=(200 if prefill else
                        getattr(self, "max_output", None) or 1200),
            # cache the (system + static course) prefix: read-billed at ~0.1x
            system=[{"type": "text", "text": self.system_prompt,
                     "cache_control": {"type": "ephemeral"}}],
            messages=messages,
        )
        if prefill:
            messages.append({"role": "assistant", "content": "{"})
            kwargs["stop_sequences"] = ["}"]
        if "temperature" not in self._dropped_params:
            kwargs["temperature"] = 0.0
        if not prefill and "output_config" not in self._dropped_params:
            kwargs["output_config"] = {
                "effort": getattr(self, "effort", None) or "low"}
        resp = self._client.messages.create(**kwargs)
        text = "".join(b.text for b in resp.content if b.type == "text")
        if prefill:
            text = "{" + text + ("}" if resp.stop_reason == "stop_sequence" else "")
        u = resp.usage
        usage = {"input_tokens": u.input_tokens,
                 "output_tokens": u.output_tokens,
                 "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", None),
                 "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", None)}
        return text, usage

    def _call_openai(self, user_text: str):
        from openai import OpenAI
        if self._client is None:
            self._client = OpenAI(timeout=REQUEST_TIMEOUT_S, max_retries=0)
        kwargs = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_text},
            ],
            max_completion_tokens=getattr(self, "max_output", None) or 2048,
        )
        if "temperature" not in self._dropped_params:
            kwargs["temperature"] = 0.0
        if "reasoning_effort" not in self._dropped_params:
            # reasoning-tier models overrun the 3 s shot clock at their
            # default effort (gpt-5.6-luna: 4.5 s default, 2.6 s at low);
            # models that reject the param drop it via the 400-retry path
            kwargs["reasoning_effort"] = getattr(self, "effort", None) or "low"
        resp = self._client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content or ""
        usage = None
        if resp.usage:
            usage = {"input_tokens": resp.usage.prompt_tokens,
                     "output_tokens": resp.usage.completion_tokens}
        return text, usage

    def _call_google(self, user_text: str):
        import logging
        from google import genai
        from google.genai import types
        if self._client is None:
            # the SDK logs a noisy AFC advisory on every generate_content call
            logging.getLogger("google_genai.models").setLevel(logging.ERROR)
            # the default Client has NO http timeout — one silently dropped
            # connection would hang generate_content, and decide(), forever
            self._client = genai.Client(http_options=types.HttpOptions(
                timeout=int(REQUEST_TIMEOUT_S * 1000)))      # milliseconds
        cfg = types.GenerateContentConfig(
            system_instruction=self.system_prompt,
            max_output_tokens=getattr(self, "max_output", None) or 2048,
        )
        if "temperature" not in self._dropped_params:
            cfg.temperature = 0.0
        if "thinking" not in self._dropped_params:
            # cap internal reasoning spend; dropped automatically on models
            # that reject a thinking config
            cfg.thinking_config = types.ThinkingConfig(thinking_budget=256)
        frames = getattr(self, "_frames", None)
        if frames is None and getattr(self, "_frame", None) is not None:
            frames = [self._frame]
        contents = user_text
        if frames:
            import io
            from PIL import Image
            parts = []
            for f in frames:
                buf = io.BytesIO()
                Image.fromarray(f).save(buf, format="JPEG", quality=80)
                parts.append(types.Part.from_bytes(data=buf.getvalue(),
                                                   mime_type="image/jpeg"))
            contents = parts + [user_text]
        resp = self._client.models.generate_content(
            model=self.model, contents=contents, config=cfg)
        try:
            text = resp.text or ""
        except (ValueError, AttributeError):
            text = ""
        usage = None
        um = getattr(resp, "usage_metadata", None)
        if um:
            usage = {"input_tokens": um.prompt_token_count,
                     "output_tokens": um.candidates_token_count}
        return text, usage

    def _call_ollama(self, user_text: str):
        """Local models via Ollama's OpenAI-compatible endpoint.

        The realistic entrant class for realtime mode: a quantized 4-8B model
        on this machine stands in for what fits a robot's onboard computer.
        Model names may contain colons (e.g. llm:ollama:qwen3:8b).
        """
        import os
        from openai import OpenAI
        if self._client is None:
            base = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
            self._client = OpenAI(base_url=f"{base}/v1", api_key="ollama",
                                  timeout=REQUEST_TIMEOUT_S)
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_text},
            ],
            max_tokens=300,
            temperature=0.0,
        )
        text = resp.choices[0].message.content or ""
        usage = None
        if resp.usage:
            usage = {"input_tokens": resp.usage.prompt_tokens,
                     "output_tokens": resp.usage.completion_tokens}
        return text, usage

    def _call_zhipu(self, user_text: str):
        """Zhipu GLM via its OpenAI-compatible endpoint."""
        import os
        from openai import OpenAI
        if self._client is None:
            base = os.environ.get("ZHIPU_BASE_URL",
                                  "https://open.bigmodel.cn/api/paas/v4")
            self._client = OpenAI(base_url=base,
                                  api_key=os.environ["ZHIPU_API_KEY"],
                                  timeout=REQUEST_TIMEOUT_S, max_retries=0)
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_text},
            ],
            max_tokens=getattr(self, "max_output", None) or 2048,
            temperature=0.0,
        )
        text = resp.choices[0].message.content or ""
        usage = None
        if resp.usage:
            usage = {"input_tokens": resp.usage.prompt_tokens,
                     "output_tokens": resp.usage.completion_tokens}
        return text, usage

    def _call_aiml(self, user_text: str):
        """Every gaffer model behind ONE key (AIMLAPI, OpenAI-compatible).

        Model ids carry a vendor prefix and a slash — `llm:aiml:zhipu/glm-5.3`
        — which survives the bounded `split(":", 2)` that parses a model spec.
        The vendor half is part of the model id, not a provider of its own:
        routing is AIMLAPI's job, and the station holds a single spend-capped
        credential rather than one key per lab.

        Why this exists: season 3 runs the gaffer sessions unattended on the
        station, and USER.md's old rule (Robin runs them himself, the station
        never holds those keys) could not survive that. One key that can only
        burn its own balance is the trade.
        """
        import os
        from openai import OpenAI
        base = os.environ.get("AIML_BASE_URL",
                              "https://api.aimlapi.com/v1").rstrip("/")
        key = os.environ.get("AIML_API_KEY")
        if not key:
            raise RuntimeError(
                "AIML_API_KEY missing from .env — run "
                "scripts/add_aiml_key.sh")
        # THE REASONING IS THE SHOW, but the big labs only hand it over when
        # asked, and two of them only on a different endpoint. Probed
        # 2026-09-01 on the live key (raw responses in the daily log):
        #   anthropic/  chat endpoint + `thinking` -> reasoning_content EMPTY;
        #               the native /v1/messages endpoint returns thinking
        #               blocks (summarised by Anthropic; nobody gets the raw
        #               trace).
        #   openai/     chat endpoint never returns reasoning, same as
        #               OpenAI direct; /v1/responses with
        #               reasoning.summary="auto" returns a summary.
        #   google/     thinks (reasoning_tokens > 0) but AIMLAPI returns only
        #               a thought_signature — four passthrough spellings
        #               tried, none surfaces the thoughts. Stays on chat.
        #   deepseek/ zhipu/ meta/  return reasoning_content unasked.
        # Only a caller that publishes the thinking pays for the detour.
        if getattr(self, "want_reasoning", False):
            vendor = self.model.split("/", 1)[0] if "/" in self.model else ""
            if vendor == "anthropic":
                return self._call_aiml_anthropic(user_text, base, key)
            if vendor == "openai":
                return self._call_aiml_responses(user_text, base, key)
        if self._client is None:
            self._client = OpenAI(base_url=base, api_key=key,
                                  timeout=REQUEST_TIMEOUT_S, max_retries=0)
        kwargs = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_text},
            ],
        )
        # Vendors behind the aggregator disagree on the spelling; send the
        # OpenAI-style one and let the 400-retry path drop whichever is
        # rejected rather than guessing per vendor.
        if "max_tokens" not in self._dropped_params:
            kwargs["max_tokens"] = getattr(self, "max_output", None) or 2048
        if "temperature" not in self._dropped_params:
            kwargs["temperature"] = 0.0
        # Cleared per call: a failed call must not leave the PREVIOUS turn's
        # reasoning lying around to be attributed to this one.
        self.last_reasoning = None
        self.last_reasoning_tokens = 0
        resp = self._client.chat.completions.create(**kwargs)
        # An aggregator does NOT guarantee the shapes the SDK's types imply.
        # Measured on 2026-08-29: gemini-3.7-flash came back with
        # `choices[0].message` set to None, and `.content` on that raised an
        # AttributeError that the retry classifier read as a transient fault
        # — five backoff rounds for a reply that was never coming. Reasoning
        # models also return an EMPTY content when thinking consumed the
        # whole token allowance. Both are "no answer", and the caller already
        # handles an empty answer (it costs a parse strike), so decode
        # defensively and let the session logic deal with it.
        text = ""
        choices = getattr(resp, "choices", None) or []
        # A DELIVERY failure is not the model saying nothing, and the caller
        # cannot tell the difference from an empty string. Observed on
        # gemini-3.7-flash: finish_reason 'malformed_function_call' with
        # message: null — the model reached for a function call, mangled it,
        # and the API returned no message at all. Intermittent, so the same
        # prompt succeeds on a retry. Treated as an empty reply it cost a
        # parse strike, and three in a row ended Gemini Flash FC's session
        # having never heard from it. Raise instead, so the retry path with
        # its backoff gets a go, and a persistent failure is reported as a
        # model error rather than blamed on the club.
        if choices:
            fr = getattr(choices[0], "finish_reason", None)
            if getattr(choices[0], "message", None) is None:
                raise RuntimeError(
                    f"no message in response (finish_reason={fr!r}) — "
                    f"delivery failure, not an empty reply")
        if choices:
            msg = getattr(choices[0], "message", None)
            text = (getattr(msg, "content", None) or "") if msg else ""
            # A model may answer in the NATIVE tool-calling protocol instead
            # of writing our JSON into the message body — muse-spark-1.2
            # does, every time, because the gaffer prompt is obviously about
            # tools. It returns finish_reason='tool_calls' with content='',
            # which reads to a content-only caller as an empty reply: three
            # of those and the harness strikes the club out. Muse's founding
            # night died exactly that way, having said nothing wrong.
            #
            # So accept both protocols and normalise to ours. The model gets
            # to speak whichever dialect it prefers.
            if not text.strip():
                calls = getattr(msg, "tool_calls", None) or [] if msg else []
                for tc in calls:
                    fn = getattr(tc, "function", None)
                    name = getattr(fn, "name", None) if fn else None
                    if not name:
                        continue
                    try:
                        args = json.loads(getattr(fn, "arguments", "") or "{}")
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                    if not isinstance(args, dict):
                        args = {}
                    text = json.dumps({"tool": name, **args})
                    break
            # THE REASONING IS THE SHOW. A reasoning model puts its working
            # in a SEPARATE field and only its conclusion in `content`, and
            # the league pays for both — measured 2026-08-29: glm-5.3 spent
            # 619 of 692 output tokens (89%) on reasoning, deepseek 125 of
            # 154. Reading only `content` means buying the thinking and
            # binning it, which would have quietly gutted the whole point of
            # publishing gaffer sessions. Field name varies by vendor.
            if msg is not None:
                for cand in ("reasoning_content", "reasoning", "thinking"):
                    r = getattr(msg, cand, None)
                    if isinstance(r, str) and r.strip():
                        self.last_reasoning = r
                        break
        usage = None
        if getattr(resp, "usage", None):
            # Cached-prompt reads are reported under different names (or not
            # at all) per vendor; read them when present so estimate_cost can
            # charge them at the cheaper rate, and treat absence as zero.
            details = getattr(resp.usage, "prompt_tokens_details", None)
            cached = getattr(details, "cached_tokens", 0) if details else 0
            cdet = getattr(resp.usage, "completion_tokens_details", None)
            self.last_reasoning_tokens = (
                getattr(cdet, "reasoning_tokens", 0) or 0) if cdet else 0
            usage = {"input_tokens": getattr(resp.usage, "prompt_tokens", 0) or 0,
                     "output_tokens": getattr(resp.usage, "completion_tokens", 0) or 0,
                     "cache_read_input_tokens": cached or 0,
                     # Already inside output_tokens — reported so a session
                     # can show what share of its purse went on thinking,
                     # NOT to be added to the cost again.
                     "reasoning_tokens": self.last_reasoning_tokens}
        return text, usage

    def _call_aiml_anthropic(self, user_text: str, base: str, key: str):
        """An Anthropic model through AIMLAPI's NATIVE Messages endpoint,
        thinking on — the only route through the aggregator that returns
        the thinking blocks (see _call_aiml). Same key, same spend cap."""
        import anthropic
        if self._aclient is None:
            # the anthropic SDK appends /v1/messages itself
            root = base[:-3] if base.endswith("/v1") else base
            self._aclient = anthropic.Anthropic(
                base_url=root, api_key=key, timeout=REQUEST_TIMEOUT_S,
                max_retries=0)
        max_tokens = getattr(self, "max_output", None) or 4096
        # Cache the system prefix, read-billed at ~0.1x. _call_anthropic has
        # done this all along; this path did not, so every retry re-bought the
        # whole ~12,850-token prefix at full price — roughly a quarter of one
        # 2026-09-02 session's total cost went on re-buying it.
        if "cache_control" not in self._dropped_params:
            system = [{"type": "text", "text": self.system_prompt,
                       "cache_control": {"type": "ephemeral"}}]
        else:
            system = self.system_prompt
        kwargs = dict(model=self.model, max_tokens=max_tokens,
                      system=system,
                      messages=[{"role": "user", "content": user_text}])
        if "thinking" not in self._dropped_params:
            # must be >= 1024 and < max_tokens; a ceiling, not a target —
            # measured 31 tokens used of 1500 offered on a one-line note
            budget = max(1024, min(int(getattr(self, "reasoning_budget", 2048)
                                       or 2048), max_tokens - 1024))
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
        # No temperature: Anthropic rejects one while thinking is enabled.
        self.last_reasoning = None
        self.last_reasoning_tokens = 0
        resp = self._aclient.messages.create(**kwargs)
        blocks = getattr(resp, "content", None) or []
        thoughts = [b.thinking for b in blocks
                    if getattr(b, "type", "") == "thinking"
                    and getattr(b, "thinking", None)]
        text = "".join(getattr(b, "text", "") or "" for b in blocks
                       if getattr(b, "type", "") == "text")
        if thoughts:
            self.last_reasoning = "\n\n".join(thoughts)
        usage = None
        u = getattr(resp, "usage", None)
        if u is not None:
            ud = u.model_dump() if hasattr(u, "model_dump") else dict(u)
            det = ud.get("output_tokens_details") or {}
            self.last_reasoning_tokens = int(det.get("thinking_tokens") or 0)
            usage = {"input_tokens": ud.get("input_tokens") or 0,
                     "output_tokens": ud.get("output_tokens") or 0,
                     "cache_read_input_tokens":
                         ud.get("cache_read_input_tokens") or 0,
                     "cache_creation_input_tokens":
                         ud.get("cache_creation_input_tokens") or 0,
                     # inside output_tokens already — reported, not re-billed
                     "reasoning_tokens": self.last_reasoning_tokens}
        return text, usage

    def _call_aiml_responses(self, user_text: str, base: str, key: str):
        """An OpenAI model through AIMLAPI's /v1/responses endpoint with
        reasoning summaries on — the only route that returns any of the
        thinking (see _call_aiml). Effort is left at the provider default so
        a club's spend per turn is what it was before summaries were asked
        for; only the summary is new."""
        from openai import OpenAI
        if self._client is None:
            self._client = OpenAI(base_url=base, api_key=key,
                                  timeout=REQUEST_TIMEOUT_S, max_retries=0)
        kwargs = dict(model=self.model, instructions=self.system_prompt,
                      input=user_text,
                      max_output_tokens=getattr(self, "max_output", None)
                      or 4096)
        if "reasoning" not in self._dropped_params:
            kwargs["reasoning"] = {"summary": "auto"}
        self.last_reasoning = None
        self.last_reasoning_tokens = 0
        resp = self._client.responses.create(**kwargs)
        text_parts, thoughts = [], []
        for item in getattr(resp, "output", None) or []:
            kind = getattr(item, "type", None)
            if kind == "reasoning":
                for part in getattr(item, "summary", None) or []:
                    st = getattr(part, "text", None)
                    if st:
                        thoughts.append(st)
            elif kind == "message":
                for part in getattr(item, "content", None) or []:
                    ct = getattr(part, "text", None)
                    if ct:
                        text_parts.append(ct)
        text = "".join(text_parts)
        if thoughts:
            self.last_reasoning = "\n\n".join(thoughts)
        usage = None
        u = getattr(resp, "usage", None)
        if u is not None:
            idet = getattr(u, "input_tokens_details", None)
            odet = getattr(u, "output_tokens_details", None)
            self.last_reasoning_tokens = int(
                (getattr(odet, "reasoning_tokens", 0) or 0) if odet else 0)
            usage = {"input_tokens": getattr(u, "input_tokens", 0) or 0,
                     "output_tokens": getattr(u, "output_tokens", 0) or 0,
                     "cache_read_input_tokens":
                         (getattr(idet, "cached_tokens", 0) or 0) if idet else 0,
                     "reasoning_tokens": self.last_reasoning_tokens}
        return text, usage

    def _call_mock(self, user_text: str):
        """Offline test double: crude proportional steering, no pathfinding.

        model="flaky" returns an unparseable reply every 7th tick to exercise
        the invalid-action path. model="slow" answers like "ok" but sleeps
        1.2 s first — a synthetic frontier-API latency for validating
        realtime mode without keys.
        """
        self._mock_count += 1
        if self.model == "slow":
            time.sleep(1.2)
        if self.model == "crawl":  # frontier-class latency for watchdog tests
            time.sleep(4.0)
        if self.model == "flaky" and self._mock_count % 7 == 0:
            return "I think I should probably head north-ish?", {"mock": True}
        if '"detections"' in user_text:      # football (rfl-0.3 SDK obs):
            # chase the ball, kick at the goal when on it, scan when blind —
            # enough behaviour that the keyless demo match looks like football
            obs = json.loads(user_text.split(
                "Current observation:\n", 1)[1].split("\n")[0])
            ball = (obs.get("detections") or {}).get("ball")
            gx, gy = obs["you"]["attack_goal_xy"]
            if ball and ball.get("distance_m", 99.0) < 0.9:
                reply = {"skill": "kick_toward", "target": [gx, gy]}
            elif ball:
                reply = {"skill": "go_to_ball"}
            else:
                reply = {"skill": "turn_to", "target": [0.0, 0.0]}
            if self._mock_count % 9 == 3:
                reply["say"] = "on it - cover the goal!"
            return json.dumps(reply), {"mock": True}
        obs = json.loads(user_text.split("Current observation:\n", 1)[1].split("\n")[0])
        px, py = obs["self"]["position"]
        h = float(obs["self"]["heading_rad"])
        tx, ty = obs["next_checkpoint"]["position"]
        interval = float(obs.get("decision_interval_s", 0.5))
        # Interval-aware steering with dead-reckoning — exactly what the
        # prompts teach, so the mock doubles as an offline validator of that
        # guidance at any latency: predict where the in-flight command leaves
        # us, then aim from there.
        last = getattr(self, "_mock_last_cmd", None)
        if last is not None and interval > 0.6:
            rot = last["wz"] * min(interval, 2.0)
            mid = h + rot / 2
            dist = last["vx"] * interval
            px += dist * float(np.cos(mid))
            py += dist * float(np.sin(mid))
            h += rot
        err = wrap_angle(float(np.arctan2(ty - py, tx - px)) - h)
        wz = float(np.clip(err / max(0.2, min(interval, 2.0)), -1.0, 1.0))
        vx = 0.9 * max(0.0, float(np.cos(err)))
        cmd = {"vx": round(vx, 2), "vy": 0.0, "wz": round(wz, 2)}
        self._mock_last_cmd = cmd
        return json.dumps(cmd), {"mock": True}
