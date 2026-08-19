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


PRICES = [
    ("claude-fable-5", 10.0, 50.0),
    # TODO(robin): confirm launch pricing for the season-2 gaffer models
    ("gpt-5.6", 5.0, 20.0),          # placeholder until confirmed
    ("gemini-3.7-flash", 0.5, 3.0),  # placeholder until confirmed
    ("glm-5.3", 1.0, 4.0),           # placeholder until confirmed
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
