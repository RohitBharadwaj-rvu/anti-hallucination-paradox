"""
src/api_router.py — OpenAI-compatible API router for three providers:
  nim / fireworks (cloud, billed) and lmstudio (local, not billed).
Implements:
  - Separate target & judge semaphore/rate gates
  - LM Studio pre-flight (GET /v1/models, require_loaded_model, warmup)
  - Auto-resolve model ID for lmstudio on 404/400
  - Tenacity retry on status codes + timeout
  - ProviderBudget token kill-switch (cloud only)
  - Reasoning trace extraction (both API reasoning_content and <think>/<thought>/tagless CoT)
  - Automatic anti-overthinking retry if a reasoning model truncates mid-CoT
"""
import asyncio
import threading
import time
from typing import Any

import requests
import openai
from openai import OpenAI, BadRequestError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
    RetryError,
)

from src.utils import get_logger
from src.env_loader import get_api_key, get_base_url, is_billed
from src.thinking import strip_thinking, count_answer_tokens

logger = get_logger("ahp.api_router")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class BudgetExceeded(Exception):
    """Raised when a cloud provider's token budget is exceeded."""


class LMStudioNotReachable(Exception):
    """Raised if LM Studio server is not reachable at startup."""


class LMStudioNoModelLoaded(Exception):
    """Raised if LM Studio server has no model loaded."""


# ---------------------------------------------------------------------------
# ProviderBudget — token kill-switch (cloud only)
# ---------------------------------------------------------------------------

class ProviderBudget:
    """
    Accumulates total_tokens for billed providers only. Checks budget before each call.
    """

    def __init__(self, settings: Any):
        self._lock = threading.Lock()
        self._totals: dict[str, int] = {}
        self._limits = {
            "nim": settings.budget.get("nim_max_tokens", 900_000),
            "fireworks": settings.budget.get("fireworks_max_tokens", 2_000_000),
        }
        self._hard_stop_fraction: float = settings.budget.get("hard_stop_at_fraction", 0.95)

    def check(self, provider: str) -> None:
        """Raise BudgetExceeded if this provider is over budget."""
        if not is_billed(provider):
            return
        limit = self._limits.get(provider, 0)
        if limit == 0:
            return
        with self._lock:
            used = self._totals.get(provider, 0)
        fraction = used / limit
        if fraction >= self._hard_stop_fraction:
            raise BudgetExceeded(
                f"Provider '{provider}' budget exceeded: "
                f"{used:,} / {limit:,} tokens used ({fraction:.1%} >= {self._hard_stop_fraction:.1%})."
            )

    def add(self, provider: str, tokens: int) -> None:
        if not is_billed(provider):
            return
        with self._lock:
            self._totals[provider] = self._totals.get(provider, 0) + tokens

    def fraction(self, provider: str) -> float:
        limit = self._limits.get(provider, 0)
        if limit == 0:
            return 0.0
        with self._lock:
            return self._totals.get(provider, 0) / limit

    def used(self, provider: str) -> int:
        with self._lock:
            return self._totals.get(provider, 0)


# ---------------------------------------------------------------------------
# RPM rate limiter (simple token bucket per provider)
# ---------------------------------------------------------------------------

class RpmGate:
    """Simple inter-request delay gate to respect RPM limits."""

    def __init__(self, rpm: int):
        self._min_interval = (60.0 / rpm) if rpm > 0 else 0.0
        self._lock = threading.Lock()
        self._last_call: float = 0.0

    def acquire(self) -> None:
        if self._min_interval == 0:
            return
        with self._lock:
            now = time.monotonic()
            sleep_for = self._min_interval - (now - self._last_call)
            if sleep_for > 0:
                time.sleep(sleep_for)
            self._last_call = time.monotonic()


# ---------------------------------------------------------------------------
# APIRouter — the main class
# ---------------------------------------------------------------------------

class APIRouter:
    """
    Manages OpenAI clients for all providers, dual semaphore gates, pre-flight,
    auto-resolve, retry, and budget tracking.
    """

    def __init__(self, settings: Any, run_cfg: dict):
        self._settings = settings
        self._run_cfg = run_cfg

        target_cfg = run_cfg["target"]
        judge_cfg = run_cfg["judge"]
        self._target_provider: str = target_cfg["provider"]
        self._target_model_id: str = target_cfg["model_id"]
        self._judge_provider: str = judge_cfg["provider"]
        self._judge_model_id: str = judge_cfg["model_id"]

        # Build one client per provider (lazy; only those referenced)
        self._clients: dict[str, OpenAI] = {}
        self._resolved_judge_model: str = self._judge_model_id
        self._loaded_judge_models: list[str] = []

        # Semaphore gates
        conc_target = int(getattr(settings, "concurrency_target", 1))
        conc_judge = int(getattr(settings, "concurrency_judge", 1))
        self._target_semaphore = threading.Semaphore(conc_target)
        self._judge_semaphore = threading.Semaphore(conc_judge)

        # RPM gates
        rate = getattr(settings, "rate", {})
        self._rpm_gates: dict[str, RpmGate] = {
            "nim": RpmGate(rate.get("nim_rpm", 35)),
            "fireworks": RpmGate(rate.get("fireworks_rpm", 60)),
            "lmstudio": RpmGate(0),
        }

        # Budget
        self.budget = ProviderBudget(settings)

        # Retry config
        retry_cfg = getattr(settings, "retry", {})
        self._retry_attempts: int = retry_cfg.get("attempts", 6)
        self._retry_wait_min: float = retry_cfg.get("wait_min_s", 2)
        self._retry_wait_max: float = retry_cfg.get("wait_max_s", 30)
        self._retry_backoff: float = retry_cfg.get("backoff_multiplier", 2)
        self._retry_statuses: set[int] = set(retry_cfg.get("retry_on_status", [429, 500, 502, 503, 504]))
        self._retry_on_timeout: bool = retry_cfg.get("retry_on_timeout", True)

        # LM Studio config
        lmstudio_cfg = getattr(settings, "lmstudio", {})
        self._lmstudio_timeout: int = lmstudio_cfg.get("timeout_s", 300)
        self._lmstudio_warmup: bool = lmstudio_cfg.get("warmup", True)
        self._lmstudio_auto_resolve: bool = lmstudio_cfg.get("auto_resolve_model", True)
        self._lmstudio_require_loaded: bool = lmstudio_cfg.get("require_loaded_model", True)

        # Decoding config
        decoding = getattr(settings, "decoding", {})
        self._temperature: float = decoding.get("temperature", 0.0)
        self._temperature_fallback: float = decoding.get("temperature_fallback", 0.1)
        self._seed: int = decoding.get("seed", 42)
        self._max_tokens_target: int = decoding.get("max_tokens_target", 8192)
        self._max_tokens_judge: int = decoding.get("max_tokens_judge", 256)
        self._top_p: float = decoding.get("top_p", 1.0)

        # Judge latency tracking
        self._judge_latencies: list[float] = []
        self._latency_lock = threading.Lock()

        # Target latency tracking
        self._target_latencies: list[float] = []

        self._preflight_done: bool = False

    # -----------------------------------------------------------------------
    # Client factory
    # -----------------------------------------------------------------------

    def _get_client(self, provider: str) -> OpenAI:
        if provider in self._clients:
            return self._clients[provider]

        api_key = get_api_key(provider)
        base_url = get_base_url(provider)
        timeout = self._lmstudio_timeout if provider == "lmstudio" else None

        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )
        self._clients[provider] = client
        logger.debug("Created OpenAI client for provider=%s base_url=%s", provider, base_url)
        return client

    # -----------------------------------------------------------------------
    # LM Studio pre-flight
    # -----------------------------------------------------------------------

    def preflight_lmstudio(self) -> None:
        """
        Run LM Studio pre-flight: GET /v1/models, require >=1 model, warmup.
        Raises LMStudioNotReachable or LMStudioNoModelLoaded on failure.
        """
        if self._judge_provider != "lmstudio":
            self._preflight_done = True
            return

        base_url = get_base_url("lmstudio")
        models_url = base_url.rstrip("/") + "/models"

        try:
            resp = requests.get(models_url, timeout=10)
            resp.raise_for_status()
            models_data = resp.json().get("data", [])
        except requests.exceptions.ConnectionError:
            raise LMStudioNotReachable(
                f"LM Studio server not reachable at {base_url}. "
                "Start the local server (LM Studio -> Developer -> Start Server) "
                "and ensure LMSTUDIO_BASE_URL matches the port."
            )
        except Exception as e:
            raise LMStudioNotReachable(
                f"LM Studio server not reachable at {base_url}. "
                f"Start the local server (LM Studio -> Developer -> Start Server) "
                f"and ensure LMSTUDIO_BASE_URL matches the port. Error: {e}"
            )

        self._loaded_judge_models = [m.get("id", "") for m in models_data]

        if not self._loaded_judge_models and self._lmstudio_require_loaded:
            raise LMStudioNoModelLoaded(
                "No model is loaded in LM Studio. Load a model (e.g. qwen2.5-7b-instruct) "
                "and start the server before running."
            )

        logger.info(
            "LM Studio pre-flight: reachable at %s | %d model(s) loaded: %s",
            base_url,
            len(self._loaded_judge_models),
            self._loaded_judge_models,
        )

        # If configured model_id not in loaded list, try first
        if (
            self._lmstudio_auto_resolve
            and self._loaded_judge_models
            and self._judge_model_id not in self._loaded_judge_models
        ):
            resolved = self._loaded_judge_models[0]
            logger.info(
                "judge model auto-resolved: %s -> %s",
                self._judge_model_id,
                resolved,
            )
            self._resolved_judge_model = resolved
        else:
            self._resolved_judge_model = self._judge_model_id

        # Warmup
        if self._lmstudio_warmup:
            t0 = time.monotonic()
            try:
                client = self._get_client("lmstudio")
                client.chat.completions.create(
                    model=self._resolved_judge_model,
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=1,
                )
                warmup_ms = (time.monotonic() - t0) * 1000
                logger.info("LM Studio warmup complete: %.0f ms", warmup_ms)
            except Exception as e:
                logger.warning("LM Studio warmup call failed (non-fatal): %s", e)

        self._preflight_done = True

    # -----------------------------------------------------------------------
    # Should-retry predicate
    # -----------------------------------------------------------------------

    def _should_retry(self, exc: Exception) -> bool:
        if isinstance(exc, openai.APIStatusError):
            return exc.status_code in self._retry_statuses
        if self._retry_on_timeout and isinstance(exc, openai.APITimeoutError):
            return True
        if isinstance(exc, openai.APIConnectionError):
            return True
        return False

    # -----------------------------------------------------------------------
    # Low-level call (with retry + temperature fallback + seed handling)
    # -----------------------------------------------------------------------

    def _raw_call(
        self,
        client: OpenAI,
        provider: str,
        model_id: str,
        messages: list[dict],
        max_tokens: int,
    ) -> tuple[str, dict]:
        """
        Make one API call with retry. Returns (text, usage_dict).
        Handles temperature=0.0 fallback, seed swallowing, and reasoning_content capture.
        """
        attempts = self._retry_attempts

        def _do_call(temperature: float) -> openai.types.chat.ChatCompletion:
            kwargs: dict = {
                "model": model_id,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": self._top_p,
            }
            try:
                return client.chat.completions.create(**kwargs, seed=self._seed)
            except (BadRequestError, openai.APIStatusError) as e:
                if "seed" in str(e).lower():
                    return client.chat.completions.create(**kwargs)
                raise

        last_exc: Exception | None = None
        temperature = self._temperature
        wait = self._retry_wait_min

        for attempt in range(attempts):
            try:
                completion = _do_call(temperature)
                msg = completion.choices[0].message
                text = (msg.content or "").strip()
                # Extract provider-separated reasoning_content if present (Fireworks / DeepSeek / LM Studio)
                extra = getattr(msg, "model_extra", None) or {}
                api_reasoning = (
                    getattr(msg, "reasoning_content", None)
                    or getattr(msg, "reasoning", None)
                    or extra.get("reasoning_content")
                    or extra.get("reasoning")
                    or ""
                ).strip()
                usage_obj = completion.usage
                usage = {
                    "prompt_tokens": getattr(usage_obj, "prompt_tokens", 0) or 0,
                    "completion_tokens": getattr(usage_obj, "completion_tokens", 0) or 0,
                    "total_tokens": getattr(usage_obj, "total_tokens", 0) or 0,
                    "api_reasoning_content": api_reasoning,
                }
                return text, usage

            except BadRequestError as e:
                if temperature == 0.0 and ("temperature" in str(e).lower() or "invalid" in str(e).lower()):
                    logger.debug("Temperature 0.0 rejected by %s; retrying with %.1f", provider, self._temperature_fallback)
                    temperature = self._temperature_fallback
                    last_exc = e
                    continue
                raise

            except Exception as e:
                if self._should_retry(e):
                    last_exc = e
                    logger.warning(
                        "Attempt %d/%d failed for provider=%s model=%s: %s. Retrying in %.0fs...",
                        attempt + 1, attempts, provider, model_id, type(e).__name__, wait,
                    )
                    time.sleep(min(wait, self._retry_wait_max))
                    wait = min(wait * self._retry_backoff, self._retry_wait_max)
                else:
                    raise

        raise last_exc or RuntimeError("All retry attempts exhausted")

    # -----------------------------------------------------------------------
    # Public call — target
    # -----------------------------------------------------------------------

    def call_target(
        self,
        messages: list[dict],
    ) -> tuple[str, dict]:
        """
        Call the target model. Acquires target_semaphore + RPM gate.
        Checks and updates budget. Returns (answer_text, usage).

        Separates thinking traces (both API `reasoning_content` and inline `<think>` /
        tagless truncated CoT) from the final answer. If the model truncated mid-CoT
        and produced an empty answer, retries once with a direct conciseness instruction.
        """
        provider = self._target_provider
        model_id = self._target_model_id

        self.budget.check(provider)

        def _invoke(msgs: list[dict]) -> tuple[str, str, dict]:
            rpm_gate = self._rpm_gates.get(provider, RpmGate(0))
            with self._target_semaphore:
                rpm_gate.acquire()
                client = self._get_client(provider)
                t0 = time.monotonic()
                r_text, u_dict = self._raw_call(client, provider, model_id, msgs, self._max_tokens_target)
                elapsed = time.monotonic() - t0
            with self._latency_lock:
                self._target_latencies.append(elapsed)

            ans_txt, inline_think = strip_thinking(r_text)
            api_think = u_dict.pop("api_reasoning_content", "")
            combined_think = "\n\n".join(p for p in (api_think, inline_think) if p).strip()
            return r_text, ans_txt, {**u_dict, "thinking_text": combined_think}

        raw_text, answer_text, usage = _invoke(messages)

        # If reasoning model truncated mid-thought (answer_text is empty, thinking_text non-empty),
        # retry once with an explicit anti-overthinking instruction to get a clean final answer.
        if not answer_text.strip() and usage.get("thinking_text"):
            logger.info("Target truncated mid-CoT without final answer; retrying with concise instruction.")
            retry_msgs = [
                messages[0],
                {
                    "role": "user",
                    "content": (
                        messages[-1]["content"]
                        + "\n\n[IMPORTANT: Do not overthink. Output your final concise answer immediately in a few words.]"
                    ),
                },
            ]
            raw_text2, answer_text2, usage2 = _invoke(retry_msgs)
            if answer_text2.strip():
                # Keep original thinking trace if second call had none, and sum token usage
                kept_think = usage2.get("thinking_text") or usage.get("thinking_text", "")
                raw_text = raw_text2
                answer_text = answer_text2
                usage = {
                    "prompt_tokens": usage.get("prompt_tokens", 0) + usage2.get("prompt_tokens", 0),
                    "completion_tokens": usage2.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0) + usage2.get("total_tokens", 0),
                    "thinking_text": kept_think,
                }

        raw_completion_tokens = usage.get("completion_tokens", 0)
        usage["raw_completion_tokens"] = raw_completion_tokens
        thinking_text = usage.get("thinking_text", "")

        if thinking_text and raw_text:
            full_combined = f"{thinking_text}\n{answer_text}"
            answer_tokens = count_answer_tokens(full_combined, answer_text, raw_completion_tokens)
            usage["completion_tokens"] = answer_tokens
            usage["total_tokens"] = usage.get("prompt_tokens", 0) + answer_tokens
            self.budget.add(provider, usage["total_tokens"])
        else:
            self.budget.add(provider, usage.get("total_tokens", 0))

        return answer_text, usage

    # -----------------------------------------------------------------------
    # Public call — judge
    # -----------------------------------------------------------------------

    def call_judge(
        self,
        messages: list[dict],
        nudge_choices: str | None = None,
    ) -> tuple[str, dict, float]:
        """
        Call the judge model. Acquires judge_semaphore. Handles auto-resolve on 404.
        Strips any <think>/<thought> reasoning traces before returning the verdict text.
        Returns (text, usage, latency_ms) — 3-tuple; latency_ms is per-call wall time.
        """
        provider = self._judge_provider
        model_id = self._resolved_judge_model

        with self._judge_semaphore:
            client = self._get_client(provider)
            t0 = time.monotonic()
            try:
                text, usage = self._raw_call(client, provider, model_id, messages, self._max_tokens_judge)
            except openai.APIStatusError as e:
                if (
                    provider == "lmstudio"
                    and self._lmstudio_auto_resolve
                    and e.status_code in (400, 404)
                    and self._loaded_judge_models
                ):
                    resolved = self._loaded_judge_models[0]
                    if resolved != model_id:
                        logger.info(
                            "judge model auto-resolved on call: %s -> %s",
                            model_id, resolved,
                        )
                        self._resolved_judge_model = resolved
                        text, usage = self._raw_call(client, provider, resolved, messages, self._max_tokens_judge)
                    else:
                        raise
                else:
                    raise
            elapsed = time.monotonic() - t0

        # Strip any <think> or <thought> blocks emitted by local thinking judges (e.g. gemma-4)
        clean_text, _ = strip_thinking(text)
        if clean_text.strip():
            text = clean_text.strip()

        elapsed_ms = elapsed * 1000
        with self._latency_lock:
            self._judge_latencies.append(elapsed)

        return text, usage, elapsed_ms

    # -----------------------------------------------------------------------
    # Stats accessors
    # -----------------------------------------------------------------------

    def mean_judge_latency_ms(self) -> float:
        with self._latency_lock:
            if not self._judge_latencies:
                return 0.0
            return sum(self._judge_latencies) / len(self._judge_latencies) * 1000

    def mean_target_latency_ms(self) -> float:
        with self._latency_lock:
            if not self._target_latencies:
                return 0.0
            return sum(self._target_latencies) / len(self._target_latencies) * 1000

    def total_judge_calls(self) -> int:
        with self._latency_lock:
            return len(self._judge_latencies)

    def total_target_calls(self) -> int:
        with self._latency_lock:
            return len(self._target_latencies)

    @property
    def resolved_judge_model(self) -> str:
        return self._resolved_judge_model

    @property
    def loaded_judge_models(self) -> list[str]:
        return self._loaded_judge_models

    @property
    def judge_provider(self) -> str:
        return self._judge_provider

    @property
    def target_provider(self) -> str:
        return self._target_provider

    @property
    def target_model_id(self) -> str:
        return self._target_model_id
