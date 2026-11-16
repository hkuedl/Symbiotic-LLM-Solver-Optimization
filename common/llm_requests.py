"""Negotiate optional capabilities of OpenAI-compatible model endpoints."""

import asyncio
import json
import re


def logprob_content(response):
    """Return token log-probabilities only when the endpoint actually supplies them."""
    choices = getattr(response, "choices", None)
    if not choices:
        return None
    probabilities = getattr(choices[0], "logprobs", None)
    return getattr(probabilities, "content", None) or None


def _unsupported_parameter(error, options):
    if getattr(error, "status_code", None) not in (400, 422):
        return None
    body = getattr(error, "body", None)
    detail = (str(error) + " " + json.dumps(body, default=str)).lower()
    rejection = (
        "not supported",
        "unsupported",
        "not support",
        "does not support",
        "not available",
        "unavailable",
        "unknown parameter",
        "unrecognized",
        "not allowed",
        "not permitted",
        "unexpected",
        "extra inputs",
    )
    if not any(phrase in detail for phrase in rejection):
        return None
    for parameter in ("top_logprobs", "logprobs", "response_format"):
        if parameter in options and re.search(r"\b" + parameter + r"\b", detail):
            return parameter
    return None


class ChatRequests:
    """Cache capabilities per client and model, without a provider-name blacklist.

    The first real request probes logprobs. Subsequent calls reuse the result.
    Only explicit unsupported-parameter responses trigger capability fallback.
    """

    def __init__(self, client):
        self.client = client
        self._capabilities = {}
        self._locks = {}

    def _options(self, kwargs):
        options = dict(kwargs)
        model = options["model"]
        if self._capabilities.get((model, "logprobs")) is not False:
            options["logprobs"] = True
            if self._capabilities.get((model, "top_logprobs")) is not False:
                options["top_logprobs"] = 5
        if self._capabilities.get((model, "response_format")) is False:
            options.pop("response_format", None)
        return options

    def _fallback(self, error, options):
        parameter = _unsupported_parameter(error, options)
        if parameter is None:
            return False
        self._capabilities[(options["model"], parameter)] = False
        options.pop(parameter, None)
        if parameter == "logprobs":
            options.pop("top_logprobs", None)
        return True

    def _record(self, response, options):
        if options.get("logprobs"):
            self._capabilities[(options["model"], "logprobs")] = bool(logprob_content(response))
        return response

    def create(self, **kwargs):
        """Call a synchronous endpoint with capability-specific retries."""
        options = self._options(kwargs)
        while True:
            try:
                response = self.client.chat.completions.create(**options)
            except Exception as error:
                if not self._fallback(error, options):
                    raise
            else:
                return self._record(response, options)

    async def _create_async(self, kwargs):
        options = self._options(kwargs)
        while True:
            try:
                response = await self.client.chat.completions.create(**options)
            except Exception as error:
                if not self._fallback(error, options):
                    raise
            else:
                return self._record(response, options)

    async def acreate(self, **kwargs):
        """Probe once before releasing the rest of a concurrent sample batch."""
        model = kwargs["model"]
        if (model, "logprobs") in self._capabilities:
            return await self._create_async(kwargs)
        lock_key = (asyncio.get_running_loop(), model)
        lock = self._locks.setdefault(lock_key, asyncio.Lock())
        async with lock:
            if (model, "logprobs") not in self._capabilities:
                return await self._create_async(kwargs)
        return await self._create_async(kwargs)
