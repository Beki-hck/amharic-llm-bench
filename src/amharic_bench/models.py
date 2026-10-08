"""Model clients from evalforge, with a cap on answer length.

Small models often fall into a repetition loop in Amharic ("ከየት ነው ከየት ነው ...")
and generate until the request times out. Capping the output keeps a run from
spending five minutes on one broken answer, and the loop still scores as wrong.
"""

from __future__ import annotations

from dataclasses import dataclass

from evalforge.models import AnthropicModel, Model, OllamaModel, _post_json
from evalforge.models import get_model as _get_model

DEFAULT_MAX_TOKENS = 512


@dataclass
class CappedOllamaModel(OllamaModel):
    max_tokens: int = DEFAULT_MAX_TOKENS

    def generate(self, prompt: str, system: str | None = None) -> str:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        data = _post_json(
            f"{self.host.rstrip('/')}/api/chat",
            {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": self.temperature, "num_predict": self.max_tokens},
            },
            {},
            self.timeout,
        )
        return data["message"]["content"]


def get_model(spec: str, max_tokens: int = DEFAULT_MAX_TOKENS) -> Model:
    """Like evalforge's get_model, but Ollama and Anthropic answers stop at `max_tokens`."""
    model = _get_model(spec)
    if isinstance(model, OllamaModel):
        return CappedOllamaModel(model.model, max_tokens=max_tokens)
    if isinstance(model, AnthropicModel):
        model.max_tokens = max_tokens
    return model
