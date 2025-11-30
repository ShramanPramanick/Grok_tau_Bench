import os

from tau_bench.model_utils.api.datapoint import Datapoint
from tau_bench.model_utils.model.chat import ChatModel, Message
from tau_bench.model_utils.model.completion import approx_cost_for_datapoint, approx_prompt_str
from tau_bench.model_utils.model.general_model import wrap_temperature
from tau_bench.model_utils.model.utils import approx_num_tokens

# Grok (xAI) configuration
DEFAULT_GROK_MODEL = "grok-4-1-fast-reasoning"
API_KEY_ENV_VAR = "XAI_API_KEY"
API_BASE_URL = "https://api.x.ai/v1"

# Pricing from xAI docs: $0.20 / 1M input tokens for grok-4-1-fast-reasoning
PRICE_PER_INPUT_TOKEN_MAP = {
    "grok-4-1-fast-reasoning": 0.20 / 1_000_000,
    "grok-4-1-fast-non-reasoning": 0.20 / 1_000_000,
}
INPUT_PRICE_PER_TOKEN_FALLBACK = 0.20 / 1_000_000

# Rough capability score (relative to GPT-4o-like models)
CAPABILITY_SCORE_MAP = {
    "grok-4-1-fast-reasoning": 0.9,
    "grok-4-1-fast-non-reasoning": 0.9,
}
CAPABILITY_SCORE_FALLBACK = 0.8

# TODO: implement if you care about latency estimates
LATENCY_MS_PER_OUTPUT_TOKEN_MAP = {}
LATENCY_MS_PER_OUTPUT_TOKEN_FALLBACK = 0.0

# Context window: 2M tokens for Grok 4.1 Fast
MAX_CONTEXT_LENGTH_MAP = {
    "grok-4-1-fast-reasoning": 2_000_000,
    "grok-4-1-fast-non-reasoning": 2_000_000,
}
MAX_CONTEXT_LENGTH_FALLBACK = 2_000_000


class GrokModel(ChatModel):
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        # xAI is OpenAI-compatible; we just change base_url + API key
        from openai import AsyncOpenAI, OpenAI

        if model is None:
            self.model = DEFAULT_GROK_MODEL
        else:
            self.model = model

        if api_key is None:
            api_key = os.getenv(API_KEY_ENV_VAR)
            if api_key is None:
                raise ValueError(f"{API_KEY_ENV_VAR} environment variable is not set")

        self.client = OpenAI(api_key=api_key, base_url=API_BASE_URL)
        self.async_client = AsyncOpenAI(api_key=api_key, base_url=API_BASE_URL)
        self.temperature = temperature

    def generate_message(
        self,
        messages: list[Message],
        force_json: bool,
        temperature: float | None = None,
    ) -> Message:
        if temperature is None:
            temperature = self.temperature
        msgs = self.build_generate_message_state(messages)
        res = self.client.chat.completions.create(
            model=self.model,
            messages=msgs,
            temperature=wrap_temperature(temperature),
            # xAI supports OpenAI-style JSON mode; keep this identical
            response_format={"type": "json_object" if force_json else "text"},
        )
        return self.handle_generate_message_response(
            prompt=msgs,
            content=res.choices[0].message.content,
            force_json=force_json,
        )

    def get_approx_cost(self, dp: Datapoint) -> float:
        cost_per_token = PRICE_PER_INPUT_TOKEN_MAP.get(
            self.model, INPUT_PRICE_PER_TOKEN_FALLBACK
        )
        return approx_cost_for_datapoint(dp=dp, price_per_input_token=cost_per_token)

    def get_latency(self, dp: Datapoint) -> float:
        latency_per_output_token = LATENCY_MS_PER_OUTPUT_TOKEN_MAP.get(
            self.model, LATENCY_MS_PER_OUTPUT_TOKEN_FALLBACK
        )
        return approx_cost_for_datapoint(
            dp=dp, price_per_input_token=latency_per_output_token
        )

    def get_capability(self) -> float:
        return CAPABILITY_SCORE_MAP.get(self.model, CAPABILITY_SCORE_FALLBACK)

    def supports_dp(self, dp: Datapoint) -> bool:
        prompt = approx_prompt_str(dp)
        return approx_num_tokens(prompt) <= MAX_CONTEXT_LENGTH_MAP.get(
            self.model, MAX_CONTEXT_LENGTH_FALLBACK
        )
