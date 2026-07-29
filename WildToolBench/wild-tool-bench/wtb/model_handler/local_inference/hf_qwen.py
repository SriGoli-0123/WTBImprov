"""Native Hugging Face inference for Qwen2.5 tool calling.

This handler intentionally bypasses OpenAI-compatible servers and parsers.  It
loads the model once with Transformers, formats the unchanged WTB messages and
supplied tool schemas with Qwen's own tokenizer chat template, and calls
``model.generate`` directly.

The WildToolBench paper reports using the Hugging Face chat template and default
generation settings for open-source models, changing only
``max_new_tokens=512``.  Accordingly, this handler does not add a prompt,
sampling policy, repair pass, or evaluator-derived feature.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid

from copy import deepcopy
from dataclasses import dataclass

from wtb.model_handler.base_handler import BaseHandler


@dataclass(frozen=True)
class NativeGeneration:
    """Small transport object between generation and WTB response parsing."""

    text: str
    prompt_tokens: int
    completion_tokens: int


class HuggingFaceQwenHandler(BaseHandler):
    """Run Qwen2.5 directly through Transformers on one shared model."""

    MAX_NEW_TOKENS = 512
    _TOOL_CALL_RE = re.compile(
        r"<tool_call>\s*(.*?)\s*</tool_call>",
        flags=re.DOTALL,
    )

    def __init__(self, model_name, temperature):
        super().__init__(model_name, temperature)

        # Imports are deliberately lazy.  The scorer and parser tests should
        # not need CUDA, Torch, or the model weights.
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Native Hugging Face inference requires torch, accelerate, "
                "and transformers==4.51.0. Install requirements-hf.txt."
            ) from exc

        model_source = os.getenv("WTB_HF_MODEL_PATH", model_name)
        revision = os.getenv("WTB_HF_REVISION") or None
        local_only = os.getenv(
            "WTB_HF_LOCAL_FILES_ONLY", "0"
        ).strip().lower() in {"1", "true", "yes"}
        dtype_name = os.getenv("WTB_HF_DTYPE", "auto").strip().lower()
        dtype = {
            "auto": "auto",
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }.get(dtype_name)
        if dtype is None:
            raise ValueError(
                "WTB_HF_DTYPE must be auto, bfloat16/bf16, "
                "float16/fp16, or float32/fp32"
            )

        load_kwargs = {
            "local_files_only": local_only,
            "trust_remote_code": True,
        }
        if revision is not None:
            load_kwargs["revision"] = revision
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_source,
            **load_kwargs,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_source,
            torch_dtype=dtype,
            device_map=os.getenv("WTB_HF_DEVICE_MAP", "auto"),
            **load_kwargs,
        )
        self.model.eval()
        self._torch = torch
        # WTB's outer thread pool shares one handler. Serialising generate()
        # prevents eight threads from launching competing forwards against the
        # same model and exhausting GPU memory. Use --num-threads 1 for the
        # reproducibility run; the lock is a final safety boundary.
        self._generation_lock = threading.Lock()

    @staticmethod
    def _native_messages(messages):
        """Convert OpenAI-style history into Qwen chat-template values.

        WTB stores assistant function arguments as JSON strings for API
        compatibility. Qwen's native Jinja template expects argument objects;
        leaving them as strings would double-encode historical tool calls.
        """
        native = deepcopy(messages)
        for message in native:
            if message.get("role") != "assistant":
                continue
            for tool_call in message.get("tool_calls") or []:
                function = tool_call.get("function", {})
                arguments = function.get("arguments")
                if not isinstance(arguments, str):
                    continue
                try:
                    parsed = json.loads(arguments)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    function["arguments"] = parsed
        return native

    def _request_tool_call(self, inference_data):
        """Generate once from the unchanged messages and supplied tools."""
        messages = self._native_messages(inference_data["messages"])
        tools = deepcopy(inference_data["tools"])

        started = time.monotonic()
        with self._generation_lock, self._torch.inference_mode():
            model_inputs = self.tokenizer.apply_chat_template(
                messages,
                tools=tools,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            model_inputs = model_inputs.to(self.model.device)
            prompt_tokens = int(model_inputs["input_ids"].shape[-1])
            generated = self.model.generate(
                **model_inputs,
                max_new_tokens=self.MAX_NEW_TOKENS,
            )
            completion_ids = generated[0][prompt_tokens:]
            completion_tokens = int(completion_ids.shape[-1])
            text = self.tokenizer.decode(
                completion_ids,
                skip_special_tokens=True,
            )
        latency = time.monotonic() - started
        return NativeGeneration(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ), latency

    @classmethod
    def _parse_native_tool_calls(cls, text):
        """Parse only complete native Qwen ``<tool_call>`` blocks.

        This is a parser, not a repair mechanism. If any emitted block is
        malformed, the complete response remains text so the benchmark records
        the model's actual failure instead of receiving a fabricated call.
        """
        matches = cls._TOOL_CALL_RE.findall(text)
        if not matches:
            return None

        calls = []
        for raw in matches:
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                return None
            if not isinstance(payload, dict):
                return None
            name = payload.get("name")
            arguments = payload.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    return None
            if not isinstance(name, str) or not isinstance(arguments, dict):
                return None
            calls.append({
                "id": "hf_call_" + uuid.uuid4().hex,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            })
        return calls

    def _parse_api_response(self, api_response):
        tool_calls = self._parse_native_tool_calls(api_response.text)
        if tool_calls:
            content = self._TOOL_CALL_RE.sub("", api_response.text).strip()
        else:
            content = api_response.text.strip()

        return {
            "reasoning_content": None,
            "content": content,
            "tool_calls": tool_calls,
            "input_token": api_response.prompt_tokens,
            "output_token": api_response.completion_tokens,
        }
