"""Hugging Face wrapper and checkpoint construction for FireRedASR-LLM.

The upstream encoder and adapter stay in the pinned external FireRedASR tree.
No training-engine imports belong in this module.
"""

import copy
import importlib.util
import logging
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import cast

import torch
from transformers import AutoTokenizer, GenerationMixin, PretrainedConfig, PreTrainedModel

from audio_swift.plugins.fireredasr_llm.audio import DEFAULT_SPEECH_TOKEN

logger = logging.getLogger(__name__)


def ensure_firered_source() -> Path | None:
    """Use an explicit source tree or the current project's prepared checkout."""
    configured = os.environ.get("FIRERED_ASR_REPO")
    candidate = Path(configured).expanduser().resolve() if configured else Path.cwd() / ".vendor" / "FireRedASR"
    if configured or candidate.is_dir():
        if not (candidate / "fireredasr").is_dir():
            raise ImportError(f"FIRERED_ASR_REPO must contain the fireredasr package: {candidate}")
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        return candidate
    if importlib.util.find_spec("fireredasr") is not None:
        return None
    raise ImportError(
        "FireRedASR source is missing. Run bash examples/fireredasr_llm/prepare_source.sh from the project root, "
        "or set FIRERED_ASR_REPO to an existing checkout. Install audio dependencies with uv sync --locked."
    )


def resolve_llm_dir(model_dir: str | Path) -> Path:
    for name in ("Qwen2-7B-Instruct", "qwen2-7b-instruct"):
        candidate = Path(model_dir) / name
        if (candidate / "config.json").is_file():
            return candidate
    raise FileNotFoundError(f"Expected {model_dir}/Qwen2-7B-Instruct/config.json; see the FireRed recipe.")


def load_model_package(model_dir: str | Path):
    checkpoint = Path(model_dir) / "model.pth.tar"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing FireRedASR checkpoint: {checkpoint}")
    # Official checkpoints pickle argparse.Namespace. Keep PyTorch >= 2.6
    # compatibility local to these trusted model checkpoints.
    return torch.load(checkpoint, map_location="cpu", weights_only=False)


@lru_cache(maxsize=1)
def load_model_args(model_dir: str):
    # Do not retain checkpoint tensors merely to read the downsample rate.
    return load_model_package(model_dir)["args"]


def load_tokenizer(model_dir: str | Path):
    tokenizer = AutoTokenizer.from_pretrained(resolve_llm_dir(model_dir))
    if DEFAULT_SPEECH_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [DEFAULT_SPEECH_TOKEN]})
    return tokenizer


class FireRedASRLLMForCausalLM(PreTrainedModel, GenerationMixin):
    """Compute causal loss after replacing aligned speech placeholders."""

    config_class = PretrainedConfig
    supports_gradient_checkpointing = True

    def __init__(self, firered_model):
        super().__init__(firered_model.llm.config)
        self.firered_model = firered_model
        self.generation_config = firered_model.llm.generation_config

    def get_input_embeddings(self):
        return self.firered_model.llm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.firered_model.llm.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.firered_model.llm.get_output_embeddings()

    def set_output_embeddings(self, value):
        self.firered_model.llm.set_output_embeddings(value)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.firered_model.llm.prepare_inputs_for_generation(*args, **kwargs)

    def save_pretrained(self, save_directory, *args, **kwargs):
        raise NotImplementedError(
            "FireRedASR-LLM full checkpoint export is unsupported. Save a lora or lora_llm adapter "
            "and retain the original FireRedASR/Qwen checkpoint directory."
        )

    def forward(self, input_ids, attention_mask=None, labels=None, padded_feat=None, feat_lengths=None, **kwargs):
        if padded_feat is None or feat_lengths is None:
            raise ValueError("FireRedASR SFT requires padded_feat and feat_lengths for every sample.")
        encoder = self.firered_model.encoder
        parameter = next(encoder.parameters())
        padded_feat = padded_feat.to(device=parameter.device, dtype=parameter.dtype)
        feat_lengths = feat_lengths.to(parameter.device)
        encoder_outs, enc_lengths, _ = encoder(padded_feat, feat_lengths)
        speech_features, speech_lens = self.firered_model.encoder_projector(encoder_outs, enc_lengths)
        inputs_embeds = self.get_input_embeddings()(input_ids)
        speech_mask = input_ids.eq(self.config.default_speech_token_id)
        counts = speech_mask.sum(dim=-1).to(speech_lens.device)
        if not torch.equal(counts, speech_lens):
            raise ValueError(
                f"Speech placeholders and encoded audio must align: tokens={counts.tolist()}, "
                f"audio={speech_lens.tolist()}. Use truncation_strategy=raise and disable packing."
            )
        features = torch.cat([speech_features[i, :length] for i, length in enumerate(speech_lens.tolist())])
        inputs_embeds = inputs_embeds.masked_scatter(
            speech_mask.unsqueeze(-1).expand_as(inputs_embeds), features.to(inputs_embeds)
        )
        kwargs.pop("inputs_embeds", None)
        return self.firered_model.llm(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, **kwargs
        )


def build_model(model_dir, tokenizer, *, torch_dtype=None, device=None, attn_impl: str | None = "eager"):
    """Load the official delta, merge its LoRA, and expose a trainable base."""
    ensure_firered_source()
    try:
        from fireredasr.models.fireredasr_aed import FireRedAsrAed
        from fireredasr.models.fireredasr_llm import FireRedAsrLlm
    except ImportError as exc:
        raise ImportError("FireRedASR dependencies are incomplete; run uv sync --locked.") from exc

    class CompatibleFireRedAsrLlm(FireRedAsrLlm):
        @classmethod
        def load_encoder(cls, model_path):
            package = torch.load(model_path, map_location="cpu", weights_only=False)
            encoder_model = FireRedAsrAed.from_args(package["args"])
            if "model_state_dict" in package:
                encoder_model.load_state_dict(package["model_state_dict"], strict=False)
            return encoder_model.encoder, encoder_model.encoder.odim

    encoder_path = Path(model_dir) / "asr_encoder.pth.tar"
    if not encoder_path.is_file():
        raise FileNotFoundError(f"Missing FireRedASR encoder checkpoint: {encoder_path}")
    if attn_impl not in (None, "eager", "sdpa", "flash_attn", "flash_attention_2"):
        raise ValueError("FireRedASR supports attn_impl=eager, sdpa or flash_attention_2.")
    if device == "auto" or isinstance(device, dict):
        raise ValueError("Use one device per process; device_map=auto/dict is unsupported.")
    package = load_model_package(model_dir)
    args = copy.deepcopy(package["args"])
    args.encoder_path = str(encoder_path)
    args.llm_dir = str(resolve_llm_dir(model_dir))
    args.use_flash_attn = attn_impl in ("flash_attn", "flash_attention_2")
    args.use_fp16 = torch_dtype == torch.float16
    if args.use_flash_attn and torch_dtype != torch.float16:
        raise ValueError("Upstream FireRedASR flash attention requires torch_dtype=float16.")
    native = CompatibleFireRedAsrLlm.from_args(args)
    if len(tokenizer) > native.llm.get_input_embeddings().weight.shape[0]:
        native.llm.resize_token_embeddings(len(tokenizer))
    incompatible = native.load_state_dict(package["model_state_dict"], strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(f"Unexpected FireRedASR checkpoint keys: {incompatible.unexpected_keys[:10]}")
    logger.info("Loaded FireRedASR delta checkpoint; missing keys: %s", len(incompatible.missing_keys))
    if hasattr(native.llm, "merge_and_unload"):
        native.llm = native.llm.merge_and_unload()
    model = FireRedASRLLMForCausalLM(native)
    # Upstream from_args only distinguishes FlashAttention from eager. Apply
    # the requested Qwen backend after merging its pretrained adapter; the
    # wrapper shares this config and delegates all LLM attention to Qwen.
    requested_attn_impl = "flash_attention_2" if attn_impl == "flash_attn" else (attn_impl or "eager")
    native.llm.set_attn_implementation(requested_attn_impl)
    if native.llm.config._attn_implementation != requested_attn_impl:
        raise ValueError(f"FireRedASR LLM did not apply attn_impl={requested_attn_impl}.")
    if torch_dtype is not None:
        cast(torch.nn.Module, model).to(dtype=torch_dtype)
    if device is not None:
        cast(torch.nn.Module, model).to(device=device)
    model.config.default_speech_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_SPEECH_TOKEN)
    model.config.use_cache = False
    return model
