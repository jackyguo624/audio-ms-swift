"""Swift registration adapter for the project's FireRed model and audio code.

Import this module through Swift's --external_plugins only for FireRed recipes.
The external upstream source tree is needed when loading the model, not here.
"""

import copy
import os
from pathlib import Path
from typing import Any, cast

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoConfig, PreTrainedTokenizerBase

from swift.dataset import DatasetMeta, ResponsePreprocessor, register_dataset
from swift.model import Model, ModelGroup, ModelLoader, ModelMeta, MultiModelKeys, register_model, register_model_arch
from swift.template import Template, TemplateMeta, register_template
from swift.template.template_inputs import StdTemplateInputs

from audio_swift.plugins.fireredasr_llm.audio import (
    DEFAULT_SPEECH_TOKEN,
    FireRedFeatureExtractor,
    clean_transcript,
    speech_token_length,
)
from audio_swift.plugins.fireredasr_llm.model import build_model, load_model_args, load_tokenizer, resolve_llm_dir


class FireRedASRLLMLoader(ModelLoader):
    def get_config(self, model_dir: str):
        return AutoConfig.from_pretrained(resolve_llm_dir(model_dir))

    def get_processor(self, model_dir: str, config):
        return load_tokenizer(model_dir)

    def get_model(self, model_dir: str, config, processor, model_kwargs):
        if model_kwargs.get("quantization_config") is not None or self.return_dummy_model:
            raise ValueError("FireRedASR does not support quantization or dummy-model loading.")
        return build_model(
            model_dir,
            processor,
            torch_dtype=self.torch_dtype,
            device=model_kwargs.get("device_map"),
            attn_impl=self.attn_impl,
        )


class FireRedASRTemplate(Template):
    support_padding_free = False
    skip_prompt = False

    def init_processor(self, processor):
        super().init_processor(processor)
        tokenizer = cast(PreTrainedTokenizerBase, processor)
        speech_token_id = tokenizer.get_vocab().get(DEFAULT_SPEECH_TOKEN)
        if not isinstance(speech_token_id, int):
            raise ValueError("FireRedASR tokenizer must define the <speech> token.")
        self._speech_token_id = speech_token_id
        if self.padding_free or self.truncation_strategy != "raise":
            raise ValueError("FireRedASR SFT requires padding_free=false and truncation_strategy=raise.")
        model_dir = self.model_info.model_dir
        self._feat_extractor = FireRedFeatureExtractor(Path(model_dir) / "cmvn.ark")
        self.encoder_downsample_rate = int(getattr(load_model_args(model_dir), "encoder_downsample_rate", 2))
        if self.encoder_downsample_rate < 1:
            raise ValueError("encoder_downsample_rate must be positive.")

    def replace_tag(self, media_type: str, index: int, inputs: StdTemplateInputs):
        if media_type != "audio":
            raise ValueError("FireRedASR supports audio input only.")
        return [[self._speech_token_id] * inputs.extra_kwargs["firered_speech_token_count"]]

    def _encode(self, inputs: StdTemplateInputs) -> dict[str, Any]:
        if len(inputs.audios) != 1:
            raise ValueError(f"FireRedASR requires exactly one audio per sample, got {len(inputs.audios)}.")
        padded_feat, feat_lengths, _ = self._feat_extractor(inputs.audios)
        feat_length = int(feat_lengths[0])
        token_count = speech_token_length(feat_length, self.encoder_downsample_rate)
        if token_count < 1:
            raise ValueError("Audio is too short for the FireRedASR encoder and adapter.")
        inputs.extra_kwargs["firered_speech_token_count"] = token_count
        encoded = super()._encode(inputs)
        actual_count = encoded["input_ids"].count(self._speech_token_id)
        if actual_count != token_count:
            raise ValueError("Each sample must contain exactly one <audio> placeholder and no literal <speech> tokens.")
        encoded["padded_feat"] = padded_feat[0]
        encoded["feat_length"] = feat_length
        return encoded

    def _data_collator(self, batch, *, padding_to=None):
        features = [sample["padded_feat"] for sample in batch]
        lengths = [sample["feat_length"] for sample in batch]
        result = super()._data_collator(batch, padding_to=padding_to)
        # Arrow cache yields lists; lazy tokenization yields tensors.
        result["padded_feat"] = pad_sequence(
            [torch.as_tensor(feature, dtype=torch.float32) for feature in features],
            batch_first=True,
            padding_value=0.0,
        )
        result["feat_lengths"] = torch.tensor(lengths, dtype=torch.long)
        return result


class FireRedASRDatasetPreprocessor(ResponsePreprocessor):
    AUDIO_KEYS = ("audios", "audio", "wav", "wav_path", "path")
    TEXT_KEYS = ("response", "target", "txt", "text", "transcript", "transcription")

    def preprocess(self, row):
        audio = next((row[key] for key in self.AUDIO_KEYS if row.get(key)), None)
        if audio is None:
            raise ValueError("Missing audio path in FireRedASR sample.")
        if isinstance(audio, (list, tuple)):
            if len(audio) != 1:
                raise ValueError("Each FireRedASR sample must have exactly one audio.")
            audio = audio[0]
        if not isinstance(audio, str):
            raise ValueError("FireRedASR expects a local audio path string.")
        if row.get("messages") is not None:
            messages = copy.deepcopy(row["messages"])
            for message in messages:
                if message.get("role") == "assistant" and isinstance(message.get("content"), str):
                    message["content"] = clean_transcript(message["content"])
            return {"messages": messages, "audios": [audio]}
        text = next((row[key] for key in self.TEXT_KEYS if row.get(key) is not None), None)
        if not isinstance(text, str):
            raise ValueError("Missing transcription string in FireRedASR sample.")
        query = row.get("query") or row.get("instruction") or "请转写音频为文字"
        if "<audio>" not in query:
            query = f"<audio>{query}"
        return super().preprocess({"query": query, "response": clean_transcript(text), "audios": [audio]})


register_template(
    TemplateMeta(
        template_type="fireredasr_llm",
        template_cls=FireRedASRTemplate,
        prefix=[],
        prompt=["<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n"],
        chat_sep=["<|im_end|>\n"],
        suffix=["<|im_end|>"],
    )
)
register_model_arch(
    MultiModelKeys(
        arch_name="fireredasr_llm",
        language_model=["firered_model.llm.model", "firered_model.llm.lm_head"],
        aligner="firered_model.encoder_projector",
        vision_tower="firered_model.encoder",
    )
)
register_model(
    ModelMeta(
        model_type="fireredasr_llm",
        model_groups=[ModelGroup([Model("FireRedTeam/FireRedASR-LLM-L", "FireRedTeam/FireRedASR-LLM-L")])],
        loader=FireRedASRLLMLoader,
        template="fireredasr_llm",
        model_arch="fireredasr_llm",
        is_multimodal=True,
        tags=["audio", "asr"],
    )
)
for _name, _variable in (
    ("firered_asr_jsonl", "FIRERED_ASR_DATASET_PATH"),
    ("firered_asr_jsonl_val", "FIRERED_ASR_VAL_DATASET_PATH"),
):
    if _path := os.environ.get(_variable):
        register_dataset(
            DatasetMeta(
                dataset_name=_name,
                dataset_path=str(Path(_path).expanduser().resolve()),
                preprocess_func=FireRedASRDatasetPreprocessor(),
                tags=["audio", "asr"],
            )
        )
