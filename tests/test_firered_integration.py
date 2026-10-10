"""CPU integration with real fbank, a tiny encoder/Qwen and LoRA checkpoints.

Run after `uv sync --locked` and preparing upstream source:
    FIRERED_ASR_REPO=/path/to/FireRedASR uv run pytest tests/test_firered_integration.py
The test builds all model weights locally and never downloads a model.
"""

import importlib
import json
import pickle
import wave
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

if TYPE_CHECKING:
    from swift.arguments import SftArguments

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
kaldiio = pytest.importorskip("kaldiio")
pytest.importorskip("kaldi_native_fbank")
pytest.importorskip("swift")
pytest.importorskip("peft")


@pytest.fixture()
def tiny_firered(tmp_path, monkeypatch):
    from swift.utils import import_external_file
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    from audio_swift.plugins.fireredasr_llm.model import ensure_firered_source

    try:
        ensure_firered_source()
    except ImportError as exc:
        pytest.skip(str(exc))
    from fireredasr.models.fireredasr_aed import FireRedAsrAed
    from fireredasr.models.fireredasr_llm import FireRedAsrLlm

    model_dir = tmp_path / "model"
    llm_dir = model_dir / "Qwen2-7B-Instruct"
    llm_dir.mkdir(parents=True)
    vocab = {
        "<unk>": 0,
        "<|endoftext|>": 1,
        "<|im_start|>": 2,
        "<|im_end|>": 3,
        "<speech>": 4,
        "user": 5,
        "assistant": 6,
        "你好": 7,
        "world": 8,
        "请转写音频为文字": 9,
    }
    fast = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    fast.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=fast,
        unk_token="<unk>",
        pad_token="<|endoftext|>",
        eos_token="<|im_end|>",
        additional_special_tokens=["<|im_start|>", "<speech>"],
    )
    tokenizer.save_pretrained(llm_dir)
    config = Qwen2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        pad_token_id=1,
        eos_token_id=3,
    )
    Qwen2ForCausalLM(config).save_pretrained(llm_dir)
    config.save_pretrained(model_dir)
    encoder_args = SimpleNamespace(
        idim=80,
        n_layers_enc=1,
        n_layers_dec=1,
        n_head=2,
        d_model=16,
        residual_dropout=0.0,
        dropout_rate=0.0,
        kernel_size=3,
        pe_maxlen=5000,
        sos_id=0,
        eos_id=1,
        pad_id=2,
        odim=16,
    )
    encoder = FireRedAsrAed.from_args(encoder_args)
    torch.save({"args": encoder_args, "model_state_dict": encoder.state_dict()}, model_dir / "asr_encoder.pth.tar")
    args = SimpleNamespace(
        encoder_path=str(model_dir / "asr_encoder.pth.tar"),
        llm_dir=str(llm_dir),
        freeze_encoder=True,
        freeze_llm=False,
        use_lora=True,
        use_fp16=False,
        use_flash_attn=False,
        encoder_downsample_rate=2,
    )

    class TinyFireRedAsrLlm(FireRedAsrLlm):
        @classmethod
        def load_encoder(cls, model_path):
            package = torch.load(model_path, map_location="cpu", weights_only=False)
            model = FireRedAsrAed.from_args(package["args"])
            model.load_state_dict(package["model_state_dict"])
            return model.encoder, model.encoder.odim

    native = TinyFireRedAsrLlm.from_args(args)
    delta = {
        name: value
        for name, value in native.state_dict().items()
        if name.startswith("encoder_projector.") or ".lora_" in name
    }
    torch.save({"args": args, "model_state_dict": delta}, model_dir / "model.pth.tar")
    cmvn = np.zeros((2, 81), dtype=np.float64)
    means = np.linspace(-0.5, 0.5, 80)
    variances = np.linspace(0.8, 1.2, 80)
    cmvn[0, :80] = 4 * means
    cmvn[0, -1] = 4
    cmvn[1, :80] = 4 * (variances + means**2)
    kaldiio.save_mat(str(model_dir / "cmvn.ark"), cmvn)
    paths = []
    for i, sample_count in enumerate((16000, 24000)):
        path = tmp_path / f"audio{i}.wav"
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes((np.sin(np.arange(sample_count) / 30) * 1000).astype(np.int16).tobytes())
        paths.append(str(path))
    rows = [
        {"audio": paths[0], "text": "你好，World！"},
        {
            "messages": [
                {"role": "user", "content": "<audio>请转写音频为文字"},
                {"role": "assistant", "content": "你好，World！"},
            ],
            "audios": [paths[1]],
        },
    ]
    dataset = tmp_path / "train.jsonl"
    dataset.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows))
    monkeypatch.setenv("FIRERED_ASR_DATASET_PATH", str(dataset))
    shim_path = Path(__file__).resolve().parents[1] / "examples" / "fireredasr_llm" / "plugin.py"
    shim = import_external_file(str(shim_path))
    plugin = importlib.import_module("audio_swift.plugins.fireredasr_llm.plugin")
    assert shim.plugin is plugin
    return model_dir, rows, plugin


@pytest.mark.integration
@pytest.mark.parametrize("attn_impl", ["eager", "sdpa"])
def test_firered_audio_sft_and_lora_restore(tiny_firered, tmp_path, attn_impl):
    from fireredasr.data.asr_feat import ASRFeatExtractor
    from peft import PeftModel
    from swift.model import get_model_processor
    from swift.template import get_template
    from swift.tuner_plugin.lora_llm import LoRALLMTuner

    from audio_swift.plugins.fireredasr_llm.audio import FireRedFeatureExtractor, speech_token_length
    from audio_swift.plugins.fireredasr_llm.model import FireRedASRLLMForCausalLM

    model_dir, rows, plugin = tiny_firered
    paths = [rows[0]["audio"], rows[1]["audios"][0]]
    # Protect the actual upstream preprocessing contract while owning the audio code.
    expected, expected_lengths, _ = ASRFeatExtractor(str(model_dir / "cmvn.ark"))(paths)
    extractor = FireRedFeatureExtractor(model_dir / "cmvn.ark")
    extracted, lengths, _ = extractor(paths)
    torch.testing.assert_close(extracted, expected)
    torch.testing.assert_close(lengths, expected_lengths)
    recreated = pickle.loads(pickle.dumps(extractor))
    torch.testing.assert_close(recreated(paths)[0], extracted)
    model, processor = get_model_processor(
        str(model_dir),
        model_type="fireredasr_llm",
        torch_dtype=torch.float32,
        device_map="cpu",
        attn_impl=attn_impl,
    )
    assert isinstance(model, FireRedASRLLMForCausalLM)
    assert not isinstance(model.firered_model.llm, PeftModel), "Pretrained LoRA must be merged before fresh SFT."
    assert model.firered_model.llm.config._attn_implementation == attn_impl
    with pytest.raises(NotImplementedError, match="adapter"):
        model.save_pretrained(tmp_path / "unsupported-full-checkpoint")
    template = get_template(
        processor,
        template_type="fireredasr_llm",
        max_length=256,
        truncation_strategy="raise",
    )
    template.set_mode("train")
    normalized = [plugin.FireRedASRDatasetPreprocessor().preprocess(row) for row in rows]
    assert normalized[0]["messages"][1]["content"] == normalized[1]["messages"][1]["content"] == "你好world"
    encoded = [template.encode(row) for row in normalized]
    speech_token = processor.convert_tokens_to_ids("<speech>")
    for row in encoded:
        assert row["input_ids"].count(speech_token) == speech_token_length(row["feat_length"], 2)
        assert all(
            label == -100 for token, label in zip(row["input_ids"], row["labels"], strict=True) if token == speech_token
        )
    batch = template.data_collator(encoded)
    output = model(**batch, num_items_in_batch=torch.tensor(6))
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert model.firered_model.encoder_projector.linear1.weight.grad is not None
    model.zero_grad()

    # Isolate the tuner using precisely the three CLI fields it consumes.
    tuner_args = cast("SftArguments", SimpleNamespace(task_type="causal_lm", lora_rank=2, lora_alpha=4))
    lora_model = LoRALLMTuner.prepare_model(tuner_args, model)
    assert isinstance(lora_model, PeftModel)
    assert lora_model.get_base_model() is model
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in lora_model.parameters() if parameter.requires_grad], lr=1e-4
    )
    loss = lora_model(**batch).loss
    assert torch.isfinite(loss)
    loss.backward()
    for module in (".lora_B.", "encoder.", "encoder_projector."):
        assert any(parameter.grad is not None for name, parameter in lora_model.named_parameters() if module in name)
    optimizer.step()
    checkpoint = tmp_path / "adapter"
    LoRALLMTuner.save_pretrained(lora_model, str(checkpoint))
    assert (checkpoint / "vit.safetensors").is_file()
    base, _ = get_model_processor(
        str(model_dir),
        model_type="fireredasr_llm",
        torch_dtype=torch.float32,
        device_map="cpu",
        attn_impl=attn_impl,
    )
    assert isinstance(base, FireRedASRLLMForCausalLM)
    assert base.firered_model.llm.config._attn_implementation == attn_impl
    restored = LoRALLMTuner.from_pretrained(base, str(checkpoint))
    assert torch.isfinite(restored(**batch).loss)

    cached = [{**row, "padded_feat": row["padded_feat"].tolist()} for row in encoded]
    torch.testing.assert_close(template.data_collator(cached)["padded_feat"], batch["padded_feat"])
    bad_batch = {**batch, "input_ids": batch["input_ids"].clone()}
    bad_batch["input_ids"][0, (bad_batch["input_ids"][0] == speech_token).nonzero()[0]] = 0
    with pytest.raises(ValueError, match="align"):
        restored(**bad_batch)
    for strategy in ("left", "right", "split"):
        with pytest.raises(ValueError, match="truncation_strategy=raise"):
            get_template(processor, template_type="fireredasr_llm", truncation_strategy=strategy)
    with pytest.raises(ValueError, match="padding_free=false"):
        get_template(processor, template_type="fireredasr_llm", truncation_strategy="raise", padding_free=True)
