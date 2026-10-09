"""FireRed's 80-bin fbank, pretrained CMVN and speech placeholder lengths.

Feature conventions follow FireRedTeam/FireRedASR (Apache-2.0). These utilities
have no dependency on the training engine or the external FireRed source tree.
"""

import os
import re
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

DEFAULT_SPEECH_TOKEN = "<speech>"


def speech_token_length(feat_length: int, downsample_rate: int) -> int:
    """Length after two kernel-3 stride-2 convolutions and the adapter."""
    if downsample_rate < 1:
        raise ValueError("encoder_downsample_rate must be positive.")
    length = ((feat_length - 3) // 2 + 1 - 3) // 2 + 1
    return max(length, 0) // downsample_rate


def clean_transcript(text: str) -> str:
    """Match the transcript normalization used by FireRed's LLM tokenizer."""
    text = re.sub(r"[，。？！,\.!?《》（）·“”、\\/]", "", text)
    text = re.sub(r"\s+", " ", text)
    parts = re.split(r"([\u3400-\u4dbf\u4e00-\u9fff])", text.strip())
    return "".join(part for part in parts if part.strip()).strip().lower()


class FireRedFeatureExtractor:
    """Extract fbank and apply checkpoint CMVN, recreating native state per PID."""

    def __init__(self, cmvn_file: str | Path):
        self.cmvn_file = Path(cmvn_file)
        if not self.cmvn_file.is_file():
            raise FileNotFoundError(f"Missing pretrained FireRedASR CMVN statistics: {self.cmvn_file}")
        self._pid = None
        self._options = None
        self._means = None
        self._inverse_std = None

    def __getstate__(self):
        # Native FbankOptions cannot be pickled by spawned dataset workers.
        state = self.__dict__.copy()
        state.update(_pid=None, _options=None, _means=None, _inverse_std=None)
        return state

    def _initialize(self):
        try:
            import kaldi_native_fbank as knf
            import kaldiio
        except ImportError as exc:
            raise ImportError("FireRed audio features require: uv sync --locked") from exc
        stats = kaldiio.load_mat(str(self.cmvn_file))
        if not isinstance(stats, np.ndarray) or stats.shape != (2, 81) or stats[0, -1] < 1:
            raise ValueError("FireRedASR CMVN must contain valid 2 x 81 Kaldi statistics.")
        count = stats[0, -1]
        self._means = stats[0, :80] / count
        variance = stats[1, :80] / count - self._means**2
        self._inverse_std = 1 / np.sqrt(np.maximum(variance, 1e-20))
        options = knf.FbankOptions()
        options.frame_opts.dither = 0.0
        options.frame_opts.frame_length_ms = 25.0
        options.frame_opts.frame_shift_ms = 10.0
        options.frame_opts.snip_edges = True
        options.mel_opts.num_bins = 80
        options.mel_opts.debug_mel = False
        self._options = options
        self._pid = os.getpid()

    def __call__(self, audio_paths):
        if self._pid != os.getpid():
            self._initialize()
        assert self._options is not None and self._means is not None and self._inverse_std is not None
        import kaldi_native_fbank as knf
        from kaldiio.wavio import read_wav

        features, durations = [], []
        for path in audio_paths:
            # Manifest paths are local files, never Kaldi command/pipe specifiers.
            # read_wav preserves integer PCM amplitudes used by pretrained fbank.
            with Path(path).open("rb") as audio_file:
                audio = read_wav(audio_file)
            if not isinstance(audio, tuple) or len(audio) != 2:
                raise ValueError(f"Expected a local WAV file for FireRedASR: {path}")
            sample_rate, waveform = audio
            if (
                not isinstance(sample_rate, (int, float))
                or sample_rate != 16000
                or not isinstance(waveform, np.ndarray)
                or waveform.ndim != 1
            ):
                raise ValueError(f"FireRedASR expects mono 16 kHz audio: {path}")
            fbank = knf.OnlineFbank(self._options)
            # The mono check guarantees a flat sequence; integer PCM values are
            # accepted unchanged by the native float-waveform interface.
            fbank.accept_waveform(sample_rate, cast(list[float], waveform.tolist()))
            if fbank.num_frames_ready == 0:
                feat = np.zeros((0, 80), dtype=np.float32)
            else:
                feat = np.vstack([fbank.get_frame(i) for i in range(fbank.num_frames_ready)])
                feat = (feat - self._means) * self._inverse_std
            features.append(torch.from_numpy(feat).float())
            durations.append(waveform.shape[0] / sample_rate)
        if not features:
            raise ValueError("At least one audio path is required.")
        lengths = torch.tensor([feat.shape[0] for feat in features], dtype=torch.long)
        return pad_sequence(features, batch_first=True, padding_value=0.0), lengths, durations
