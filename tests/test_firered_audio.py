"""Local WAV reads cannot interpret dataset paths as Kaldi shell commands."""

import shlex
import wave

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
kaldiio = pytest.importorskip("kaldiio")
pytest.importorskip("kaldi_native_fbank")


@pytest.fixture()
def audio_fixture(tmp_path):
    from audio_swift.plugins.fireredasr_llm.audio import FireRedFeatureExtractor

    cmvn = np.zeros((2, 81), dtype=np.float64)
    cmvn[0, -1] = 1
    cmvn[1, :80] = 1
    cmvn_path = tmp_path / "cmvn.ark"
    kaldiio.save_mat(str(cmvn_path), cmvn)
    samples = (np.sin(np.arange(16000) / 30) * 1000).astype(np.int16)
    regular = tmp_path / "speech.wav"
    with wave.open(str(regular), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(samples.tobytes())
    return FireRedFeatureExtractor(cmvn_path), regular, samples


def test_audio_manifest_cannot_execute_shell(audio_fixture, tmp_path):
    extractor, _, _ = audio_fixture
    marker = tmp_path / "shell-command-executed"
    malicious_audio = f"touch {shlex.quote(str(marker))} |"
    with pytest.raises(FileNotFoundError):
        extractor([malicious_audio])
    assert not marker.exists(), "Dataset audio paths must never execute a shell command"


def test_pipe_filename_is_read_as_local_pcm_wav(audio_fixture, tmp_path):
    from kaldiio.wavio import read_wav

    extractor, regular, samples = audio_fixture
    pipe_filename = tmp_path / "speech.wav|"
    pipe_filename.write_bytes(regular.read_bytes())
    with pipe_filename.open("rb") as handle:
        rate, pcm = read_wav(handle)
    assert rate == 16000
    np.testing.assert_array_equal(pcm, samples)
    # Kaldi's safe regular-file path confirms that FD reads preserve its PCM scale.
    expected_rate, expected_pcm = kaldiio.load_mat(str(regular))
    assert rate == expected_rate
    np.testing.assert_array_equal(pcm, expected_pcm)
    actual_features, actual_lengths, actual_durations = extractor([pipe_filename])
    expected_features, expected_lengths, expected_durations = extractor([regular])
    torch.testing.assert_close(actual_features, expected_features, rtol=0, atol=0)
    torch.testing.assert_close(actual_lengths, expected_lengths)
    assert actual_durations == expected_durations
