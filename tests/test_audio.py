from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from orato_asr import audio
from orato_asr.exceptions import AudioValidationError, DependencyError


class _Mono:
    def __init__(self, size: int) -> None:
        self.size = size
        self.astype_calls: list[tuple[str, bool]] = []

    def astype(self, dtype: str, copy: bool = True) -> "_Mono":
        self.astype_calls.append((dtype, copy))
        return self


class _Samples:
    ndim = 2

    def __init__(self, frames: int, channels: int) -> None:
        self.shape = (frames, channels)
        self.mean_calls: list[tuple[int, str]] = []
        self.mono = _Mono(frames)

    def mean(self, axis: int, dtype: str) -> _Mono:
        self.mean_calls.append((axis, dtype))
        return self.mono

    def __getitem__(self, key: object) -> _Mono:
        assert key == (slice(None), 0)
        return self.mono


def _audio_file(tmp_path: Path, suffix: str = ".wav") -> Path:
    path = tmp_path / f"sample{suffix}"
    path.write_bytes(b"not-empty")
    return path


def test_decode_downmixes_and_resamples_in_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    samples = _Samples(frames=8_000, channels=2)
    resampled = _Mono(16_000)
    calls: list[tuple[object, int, int, str]] = []
    fake_soundfile = SimpleNamespace(read=lambda *args, **kwargs: (samples, 8_000))

    def fake_resample(values: object, source: int, target: int, quality: str) -> _Mono:
        calls.append((values, source, target, quality))
        return resampled

    modules = {
        "soundfile": fake_soundfile,
        "soxr": SimpleNamespace(resample=fake_resample),
    }
    monkeypatch.setattr(audio.importlib, "import_module", modules.__getitem__)
    source = _audio_file(tmp_path, ".flac")
    original = source.read_bytes()

    decoded = audio.decode_audio(source)

    assert decoded.samples is resampled
    assert decoded.original_sample_rate == 8_000
    assert decoded.sample_rate == 16_000
    assert decoded.original_channels == 2
    assert decoded.channels == 1
    assert decoded.duration_seconds == 1.0
    assert decoded.downmixed is True
    assert decoded.resampled is True
    assert samples.mean_calls == [(1, "float32")]
    assert calls == [(samples.mono, 8_000, 16_000, "HQ")]
    assert source.read_bytes() == original


def test_decode_mono_16khz_does_not_import_soxr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    samples = _Samples(frames=16_000, channels=1)

    def fake_import(name: str) -> object:
        assert name == "soundfile"
        return SimpleNamespace(read=lambda *args, **kwargs: (samples, 16_000))

    monkeypatch.setattr(audio.importlib, "import_module", fake_import)
    decoded = audio.decode_audio(_audio_file(tmp_path))

    assert decoded.samples is samples.mono
    assert decoded.downmixed is False
    assert decoded.resampled is False


@pytest.mark.parametrize("kind", ["missing", "directory", "empty", "extension"])
def test_audio_path_validation(tmp_path: Path, kind: str) -> None:
    path = tmp_path / "sample.wav"
    if kind == "directory":
        path.mkdir()
    elif kind == "empty":
        path.touch()
    elif kind == "extension":
        path = _audio_file(tmp_path, ".mp3")

    with pytest.raises(AudioValidationError):
        audio.decode_audio(path)


def test_decode_failure_is_actionable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failed_read(*args: object, **kwargs: object) -> object:
        raise RuntimeError("bad container")

    monkeypatch.setattr(
        audio.importlib,
        "import_module",
        lambda _: SimpleNamespace(read=failed_read),
    )
    with pytest.raises(AudioValidationError, match="Could not decode"):
        audio.decode_audio(_audio_file(tmp_path))


def test_missing_decoder_dependency_is_actionable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def missing(_: str) -> object:
        raise ImportError("no module")

    monkeypatch.setattr(audio.importlib, "import_module", missing)
    with pytest.raises(DependencyError, match="requirements/inference.txt"):
        audio.decode_audio(_audio_file(tmp_path))
