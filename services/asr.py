from __future__ import annotations

import re
import subprocess
from pathlib import Path

from config import NEMO_MODEL
from utils.text import clean, concept_terms

DEFAULT_AUDIO_DEVICE = "Microphone Array (2- Realtek(R) Audio)"


def get_available_audio_devices() -> list[str]:
    """List available DirectShow audio input devices via FFmpeg."""
    try:
        res = subprocess.run(
            ["ffmpeg", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
        )
        output = res.stderr or ""
        return re.findall(r'"([^"]+)"\s*\(audio\)', output)
    except Exception:
        return []


def preprocess_audio(input_wav: Path) -> Path:
    """
    Preprocess recorded audio before passing to Nemotron ASR:
    - Highpass filter (80 Hz) to remove low frequency hum/rumble
    - Lowpass filter (7500 Hz) to eliminate high frequency hiss
    - Mild voice-activity silence trimming without clipping sentence start/end
    - Loudness normalization (loudnorm filter) to standard 16 kHz mono WAV
    """
    input_wav = Path(input_wav)
    if not input_wav.exists() or input_wav.stat().st_size < 1000:
        return input_wav

    proc_path = input_wav.parent / f"proc_{input_wav.name}"
    filter_graph = (
        "highpass=f=80,"
        "lowpass=f=7500,"
        "silenceremove=start_periods=1:start_duration=0.15:start_threshold=-45dB:stop_periods=1:stop_duration=0.5:stop_threshold=-45dB,"
        "loudnorm"
    )
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_wav),
        "-af",
        filter_graph,
        "-ar",
        "16000",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        str(proc_path),
    ]
    try:
        res = subprocess.run(command, capture_output=True, text=True, timeout=20)
        if res.returncode == 0 and proc_path.exists() and proc_path.stat().st_size > 1000:
            return proc_path
    except Exception:
        pass
    return input_wav


def record_audio_dshow(
    output_path: Path,
    duration: int = 15,
    device_name: str = DEFAULT_AUDIO_DEVICE,
) -> Path:
    """
    Capture audio from Windows microphone via FFmpeg DirectShow into 16kHz mono PCM 16-bit WAV.
    Auto-detects alternative audio input device if specified device fails.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        try:
            output_path.unlink()
        except Exception:
            pass

    duration = min(max(int(duration), 3), 60)
    devices_to_try = [device_name]
    detected = get_available_audio_devices()
    for d in detected:
        if d not in devices_to_try:
            devices_to_try.append(d)

    last_error = ""
    for target_device in devices_to_try:
        command = [
            "ffmpeg",
            "-y",
            "-f",
            "dshow",
            "-i",
            f"audio={target_device}",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-acodec",
            "pcm_s16le",
            "-t",
            str(duration),
            str(output_path),
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=duration + 10,
            )
            if result.returncode == 0 and output_path.exists() and output_path.stat().st_size >= 1000:
                # Successfully recorded; apply audio preprocessing
                processed = preprocess_audio(output_path)
                return processed
            last_error = (result.stderr or result.stdout or "Unknown FFmpeg error").strip()
        except subprocess.TimeoutExpired:
            last_error = f"Audio recording timed out after {duration} seconds."
        except FileNotFoundError:
            raise RuntimeError("FFmpeg is not installed or not available on system PATH.")

    raise RuntimeError(
        f"Failed to record audio from microphone ({device_name}): {last_error[-400:]}"
    )


def normalize_transcript(text: str, manual: str) -> str:
    replacements = {
        r"\bsim li a regression\b": "simple linear regression",
        r"\bsim linear regression\b": "simple linear regression",
        r"\btrain test split\b": "train_test_split",
        r"\btrain-test split\b": "train_test_split",
    }
    manual_l = (manual or "").lower()
    for pattern, replacement in replacements.items():
        needle = replacement.replace("_", " ").lower()
        if needle in manual_l or replacement.lower() in manual_l:
            text = re.sub(pattern, replacement, text, flags=re.I)
    return clean(text)


def speech_service_available() -> bool:
    try:
        result = subprocess.run(
            ["nemo-speech", "--version"],
            capture_output=True,
            text=True,
            timeout=8,
        )
        return result.returncode == 0
    except Exception:
        return False


def transcribe_audio(path: Path, experiment_text: str) -> str:
    if not path.exists():
        raise RuntimeError("Recorded audio file was not found.")
    if path.stat().st_size < 1000:
        raise RuntimeError("The recording is empty or too short. Speak again.")

    # Preprocess raw audio first if not already preprocessed
    clean_audio_path = preprocess_audio(path)

    command = [
        "nemo-speech",
        "transcribe",
        str(clean_audio_path),
        "--model",
        NEMO_MODEL,
        "--language",
        "en",
        "--device",
        "cpu",
        "--format",
        "text",
    ]
    seen: set[str] = set()
    for phrase in concept_terms(experiment_text, 12):
        phrase = clean(phrase)
        key = phrase.lower()
        if phrase and key not in seen:
            command += ["--speech-context", phrase, "--speech-context-boost", "2.0"]
            seen.add(key)

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=45,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "Speech recognition service (nemo-speech) is not installed on this workstation."
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            "Nemotron speech recognition timed out after 45 seconds. Try recording a shorter answer."
        )

    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "Unknown speech recognition error").strip()
        raise RuntimeError(f"Speech recognition error: {detail[-1000:]}")

    transcript = re.sub(r"(?im)^\[.*?\].*$", "", result.stdout)
    transcript = normalize_transcript(clean(transcript), experiment_text)
    if not transcript:
        raise RuntimeError("No speech was recognised. Record again in a quieter environment.")
    return transcript
