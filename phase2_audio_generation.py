# phase2_audio_generation.py
"""
Phase 2: Audio and Voice Generation (Piper TTS - 100% Free & Offline)
======================================================================
Reads outputs/dialogue.json from Phase 1.
Generates one .wav file per dialogue turn using Piper TTS.
Two distinct voices: ARIA (female/lessac) and MARCUS (male/ryan).

Output Contract (consumed by Phase 4):
    outputs/audio/turn_XX_SPEAKER.wav  — one file per turn
    outputs/audio/audio_manifest.json  — timing metadata for Phase 5

Dependencies:
    pip install piper-tts soundfile numpy pydub librosa
"""

from __future__ import annotations

import json
import time
import wave
import struct
from pathlib import Path

import numpy as np
import soundfile as sf


# ─────────────────────────────────────────────
# 1. VOICE CONFIGURATION
# ─────────────────────────────────────────────

VOICE_CONFIG = {
    "ARIA": {
        "model":       Path("voices/en_US-lessac-medium.onnx"),
        "description": "Female voice, warm and clear",
        "speed":       1.0,

    },
    "MARCUS": {
        "model":       Path("voices/en_US-ryan-medium.onnx"),
        "description": "Male voice, deep and deliberate",
        "speed":       1.0,

    }
}


# ─────────────────────────────────────────────
# 2. AUDIO UTILITIES
# ─────────────────────────────────────────────

def get_audio_duration(wav_path: Path) -> float:
    """Returns duration of a .wav file in seconds."""
    data, sample_rate = sf.read(str(wav_path))
    return len(data) / sample_rate


def add_silence(wav_path: Path, silence_ms: int = 350) -> None:
    """
    Appends a natural pause at end of each turn.
    This creates breathing room between speakers in Phase 5.
    """
    data, sample_rate = sf.read(str(wav_path))
    silence_samples   = int(sample_rate * silence_ms / 1000)

    if data.ndim == 1:
        silence = np.zeros(silence_samples, dtype=data.dtype)
    else:
        silence = np.zeros((silence_samples, data.shape[1]), dtype=data.dtype)

    padded = np.concatenate([data, silence])
    sf.write(str(wav_path), padded, sample_rate)


def normalize_audio(wav_path: Path, target_db: float = -18.0) -> None:
    """
    Normalizes loudness so ARIA and MARCUS have consistent volume.
    Uses RMS normalization to broadcast standard (-18 dB).
    """
    data, sample_rate = sf.read(str(wav_path))
    data_f32 = data.astype(np.float32)
    rms = np.sqrt(np.mean(data_f32 ** 2))
    if rms < 1e-9:
        return
    target_rms = 10 ** (target_db / 20)
    gain       = target_rms / rms
    normalized = np.clip(data_f32 * gain, -1.0, 1.0)
    sf.write(str(wav_path), normalized, sample_rate)


# ─────────────────────────────────────────────
# 3. PIPER TTS ENGINE
# ─────────────────────────────────────────────

class PiperTTSEngine:
    """
    Wraps Piper TTS with one model instance per speaker.
    Models are loaded once and reused for all turns of that speaker.

    How Piper works internally:
    Text → Phonemizer → VITS acoustic model (.onnx) → raw PCM audio
    The .onnx file runs via ONNXRuntime — no GPU needed, CPU only.
    """

    def __init__(self) -> None:
        self._models: dict = {}
        self._verify_voice_files()

    def _verify_voice_files(self) -> None:
        """Check all voice files exist before starting."""
        print("[Phase 2] Verifying voice files...")
        for speaker, config in VOICE_CONFIG.items():
            model_path = config["model"]
            json_path  = Path(str(model_path) + ".json")
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Voice model not found for {speaker}: {model_path}\n"
                    f"Please download it from HuggingFace (see instructions above)."
                )
            if not json_path.exists():
                raise FileNotFoundError(
                    f"Voice config not found for {speaker}: {json_path}\n"
                    f"Download the .onnx.json file alongside the .onnx file."
                )
            print(f"  ✓ {speaker}: {model_path.name}")

    def _load_model(self, speaker: str):
        """Lazy-loads Piper model for a speaker."""
        if speaker not in self._models:
            from piper import PiperVoice

            config     = VOICE_CONFIG[speaker]
            model_path = config["model"]
            print(f"\n[Phase 2] Loading voice model for {speaker}...")
            print(f"          {config['description']}")

            voice = PiperVoice.load(
                str(model_path),
                config_path=str(model_path) + ".json",
                use_cuda=False   # CPU only — works on any machine
            )
            self._models[speaker] = voice
            print(f"  ✓ {speaker} voice ready")

        return self._models[speaker]

    def synthesize(self, speaker: str, text: str, output_path: Path) -> float:
        """
        Synthesizes one dialogue turn to a .wav file.
        Uses synthesize_wav() with a wave.Wave_write object (correct API).
        """
        import wave as wave_module
        voice = self._load_model(speaker)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"\n  Synthesizing turn for {speaker}:")
        print(f"  \"{text[:70]}{'...' if len(text) > 70 else ''}\"")

        # synthesize_wav() needs an open Wave_write object, not a path
        with wave_module.open(str(output_path), "wb") as wav_file:
            voice.synthesize_wav(text, wav_file)

        # Verify file was written properly
        size = output_path.stat().st_size
        print(f"  File size: {size} bytes")
        if size < 1000:
            raise RuntimeError(f"Audio file too small ({size} bytes) — synthesis failed silently")

        # Post-process: normalize volume + add natural pause
        normalize_audio(output_path)
        add_silence(output_path, silence_ms=350)

        duration = get_audio_duration(output_path)
        print(f"  ✓ {output_path.name} → {duration:.2f}s")
        return duration
        return duration


# ─────────────────────────────────────────────
# 4. MANIFEST BUILDER
# ─────────────────────────────────────────────

def build_manifest(turns_metadata: list[dict], output_dir: Path) -> Path:
    """
    Builds audio_manifest.json with absolute timeline positions.
    This is the Phase 5 synchronization contract.

    Timeline structure:
    [turn_0_start ... turn_0_end][turn_1_start ... turn_1_end] ...
    Turns are placed end-to-end (no gap — silence already baked in).
    """
    manifest_turns = []
    cursor = 0.0

    for meta in turns_metadata:
        entry = {
            "turn_id":                 meta["turn_id"],
            "speaker":                 meta["speaker"],
            "text":                    meta["text"],
            "emotion_tag":             meta["emotion_tag"],
            "word_count":              meta["word_count"],
            "wav_path":                meta["wav_path"],
            "actual_duration_seconds": round(meta["actual_duration"], 3),
            "start_time_seconds":      round(cursor, 3),
            "end_time_seconds":        round(cursor + meta["actual_duration"], 3),
        }
        cursor += meta["actual_duration"]
        manifest_turns.append(entry)

    manifest = {
        "total_duration_seconds": round(cursor, 3),
        "total_turns":            len(manifest_turns),
        "turns":                  manifest_turns,
    }

    manifest_path = output_dir / "audio_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return manifest_path


# ─────────────────────────────────────────────
# 5. MAIN PIPELINE
# ─────────────────────────────────────────────

def run_phase2(
    dialogue_path: Path = Path("outputs/dialogue.json"),
    audio_dir:     Path = Path("outputs/audio"),
) -> Path:
    """Full Phase 2 execution."""

    # ── Load Phase 1 dialogue ──
    print("[Phase 2] Reading dialogue.json...")
    with open(dialogue_path, encoding="utf-8") as f:
        dialogue = json.load(f)

    turns  = dialogue["turns"]
    engine = PiperTTSEngine()

    print(f"\n[Phase 2] Generating audio for {len(turns)} turns...")
    print("─" * 60)

    turns_metadata = []

    for turn in turns:
        turn_id = turn["turn_id"]
        speaker = turn["speaker"]
        text    = turn["text"]
        emotion = turn["emotion_tag"]

        filename    = f"turn_{turn_id:02d}_{speaker}.wav"
        output_path = audio_dir / filename

        print(f"\n── Turn {turn_id:02d} | {speaker} | [{emotion}]")

        if output_path.exists():
            # Resume support — skip already generated turns
            duration = get_audio_duration(output_path)
            print(f"  Already exists → {duration:.2f}s  (skipping)")
        else:
            t0       = time.time()
            duration = engine.synthesize(speaker, text, output_path)
            elapsed  = time.time() - t0
            print(f"  Generated in {elapsed:.1f}s")

        turns_metadata.append({
            "turn_id":        turn_id,
            "speaker":        speaker,
            "text":           text,
            "emotion_tag":    emotion,
            "word_count":     turn["word_count"],
            "wav_path":       str(output_path),
            "actual_duration": get_audio_duration(output_path),
        })

    # ── Build manifest ──
    print("\n" + "─" * 60)
    print("[Phase 2] Building audio manifest...")
    manifest_path = build_manifest(turns_metadata, audio_dir)

    # ── Final summary ──
    total_dur = sum(m["actual_duration"] for m in turns_metadata)
    print("\n" + "═" * 60)
    print("  PHASE 2 COMPLETE")
    print("═" * 60)
    for m in turns_metadata:
        bar = "█" * int(m["actual_duration"] * 1.5)
        print(f"  {m['turn_id']:02d} | {m['speaker']:<8} | {bar:<30} | {m['actual_duration']:.1f}s")
    print("─" * 60)
    print(f"  Total audio duration: {total_dur:.1f}s ({total_dur/60:.2f} min)")
    print(f"  Manifest saved → {manifest_path}")
    print("═" * 60)

    return manifest_path


if __name__ == "__main__":
    run_phase2()