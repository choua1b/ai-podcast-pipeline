# phase4_lipsync.py
"""
Phase 4: Animation and Lip-Synchronization (Wav2Lip, 100% Free)
================================================================
Reads audio_manifest.json from Phase 2.
For each turn, runs Wav2Lip to animate the correct avatar
(ARIA or MARCUS) with the corresponding audio.

Output Contract (consumed by Phase 5):
    outputs/video_turns/turn_00_ARIA.mp4
    outputs/video_turns/turn_01_MARCUS.mp4
    ...
    outputs/video_turns/lipsync_manifest.json

Dependencies:
    pip install opencv-python torch torchvision ffmpeg-python
    git clone https://github.com/Rudrabha/Wav2Lip.git
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────

CONFIG = {
    "wav2lip_dir":    Path("Wav2Lip"),
    "checkpoint":     Path("Wav2Lip/checkpoints/wav2lip_gan.pth"),
    "avatar_paths": {
        "ARIA":   Path("outputs/visuals/aria_avatar.png"),
        "MARCUS": Path("outputs/visuals/marcus_avatar.png"),
    },
    "output_dir":     Path("outputs/video_turns"),
    "fps":            25,          # Wav2Lip default — do not change
    "resize_factor":  1,           # 1 = full resolution
    "pad_top":        0,
    "pad_bottom":     10,          # Slight bottom padding improves chin sync
    "pad_left":       0,
    "pad_right":      0,
    "nosmooth":       False,       # Smoothing = more natural movement
}


# ─────────────────────────────────────────────
# 2. ENVIRONMENT VALIDATION
# ─────────────────────────────────────────────

def validate_environment() -> None:
    """Checks all required files exist before processing."""
    print("[Phase 4] Validating environment...")

    # Wav2Lip repo
    if not CONFIG["wav2lip_dir"].exists():
        raise FileNotFoundError(
            "Wav2Lip directory not found.\n"
            "Run: git clone https://github.com/Rudrabha/Wav2Lip.git"
        )

    # Model weights
    if not CONFIG["checkpoint"].exists():
        raise FileNotFoundError(
            f"Wav2Lip checkpoint not found at {CONFIG['checkpoint']}\n"
            "Download wav2lip_gan.pth and place in Wav2Lip/checkpoints/"
        )

    # Avatar images
    for speaker, path in CONFIG["avatar_paths"].items():
        if not path.exists():
            raise FileNotFoundError(
                f"Avatar not found for {speaker}: {path}\n"
                "Complete Phase 3 first."
            )

    print("  ✓ Wav2Lip directory found")
    print("  ✓ Model checkpoint found")
    print("  ✓ Both avatar images found")


# ─────────────────────────────────────────────
# 3. WAV2LIP RUNNER
# ─────────────────────────────────────────────

def run_wav2lip(
    speaker:    str,
    audio_path: Path,
    output_path: Path,
) -> bool:
    """
    Runs Wav2Lip inference for one dialogue turn.

    Wav2Lip takes:
      --face   : the portrait image (or video) to animate
      --audio  : the speech audio (.wav)
      --outfile: where to save the output .mp4

    Internally it:
    1. Detects face bounding box using S3FD detector
    2. Extracts mel spectrogram from audio (80 bins, 16kHz)
    3. Generates mouth region frame-by-frame
    4. Blends generated mouth back into original face
    5. Encodes final video with ffmpeg
    """
    avatar_path = CONFIG["avatar_paths"][speaker]

    # Build the Wav2Lip inference command
    cmd = [
        sys.executable,                          # python
        str(Path("Wav2Lip/inference.py").resolve()),
        "--checkpoint_path", str(CONFIG["checkpoint"]),
        "--face",            str(avatar_path),
        "--audio",           str(audio_path),
        "--outfile",         str(output_path),
        "--fps",             str(CONFIG["fps"]),
        "--resize_factor",   str(CONFIG["resize_factor"]),
        "--pads",
            str(CONFIG["pad_top"]),
            str(CONFIG["pad_bottom"]),
            str(CONFIG["pad_left"]),
            str(CONFIG["pad_right"]),
    ]

    if CONFIG["nosmooth"]:
        cmd.append("--nosmooth")

    # Run from Wav2Lip directory so its imports resolve correctly
    env = os.environ.copy()
    env["PYTHONPATH"] = str(CONFIG["wav2lip_dir"])

    print(f"\n  Running Wav2Lip for {speaker}...")
    print(f"  Audio : {audio_path.name}")
    print(f"  Avatar: {avatar_path.name}")
    print(f"  Output: {output_path.name}")

    result = subprocess.run(
        cmd,
        cwd=str(Path.cwd()),   # Run from PROJECT root, not Wav2Lip subfolder
        env=env,
        capture_output=False,
        text=True,
    )

    if result.returncode != 0:
        print(f"  ✗ Wav2Lip failed for turn (return code {result.returncode})")
        return False

    if not output_path.exists() or output_path.stat().st_size < 10000:
        print(f"  ✗ Output file missing or too small")
        return False

    print(f"  ✓ Generated: {output_path.name}")
    return True


# ─────────────────────────────────────────────
# 4. MANIFEST BUILDER
# ─────────────────────────────────────────────

def build_lipsync_manifest(
    results: list[dict],
    output_dir: Path
) -> Path:
    """
    Builds lipsync_manifest.json — the Phase 5 assembly contract.
    Maps each turn to its animated .mp4 file with timing data.
    """
    manifest = {
        "total_turns": len(results),
        "fps":         CONFIG["fps"],
        "turns":       results,
    }

    path = output_dir / "lipsync_manifest.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n[Phase 4] ✓ Manifest saved → {path}")
    return path


# ─────────────────────────────────────────────
# 5. MAIN PIPELINE
# ─────────────────────────────────────────────

def run_phase4(
    audio_manifest_path: Path = Path("outputs/audio/audio_manifest.json"),
    output_dir:          Path = Path("outputs/video_turns"),
) -> Path:
    """Full Phase 4 execution."""

    validate_environment()

    # Load Phase 2 manifest
    print("\n[Phase 4] Loading audio manifest...")
    with open(audio_manifest_path, encoding="utf-8") as f:
        audio_manifest = json.load(f)

    turns = audio_manifest["turns"]
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Phase 4] Processing {len(turns)} turns...\n")
    print("─" * 60)

    results      = []
    failed_turns = []

    for turn in turns:
        turn_id    = turn["turn_id"]
        speaker    = turn["speaker"]
        audio_path = Path(turn["wav_path"])
        emotion    = turn["emotion_tag"]

        filename    = f"turn_{turn_id:02d}_{speaker}.mp4"
        output_path = output_dir / filename

        print(f"\n── Turn {turn_id:02d} | {speaker} | [{emotion}]")

        # Resume support
        if output_path.exists() and output_path.stat().st_size > 10000:
            print(f"  Already exists → skipping")
            success = True
        else:
            t0      = time.time()
            success = run_wav2lip(speaker, audio_path, output_path)
            elapsed = time.time() - t0
            if success:
                print(f"  Completed in {elapsed:.1f}s")
            else:
                failed_turns.append(turn_id)

        results.append({
            "turn_id":                 turn_id,
            "speaker":                 speaker,
            "text":                    turn["text"],
            "emotion_tag":             emotion,
            "mp4_path":                str(output_path),
            "wav_path":                str(audio_path),
            "actual_duration_seconds": turn["actual_duration_seconds"],
            "start_time_seconds":      turn["start_time_seconds"],
            "end_time_seconds":        turn["end_time_seconds"],
            "success":                 success,
        })

    # Summary
    print("\n" + "═" * 60)
    print("  PHASE 4 COMPLETE")
    print("═" * 60)
    succeeded = sum(1 for r in results if r["success"])
    print(f"  Successful : {succeeded}/{len(turns)} turns")
    if failed_turns:
        print(f"  Failed     : turns {failed_turns}")
        print(f"  Re-run script to retry failed turns")
    print("═" * 60)

    manifest_path = build_lipsync_manifest(results, output_dir)
    return manifest_path


if __name__ == "__main__":
    run_phase4()