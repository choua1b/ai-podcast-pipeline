# phase5_assembly.py  (REVISED - facing each other, shared background)
"""
Phase 5: Multimodal Integration and Final Assembly
===================================================
Characters face each other on a single shared studio background.
ARIA on left (faces right), MARCUS on right (faces left / mirrored).
"""

from __future__ import annotations
import json, subprocess
from pathlib import Path
import cv2
import numpy as np

# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────

CONFIG = {
    "output_width":   1280,
    "output_height":  720,
    "fps":            25,
    "background":     Path("outputs/visuals/studio_background.png"),
    "output_dir":     Path("outputs"),
    "final_output":   Path("outputs/video_face.mp4"),
}

# Side-by-side layout — each avatar takes ~40% of screen width
LAYOUT = {
    "ARIA": {
        "x": 40,     # Left side
        "y": 80,
        "w": 460,
        "h": 460,
        "flip": False,  # ARIA faces right naturally
        "label_x": 180,
    },
    "MARCUS": {
        "x": 780,    # Right side
        "y": 80,
        "w": 460,
        "h": 460,
        "flip": True,   # MARCUS mirrored so he faces LEFT toward ARIA
        "label_x": 920,
    }
}


# ─────────────────────────────────────────────
# 2. ASSET PREPARATION
# ─────────────────────────────────────────────

def prepare_background() -> np.ndarray:
    bg = cv2.imread(str(CONFIG["background"]))
    if bg is None:
        bg = np.zeros((CONFIG["output_height"], CONFIG["output_width"], 3), dtype=np.uint8)
        bg[:] = (25, 20, 35)   # Dark fallback
    bg = cv2.resize(bg, (CONFIG["output_width"], CONFIG["output_height"]))
    # Darken background slightly so avatars pop
    bg = (bg * 0.75).astype(np.uint8)
    return bg

def prepare_static_frame(speaker: str) -> np.ndarray:
    """Loads the avatar image as a static fallback for the inactive speaker."""
    path_key = "aria_avatar" if speaker == "ARIA" else "marcus_avatar"
    path_map = {
        "aria_avatar":   Path("outputs/visuals/aria_avatar.png"),
        "marcus_avatar": Path("outputs/visuals/marcus_avatar.png"),
    }
    img = cv2.imread(str(path_map[path_key]))
    if img is None:
        img = np.zeros((460, 460, 3), dtype=np.uint8)
    l = LAYOUT[speaker]
    img = cv2.resize(img, (l["w"], l["h"]))
    if l["flip"]:
        img = cv2.flip(img, 1)   # Mirror so MARCUS faces left
    return img


# ─────────────────────────────────────────────
# 3. PER-TURN COMPOSITING
# ─────────────────────────────────────────────

def composite_turn(
    turn: dict,
    background: np.ndarray,
    static_frames: dict,
    output_path: Path,
) -> bool:
    """
    Composites one turn onto the shared studio background.
    Active speaker: animated (from Wav2Lip), full brightness.
    Inactive speaker: static avatar, dimmed to 60%.
    MARCUS is horizontally flipped so both characters face each other.
    """
    speaker  = turn["speaker"]
    mp4_path = Path(turn["mp4_path"])

    if not mp4_path.exists():
        print(f"  ✗ Missing: {mp4_path}")
        return False

    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        print(f"  ✗ Cannot open: {mp4_path}")
        return False

    fps = CONFIG["fps"]
    W   = CONFIG["output_width"]
    H   = CONFIG["output_height"]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H))

    active_layout   = LAYOUT[speaker]
    inactive_spk    = "MARCUS" if speaker == "ARIA" else "ARIA"
    inactive_layout = LAYOUT[inactive_spk]

    # Dimmed static frame for inactive speaker
    inactive_img = (static_frames[inactive_spk].copy() * 0.55).astype(np.uint8)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        canvas = background.copy()

        # ── Place inactive speaker (static, dimmed) ──
        il = inactive_layout
        inactive_resized = cv2.resize(inactive_img, (il["w"], il["h"]))
        canvas[il["y"]:il["y"]+il["h"], il["x"]:il["x"]+il["w"]] = inactive_resized

        # ── Place active speaker (animated from Wav2Lip) ──
        al = active_layout
        active_frame = cv2.resize(frame, (al["w"], al["h"]))
        if al["flip"]:
            active_frame = cv2.flip(active_frame, 1)  # Mirror MARCUS
        canvas[al["y"]:al["y"]+al["h"], al["x"]:al["x"]+al["w"]] = active_frame

        # ── Speaker name label below avatar ──
        label_y = al["y"] + al["h"] + 35
        cv2.putText(
            canvas, speaker,
            (al["label_x"], label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0, (255, 255, 255), 2, cv2.LINE_AA
        )

        # ── Inactive speaker name (dimmed) ──
        inactive_label_y = il["y"] + il["h"] + 35
        cv2.putText(
            canvas, inactive_spk,
            (il["label_x"], inactive_label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0, (150, 150, 150), 2, cv2.LINE_AA
        )

        writer.write(canvas)
        frame_count += 1

    cap.release()
    writer.release()
    print(f"  ✓ {frame_count} frames composited → {output_path.name}")
    return True


# ─────────────────────────────────────────────
# 4. CONCATENATION + AUDIO MERGE
# ─────────────────────────────────────────────

def run_ffmpeg(cmd: list, label: str) -> bool:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ✗ {label} failed:\n{result.stderr[-600:]}")
        return False
    print(f"  ✓ {label}")
    return True

def concatenate_and_merge(
    composited_paths: list[Path],
    audio_paths:      list[str],
    output_path:      Path,
) -> bool:
    print("\n[Phase 5] Concatenating video and audio...")
    Path("temp").mkdir(exist_ok=True)

    # Video concat list
    vlist = Path("temp/vconcat.txt")
    vlist.write_text("\n".join(f"file '{p.resolve()}'" for p in composited_paths))

    # Audio concat list
    alist = Path("temp/aconcat.txt")
    alist.write_text("\n".join(f"file '{Path(p).resolve()}'" for p in audio_paths))

    temp_vid   = Path("temp/video_only.mp4")
    temp_audio = Path("temp/audio_only.wav")

    ok = run_ffmpeg([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(vlist), "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        str(temp_vid)
    ], "Video concatenation")
    if not ok: return False

    ok = run_ffmpeg([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(alist), "-c", "copy", str(temp_audio)
    ], "Audio concatenation")
    if not ok: return False

    ok = run_ffmpeg([
        "ffmpeg", "-y",
        "-i", str(temp_vid),
        "-i", str(temp_audio),
        "-c:v", "copy", "-c:a", "aac", "-shortest",
        str(output_path)
    ], "Audio-video merge")
    return ok


# ─────────────────────────────────────────────
# 5. SUBTITLES
# ─────────────────────────────────────────────

def generate_srt(turns: list[dict], out_path: Path) -> Path:
    def ts(s):
        h, m = int(s//3600), int((s%3600)//60)
        sec, ms = int(s%60), int((s%1)*1000)
        return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"

    lines = []
    for i, t in enumerate(turns, 1):
        lines += [str(i),
                  f"{ts(t['start_time_seconds'])} --> {ts(t['end_time_seconds'])}",
                  f"[{t['speaker']}] {t['text']}", ""]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  ✓ Subtitles → {out_path}")
    return out_path

def burn_subtitles(video: Path, srt: Path, output: Path) -> bool:
    sub = str(srt.resolve()).replace("\\", "/").replace(":", "\\:")
    ok = run_ffmpeg([
        "ffmpeg", "-y", "-i", str(video),
        "-vf", f"subtitles='{sub}':force_style="
               f"'FontName=Arial,FontSize=14,PrimaryColour=&Hffffff,"
               f"OutlineColour=&H000000,Outline=2,Alignment=2,MarginV=20'",
        "-c:a", "copy", "-preset", "fast", str(output)
    ], "Subtitle burn-in")
    if not ok:
        import shutil; shutil.copy(video, output)
        print("  ↳ Saved without subtitles as fallback")
    return ok


# ─────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────

def run_phase5(
    lipsync_path: Path = Path("outputs/video_turns/lipsync_manifest.json"),
    audio_path:   Path = Path("outputs/audio/audio_manifest.json"),
):
    Path("temp").mkdir(exist_ok=True)

    print("[Phase 5] Loading manifests...")
    with open(lipsync_path, encoding="utf-8") as f:
        lm = json.load(f)
    with open(audio_path, encoding="utf-8") as f:
        am = json.load(f)

    turns       = lm["turns"]
    audio_turns = am["turns"]

    print("[Phase 5] Preparing assets...")
    background    = prepare_background()
    static_frames = {
        "ARIA":   prepare_static_frame("ARIA"),
        "MARCUS": prepare_static_frame("MARCUS"),
    }
    print("  ✓ Background and avatars ready")

    print(f"\n[Phase 5] Compositing {len(turns)} turns...")
    print("─" * 55)

    composited = []
    audio_wavs = []

    for turn in turns:
        tid  = turn["turn_id"]
        spk  = turn["speaker"]
        cout = Path(f"temp/comp_{tid:02d}_{spk}.mp4")
        print(f"\n  Turn {tid:02d} | {spk}")

        if cout.exists() and cout.stat().st_size > 10000:
            print(f"  Already done → skipping")
        else:
            composite_turn(turn, background, static_frames, cout)

        composited.append(cout)
        audio_wavs.append(audio_turns[tid]["wav_path"])

    # Merge
    merged = Path("temp/merged.mp4")
    if not concatenate_and_merge(composited, audio_wavs, merged):
        print("[Phase 5] ✗ Merge failed"); return

    # Subtitles
    print("\n[Phase 5] Generating subtitles...")
    srt = Path("outputs/subtitles.srt")
    generate_srt(audio_turns, srt)

    # Burn subtitles
    final = CONFIG["final_output"]
    burn_subtitles(merged, srt, final)

    size = final.stat().st_size / (1024*1024) if final.exists() else 0
    dur  = sum(t["actual_duration_seconds"] for t in audio_turns)

    print("\n" + "═"*55)
    print("  PHASE 5 COMPLETE")
    print("═"*55)
    print(f"  Output : {final}")
    print(f"  Size   : {size:.1f} MB")
    print(f"  Duration: {dur:.0f}s ({dur/60:.2f} min)")
    print("═"*55)

if __name__ == "__main__":
    run_phase5()