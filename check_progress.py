from pathlib import Path
import json

print("=" * 55)
print("  PROJECT PROGRESS CHECK")
print("=" * 55)

# Phase 1
f = Path("outputs/dialogue.json")
if f.exists():
    data = json.loads(f.read_text(encoding="utf-8"))
    turns = len(data.get("turns", []))
    words = data.get("total_word_count", 0)
    print(f"\n✓ Phase 1 — dialogue.json ({turns} turns, {words} words)")
else:
    print(f"\n✗ Phase 1 — dialogue.json MISSING")

# Phase 2
wav_files = list(Path("outputs/audio").glob("*.wav")) if Path("outputs/audio").exists() else []
manifest  = Path("outputs/audio/audio_manifest.json")
print(f"\n{'✓' if len(wav_files)==12 else '✗'} Phase 2 — Audio ({len(wav_files)}/12 wav files)")
if manifest.exists():
    m = json.loads(manifest.read_text(encoding="utf-8"))
    print(f"           Manifest: {m['total_duration_seconds']:.1f}s total")

# Phase 3
visuals = {
    "aria_avatar.png":        Path("outputs/visuals/aria_avatar.png"),
    "marcus_avatar.png":      Path("outputs/visuals/marcus_avatar.png"),
    "studio_background.png":  Path("outputs/visuals/studio_background.png"),
}
v_count = sum(1 for p in visuals.values() if p.exists())
print(f"\n{'✓' if v_count==3 else '✗'} Phase 3 — Visuals ({v_count}/3 images)")
for name, path in visuals.items():
    print(f"           {'✓' if path.exists() else '✗'} {name}")

# Phase 4
mp4_files = list(Path("outputs/video_turns").glob("*.mp4")) if Path("outputs/video_turns").exists() else []
lipsync   = Path("outputs/video_turns/lipsync_manifest.json")
print(f"\n{'✓' if len(mp4_files)==12 else '✗'} Phase 4 — Lip-sync ({len(mp4_files)}/12 mp4 files)")

# Phase 5
final = Path("outputs/final_podcast.mp4")
print(f"\n{'✓' if final.exists() else '✗'} Phase 5 — Final video {'EXISTS' if final.exists() else 'NOT YET'}")

# ffmpeg
import shutil
ffmpeg = shutil.which("ffmpeg")
print(f"\n{'✓' if ffmpeg else '✗'} ffmpeg — {'found at ' + ffmpeg if ffmpeg else 'NOT FOUND'}")

print("\n" + "=" * 55)