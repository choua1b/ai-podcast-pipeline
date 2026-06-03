# AI-Generated Multimodal Podcast Pipeline

**Student:** Chouaib Jbel  
**Academic Year:** 2025–2026  

## Overview
A fully automated 6-phase Python pipeline that generates a professional
podcast video between two AI virtual characters (ARIA & MARCUS) discussing
the impact of Artificial Intelligence on society, education, work and creativity.

## Pipeline Phases
| Phase | Description | Technology |
|-------|-------------|------------|
| 1 | Dialogue Script Generation | Anthropic Claude |
| 2 | Audio Voice Synthesis | Piper TTS (VITS) |
| 3 | Avatar & Background Images | Bing Image Creator |
| 4 | Lip-Synchronization | Wav2Lip GAN |
| 5 | Final Video Assembly | ffmpeg + OpenCV |
| 6 | Evaluation & Report | LaTeX |

## Results
- Final video duration: **178.7 seconds (2.98 minutes)**
- Total spoken words: **334 words**
- Dialogue turns: **12 turns**
- Total cost: **$0.00 (fully open source)**

## Installation
```bash
pip install piper-tts soundfile numpy pydub librosa opencv-python torch torchvision
```

## How to Run
```bash
python phase1_dialogue_generation.py
python phase2_audio_generation.py
python phase4_lipsync.py
python phase5_assembly.py
```

## Project Structure
```
ai-podcast-pipeline/
├── phase1_dialogue_generation.py
├── phase2_audio_generation.py
├── phase4_lipsync.py
├── phase5_assembly.py
├── check_progress.py
├── outputs/
│   ├── dialogue.json
│   └── subtitles.srt
└── Wav2Lip/
``` 
