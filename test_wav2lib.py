import sys
sys.path.insert(0, 'Wav2Lip')

import torch
import numpy as np
import cv2

print("PyTorch:", torch.__version__)
print("NumPy:", np.__version__)
print("OpenCV:", cv2.__version__)

# Test that Wav2Lip core files exist
from pathlib import Path
required = [
    'Wav2Lip/inference.py',
    'Wav2Lip/models/wav2lip.py',
    'Wav2Lip/checkpoints/wav2lip_gan.pth',
]
for f in required:
    exists = Path(f).exists()
    print(f"  {'✓' if exists else '✗'} {f}")