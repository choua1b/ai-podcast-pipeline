import torch
import cv2
import numpy

print("PyTorch  :", torch.__version__)
print("OpenCV   :", cv2.__version__)
print("NumPy    :", numpy.__version__)
print("CUDA available:", torch.cuda.is_available())
print("All good!")