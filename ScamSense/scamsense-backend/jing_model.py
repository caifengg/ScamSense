"""
Deepfake detector — extracted from deepfake_finetune_documented.ipynb.

Wraps the trained EfficientNet-B0 model + MTCNN face detection into a
loadable, callable Detector for use in a backend service. No training,
dataset, or notebook-only code lives here — this is inference only.
"""

from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import timm
from PIL import Image
from torchvision import transforms
from facenet_pytorch import MTCNN

# ── Settings (must match what the model was trained with — Cell 3 of the notebook) ──
IMG_SIZE   = 224
THRESHOLD  = 0.45                 # decision cutoff used during training/eval
MODEL_PATH = Path(__file__).parent / "deepfake_finetuned.pth"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ── Model architecture (identical to Cell 8 of the notebook) ────────────────────
class DeepfakeNet(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        # pretrained=False here — we load YOUR fine-tuned weights below, not ImageNet ones
        self.backbone = timm.create_model('efficientnet_b0', pretrained=False, num_classes=0)
        feat_dim = self.backbone.num_features   # 1280 for B0
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, 1),   # single logit → sigmoid → fake probability
        )

    def forward(self, x):
        return self.head(self.backbone(x)).squeeze(1)


# ── Face detection (identical to Cell 4 of the notebook) ───────────────────────
def _build_mtcnn(device):
    primary = MTCNN(
        image_size=160, margin=14, min_face_size=20,
        thresholds=[0.5, 0.6, 0.6],          # relaxed — catches small low-res faces
        keep_all=False, post_process=False, device=device,
    )
    sensitive = MTCNN(
        image_size=160, margin=10, min_face_size=10,
        thresholds=[0.4, 0.5, 0.5],          # even more relaxed for 2x upscaled frames
        keep_all=False, post_process=False, device=device,
    )
    return primary, sensitive


def get_face_pil(frame_rgb: np.ndarray, mtcnn: MTCNN, mtcnn_sensitive: MTCNN):
    """Returns (PIL crop, face_detected:bool). Never returns None — falls back
    to a centre-crop if MTCNN can't find a face."""
    h, w = frame_rgb.shape[:2]
    pil = Image.fromarray(frame_rgb)

    face = mtcnn(pil)
    if face is not None:
        return Image.fromarray(face.permute(1, 2, 0).byte().numpy()), True

    up = pil.resize((w * 2, h * 2), Image.BILINEAR)
    face = mtcnn_sensitive(up)
    if face is not None:
        return Image.fromarray(face.permute(1, 2, 0).byte().numpy()), True

    crop = min(h, w)
    t, l = (h - crop) // 2, (w - crop) // 2
    return pil.crop((l, t, l + crop, t + crop)), False


# ── Single-frame detector — loads once, reused across every request ────────────
class Detector:
    """Loads the model + face detector once at startup. score_frame() is safe
    to call repeatedly with no reload overhead — this is what your API layer
    should hold a single instance of."""

    def __init__(self, model_path: Path = MODEL_PATH, device: Optional[str] = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = DeepfakeNet().to(self.device)
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()
        self.mtcnn, self.mtcnn_sensitive = _build_mtcnn(self.device)

    @torch.no_grad()
    def score_frame(self, frame_rgb: np.ndarray) -> dict:
        """Score a single RGB frame (HxWx3 numpy array). Returns this frame's
        probability alone — no temporal smoothing (see RollingVerdict for that)."""
        face_pil, face_found = get_face_pil(frame_rgb, self.mtcnn, self.mtcnn_sensitive)
        face_pil = face_pil.resize((IMG_SIZE, IMG_SIZE))
        x = eval_tf(face_pil).unsqueeze(0).to(self.device)
        prob = torch.sigmoid(self.model(x)).item()
        return {
            "prob": prob,
            "verdict": "DEEPFAKE" if prob > THRESHOLD else "REAL",
            "face_found": face_found,
        }


# ── Rolling aggregation for live video ──────────────────────────────────────────
# Mirrors the notebook's per-video weighted-mean logic (Cell 12: MTCNN-detected
# frames weighted 3x over fallback crops) but as a sliding window over the most
# recent frames instead of waiting for an entire video to finish.
class RollingVerdict:
    def __init__(self, window: int = 25):
        self.window = window
        self.probs = deque(maxlen=window)
        self.weights = deque(maxlen=window)

    def update(self, prob: float, face_found: bool) -> dict:
        self.probs.append(prob)
        self.weights.append(3.0 if face_found else 1.0)
        mean_score = sum(p * w for p, w in zip(self.probs, self.weights)) / sum(self.weights)
        return {
            "prob": mean_score,
            "verdict": "DEEPFAKE" if mean_score > THRESHOLD else "REAL",
            "frames_in_window": len(self.probs),
        }
