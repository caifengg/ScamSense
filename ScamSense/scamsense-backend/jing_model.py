"""
Deepfake detector — extracted from deepfake_finetune_documented.ipynb.

Wraps the trained EfficientNet-B0 model + MTCNN face detection into a
loadable, callable Detector for use in a backend service. No training,
dataset, or notebook-only code lives here — this is inference only.

Heatmap addition: score_frame_with_heatmap() and process_video_file() add a
Grad-CAM/Layer-CAM explainability overlay on top of the same model — a
green(real)->yellow(uncertain)->orange(suspicious)->red(fake) suspicion map
with a self-sizing legend strip. This does NOT train or modify the model in
any way — it only uses .backward() to compute which pixels influenced the
existing frozen prediction, the same way score_frame() already does a plain
forward pass. deepfake_finetuned.pth itself is never written to.
"""

from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from torchvision import transforms
from facenet_pytorch import MTCNN

# ── Settings (must match what the model was trained with — Cell 3 of the notebook) ──
IMG_SIZE       = 224
THRESHOLD      = 0.12                # decision cutoff used during training/eval
MODEL_PATH     = Path(__file__).parent / "deepfake_finetuned.pth"
HEATMAP_STRIDE = 3                   # recompute Grad-CAM every N frames; forward-only on the rest

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ── Heatmap settings (new — doesn't affect score_frame()'s behaviour at all) ────
CLIP_PERCENTILE = (2, 98)    # clip outlier CAM pixels before normalizing
BLUR_SIGMA_FRAC = 0.06       # Gaussian blur sigma as a fraction of working resolution
CONTRAST_GAMMA  = 1.3
WORK_RES        = 56         # shared working resolution for combining CAM layers

COLOR_STOPS = [
    (0.00, '#2ecc71'),   # green  — real
    (0.40, '#f1c40f'),   # yellow — uncertain
    (0.65, '#e67e22'),   # orange — suspicious
    (1.00, '#ff1744'),   # red    — fake
]
BASE_ALPHA_MIN = 0.18    # opacity at score=0 (fully real) — faint but visible
BASE_ALPHA_MAX = 0.88    # opacity at score=1 (fully fake) — bright & visible
SHOW_LEGEND    = True
LEGEND_ITEMS = [('Real', '#2ecc71'), ('Uncertain', '#f1c40f'), ('Suspicious', '#e67e22'), ('Fake', '#ff1744')]

LEGEND_FONT = cv2.FONT_HERSHEY_SIMPLEX
LEGEND_FONT_SCALE = 0.42
LEGEND_FONT_THICKNESS = 1
LEGEND_CHIP_W, LEGEND_CHIP_H = 20, 16
LEGEND_PAD = 10
LEGEND_GAP_CHIP_TEXT = 6
LEGEND_GAP_ITEMS = 20
LEGEND_HEIGHT = 40


def _hex_to_rgb255(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _legend_item_width(label):
    (tw, _), _ = cv2.getTextSize(label, LEGEND_FONT, LEGEND_FONT_SCALE, LEGEND_FONT_THICKNESS)
    return LEGEND_CHIP_W + LEGEND_GAP_CHIP_TEXT + tw


def build_legend_strip():
    """Sizes itself to fit its own text — avoids overlapping labels."""
    item_widths = [_legend_item_width(label) for label, _ in LEGEND_ITEMS]
    total_w = LEGEND_PAD * 2 + sum(item_widths) + LEGEND_GAP_ITEMS * (len(LEGEND_ITEMS) - 1)
    strip = np.ones((LEGEND_HEIGHT, total_w, 3), dtype=np.uint8) * 255
    x = LEGEND_PAD
    for (label, hexcol), iw in zip(LEGEND_ITEMS, item_widths):
        color = _hex_to_rgb255(hexcol)
        chip_y0 = (LEGEND_HEIGHT - LEGEND_CHIP_H) // 2
        cv2.rectangle(strip, (x, chip_y0), (x + LEGEND_CHIP_W, chip_y0 + LEGEND_CHIP_H), color, -1)
        text_x = x + LEGEND_CHIP_W + LEGEND_GAP_CHIP_TEXT
        text_y = LEGEND_HEIGHT // 2 + 5
        cv2.putText(strip, label, (text_x, text_y), LEGEND_FONT, LEGEND_FONT_SCALE, (30, 30, 30), LEGEND_FONT_THICKNESS, cv2.LINE_AA)
        x += iw + LEGEND_GAP_ITEMS
    return strip


def _pad_to_width(img, target_w):
    if img.shape[1] >= target_w:
        return img
    pad_total = target_w - img.shape[1]
    pad_l, pad_r = pad_total // 2, pad_total - pad_total // 2
    return cv2.copyMakeBorder(img, 0, 0, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(255, 255, 255))


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
    # post_process=False keeps pixel values in [0,255] uint8; do NOT change this
    # or the .byte() conversion in get_face_pil will silently corrupt the crop.
    # image_size=224 so MTCNN outputs the exact resolution eval_tf expects —
    # avoids the extra 160→224 upscale that was not present at training time.
    primary = MTCNN(
        image_size=224, margin=14, min_face_size=20,
        thresholds=[0.5, 0.6, 0.6],          # relaxed — catches small low-res faces
        keep_all=False, post_process=False, device=device,
    )
    sensitive = MTCNN(
        image_size=224, margin=14, min_face_size=10,
        thresholds=[0.4, 0.5, 0.5],          # even more relaxed for 2x upscaled frames
        keep_all=False, post_process=False, device=device,
    )
    return primary, sensitive


def get_face_pil(frame_rgb: np.ndarray, mtcnn: MTCNN, mtcnn_sensitive: MTCNN,
                 last_good_crop: Optional[Image.Image] = None):
    """Returns (PIL crop, face_detected:bool).
    Detection order:
      1. Primary MTCNN on the original frame.
      2. Sensitive MTCNN on a 2x upscaled copy.
      3. Last known-good face crop from a previous frame (avoids feeding an
         out-of-distribution centre-crop to the model).
      4. Centre-crop of the full frame as a last resort (only when no prior
         crop is available at all).
    """
    pil = Image.fromarray(frame_rgb)

    face = mtcnn(pil)
    if face is not None:
        return Image.fromarray(face.permute(1, 2, 0).byte().numpy()), True

    h, w = frame_rgb.shape[:2]
    up = pil.resize((w * 2, h * 2), Image.BILINEAR)
    face = mtcnn_sensitive(up)
    if face is not None:
        return Image.fromarray(face.permute(1, 2, 0).byte().numpy()), True

    # Use the last frame where a face was successfully detected instead of a
    # random centre-crop — keeps the model input in-distribution.
    if last_good_crop is not None:
        return last_good_crop, False

    # Absolute last resort: centre-crop (only on the very first frames before
    # any face has ever been detected).
    crop = min(h, w)
    t, l = (h - crop) // 2, (w - crop) // 2
    return pil.crop((l, t, l + crop, t + crop)), False


# ── Single-frame detector — loads once, reused across every request ────────────
class Detector:
    """Loads the model + face detector once at startup. score_frame() is safe
    to call repeatedly with no reload overhead — this is what your API layer
    should hold a single instance of.

    score_frame_with_heatmap() and process_video_file() are additions for the
    Grad-CAM explainability overlay — same loaded model, no retraining, no
    changes to deepfake_finetuned.pth."""

    def __init__(self, model_path: Path = MODEL_PATH, device: Optional[str] = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = DeepfakeNet().to(self.device)
        state = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state)
        self.model.eval()
        self.mtcnn, self.mtcnn_sensitive = _build_mtcnn(self.device)
        self._frame_count = 0     # debug: counts frames scored
        self._last_good_crop: Optional[Image.Image] = None  # last frame with a detected face

        # ── Grad-CAM hooks (new) — registered once, reused every heatmap call ──
        self._acts = {}
        self._grads = {}

        def _make_hooks(name):
            def fwd(module, inp, out):
                self._acts[name] = out.detach()
            def bwd(module, grad_in, grad_out):
                self._grads[name] = grad_out[0].detach()
            return fwd, bwd

        fwd_shallow, bwd_shallow = _make_hooks('shallow')
        fwd_deep, bwd_deep = _make_hooks('deep')

        self._shallow_layer = self.model.backbone.blocks[3]   # 14x14 — spatial detail
        self._deep_layer = self.model.backbone.conv_head      # 7x7  — semantic gate

        self._shallow_layer.register_forward_hook(fwd_shallow)
        self._shallow_layer.register_full_backward_hook(bwd_shallow)
        self._deep_layer.register_forward_hook(fwd_deep)
        self._deep_layer.register_full_backward_hook(bwd_deep)

        self._cmap = LinearSegmentedColormap.from_list('fake_real', COLOR_STOPS)
        self._legend = build_legend_strip()   # built once — layout is static

        # ── Heatmap throttle cache ────────────────────────────────────────────
        self._heatmap_tick: int = 0
        self._cached_score_map: Optional[np.ndarray] = None   # last computed score map
        self._cached_heatmap_bbox: Optional[tuple] = None     # last (x1,y1,x2,y2) in frame coords

    @torch.no_grad()
    def score_frame(self, frame_rgb: np.ndarray) -> dict:
        """Score a single RGB frame (HxWx3 numpy array). Returns this frame's
        probability alone — no temporal smoothing (see RollingVerdict for that)."""
        face_pil, face_found = get_face_pil(
            frame_rgb, self.mtcnn, self.mtcnn_sensitive, self._last_good_crop
        )
        if face_found:
            self._last_good_crop = face_pil  # cache for use on missed-face frames
        # MTCNN now outputs 224×224 directly; eval_tf's Resize is a no-op but kept
        # for safety in case image_size is ever changed above.
        x = eval_tf(face_pil).unsqueeze(0).to(self.device)
        prob = torch.sigmoid(self.model(x)).item()
        verdict = "DEEPFAKE" if prob > THRESHOLD else "REAL"

        # ── Debug output ────────────────────────────────────────────────────────
        self._frame_count += 1
        print(
            f"[debug] frame={self._frame_count:05d} "
            f"prob={prob:.4f} verdict={verdict} face_found={face_found}"
        )
        if self._frame_count % 100 == 0:
            debug_path = Path(__file__).parent / f"debug_{self._frame_count}.jpg"
            face_pil.save(debug_path)
            print(f"[debug] saved face crop → {debug_path}")
        # ── End debug ────────────────────────────────────────────────────────────

        return {
            "prob": prob,
            "verdict": verdict,
            "face_found": face_found,
        }

    # ═══════════════════════════════════════════════════════════════════════
    # Grad-CAM heatmap (new) — everything below is additive, score_frame()
    # above is completely untouched.
    # ═══════════════════════════════════════════════════════════════════════

    def _layer_cam_raw(self, name):
        """Per-pixel gradient weighting (Layer-CAM) — preserves spatial detail."""
        grads = self._grads[name]
        acts = self._acts[name]
        weights = F.relu(grads)
        cam = F.relu((weights * acts).sum(dim=1, keepdim=True))
        return cam.squeeze().cpu().numpy()

    def _grad_cam_raw(self, name):
        """Standard Grad-CAM (global-average-pooled weight) — coarse semantic mask."""
        grads = self._grads[name]
        acts = self._acts[name]
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * acts).sum(dim=1, keepdim=True))
        return cam.squeeze().cpu().numpy()

    def _compute_fused_cam(self, face_pil: Image.Image):
        """Fused shallow(14x14 Layer-CAM) + deep(7x7 Grad-CAM) attention map,
        averaged (not multiplied), percentile-clipped, blurred, gamma-adjusted.
        Backprop target is the model's own predicted class, so this explains
        whichever verdict the model actually gave."""
        x = eval_tf(face_pil).unsqueeze(0).to(self.device)
        self.model.zero_grad(set_to_none=True)
        logit = self.model(x)
        prob = torch.sigmoid(logit).item()
        verdict = "DEEPFAKE" if prob > THRESHOLD else "REAL"

        target = logit if verdict == "DEEPFAKE" else -logit
        target.backward()

        shallow_raw = self._layer_cam_raw('shallow')
        deep_raw = self._grad_cam_raw('deep')

        shallow_up = cv2.resize(shallow_raw, (WORK_RES, WORK_RES), interpolation=cv2.INTER_CUBIC)
        deep_up = cv2.resize(deep_raw, (WORK_RES, WORK_RES), interpolation=cv2.INTER_CUBIC)

        fused = 0.5 * shallow_up + 0.5 * deep_up
        lo, hi = np.percentile(fused, CLIP_PERCENTILE)
        fused = np.clip(fused, lo, hi)
        fused = fused - fused.min()
        fused = fused / (fused.max() + 1e-8)

        sigma = WORK_RES * BLUR_SIGMA_FRAC
        fused = cv2.GaussianBlur(fused, (0, 0), sigmaX=sigma)
        fused = np.clip(fused, 0, 1) ** CONTRAST_GAMMA

        return fused, prob, verdict

    def _compute_score_map(self, face_pil: Image.Image):
        """Remaps the CAM into the real(0)<->fake(1) scale, anchored at 0.5=uncertain.
        Verdict DEEPFAKE pushes attention up toward red (0.5->1.0);
        verdict REAL pushes attention down toward green (0.5->0.0)."""
        cam, prob, verdict = self._compute_fused_cam(face_pil)
        score_map = 0.5 + 0.5 * cam if verdict == "DEEPFAKE" else 0.5 - 0.5 * cam
        return score_map, prob, verdict

    def _make_overlay(self, face_pil: Image.Image, score_map_low_res: np.ndarray) -> np.ndarray:
        score_224 = cv2.resize(score_map_low_res, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
        score_224 = np.clip(score_224, 0, 1)

        heat_rgb = self._cmap(score_224)[..., :3]
        face_np = np.array(face_pil.convert('RGB')).astype(np.float32) / 255.0

        # Opacity scales across the full 0..1 range: faint/transparent on the
        # real side, bright/visible on the fake side.
        alpha_map = (BASE_ALPHA_MIN + (BASE_ALPHA_MAX - BASE_ALPHA_MIN) * score_224)[..., None]
        overlay = np.clip(face_np * (1 - alpha_map) + heat_rgb * alpha_map, 0, 1)
        overlay_uint8 = (overlay * 255).astype(np.uint8)

        if SHOW_LEGEND:
            legend = self._legend
            canvas_w = max(overlay_uint8.shape[1], legend.shape[1])
            overlay_uint8 = _pad_to_width(overlay_uint8, canvas_w)
            legend = _pad_to_width(legend, canvas_w)
            overlay_uint8 = np.vstack([overlay_uint8, legend])

        return overlay_uint8

    def _make_fullframe_overlay(self, frame_rgb: np.ndarray, score_map_low_res: np.ndarray, bbox: tuple) -> np.ndarray:
        """Blend the heatmap only over the detected face bbox and return the full
        original frame so the rest of the video stays visible and un-zoomed.
        The legend strip is appended to the bottom edge as before."""
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0:
            return frame_rgb.copy()

        # Resize the low-res score map to exactly the face bbox dimensions
        score_bbox = cv2.resize(score_map_low_res, (bw, bh), interpolation=cv2.INTER_CUBIC)
        score_bbox = np.clip(score_bbox, 0, 1)

        heat_rgb  = self._cmap(score_bbox)[..., :3]   # HxWx3 float [0,1]
        result    = frame_rgb.astype(np.float32) / 255.0
        face_rgn  = result[y1:y2, x1:x2]
        alpha_map = (BASE_ALPHA_MIN + (BASE_ALPHA_MAX - BASE_ALPHA_MIN) * score_bbox)[..., None]
        result[y1:y2, x1:x2] = np.clip(face_rgn * (1 - alpha_map) + heat_rgb * alpha_map, 0, 1)

        result_uint8 = (result * 255).astype(np.uint8)
        if SHOW_LEGEND:
            legend   = self._legend
            canvas_w = max(result_uint8.shape[1], legend.shape[1])
            result_uint8 = _pad_to_width(result_uint8, canvas_w)
            legend       = _pad_to_width(legend, canvas_w)
            result_uint8 = np.vstack([result_uint8, legend])
        return result_uint8

    def _make_heatmap_rgba_crop(self, score_map_low_res: np.ndarray, bbox: tuple) -> np.ndarray:
        """Returns an RGBA uint8 array sized to the face bbox — pure heatmap
        colour with per-pixel alpha encoding opacity. Designed to be composited
        by the browser canvas directly over the live video feed so the video
        keeps running at full frame rate underneath."""
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0:
            return np.zeros((0, 0, 4), dtype=np.uint8)

        score_bbox = cv2.resize(score_map_low_res, (bw, bh), interpolation=cv2.INTER_CUBIC)
        score_bbox = np.clip(score_bbox, 0, 1)

        heat_rgb = self._cmap(score_bbox)[..., :3]   # HxWx3 float [0,1]
        alpha_f  = BASE_ALPHA_MIN + (BASE_ALPHA_MAX - BASE_ALPHA_MIN) * score_bbox

        rgba = np.zeros((bh, bw, 4), dtype=np.uint8)
        rgba[..., :3] = (heat_rgb * 255).astype(np.uint8)
        rgba[..., 3]  = (alpha_f  * 255).astype(np.uint8)
        return rgba

    def score_frame_with_heatmap(self, frame_rgb: np.ndarray) -> dict:
        """Full-frame heatmap: detects the face bbox, blends the Grad-CAM heat
        only over that region, and returns the complete webcam frame so the
        rest of the video stays visible and un-zoomed.

        Throttled via HEATMAP_STRIDE — Grad-CAM backprop runs only every N
        calls; intermediate frames do a cheap forward-only score pass and
        re-apply the last computed score map + bbox to the current pixel data
        so the overlay stays spatially anchored without repeated backprop.
        """
        self._heatmap_tick += 1
        run_cam = (self._heatmap_tick % HEATMAP_STRIDE == 1) or (self._cached_score_map is None)

        # ── Always: get face crop for scoring ────────────────────────────────
        face_pil, face_found = get_face_pil(
            frame_rgb, self.mtcnn, self.mtcnn_sensitive, self._last_good_crop
        )
        if face_found:
            self._last_good_crop = face_pil

        if run_cam:
            # ── Heatmap frame: detect bbox, run full Grad-CAM ─────────────────
            pil_full = Image.fromarray(frame_rgb)
            boxes, _ = self.mtcnn.detect(pil_full)
            bbox = None
            if boxes is not None and len(boxes) > 0:
                h, w = frame_rgb.shape[:2]
                x1, y1, x2, y2 = (int(v) for v in boxes[0])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 > x1 and y2 > y1:
                    bbox = (x1, y1, x2, y2)

            score_map, prob, verdict = self._compute_score_map(face_pil)
            self._cached_score_map  = score_map
            self._cached_heatmap_bbox = bbox
        else:
            # ── Throttled frame: cheap forward-only score, reuse cached map ───
            with torch.no_grad():
                x = eval_tf(face_pil).unsqueeze(0).to(self.device)
                prob    = torch.sigmoid(self.model(x)).item()
                verdict = "DEEPFAKE" if prob > THRESHOLD else "REAL"
            score_map = self._cached_score_map
            bbox      = self._cached_heatmap_bbox

        # ── Build face-crop RGBA on Grad-CAM frames; return None on throttled frames ──
        # Frontend caches the last heatmap image and re-draws it at the new bbox
        # position every rAF tick (30 fps), decoupling render rate from compute rate.
        heatmap_rgba = (
            self._make_heatmap_rgba_crop(score_map, bbox)
            if (run_cam and bbox is not None) else None
        )

        return {
            "prob": prob,
            "verdict": verdict,
            "face_found": face_found,
            "heatmap_rgba": heatmap_rgba,   # RGBA face-crop; None on throttled frames
            "bbox": bbox,                   # always current face position (None if no face)
        }

    def process_video_file(self, video_path, output_path=None, frame_stride=2, max_frames=None) -> dict:
        """Scores an entire video file: samples every `frame_stride`-th frame,
        writes an annotated heatmap+legend .mp4 to `output_path` (if given), and
        returns one aggregate verdict — same weighted-mean logic as RollingVerdict
        below (face-detected frames weighted 3x over fallback-crop frames).
        The resolved output path is included in the returned dict.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {"error": f"Could not open {video_path}"}

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        writer = None
        if output_path is not None:
            canvas_w = max(IMG_SIZE, self._legend.shape[1] if SHOW_LEGEND else 0)
            out_h = IMG_SIZE + (LEGEND_HEIGHT if SHOW_LEGEND else 0)
            writer = cv2.VideoWriter(
                str(output_path), cv2.VideoWriter_fourcc(*'mp4v'),
                max(fps / frame_stride, 1), (canvas_w, out_h),
            )

        probs, weights = [], []
        frame_idx, processed = 0, 0
        try:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                if frame_idx % frame_stride != 0:
                    frame_idx += 1
                    continue

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                face_pil, face_found = get_face_pil(
                    frame_rgb, self.mtcnn, self.mtcnn_sensitive, self._last_good_crop
                )
                if not face_found and self._last_good_crop is None:
                    frame_idx += 1
                    continue

                crop = face_pil if face_found else self._last_good_crop
                score_map, prob, verdict = self._compute_score_map(crop)
                if face_found:
                    self._last_good_crop = face_pil

                probs.append(prob)
                weights.append(3.0 if face_found else 1.0)

                if writer is not None:
                    overlay = self._make_overlay(crop, score_map)
                    writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

                processed += 1
                frame_idx += 1
                if max_frames and processed >= max_frames:
                    break
        finally:
            cap.release()
            if writer is not None:
                writer.release()

        if not probs:
            return {"error": "No faces detected anywhere in this video."}

        mean_prob = sum(p * w for p, w in zip(probs, weights)) / sum(weights)
        return {
            "verdict": "DEEPFAKE" if mean_prob > THRESHOLD else "REAL",
            "prob": mean_prob,
            "frames_processed": processed,
            "output_video": str(Path(output_path).resolve()) if output_path else None,
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
        probs_list = list(self.probs)
        deepfake_frames = sum(1 for p in probs_list if p > THRESHOLD)
        return {
            "prob": mean_score,
            "verdict": "DEEPFAKE" if mean_score > THRESHOLD else "REAL",
            "frames_in_window": len(probs_list),
            # Extra statistics used only for LLM explanation — do not affect verdict
            "max_prob": max(probs_list),
            "min_prob": min(probs_list),
            "deepfake_frames": deepfake_frames,
            "real_frames": len(probs_list) - deepfake_frames,
        }


# ── Example usage ────────────────────────────────────────────────────────────
# detector = Detector()
#
# Plain scoring (unchanged from before):
#   result = detector.score_frame(some_rgb_frame)
#
# Scoring + heatmap for a single frame:
#   result = detector.score_frame_with_heatmap(some_rgb_frame)
#   result["heatmap_overlay_rgb"]  # uint8 RGB array, ready to JPEG-encode/stream
#
# Batch-test a whole video file:
#   result = detector.process_video_file("suspect_video.mp4", "annotated_out.mp4")
#   print(result)   # result["output_video"] is the absolute path to the saved file