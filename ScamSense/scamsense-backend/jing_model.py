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

Stylized addition: score_frame_with_stylized_heatmap() adds a full-frame Jet
colormap version matching the ScamSense mockup — genuine Eigen-CAM detail
inside the detected face box, a decorative radial feather fading outward over
shoulders/background (clearly labeled as such via an on-screen disclaimer),
and a 3-tier High/Medium/Low Suspicion legend.

Bugfix: _make_heatmap_rgba_crop previously coloured pixels using the raw
Grad-CAM activation, which is always min-max normalized to 0..1 WITHIN each
frame — so even a fully REAL face always had some "hottest" pixel pushed
toward red/blue. Colour is now derived from the verdict-remapped score_map
(same green(real)->red(fake) scale used everywhere else in this file), so a
REAL verdict can no longer produce red/blue anywhere on the face.

Contrast update: opacity ranges widened (BASE_ALPHA_MIN/MAX, STYLIZED_ALPHA_
MIN/MAX, and the live-crop threshold/gamma) so the colour reads much more
strongly overall, and the red end of COLOR_STOPS was changed to a pure,
earlier-triggering red so high-suspicion regions look unmistakably red
instead of a muted pink that only appears at the very top of the scale.
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

# facenet_pytorch's extract_face — needed so we can get BOTH the face crop
# AND its box coordinates from a single detection pass.
try:
    from facenet_pytorch.models.utils.detect_face import extract_face
    _HAS_EXTRACT_FACE = True
except ImportError:
    _HAS_EXTRACT_FACE = False

# MediaPipe Face Mesh — optional; graceful fallback to soft ellipse if absent.
try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False
    mp = None

# ── Settings (must match what the model was trained with — Cell 3 of the notebook) ──
IMG_SIZE       = 224
THRESHOLD      = 0.03                # decision cutoff — anything above 3% is flagged as deepfake
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

# Red moved earlier (0.75 instead of 1.00) and made pure (#ff0000, no pink) so
# high-suspicion regions read as unmistakably red instead of a muted colour
# that only ever appeared right at the extreme top of the scale.
COLOR_STOPS = [
    (0.00, '#2ecc71'),   # green  — real
    (0.35, '#f1c40f'),   # yellow — uncertain
    (0.55, '#e67e22'),   # orange — suspicious
    (0.75, '#ff0000'),   # red    — fake (reached earlier, pure red)
    (1.00, '#ff0000'),   # stays pure red — no fade at the very top
]
BASE_ALPHA_MIN = 0.35    # opacity at score=0 (fully real) — was 0.18, now much stronger
BASE_ALPHA_MAX = 1.00    # opacity at score=1 (fully fake) — was 0.88, now fully opaque
SHOW_LEGEND    = True
LEGEND_ITEMS = [('Real', '#2ecc71'), ('Uncertain', '#f1c40f'), ('Suspicious', '#e67e22'), ('Fake', '#ff0000')]

# NOTE: LIVE_HEATMAP_COLORS / self._live_cmap are no longer used by
# _make_heatmap_rgba_crop after the colour-bug fix below — colour now comes
# from self._cmap (verdict-remapped) instead. Left in place, harmless, in
# case anything else references them; safe to delete if you want to tidy up.
LIVE_HEATMAP_COLORS = [
    (0.00, '#00ff00'),   # green - low
    (0.50, '#0066ff'),   # blue - medium
    (1.00, '#ff0000'),   # red - high
]

LEGEND_FONT = cv2.FONT_HERSHEY_SIMPLEX
LEGEND_FONT_SCALE = 0.42
LEGEND_FONT_THICKNESS = 1
LEGEND_CHIP_W, LEGEND_CHIP_H = 20, 16
LEGEND_PAD = 10
LEGEND_GAP_CHIP_TEXT = 6
LEGEND_GAP_ITEMS = 20
LEGEND_GRADIENT_H = 14   # height of the green→red gradient bar
LEGEND_HEIGHT = 40 + LEGEND_GRADIENT_H + 4   # +4 for spacing above the bar

# ── Stylized full-frame heatmap settings (Jet colormap, feathered spread) ──
JET_CMAP = plt.get_cmap('jet')
FEATHER_INNER_MULT = 1.0
FEATHER_OUTER_MULT = 3.0
STYLIZED_ALPHA_MIN = 0.30   # was 0.10 — stronger even in the decorative wash zone
STYLIZED_ALPHA_MAX = 1.00   # was 0.80 — fully opaque at peak suspicion

LEGEND_TIERS = [
    ('High Suspicion', '#ff0000'),     # pure red — matches the more vivid fake colour above
    ('Medium Suspicion', '#ff9500'),
    ('Low Suspicion', '#ffcc00'),
]
DISCLAIMER_TEXT = "Colour beyond the face is stylised, not analysed"


def _hex_to_rgb255(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _legend_item_width(label):
    (tw, _), _ = cv2.getTextSize(label, LEGEND_FONT, LEGEND_FONT_SCALE, LEGEND_FONT_THICKNESS)
    return LEGEND_CHIP_W + LEGEND_GAP_CHIP_TEXT + tw


def build_legend_strip():
    """Sizes itself to fit its own text — avoids overlapping labels.
    Includes a green→yellow→orange→red gradient bar below the chip labels
    so users can read heatmap colours at a glance."""
    item_widths = [_legend_item_width(label) for label, _ in LEGEND_ITEMS]
    total_w = LEGEND_PAD * 2 + sum(item_widths) + LEGEND_GAP_ITEMS * (len(LEGEND_ITEMS) - 1)
    chip_row_h = 40   # fixed height for the chip+label row
    strip = np.ones((LEGEND_HEIGHT, total_w, 3), dtype=np.uint8) * 255
    x = LEGEND_PAD
    for (label, hexcol), iw in zip(LEGEND_ITEMS, item_widths):
        color = _hex_to_rgb255(hexcol)
        chip_y0 = (chip_row_h - LEGEND_CHIP_H) // 2
        cv2.rectangle(strip, (x, chip_y0), (x + LEGEND_CHIP_W, chip_y0 + LEGEND_CHIP_H), color, -1)
        text_x = x + LEGEND_CHIP_W + LEGEND_GAP_CHIP_TEXT
        text_y = chip_row_h // 2 + 5
        cv2.putText(strip, label, (text_x, text_y), LEGEND_FONT, LEGEND_FONT_SCALE, (30, 30, 30), LEGEND_FONT_THICKNESS, cv2.LINE_AA)
        x += iw + LEGEND_GAP_ITEMS

    # ── Gradient bar: green(real, left) → yellow → orange → red(fake, right) ──
    bar_x0 = LEGEND_PAD
    bar_x1 = total_w - LEGEND_PAD
    bar_w  = max(bar_x1 - bar_x0, 1)
    bar_y0 = chip_row_h + 4
    bar_y1 = bar_y0 + LEGEND_GRADIENT_H
    # Sample COLOR_STOPS colormap across bar_w pixels
    _grad_cmap = LinearSegmentedColormap.from_list('_grad', COLOR_STOPS)
    for px in range(bar_w):
        t = px / max(bar_w - 1, 1)
        r, g, b, _ = _grad_cmap(t)
        bgr = (int(b * 255), int(g * 255), int(r * 255))
        cv2.line(strip, (bar_x0 + px, bar_y0), (bar_x0 + px, bar_y1 - 1), bgr, 1)
    # Thin border around the bar
    cv2.rectangle(strip, (bar_x0, bar_y0), (bar_x1 - 1, bar_y1 - 1), (180, 180, 180), 1)

    return strip


def _pad_to_width(img, target_w):
    if img.shape[1] >= target_w:
        return img
    pad_total = target_w - img.shape[1]
    pad_l, pad_r = pad_total // 2, pad_total - pad_total // 2
    return cv2.copyMakeBorder(img, 0, 0, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(255, 255, 255))


def _hex_to_bgr(h):
    h = h.lstrip('#')
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)   # cv2 draws in BGR


def draw_tier_legend_and_disclaimer(frame_bgr):
    """Draws the 3-tier legend (top-left) and disclaimer badge (bottom-left)
    directly onto a BGR frame, in place."""
    h, w = frame_bgr.shape[:2]

    x, y = 16, 16
    for label, hexcol in LEGEND_TIERS:
        color = _hex_to_bgr(hexcol)
        cv2.circle(frame_bgr, (x + 6, y + 6), 6, color, -1)
        cv2.putText(frame_bgr, label, (x + 20, y + 11), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22

    (tw, th), _ = cv2.getTextSize(DISCLAIMER_TEXT, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
    pad = 8
    badge_h = th + pad * 2
    cv2.rectangle(frame_bgr, (10, h - badge_h - 10), (10 + tw + pad * 2, h - 10),
                  (30, 30, 30), -1)
    cv2.putText(frame_bgr, DISCLAIMER_TEXT, (10 + pad, h - 10 - pad + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)

    return frame_bgr


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
    changes to deepfake_finetuned.pth.

    score_frame_with_stylized_heatmap() is the full-frame Jet-colormap version
    matching the ScamSense mockup — same loaded model, no retraining."""

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
        self._live_cmap = LinearSegmentedColormap.from_list('live_heat', LIVE_HEATMAP_COLORS)
        self._legend = build_legend_strip()   # built once — layout is static

        # ── Heatmap throttle cache ────────────────────────────────────────────
        self._heatmap_tick: int = 0
        self._cached_score_map: Optional[np.ndarray] = None   # last computed raw CAM
        self._cached_heatmap_bbox: Optional[tuple] = None     # last (x1,y1,x2,y2) in frame coords

        # ── MediaPipe Face Mesh ────────────────────────────────────────────
        # When available, FaceMesh traces the actual face oval so the heatmap
        # follows the real face shape rather than a rectangular bounding box.
        # Gracefully falls back to a soft ellipse if MediaPipe isn't installed.
        self._has_mediapipe = False
        self._mp_face_mesh = None
        self._face_oval_indices: list = []
        if _MP_AVAILABLE:
            try:
                self._mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                # FACEMESH_FACE_OVAL is a frozenset of (start_idx, end_idx) pairs
                self._face_oval_indices = sorted(
                    set(i for conn in mp.solutions.face_mesh.FACEMESH_FACE_OVAL for i in conn)
                )
                self._has_mediapipe = True
            except Exception as _mp_err:
                print(f"[jing_model] MediaPipe unavailable ({_mp_err}); using ellipse fallback.")
        self._cached_face_mask: Optional[np.ndarray] = None   # face-shape mask for current bbox

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

    def _get_face_shape_mask(self, frame_rgb: np.ndarray, bbox: tuple) -> np.ndarray:
        """Returns a float32 HxW mask (0–1) shaped to the real face oval.

        When MediaPipe is available: runs FaceMesh on the full RGB frame, maps
        the FACEMESH_FACE_OVAL landmark ring to bbox-relative pixel coords,
        fills the polygon, then feathers the boundary with a Gaussian blur.

        When MediaPipe is absent or detects no face: falls back to a soft
        ellipse approximating a head shape (slightly taller than wide).
        """
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1

        def _ellipse_fallback() -> np.ndarray:
            m = np.zeros((bh, bw), dtype=np.float32)
            cv2.ellipse(m, (bw // 2, bh // 2),
                        (int(bw * 0.48), int(bh * 0.52)),
                        angle=0, startAngle=0, endAngle=360, color=1.0, thickness=-1)
            m = cv2.GaussianBlur(m, (0, 0), sigmaX=max(bw, bh) * 0.025)
            return np.clip(m, 0, 1)

        if not self._has_mediapipe or self._mp_face_mesh is None:
            return _ellipse_fallback()

        H, W = frame_rgb.shape[:2]
        results = self._mp_face_mesh.process(frame_rgb)

        if not results.multi_face_landmarks:
            return _ellipse_fallback()

        landmarks = results.multi_face_landmarks[0].landmark

        # Map face-oval landmark ring to bbox-relative integer coordinates
        oval_pts = []
        for idx in self._face_oval_indices:
            lm = landmarks[idx]
            px = int(round(lm.x * W - x1))
            py = int(round(lm.y * H - y1))
            # Clamp to bbox bounds so fillPoly never writes outside the array
            px = max(0, min(bw - 1, px))
            py = max(0, min(bh - 1, py))
            oval_pts.append([px, py])

        if len(oval_pts) < 3:
            return _ellipse_fallback()

        mask = np.zeros((bh, bw), dtype=np.float32)
        cv2.fillPoly(mask, [np.array(oval_pts, dtype=np.int32)], 1.0)

        # Feather the boundary for smooth blending with the video
        sigma = max(bw, bh) * 0.025
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma)
        return np.clip(mask, 0, 1)

    def _make_heatmap_rgba_crop(self, cam_low_res: np.ndarray, bbox: tuple, verdict: str,
                                  face_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Returns an RGBA uint8 crop for browser-side canvas compositing.

        Colour is driven by the verdict-remapped score_map (same green(real)
        ->red(fake) scale as self._cmap elsewhere in this file), not the raw
        Grad-CAM activation. The raw activation is always min-max normalized
        to 0..1 WITHIN each frame, so even a fully REAL face always had some
        "hottest" pixel — that's why real faces were showing red/blue patches
        before this fix. Shape/transparency still comes from the raw
        activation; only the colour source changed.

        Pipeline:
          1. Upsample CAM to bbox size with cubic interpolation.
          2. Gaussian blur → smooth, organic heat blobs with feathered edges.
          3. Hard threshold at 0.10 → cam < 0.10 is FULLY TRANSPARENT (alpha=0).
             Lower threshold than before (was 0.25) so more of the face counts
             as "active" instead of being cut to invisible.
          4. Above threshold: alpha = (activation ^ 0.6) * 1.0
             → lower exponent and no opacity ceiling means mid-range
             activation reaches strong opacity much faster (was ^1.2 * 0.72).
          5. Verdict-remapped green->red colormap applied — a REAL verdict
             can only ever produce green/yellow, never orange/red, and vice
             versa for DEEPFAKE. Colour is independent of alpha; only the
             bbox position determines where the overlay appears on the video.
        """
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0:
            return np.zeros((0, 0, 4), dtype=np.uint8)

        # 1. Upsample raw CAM to face bbox dimensions
        cam_bbox = cv2.resize(cam_low_res, (bw, bh), interpolation=cv2.INTER_CUBIC)
        cam_bbox = np.clip(cam_bbox, 0, 1)

        # 2. Gaussian blur → soft, organic blobs (no hard rectangular edges)
        sigma = max(bw, bh) * 0.08
        cam_smooth = cv2.GaussianBlur(cam_bbox, (0, 0), sigmaX=sigma)
        cam_smooth = np.clip(cam_smooth, 0, 1)

        # 3. Hard threshold: everything below 0.10 is fully transparent
        _THRESH = 0.10
        activation = np.where(
            cam_smooth < _THRESH,
            0.0,
            np.clip((cam_smooth - _THRESH) / (1.0 - _THRESH), 0.0, 1.0),
        )

        # 4. Progressive alpha — near-full strength, no opacity ceiling
        alpha_f = (activation ** 0.6) * 1.0

        # 4b. Multiply by face-shape mask so heatmap is clipped to the face oval.
        #     Pixels outside the oval get alpha=0 regardless of CAM activation.
        if face_mask is not None:
            fm = face_mask if face_mask.shape == (bh, bw) else cv2.resize(
                face_mask, (bw, bh), interpolation=cv2.INTER_LINEAR)
            alpha_f = alpha_f * np.clip(fm, 0, 1)

        # 5. Colour driven by the ACTUAL verdict, not raw attention magnitude.
        # cam_smooth alone is always renormalized 0..1 per-frame, so even a
        # fully real face would show a "hottest" pixel near 1.0 — that's why
        # real faces were showing red/blue. Remap through the verdict first,
        # using the same green(real)->red(fake) scale as the rest of the file.
        score_map = 0.5 + 0.5 * cam_smooth if verdict == "DEEPFAKE" else 0.5 - 0.5 * cam_smooth
        heat_rgb = self._cmap(score_map)[..., :3]   # HxWx3 float [0,1]

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

        H, W = frame_rgb.shape[:2]
        pil_full = Image.fromarray(frame_rgb)

        if run_cam:
            # ── Grad-CAM frame: ONE mtcnn.detect for bbox + face crop ─────────────
            # Previously: get_face_pil (MTCNN) + mtcnn.detect (second MTCNN) = 2x.
            # Now: a single detect() call; face crop extracted from that box.
            boxes, _ = self.mtcnn.detect(pil_full)
            bbox = None
            face_pil = None
            face_found = False

            if boxes is not None and len(boxes) > 0:
                x1, y1, x2, y2 = (int(v) for v in boxes[0])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                if x2 > x1 and y2 > y1:
                    bbox = (x1, y1, x2, y2)
                    face_found = True
                    # Extract crop using facenet_pytorch helper (no second MTCNN pass)
                    if _HAS_EXTRACT_FACE:
                        try:
                            ft = extract_face(pil_full, boxes[0],
                                              image_size=IMG_SIZE, margin=14)
                            face_pil = Image.fromarray(ft.permute(1, 2, 0).byte().numpy())
                        except Exception:
                            face_pil = None
                    if face_pil is None:
                        # Manual margin crop — mirrors facenet_pytorch's formula
                        bw_box = boxes[0][2] - boxes[0][0]
                        bh_box = boxes[0][3] - boxes[0][1]
                        mx = int(14 * bw_box / (IMG_SIZE - 14) / 2)
                        my = int(14 * bh_box / (IMG_SIZE - 14) / 2)
                        face_pil = pil_full.crop((
                            max(0, x1 - mx), max(0, y1 - my),
                            min(W, x2 + mx), min(H, y2 + my)
                        )).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)

            if face_pil is None:
                # No face detected — fall back to last cached crop
                face_pil = self._last_good_crop
                if face_pil is None:
                    # Absolute last resort: centre-crop
                    crop = min(H, W)
                    t, l = (H - crop) // 2, (W - crop) // 2
                    face_pil = pil_full.crop((l, t, l + crop, t + crop))

            if face_found:
                self._last_good_crop = face_pil

            # Use the RAW normalized CAM (0-1) directly.
            cam, prob, verdict = self._compute_fused_cam(face_pil)
            self._cached_score_map    = cam
            self._cached_heatmap_bbox = bbox
            # MediaPipe face shape mask — clips heatmap to the actual face oval
            self._cached_face_mask = (
                self._get_face_shape_mask(frame_rgb, bbox) if bbox is not None else None
            )

        else:
            # ── Throttled frame: ZERO face detection calls ───────────────────────────
            # Re-use the cached face crop for a fast forward-only score update.
            # bbox stays at the last detected position; rAF repositions the
            # overlay canvas to follow as the face moves.
            face_found = self._cached_heatmap_bbox is not None
            face_pil   = self._last_good_crop

            if face_pil is not None:
                with torch.no_grad():
                    x = eval_tf(face_pil).unsqueeze(0).to(self.device)
                    prob    = torch.sigmoid(self.model(x)).item()
                    verdict = "DEEPFAKE" if prob > THRESHOLD else "REAL"
            else:
                prob, verdict = 0.5, "REAL"

            cam  = self._cached_score_map
            bbox = self._cached_heatmap_bbox

        # ── Build face-crop RGBA on Grad-CAM frames; None on throttled frames ───
        heatmap_rgba = (
            self._make_heatmap_rgba_crop(cam, bbox, verdict, face_mask=self._cached_face_mask)
            if (run_cam and bbox is not None and cam is not None) else None
        )

        return {
            "prob": prob,
            "verdict": verdict,
            "face_found": face_found,
            "heatmap_rgba": heatmap_rgba,
            "bbox": bbox,
        }

    # ═══════════════════════════════════════════════════════════════════════
    # Stylized full-frame heatmap (new) — Jet colormap, feathered spread over
    # shoulders/background, 3-tier legend + disclaimer badge. Uses Eigen-CAM
    # (no .backward() call) for the genuine in-face detail.
    # ═══════════════════════════════════════════════════════════════════════

    def _get_face_and_box(self, frame_rgb):
        """Returns (face_pil, box_xyxy, face_found). box_xyxy is in the ORIGINAL
        frame's coordinates — needed to know where to center the feather effect.
        Falls back to a manual crop if facenet_pytorch's extract_face isn't
        importable in your installed version."""
        pil = Image.fromarray(frame_rgb)
        boxes, probs = self.mtcnn.detect(pil)

        if boxes is None or len(boxes) == 0:
            return None, None, False

        box = boxes[0]

        if _HAS_EXTRACT_FACE:
            face_tensor = extract_face(pil, box, image_size=IMG_SIZE, margin=14)
            face_pil = Image.fromarray(face_tensor.permute(1, 2, 0).byte().numpy())
        else:
            # Manual fallback — mirrors facenet_pytorch's own margin formula.
            x1, y1, x2, y2 = box
            w, h = pil.size
            margin = 14
            margin_x = margin * (x2 - x1) / (IMG_SIZE - margin)
            margin_y = margin * (y2 - y1) / (IMG_SIZE - margin)
            cx1 = max(int(x1 - margin_x / 2), 0)
            cy1 = max(int(y1 - margin_y / 2), 0)
            cx2 = min(int(x2 + margin_x / 2), w)
            cy2 = min(int(y2 + margin_y / 2), h)
            face_pil = pil.crop((cx1, cy1, cx2, cy2)).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)

        return face_pil, box, True

    def score_frame_with_stylized_heatmap(self, frame_rgb: np.ndarray) -> dict:
        """Full-frame version matching the ScamSense mockup: Jet colormap,
        genuine Eigen-CAM detail inside the face box, a radial stylistic feather
        fading outward over shoulders/background, 3-tier legend + disclaimer
        burned into the frame. No .backward() call — Eigen-CAM only needs the
        forward-pass activations, so this is cheap enough for live video.
        """
        face_pil, box, face_found = self._get_face_and_box(frame_rgb)

        if not face_found:
            # No face this frame — just return the plain frame with legend/disclaimer
            out = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            draw_tier_legend_and_disclaimer(out)
            return {"prob": None, "verdict": "NO FACE", "face_found": False,
                    "heatmap_overlay_bgr": out}

        x = eval_tf(face_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logit = self.model(x)   # forward pass — hook captures self._acts['shallow']
        prob = torch.sigmoid(logit).item()
        verdict = "DEEPFAKE" if prob > THRESHOLD else "REAL"

        # ── Eigen-CAM: per-pixel detail WITHIN the face box (real signal) ──────
        acts = self._acts['shallow'].squeeze(0).cpu().numpy()
        C, Hc, Wc = acts.shape
        flat = acts.reshape(C, Hc * Wc)
        flat = flat - flat.mean(axis=1, keepdims=True)
        _, _, Vt = np.linalg.svd(flat, full_matrices=False)
        cam = Vt[0].reshape(Hc, Wc)
        if cam.sum() < 0:
            cam = -cam
        cam = np.maximum(cam, 0)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        # ── Overall scalar suspicion score (drives the decorative wash) ───────
        if verdict == "DEEPFAKE":
            base_score = 0.5 + 0.5 * np.clip((prob - THRESHOLD) / (1 - THRESHOLD), 0, 1)
        else:
            base_score = 0.5 * np.clip(1 - prob / THRESHOLD, 0, 1)

        H, W = frame_rgb.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, W), min(y2, H)
        face_w, face_h = max(x2 - x1, 1), max(y2 - y1, 1)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        face_radius = max(face_w, face_h) / 2

        # ── Radial feather: 1.0 at face, fading to 0.0 by FEATHER_OUTER_MULT ──
        yy, xx = np.mgrid[0:H, 0:W]
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        inner_r = face_radius * FEATHER_INNER_MULT
        outer_r = face_radius * FEATHER_OUTER_MULT
        feather = np.clip(1 - (dist - inner_r) / max(outer_r - inner_r, 1e-6), 0, 1)

        # ── Full-frame score map: decorative wash everywhere by default ───────
        score_map = base_score * feather

        # ── Overwrite with REAL per-pixel detail inside the face box ──────────
        cam_resized = cv2.resize(cam, (face_w, face_h), interpolation=cv2.INTER_CUBIC)
        face_detail = 0.5 + 0.5 * cam_resized if verdict == "DEEPFAKE" else 0.5 - 0.5 * cam_resized
        score_map[y1:y2, x1:x2] = np.clip(face_detail, 0, 1)
        score_map = np.clip(score_map, 0, 1)

        # ── Colorize + composite ───────────────────────────────────────────────
        heat_rgb = JET_CMAP(score_map)[..., :3]
        frame_np = frame_rgb.astype(np.float32) / 255.0
        alpha = STYLIZED_ALPHA_MIN + (STYLIZED_ALPHA_MAX - STYLIZED_ALPHA_MIN) * score_map
        overlay = np.clip(frame_np * (1 - alpha[..., None]) + heat_rgb * alpha[..., None], 0, 1)
        overlay_bgr = cv2.cvtColor((overlay * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        draw_tier_legend_and_disclaimer(overlay_bgr)

        return {
            "prob": prob,
            "verdict": verdict,
            "face_found": True,
            "heatmap_overlay_bgr": overlay_bgr,   # BGR uint8 — ready for cv2.imencode directly
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
# Full-frame Grad-CAM heatmap for a single frame (canvas-composited on the frontend):
#   result = detector.score_frame_with_heatmap(some_rgb_frame)
#   result["heatmap_rgba"]  # RGBA face-crop, or None on throttled frames — colour is
#                            # verdict-remapped (green=real, red=fake), so a REAL
#                            # verdict can never show red/blue patches.
#   result["bbox"]          # current (x1,y1,x2,y2) face position
#
# Full-frame stylized Jet heatmap for a single frame (matches ScamSense mockup):
#   result = detector.score_frame_with_stylized_heatmap(some_rgb_frame)
#   result["heatmap_overlay_bgr"]  # uint8 BGR array, ready to cv2.imencode/stream
#
# Batch-test a whole video file:
#   result = detector.process_video_file("suspect_video.mp4", "annotated_out.mp4")
#   print(result)   # result["output_video"] is the absolute path to the saved file