"""P7 (Edge AI) - the lightweight feature extractor that replaces the frozen I3D/VGGish backbone.

The benchmark track reads **precomputed** features (I3D 1024-d visual, VGGish 128-d audio), which is
why it can train 30 epochs on a CPU. Deploying that on a camera is impossible: the machine that
produced those features is a GPU cluster. So the edge track needs an extractor it can actually run.

Two facts shape this module, both measured on this machine (``docs/journal.md``, Session 15):

* **Decoding dominates, not the model.** Sequential H.264 decode runs at ~1800-2900 fps here (a
  4-min clip in ~3.2 s) while the trained head costs ~5 ms / 100 snippets. A CNN priced in
  milliseconds per frame is therefore *not* the bottleneck unless it is careless - the video IO is.
* **Seeking is 10x the cost of decoding.** Random access via ``CAP_PROP_POS_FRAMES`` costs 25-50 ms
  per frame (the decoder restarts from the previous keyframe). An edge sampler must walk the file
  once and keep every N-th frame, which is what :func:`frame_grid` supports.

The extractor maps sampled RGB frames to a ``(T, 1024)`` tensor - the *same shape* the frozen I3D
backbone produces - so the trained fusion head can consume it unchanged. That shape compatibility is
the whole point: it is what lets the edge track reuse the benchmark head instead of inventing a
second model. Whether the untrained features *mean* anything is a separate question, answered by
distillation (``scripts/edge_distill.py``) rather than assumed here.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

# The frozen backbone's output width (I3D RGB, official grid). The head was trained on this, so the
# extractor must match it or the head cannot be reused.
FEATURE_DIM = 1024
# 16 frames per snippet on the official grid (scripts/check_feature_grid.py) - the extractor must
# respect the same stride, otherwise its features are seconds away from the video they describe.
SNIPPET_FRAMES = 16
# Small enough to be honest about "lightweight": 112x112 is the usual budget input for mobile nets.
DEFAULT_INPUT = 112


def frame_grid(frame_count: int, n_snippets: int, snippet_frames: int = SNIPPET_FRAMES,
               offset: int = 0) -> np.ndarray:
    """Frame indices to feed the extractor: one frame per snippet, on the 16-frame grid.

    One frame per snippet (not 16) is deliberate: 16x fewer CNN calls for a 2.64 s window is a
    defensible edge trade, and the benchmark's own snippet features already summarise 16 frames
    (I3D's receptive field). ``offset`` selects the frame inside each snippet window, so a caller
    can spread samples instead of always taking the first.
    """
    if n_snippets <= 0:
        raise ValueError("n_snippets must be positive")
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    # The TRUE stride: a 5760-frame clip at 360 snippets is 16 frames/snippet, so snippet i starts
    # at i*16 and the grid reaches 5744. Deriving the step from `(frames - span)` gave 15, which
    # stopped at frame 5385 - the sampler then described only the first 93% of every clip.
    step = max(1, frame_count // n_snippets)
    span = min(snippet_frames, frame_count)
    idx = np.arange(n_snippets) * step + min(offset, span - 1)
    return np.clip(idx, 0, frame_count - 1).astype(np.int64)


class MiniRGB(nn.Module):
    """A small CNN: ``(B, T, 3, H, W)`` sampled frames -> ``(B, T, 1024)`` snippet features.

    Architecture is a plain strided-conv trunk + global average pool + linear projection. No
    ImageNet weights (torchvision is deliberately not a dependency of this CPU-only repo), so the
    network is trained from scratch **by distillation** from the frozen I3D features - see
    ``scripts/edge_distill.py``. Building it here only fixes the *shape* and the *budget*; the
    fidelity is measured, not claimed.
    """

    def __init__(self, feature_dim: int = FEATURE_DIM, width: int = 24,
                 input_size: int = DEFAULT_INPUT, dropout: float = 0.1) -> None:
        super().__init__()
        if width < 8 or width % 8:
            raise ValueError("width must be a multiple of 8 (four stride-2 stages)")
        self.feature_dim = int(feature_dim)
        self.input_size = int(input_size)
        w = int(width)
        self.trunk = nn.Sequential(
            nn.Conv2d(3, w, 5, stride=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(w, 2 * w, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(2 * w, 4 * w, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(4 * w, 8 * w, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout),
                                  nn.Linear(8 * w, self.feature_dim))

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """``(B, T, 3, H, W)`` -> ``(B, T, feature_dim)``; a bare ``(B, 3, H, W)`` also works."""
        if frames.ndim == 4:
            return self.head(self.trunk(frames))
        batch, steps = frames.shape[:2]
        flat = frames.reshape(batch * steps, *frames.shape[2:])
        return self.head(self.trunk(flat)).reshape(batch, steps, self.feature_dim)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
