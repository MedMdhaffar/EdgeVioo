"""Media artefacts for an alert card: the important frames and the sound's spectrogram.

The subject asks for "visualisation des images les plus importantes dans la decision" and
"generation de spectrogrammes". Both are produced here from the *raw* clip, because the benchmark
track works on frozen features and cannot show anything.

Everything is best-effort by design: a missing video, an unreadable codec or an absent matplotlib
must degrade the alert card (``None`` instead of an artefact) rather than break it - an operator
console that crashes because one artefact could not be rendered is worse than one missing that
artefact.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

TARGET_SR = 16000
VIDEO_ROOT = "data/raw/xd-violence/data/video"


def find_video(clip_id: str, video_root: str | Path = VIDEO_ROOT) -> Path | None:
    """Locate a clip's mp4 under the usual mirror layout (test folder or train chunks)."""
    root = Path(video_root)
    candidates = [root / "test_videos" / f"{clip_id}.mp4", root / f"{clip_id}.mp4"]
    candidates += sorted(root.glob(f"*/{clip_id}.mp4"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def frame_times(snippets, stride_frames: float, fps: float = 24.0) -> list[float]:
    """Snippet indices -> seconds at the *centre* of each snippet window (what the model scored)."""
    stride_s = stride_frames / fps
    return [float((int(i) + 0.5) * stride_s) for i in snippets]


def key_frames(video: Path, seconds: list[float], out_dir: str | Path, prefix: str = "frame",
               max_width: int = 640) -> list[dict]:
    """Write one JPEG per timestamp - the frames the model actually looked at.

    OpenCV rather than ffmpeg-per-frame: one decoder pass, and it degrades to an empty list if the
    codec is unsupported. Returns ``[{"second": .., "path": ..}]`` for the frames it could read.
    """
    import cv2

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        return []
    fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
    written: list[dict] = []
    try:
        for index, second in enumerate(seconds):
            capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(second * fps)))
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            if max_width and frame.shape[1] > max_width:
                scale = max_width / frame.shape[1]
                frame = cv2.resize(frame, (max_width, int(frame.shape[0] * scale)))
            path = out / f"{prefix}_{index:02d}_{second:07.2f}s.jpg"
            if cv2.imwrite(str(path), frame):
                written.append({"second": round(second, 2), "path": str(path)})
    finally:
        capture.release()
    return written


#: mean abs frame-difference energy below which a "motion zone" would be noise, not motion. The
#: scale is the pooled L1 difference of 8-bit grey levels over a ~20x26 px block; static scenes sit
#: at 0.2-0.8, a person walking at ~3-8, a collision/impact at 20+. The floor is deliberately
#: conservative: a missed zone is honest, a box on a still background is not.
MOTION_FLOOR = 3.0


def motion_saliency(video: Path, seconds: list[float], out_dir: str | Path,
                    prefix: str = "frame", window_s: float = 0.5,
                    grid: tuple[int, int] = (12, 16), top_blocks: int = 3,
                    max_width: int = 640) -> list[dict]:
    """Per-timestamp *spatial* motion saliency: which zones of the frame moved the most.

    Fills the subject's "les images **ou zones** significatives" - the benchmark track scores
    snippets, so this is where a card learns *where in the frame* the action happened. Around each
    timestamp (default 0.5 s, ~12 frames at 24 fps) frames are decoded with the same OpenCV capture
    as ``key_frames``, grayscaled, and the absolute difference between consecutive frames is pooled
    onto a coarse block grid (default 12 rows x 16 cols). The highest-energy blocks are dilated by
    one block and grouped into bounding boxes (connected components), returned **normalised to
    [0, 1]** so they can be drawn on any render size.

    This is motion saliency, not object detection: global camera shake or a dark noisy sensor can
    still produce a box, so callers must label it as a *motion* zone and never as an identified
    object. Best-effort like the rest of the module: unreadable video or a too-short window yields
    empty boxes, never an exception.
    """
    import cv2

    rows, cols = grid
    out: list[dict] = []
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        return [{"second": round(s, 2), "boxes": []} for s in seconds]
    fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
    n_frames = max(4, int(round(window_s * fps)))
    try:
        for second in seconds:
            start = max(0, int(round((second - window_s / 2.0) * fps)))
            capture.set(cv2.CAP_PROP_POS_FRAMES, start)
            frames: list[np.ndarray] = []
            for _ in range(n_frames):
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if max_width and gray.shape[1] > max_width:
                    scale = max_width / gray.shape[1]
                    gray = cv2.resize(gray, (max_width, int(gray.shape[0] * scale)))
                frames.append(gray.astype(np.float32))
            entry: dict = {"second": round(second, 2), "boxes": [],
                          "n_frames": len(frames)}
            if len(frames) >= 4:
                energy = np.zeros((rows, cols), dtype=np.float32)
                for previous, current in zip(frames[:-1], frames[1:], strict=True):
                    diff = np.abs(current - previous)
                    block_h, block_w = diff.shape[0] // rows, diff.shape[1] // cols
                    if block_h == 0 or block_w == 0:
                        continue
                    crop = diff[:rows * block_h, :cols * block_w]
                    energy += crop.reshape(rows, block_h, cols, block_w).mean(axis=(1, 3))
                energy /= max(1, len(frames) - 1)
                if energy.max() >= MOTION_FLOOR:
                    # top blocks, dilated by one block so a 1-block spike cannot produce a hairline
                    top = np.argpartition(energy.ravel(), -top_blocks)[-top_blocks:]
                    mask = np.zeros(energy.shape, dtype=bool)
                    for flat_index in top:
                        r, c = divmod(int(flat_index), cols)
                        mask[max(0, r - 1):r + 2, max(0, c - 1):c + 2] = True
                    # cv2 returns (number_of_components, labels); label 0 is the background, so the
                    # object components are 1 .. n_components-1
                    n_components, labels = cv2.connectedComponents(
                        mask.astype(np.uint8) * 255)
                    h, w = energy.shape
                    boxes = []
                    for component in range(1, int(n_components)):
                        ys, xs = np.where(labels == component)
                        score = float(energy[ys, xs].mean())
                        if score < MOTION_FLOOR:
                            continue
                        boxes.append({
                            "x": round(float(xs.min()) / w, 4),
                            "y": round(float(ys.min()) / h, 4),
                            "w": round((float(xs.max()) - float(xs.min()) + 1.0) / w, 4),
                            "h": round((float(ys.max()) - float(ys.min()) + 1.0) / h, 4),
                            "motion": round(score, 3),
                        })
                    boxes.sort(key=lambda box: -box["motion"])
                    entry["boxes"] = boxes[:3]
            out.append(entry)
    finally:
        capture.release()
    return out


def decode_audio(video: Path, sample_rate: int = TARGET_SR) -> np.ndarray | None:
    """Mono float32 waveform via ffmpeg (the same decoder path the feature extractor uses)."""
    cmd = ["ffmpeg", "-v", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", str(sample_rate),
           "-f", "f32le", "-"]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return np.frombuffer(proc.stdout, dtype=np.float32)


def _pyplot():
    """Headless matplotlib, imported lazily so importing this module needs no display."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt



def spectrogram(video: Path, out_path: str | Path, start_s: float = 0.0,
                end_s: float | None = None, title: str | None = None) -> Path | None:
    """Log-mel spectrogram PNG of ``[start_s, end_s]`` - the alert's audio evidence, rendered.

    Same mel parameters as the feature extractor (n_fft 400, hop 10 ms, 64 mels, 50-8 000 Hz) so the
    picture shows what the model's audio channels were built from, not a different analysis.
    """
    import librosa

    plt = _pyplot()
    waveform = decode_audio(video)
    if waveform is None or waveform.size == 0:
        return None
    first = int(max(0.0, start_s) * TARGET_SR)
    stop = int(end_s * TARGET_SR) if end_s else waveform.size
    segment = waveform[first:min(stop, waveform.size)]
    if segment.size < 400:                       # shorter than one analysis window
        return None
    mel = librosa.feature.melspectrogram(y=segment, sr=TARGET_SR, n_fft=400, hop_length=160,
                                         n_mels=64, fmin=50, fmax=8000, power=2.0)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    duration = segment.size / TARGET_SR
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 2.6), dpi=110)
    image = ax.imshow(mel_db, aspect="auto", origin="lower", cmap="magma",
                      extent=[start_s, start_s + duration, 0, 64])
    ax.set_xlabel("time (s)")
    ax.set_ylabel("mel band")
    ax.set_title(title or f"{video.name} [{start_s:.1f}-{start_s + duration:.1f} s]", fontsize=9)
    fig.colorbar(image, ax=ax, format="%+.0f dB", pad=0.01)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out
