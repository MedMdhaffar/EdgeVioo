from pathlib import Path


LIST_DIR = Path(__file__).resolve().parent

# Feature directories used by the MIX2 model (RGB I3D + VGGish audio).
FEATURE_DIRS = {
    "rgb.list": LIST_DIR / "../../i3d-features/i3d-features/RGB",
    "flow.list": LIST_DIR / "../../i3d-features/i3d-features/Flow",
    "audio.list": LIST_DIR / "../../vggish-features/vggish-features/train",
    "rgb_test.list": LIST_DIR / "../../i3d-features/i3d-features/RGBTest",
    "flow_test.list": LIST_DIR / "../../i3d-features/i3d-features/FlowTest",
    "audio_test.list": LIST_DIR / "../../vggish-features/vggish-features/test",
}


def ordered_features(directory):
    """Return abnormal features first and normal (_label_A) features last."""
    directory = directory.resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Feature directory does not exist: {directory}")

    files = sorted(directory.glob("*.npy"))
    if not files:
        raise RuntimeError(f"No .npy feature files found in: {directory}")

    abnormal = [path for path in files if "_label_A" not in path.name]
    normal = [path for path in files if "_label_A" in path.name]
    return abnormal + normal


def video_key(path, modality):
    stem = path.stem
    if modality == "audio":
        suffix = "__vggish"
        if not stem.endswith(suffix):
            raise RuntimeError(f"Unexpected VGGish filename: {path.name}")
        return stem[: -len(suffix)]

    base, separator, crop = stem.rpartition("__")
    if not separator or crop not in {"0", "1", "2", "3", "4"}:
        raise RuntimeError(f"Unexpected RGB crop filename: {path.name}")
    return base


def validate_mix2_pairing(rgb_files, audio_files, split):
    if len(rgb_files) != 5 * len(audio_files):
        raise RuntimeError(
            f"{split}: expected five RGB crops per audio file, but found "
            f"{len(rgb_files)} RGB files and {len(audio_files)} audio files"
        )

    for index, audio_path in enumerate(audio_files):
        rgb_group = rgb_files[index * 5 : (index + 1) * 5]
        audio_key = video_key(audio_path, "audio")
        rgb_keys = [video_key(path, "rgb") for path in rgb_group]
        crop_order = [path.stem.rpartition("__")[2] for path in rgb_group]

        if rgb_keys != [audio_key] * 5 or crop_order != ["0", "1", "2", "3", "4"]:
            raise RuntimeError(
                f"{split}: RGB/audio ordering mismatch near {audio_path.name}"
            )


def validate_rgb_flow_pairing(rgb_files, flow_files, split):
    if len(rgb_files) != len(flow_files):
        raise RuntimeError(
            f"{split}: found {len(rgb_files)} RGB files and "
            f"{len(flow_files)} flow files"
        )

    for rgb_path, flow_path in zip(rgb_files, flow_files):
        if rgb_path.name != flow_path.name:
            raise RuntimeError(
                f"{split}: RGB/flow ordering mismatch: "
                f"{rgb_path.name} != {flow_path.name}"
            )


def write_list(name, files):
    output = LIST_DIR / name
    with output.open("w", encoding="utf-8") as file:
        for path in files:
            file.write(f"{path.resolve()}\n")
    print(f"Wrote {len(files):5d} entries to {output}")


def main():
    features = {
        name: ordered_features(directory)
        for name, directory in FEATURE_DIRS.items()
    }

    validate_mix2_pairing(
        features["rgb.list"], features["audio.list"], "training"
    )
    validate_mix2_pairing(
        features["rgb_test.list"], features["audio_test.list"], "test"
    )
    validate_rgb_flow_pairing(
        features["rgb.list"], features["flow.list"], "training"
    )
    validate_rgb_flow_pairing(
        features["rgb_test.list"], features["flow_test.list"], "test"
    )

    for name, files in features.items():
        write_list(name, files)


if __name__ == "__main__":
    main()
