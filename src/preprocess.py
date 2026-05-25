from __future__ import annotations

import argparse
import html
import json
import random
import re
from pathlib import Path
from typing import Any

import pandas as pd

from .config import DEFAULT_DATASET_DIR, DEFAULT_PROCESSED_DIR, DEFAULT_SEED, DEFAULT_TOP_K

TEXT_COLUMNS = [
    "productDisplayName",
    "json_description",
    "brandName",
    "gender",
    "masterCategory",
    "subCategory",
    "baseColour",
    "season",
    "usage",
    "articleAttributes",
]


def clean_html_text(value: Any) -> str:
    """Convert HTML product descriptions into compact plain text."""
    if not isinstance(value, str):
        return ""

    # The dataset contains tens of thousands of short HTML snippets; regex cleanup is
    # much faster than instantiating an HTML parser for every product.
    text = re.sub(r"<br\s*/?>", " ", value, flags=re.IGNORECASE)
    text = re.sub(r"</p>|</div>|</li>|</tr>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _safe_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    value = str(value).strip()
    if value.lower() in {"nan", "none", "na", "n/a"}:
        return ""
    return value


def load_json_details(style_json_dir: Path, product_id: Any) -> dict[str, str]:
    """Read optional rich metadata from fashion-dataset/styles/{id}.json."""
    path = style_json_dir / f"{product_id}.json"
    if not path.exists():
        return {"json_description": "", "brandName": "", "articleAttributes": ""}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"json_description": "", "brandName": "", "articleAttributes": ""}

    data = payload.get("data") or {}
    descriptors = data.get("productDescriptors") or {}
    description = ""
    if isinstance(descriptors, dict):
        description = clean_html_text((descriptors.get("description") or {}).get("value", ""))

    attrs = data.get("articleAttributes") or {}
    attr_text = ""
    if isinstance(attrs, dict):
        attr_bits = []
        for key, value in attrs.items():
            key_text = _safe_text(key)
            value_text = _safe_text(value)
            if key_text and value_text:
                attr_bits.append(f"{key_text}: {value_text}")
        attr_text = ". ".join(attr_bits)

    return {
        "json_description": description,
        "brandName": _safe_text(data.get("brandName")),
        "articleAttributes": attr_text,
    }


def build_product_text(row: pd.Series, max_text_chars: int) -> str:
    """Build text input without including the target articleType directly."""
    pieces = []

    name = _safe_text(row.get("productDisplayName"))
    if name:
        pieces.append(name)

    description = _safe_text(row.get("json_description"))
    if description and description.lower() != name.lower():
        pieces.append(description)

    metadata = []
    metadata_specs = [
        ("Brand", row.get("brandName")),
        ("Gender", row.get("gender")),
        ("Category", row.get("masterCategory")),
        ("Subcategory", row.get("subCategory")),
        ("Color", row.get("baseColour")),
        ("Season", row.get("season")),
        ("Usage", row.get("usage")),
    ]
    for label, value in metadata_specs:
        value_text = _safe_text(value)
        if value_text:
            metadata.append(f"{label}: {value_text}")

    attrs = _safe_text(row.get("articleAttributes"))
    if attrs:
        metadata.append(attrs)

    if metadata:
        pieces.append(". ".join(metadata))

    text = " ".join(pieces)
    text = re.sub(r"\s+", " ", text).strip()
    if max_text_chars > 0:
        text = text[:max_text_chars].strip()
    return text


def stratified_split(
    df: pd.DataFrame,
    label_col: str = "label_id",
    val_size: float = 0.1,
    test_size: float = 0.1,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Assign train/val/test splits per class without requiring scikit-learn."""
    if not 0 < val_size < 1 or not 0 < test_size < 1 or val_size + test_size >= 1:
        raise ValueError("val_size and test_size must be positive and sum to less than 1.")

    rng = random.Random(seed)
    split_by_index: dict[int, str] = {}

    for _, group in df.groupby(label_col, sort=False):
        indices = list(group.index)
        rng.shuffle(indices)
        n = len(indices)

        test_n = max(1, round(n * test_size)) if n >= 3 else 0
        val_n = max(1, round(n * val_size)) if n - test_n >= 2 else 0

        test_indices = indices[:test_n]
        val_indices = indices[test_n : test_n + val_n]
        train_indices = indices[test_n + val_n :]

        for idx in train_indices:
            split_by_index[idx] = "train"
        for idx in val_indices:
            split_by_index[idx] = "val"
        for idx in test_indices:
            split_by_index[idx] = "test"

    result = df.copy()
    result["split"] = result.index.map(split_by_index)
    return result


def _maybe_sample_per_class(df: pd.DataFrame, sample_per_class: int | None, seed: int) -> pd.DataFrame:
    if not sample_per_class:
        return df
    sampled_groups = []
    for _, group in df.groupby("articleType", sort=False):
        sampled_groups.append(group.sample(min(len(group), sample_per_class), random_state=seed))
    return pd.concat(sampled_groups, ignore_index=True)


def create_processed_dataset(
    dataset_dir: Path,
    output_dir: Path,
    top_k: int = DEFAULT_TOP_K,
    val_size: float = 0.1,
    test_size: float = 0.1,
    seed: int = DEFAULT_SEED,
    max_text_chars: int = 1200,
    sample_per_class: int | None = None,
) -> dict[str, Path]:
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    styles_csv = dataset_dir / "styles.csv"
    image_dir = dataset_dir / "images"
    style_json_dir = dataset_dir / "styles"

    if not styles_csv.exists():
        raise FileNotFoundError(f"Missing metadata file: {styles_csv}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(styles_csv, on_bad_lines="skip", low_memory=False)
    df["id"] = df["id"].astype(str)
    image_ids = {path.stem for path in image_dir.glob("*.jpg")}
    df = df[df["id"].isin(image_ids)].copy()
    df["image_path"] = df["id"].map(lambda product_id: str((image_dir / f"{product_id}.jpg").resolve()))
    df = df.dropna(subset=["articleType"]).copy()

    top_labels = df["articleType"].value_counts().head(top_k).index.tolist()
    df = df[df["articleType"].isin(top_labels)].copy()
    df = _maybe_sample_per_class(df, sample_per_class, seed)

    label_to_id = {label: idx for idx, label in enumerate(top_labels)}
    id_to_label = {str(idx): label for label, idx in label_to_id.items()}
    df["label"] = df["articleType"]
    df["label_id"] = df["label"].map(label_to_id).astype(int)

    json_rows = [load_json_details(style_json_dir, product_id) for product_id in df["id"].tolist()]
    json_df = pd.DataFrame(json_rows, index=df.index)
    for col in ["json_description", "brandName", "articleAttributes"]:
        df[col] = json_df[col].fillna("")

    df["text"] = df.apply(lambda row: build_product_text(row, max_text_chars=max_text_chars), axis=1)
    df = stratified_split(df, val_size=val_size, test_size=test_size, seed=seed)

    output_columns = ["id", "image_path", "text", "label", "label_id", "split"] + [
        col for col in TEXT_COLUMNS if col in df.columns and col not in {"json_description"}
    ]
    metadata_path = output_dir / "metadata.csv"
    df[output_columns].to_csv(metadata_path, index=False)

    label_map_path = output_dir / "label_map.json"
    label_map = {"label_to_id": label_to_id, "id_to_label": id_to_label}
    label_map_path.write_text(json.dumps(label_map, indent=2), encoding="utf-8")

    summary = {
        "dataset_dir": str(dataset_dir.resolve()),
        "rows": int(len(df)),
        "top_k": top_k,
        "labels": top_labels,
        "split_counts": df["split"].value_counts().to_dict(),
        "class_counts": df["label"].value_counts().to_dict(),
        "missing_text_rows": int((df["text"].str.len() == 0).sum()),
    }
    summary_path = output_dir / "split_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "metadata": metadata_path,
        "label_map": label_map_path,
        "summary": summary_path,
    }


def validate_processed_dataset(metadata_path: Path, expected_classes: int = DEFAULT_TOP_K) -> dict[str, Any]:
    df = pd.read_csv(metadata_path)
    missing_images = [path for path in df["image_path"].tolist() if not Path(path).exists()]
    labels = sorted(df["label_id"].unique().tolist())
    split_counts = df["split"].value_counts().to_dict()
    class_split_counts = df.groupby(["label", "split"]).size().unstack(fill_value=0).to_dict()

    if missing_images:
        raise AssertionError(f"{len(missing_images)} processed rows point to missing images.")
    if len(labels) != expected_classes:
        raise AssertionError(f"Expected {expected_classes} classes, found {len(labels)}.")
    if set(split_counts) != {"train", "val", "test"}:
        raise AssertionError(f"Expected train/val/test splits, found {sorted(split_counts)}.")

    return {
        "rows": int(len(df)),
        "classes": int(len(labels)),
        "split_counts": split_counts,
        "class_split_counts": class_split_counts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the fashion product multimodal dataset.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-text-chars", type=int, default=1200)
    parser.add_argument("--sample-per-class", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = create_processed_dataset(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        top_k=args.top_k,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
        max_text_chars=args.max_text_chars,
        sample_per_class=args.sample_per_class,
    )
    validation = validate_processed_dataset(paths["metadata"], expected_classes=args.top_k)
    print(json.dumps({"paths": {key: str(value) for key, value in paths.items()}, "validation": validation}, indent=2))


if __name__ == "__main__":
    main()
