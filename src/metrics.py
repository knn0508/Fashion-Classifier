from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable


def confusion_matrix(y_true: Iterable[int], y_pred: Iterable[int], num_classes: int) -> list[list[int]]:
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for true, pred in zip(y_true, y_pred):
        matrix[int(true)][int(pred)] += 1
    return matrix


def macro_f1_score(y_true: Iterable[int], y_pred: Iterable[int], num_classes: int) -> tuple[float, list[dict[str, float]]]:
    matrix = confusion_matrix(y_true, y_pred, num_classes)
    rows = []
    f1_values = []

    for cls in range(num_classes):
        tp = matrix[cls][cls]
        fp = sum(matrix[row][cls] for row in range(num_classes) if row != cls)
        fn = sum(matrix[cls][col] for col in range(num_classes) if col != cls)
        support = sum(matrix[cls])

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        accuracy = tp / support if support else 0.0
        f1_values.append(f1)
        rows.append(
            {
                "class_id": cls,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "class_accuracy": accuracy,
                "support": support,
            }
        )

    macro_f1 = sum(f1_values) / num_classes if num_classes else 0.0
    return macro_f1, rows


def classification_metrics(
    y_true: list[int],
    y_pred: list[int],
    id_to_label: dict[str, str],
) -> dict[str, object]:
    num_classes = len(id_to_label)
    matrix = confusion_matrix(y_true, y_pred, num_classes)
    macro_f1, class_rows = macro_f1_score(y_true, y_pred, num_classes)
    total = len(y_true)
    correct = sum(int(true == pred) for true, pred in zip(y_true, y_pred))

    for row in class_rows:
        row["label"] = id_to_label[str(row["class_id"])]

    return {
        "macro_f1": macro_f1,
        "accuracy": correct / total if total else 0.0,
        "class_wise": class_rows,
        "confusion_matrix": matrix,
        "support": total,
    }


def write_metrics(metrics: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    class_rows = metrics.get("class_wise", [])
    with (output_dir / "class_wise_accuracy.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["class_id", "label", "precision", "recall", "f1", "class_accuracy", "support"],
        )
        writer.writeheader()
        writer.writerows(class_rows)

    matrix = metrics.get("confusion_matrix", [])
    with (output_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows(matrix)
