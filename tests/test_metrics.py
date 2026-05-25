from src.metrics import classification_metrics


def test_classification_metrics_macro_f1_and_class_accuracy():
    metrics = classification_metrics(
        y_true=[0, 0, 1, 1],
        y_pred=[0, 1, 1, 1],
        id_to_label={"0": "A", "1": "B"},
    )

    assert metrics["accuracy"] == 0.75
    assert round(metrics["macro_f1"], 4) == 0.7333
    class_accuracy = {row["label"]: row["class_accuracy"] for row in metrics["class_wise"]}
    assert class_accuracy == {"A": 0.5, "B": 1.0}
