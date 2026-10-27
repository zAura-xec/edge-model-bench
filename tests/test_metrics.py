import numpy as np
import pytest

from edge_model_bench.evaluator import ClsTaskEvaluator, SemanticSegmentationEvaluator


def test_topk_for_known_scores():
    result = ClsTaskEvaluator(top_k=(1, 2))(
        [
            {"label": 0, "pred": {0: 0.8, 1: 0.2}},
            {"label": 1, "pred": {0: 0.6, 1: 0.4}},
        ]
    )
    assert result["top_1_accuracy"] == 0.5
    assert result["top_2_accuracy"] == 1
    assert result["confusion_matrix"] == [[1, 0], [1, 0]]


def test_iou_and_ignore_pixels():
    result = SemanticSegmentationEvaluator(num_classes=2)(
        [
            {"label": np.array([[0, 1], [1, 255]]), "pred": np.array([[0, 1], [0, 9]])},
        ]
    )
    assert result["mean_iou"] == pytest.approx(0.5)
    assert result["pixel_accuracy"] == pytest.approx(2 / 3)
    assert result["num_ignored_pixels"] == 1


def test_classification_empty_is_actionable():
    with pytest.raises(ValueError, match="at least one sample"):
        ClsTaskEvaluator()([])


def test_detection_ap_for_exact_box():
    from edge_model_bench.evaluator import DetectionEvaluator

    result = DetectionEvaluator(input_size=(100, 100), num_classes=1)(
        [
            {
                "label": [{"label": 0, "box": [0.5, 0.5, 0.5, 0.5]}],
                "pred": {"boxes": [[25, 25, 75, 75]], "scores": [0.9], "labels": [0]},
            },
        ]
    )
    assert result["map50"] == pytest.approx(1)
    assert result["map50_95"] == pytest.approx(1)
