"""Classification metrics."""

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm.auto import tqdm


class ClsTaskEvaluator:
    def __init__(self, top_k: Sequence[int] = (1, 5), zero_division: int | str = 0) -> None:
        self.top_k = top_k
        self.zero_division = zero_division

    def __call__(self, samples: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        labels = []
        scores = []
        for sample in tqdm(samples, desc="Evaluating", unit="sample"):
            labels.append(sample["label"])
            scores.append(sample["pred"])
        if not labels:
            raise ValueError("Classification evaluation requires at least one sample")
        predictions = [max(score, key=score.get) for score in scores]
        classes = sorted(set(labels).union(*(score.keys() for score in scores)), key=str)

        result = {
            "num_samples": len(labels),
            "classes": classes,
            "accuracy": float(accuracy_score(labels, predictions)),
            "confusion_matrix": confusion_matrix(labels, predictions, labels=classes).tolist(),
            "classification_report": classification_report(
                labels,
                predictions,
                labels=classes,
                output_dict=True,
                zero_division=self.zero_division,
            ),
        }
        for average in ("macro", "weighted"):
            options = {
                "labels": classes,
                "average": average,
                "zero_division": self.zero_division,
            }
            result[f"precision_{average}"] = float(precision_score(labels, predictions, **options))
            result[f"recall_{average}"] = float(recall_score(labels, predictions, **options))
            result[f"f1_{average}"] = float(f1_score(labels, predictions, **options))

        for k in self.top_k:
            correct = sum(
                label in sorted(score, key=score.get, reverse=True)[:k]
                for label, score in zip(labels, scores)
            )
            result[f"top_{k}_accuracy"] = correct / len(labels)
        return result
