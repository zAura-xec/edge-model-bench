from functools import partial

import pytest

from edge_model_bench.pipeline import ModelEvaluator, ModelTask


class Dataset:
    def __init__(self, size=2):
        self.samples = [{"label": index} for index in range(size)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return index, self.samples[index]


class Runner:
    def __init__(self, model, fail=False, timings=True):
        self.latencies = []
        self.closed = False
        self.calls = 0
        self.fail = fail
        self.timings = timings

    def __call__(self, inputs):
        self.calls += 1
        if self.fail:
            raise RuntimeError("inference failed")
        if self.timings:
            self.latencies.append(0.01)
        return inputs

    def close(self):
        self.closed = True
        self.latencies.clear()


def postprocess(sample, output):
    sample["pred"] = output
    return sample


def evaluate(samples):
    return {
        "num_samples": len(samples),
        "accuracy": sum(s["pred"] == s["label"] for s in samples) / len(samples),
    }


def test_shared_evaluation_warmup_and_reuse():
    dataset = Dataset()
    runner = Runner(None)
    task = ModelTask(dataset, lambda model: runner, postprocess, evaluate, model="model", warmup=3)
    result = task()
    assert result["accuracy"] == 1
    assert result["average_latency_ms"] == pytest.approx(10)
    assert result["fps"] == pytest.approx(100)
    assert result["num_inferences"] == 2
    assert runner.calls == 5
    assert runner.closed
    assert all("pred" not in sample for sample in dataset.samples)


def test_failure_closes_runner():
    runner = Runner(None, fail=True)
    with pytest.raises(RuntimeError, match="inference failed"):
        ModelEvaluator(None, Dataset(), lambda model: runner, postprocess, evaluate)()
    assert runner.closed


def test_empty_dataset_fails_before_loading_model():
    with pytest.raises(ValueError, match="empty"):
        ModelEvaluator(None, Dataset(0), None, postprocess, evaluate)()


def test_runner_without_timings():
    result = ModelEvaluator(
        None, Dataset(), partial(Runner, timings=False), postprocess, evaluate
    )()
    assert "fps" not in result


def test_conversion_and_calibration_share_pipeline():
    calls = []
    dataset = Dataset()

    class Calibration:
        def __len__(self):
            return 2

        def __getitem__(self, index):
            calls.append(index)

    def convert():
        calls.append("convert")
        return "artifact"

    def runner(model):
        calls.append(model)
        return Runner(model)

    ModelTask(
        dataset, runner, postprocess, evaluate, converter=convert, calibration_dataset=Calibration()
    )()
    assert calls == [0, 1, "convert", "artifact"]
