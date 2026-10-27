"""Generate a tiny ONNX model and synthetic images; no network or weights needed."""

import argparse
import csv
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper
from PIL import Image


def create_demo(output: Path) -> Path:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    graph = helper.make_graph(
        [
            helper.make_node("GlobalAveragePool", ["image"], ["pooled"]),
            helper.make_node("Flatten", ["pooled"], ["logits"], axis=1),
        ],
        "color_classifier",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, 8, 8])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 3])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
    onnx.checker.check_model(model)
    onnx.save(model, output / "model.onnx")
    with (output / "validation.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image_path", "label"])
        for index in range(6):
            label = index % 3
            image = np.zeros((8, 8, 3), dtype=np.uint8)
            image[:, :, label] = 255
            path = output / f"{index}.png"
            Image.fromarray(image).save(path)
            writer.writerow([str(path), label])
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("work/demo"))
    print(create_demo(parser.parse_args().output))


if __name__ == "__main__":
    main()
