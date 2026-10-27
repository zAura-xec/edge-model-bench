# Third-party components and source review

The project is licensed under the [MIT License](LICENSE). This document inventories external integration boundaries; it does not grant rights to SDKs, models or datasets and does not replace required upstream notices.

## Python dependencies

Core dependencies are Hydra Core / OmegaConf, NumPy, OpenCV (headless distribution), Pillow, PyYAML, scikit-learn and tqdm. The optional ONNX extra adds ONNX and ONNX Runtime. Development dependencies include pytest, Ruff, build and the selected build backend. `pyproject.toml` defines requirements and `uv.lock` records resolved artifacts and hashes.

Each dependency retains its own license. Before distributing binaries or a container, review the licenses and bundled notices for the exact dependency artifacts being distributed, including transitive dependencies and native libraries.

## Vendor integrations

- QAIRT / QNN: external Qualcomm tooling and device runtime, obtained separately.
- Horizon HBR / OpenExplorer: external compiler tooling and `hbm_runtime`, obtained separately.
- MTNN / MTC: external compiler and runtime, obtained separately.

These names identify compatibility integrations, not affiliation or endorsement. No vendor SDK, binary wheel or runtime library is bundled.

## Models and data

Names of model families and datasets identify supported adapters/examples. Users must obtain model artifacts and datasets under their respective terms. The generated CPU demo uses local synthetic images and an operator-only ONNX graph; it downloads no external weights or data.

## Source provenance

The optional BPU smoke utility retains its reference to the [D-Robotics RDK model-zoo example](https://github.com/D-Robotics/rdk_model_zoo/tree/rdk_s/samples/vision/mobilenetv1). Maintainers must review whether any implementation was copied/adapted and preserve applicable notices before publication.

The pre-release refactor preserves the existing algorithm implementations. Automated scans cannot determine their authorship or licensing history. The copyright holder must review source provenance, employee/contributor ownership, and any incorporated third-party snippets before granting a project license. Keep that review in a private record; include all required public notices in the release.
