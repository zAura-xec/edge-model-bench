# Edge Model Bench

**面向边缘 AI 模型的精度评估与推理性能测试工具。**

[English](README.md) · [后端安装](docs/backends.md) · [扩展指南](docs/new_chip_benchmark_pipeline.md) · [贡献指南](CONTRIBUTING.md)

通过 YAML 组合数据集、预处理、推理后端、后处理和指标计算，在同一评测协议下比较 ONNX 浮点基线与设备产物。组件也可以直接在 Python 中复用。

```text
数据集 → 预处理 → Runner 推理 → 后处理 → Evaluator 指标 → result.json
                       ↑
                  可选模型转换
```

## 支持范围

| 能力 | 内容 |
| --- | --- |
| 任务 | 分类、目标检测、实例分割、语义分割、点云分类、BEV |
| 数据适配 | ImageNet 风格 CSV、COCO、VOC、ADE20K、ModelNet10、nuScenes |
| 推理后端 | ONNX Runtime；可选 QAIRT/QNN、Horizon HBR、MTNN 适配 |
| 模型转换 | 厂商转换适配；ONNX Runtime 静态量化和分块校准 |
| 精度指标 | Top-k、Precision/Recall/F1、AP、IoU 等，取决于任务 |
| 性能指标 | 平均延迟、p50/p90 延迟、平均推理延迟倒数 FPS |
| 产物 | JSON、顺序批量执行日志、标量指标 CSV |

普通 CI 验证通用组件、所有配置的组合以及 CPU 示例。厂商后端仍需要对应 SDK、运行时与设备，本次改造不代表已经完成各设备实测。示例模型配置对导出格式和预处理有要求，不能仅凭同名模型就认定兼容。

仓库不包含模型权重、数据集、私有 SDK 或预置跑分。

## 五分钟运行 CPU 示例

通用环境支持 Python **3.10–3.13**；厂商工具链可能要求特定 Python 版本，参考[后端安装说明](docs/backends.md)。

```bash
git clone https://github.com/zAura-xec/edge-model-bench.git
cd edge-model-bench
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[onnx]'
python examples/create_demo.py
edge-model-bench demo
```

使用 uv 和锁文件：

```bash
uv sync --extra onnx
uv run python examples/create_demo.py
uv run edge-model-bench demo
```

脚本在 `work/demo/` 本地生成六张纯色图与一个微型 ONNX 分类模型，不需要下载模型和数据。流水线先预热两次，再评估六个样本，结果写入 `work/demo/evaluation/result.json`。

预期 `top_1_accuracy=1.0`、`num_inferences=6`。这是验证安装、推理、指标和结果输出是否连通的合成示例，不是实际模型精度或设备跑分。

只使用通用组件时可以 `pip install -e .`；需要 ONNX 推理、转换辅助或量化时安装 `[onnx]`。厂商 SDK 不进入基础依赖，单独按官方说明安装。

## 接入自己的模型与数据

默认资源目录是 `./benchmark`，输出目录是 `./work`，相对路径均以**执行命令时的工作目录**为基准。

```bash
export BENCHMARK_ROOT=/path/to/benchmark
export BENCHMARK_WORK_ROOT=/path/to/results
edge-model-bench pipeline/task/cls/onnx/resnet18 --print-config
edge-model-bench pipeline/task/cls/onnx/resnet18
```

上述示例约定：

```text
benchmark/
├── onnx/cls/resnet18-onnx-float/resnet18.onnx
└── dataset/ImageNet-1K/validation_1000.csv
```

分类验证清单：

```csv
image_path,label
/path/to/images/image_a.jpg,0
/path/to/images/image_b.jpg,1
```

标签使用与模型类别顺序一致的整数。相对图片路径可以通过数据集的 `image_root` 指定根目录；未指定时按当前工作目录解释。`ImageNet1K` 读取器也能用于同样 CSV 格式的其他分类数据。

直接覆盖资源位置：

```bash
edge-model-bench pipeline/task/cls/onnx/resnet18 \
  model_path=/path/to/model.onnx \
  validation_csv=/path/to/validation.csv \
  input_size='[224,224]' \
  work_root=./work/my_cpu_run
```

评测前必须核对 RGB/BGR、缩放策略、归一化、输入输出名称、张量布局、类别映射；检测与分割尤其容易因导出格式不同而误用后处理。校准集应与验证集分开。旧 nuScenes mini 清单脚本会从同一 mini 样本池构造两类清单；正式量化对比前需自行准备独立的场景划分。

`.env.example` 仅记录可选环境变量，程序不会自动读取 `.env`，请自行 export。

## 命令行与配置

```bash
edge-model-bench demo --print-config
python tools/pipeline.py config/demo.yaml --output ./output/demo/result.json
python -m edge_model_bench demo --no-save-result
```

配置中的 `_target_` 会实例化 Python 对象，只运行你信任的配置。既支持 YAML 路径，也支持内置配置名；wheel 安装后在其他工作目录也能加载内置配置。

自定义配置树通过 `--config-root /path/to/config` 指定根目录。树外独立 YAML 默认以其父目录为根目录；若需要继承内置绝对 defaults，应明确指定根目录或把配置放入同一配置树。

| 阶段 | 用途 |
| --- | --- |
| `task`，默认 | 执行配置中的完整流水线；转换任务可先准备校准数据 |
| `calibrate` | 遍历 `pipeline.calibration_dataset`，生成校准张量 |
| `convert` | 仅调用转换器；所需校准数据应事先准备 |
| `quantize-onnx` | 调用所选转换器提供的独立 ONNX 量化功能 |
| `convert-static-onnx` | 调用支持此功能的转换器导入已量化 ONNX |
| `evaluate-onnx` | 用 ONNX Runtime 评估 `model_path` 或 `conversion.input_model` |
| `evaluate` | 使用配置中的 Runner 评估已转换产物 |
| `evaluate-qnn` | `evaluate` 的兼容别名 |

设备任务也可以独立跑 ONNX 基线：

```bash
edge-model-bench pipeline/task/det/qnn/fp16/yolov8n \
  --stage evaluate-onnx --provider CPUExecutionProvider \
  --output ./output/yolov8n_cpu/result.json
```

此阶段只实例化评测组件，不加载转换 SDK；但配置中的后处理仍必须匹配 ONNX 输出。`--provider` 作用于 `evaluate-onnx`，完整 `task` 使用 YAML 的 Runner 设置。指定的 provider 不可用时明确报错，避免把 CPU 回退结果当作 GPU 性能。

设备评测可以通过 `pipeline.model`、`input_bin` 或 `+compiled_model=/path/to/artifact` 指定产物。旧 QNN 配置可从输出目录解析唯一的 compiled bin；存在多个候选时会报错。

评测默认写入 `<work_dir>/result.json`，未设置则使用 `./work/result.json`。转换与准备阶段默认不保存结果，需要时加 `--save-result`。不同设备和精度请用 `--output` 或独立 `work_root` 隔离。

## 指标解释与结果

精度字段随任务而变化，例如分类 `top_1_accuracy`、检测 `map50_95`、实例分割 `mask_map50_95`、语义分割 `mean_iou`。

性能字段包括：

- `average_latency_ms`、`p50_latency_ms`、`p90_latency_ms`：Runner 执行调用的延迟。
- `fps`：`1000 / average_latency_ms`，是当前输入一次调用的吞吐估计。
- `num_inferences`：实际参与计时的调用数，排除预热。

模型加载、数据读取、预处理、后处理和指标计算不计入 Runner 延迟；某些运行时在调用中包含数据传输，因此对比时还要保持输入尺寸、batch、预热次数、精度、SDK 和设备状态一致。FPS 不是完整业务链路的端到端吞吐。

无预热字段的配置可以用 `+pipeline.warmup=10` 增加预热。未记录耗时的后端会省略性能字段。空验证集与非有限 JSON 数值会明确失败，结果文件在序列化成功后原子替换。

仓库 AP 与 BEV 指标是自身实现的评测协议，未经官方评估器交叉核验，不应当作 COCO/nuScenes 官方榜单成绩。FastSAM 属于候选掩码代理流程，不能当作提示驱动分割的正式精度结果。

## 批量执行与 CSV 报告

```bash
python tools/benchmark_all.py \
  config/pipeline/task/cls/onnx/resnet18.yaml \
  config/pipeline/task/cls/onnx/resnet50.yaml \
  --output-dir ./output/cpu --keep-going
python tools/export_benchmark_csv.py ./output/cpu --output ./output/cpu.csv
```

批处理按顺序执行，每个配置单独保存 `run.log` 和 `result.json`。公共覆盖参数使用可重复的 `--override key=value`。默认遇错停止，`--keep-going` 允许继续；只要有任务失败，批处理最终退出码非零。重跑前会清除相应旧结果，避免把失败任务误认作成功。脚本不隐式连接远程设备，也不提供自动恢复缓存。

CSV 统一导出标量字段并保留 `result_path`；混淆矩阵、类别列表、详细分类报告留在 JSON。

## 目录与扩展

```text
src/edge_model_bench/    可安装的 Python 包
config/                 Hydra 配置树，也打包进 wheel
examples/               合成 CPU 示例
tools/                  数据清单、量化、批量执行、报告和发布工具
tests/                  通用逻辑、配置组合、CPU 集成测试
docs/                   后端安装、扩展指南、发布检查
.github/                CI、Issue 表单、PR 模板
```

公开导入统一为 `edge_model_bench.*`，旧的 `from runner import ...` 等需迁移。部分解码器保留历史 `Qnn` 名称，它们是 NumPy 后处理，不依赖 QNN，只要输出格式一致即可跨后端复用。

新增硬件后端通常只需实现 Converter、Runner 和 YAML，详细契约与示例见[扩展指南](docs/new_chip_benchmark_pipeline.md)。Python 复用示例见 [English README](README.md#reuse-from-python)。

## 开发与公开发布

```bash
uv sync --extra onnx
uv run pytest
uv run ruff check src tools examples tests
uv run python -m build
uv run python tools/audit_release.py
```

贡献方式见 [CONTRIBUTING.md](CONTRIBUTING.md)，敏感问题报告见 [SECURITY.md](SECURITY.md)。

本项目采用目标仓库已有的 [MIT 许可证](LICENSE)，保留其版权署名。SDK、模型和数据集各自保留原有条款。参考[第三方说明](THIRD_PARTY_NOTICES.md)与[发布清单](docs/releasing.md)。
