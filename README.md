# PICFormer: Visibility-Aware 3D Human Pose Estimation

This is a Transformer-based 3D human pose estimation project implementing the PICFormer (Pose-aware Interactive Cross-modal Transformer) model.

![](C:\Users\31291\OneDrive\论文\figure\flow_chart_new.png)



## Project Features

- **Visibility-Aware Module (VFM)**: Automatically detects and utilizes keypoint visibility information
- **Spatial-Temporal Aggregation Module (SPA)**: Multi-scale spatial and temporal feature aggregation
- **Global Position-Aware Module (GPA)**: Attention-based global position awareness
- **End-to-End Training**: Supports end-to-end training from 2D keypoints to 3D poses
- **Multi-Dataset Support**: Supports Human36M, 3DPW and other datasets

## Project Structure

```
picformer_project/
├── README.md                 # Project description
├── requirements.txt          # Dependency list
├── setup.py                  # Installation script
├── config/
│   └── config.yaml          # Configuration file
├── data/
│   ├── raw/                 # Raw data
│   └── processed/           # Processed data
├── src/
│   ├── __init__.py
│   ├── config.py            # Configuration management
│   ├── datasets.py          # Dataset loading
│   ├── utils.py             # Utility functions
│   ├── train.py             # Training script
│   ├── evaluate.py          # Evaluation script
│   ├── infer.py             # Inference script
│   └── models/
│       ├── __init__.py
│       ├── vfm.py           # Visibility-aware module
│       ├── spa.py           # Spatial-temporal aggregation module
│       ├── gpa.py           # Global position-aware module
│       └── picformer.py     # Main model
├── scripts/
│   ├── train.sh             # Training script
│   ├── evaluate.sh          # Evaluation script
│   └── infer.sh             # Inference script
└── tests/
    └── test_models.py       # Model tests
```

## Installation

1. Clone the project
```bash
git clone <repository-url>
cd picformer_project
```

2. Install dependencies
```bash
pip install -r requirements.txt
pip install -e .
```

## Usage

### 1. Data Preparation

Place datasets in the `data/processed/` directory, supporting the following formats:
- Human36M: `h36m_train.h5`, `h36m_val.h5`
- 3DPW: `3dpw_val.h5`

Data format:
- `poses_2d`: 2D keypoints [N, T, J, 2]
- `poses_3d`: 3D keypoints [N, T, J, 3]
- `visibility`: Visibility labels [N, T, J]

### 2. Training

```bash
# Use script for training
bash scripts/train.sh

# Or run Python script directly
python src/train.py --config config/config.yaml --checkpoint checkpoint
```

### 3. Evaluation

```bash
# Use script for evaluation
bash scripts/evaluate.sh

# Or run Python script directly
python src/evaluate.py --config config/config.yaml --checkpoint checkpoint
```

### 4. Inference

```bash
# Use script for inference
bash scripts/infer.sh data/input_poses.npy output/results.npz

# Or run Python script directly
python src/infer.py --input data/input_poses.npy --output output/results.npz --checkpoint checkpoint/best.pth
```

### 5. Testing

```bash
python tests/test_models.py
```

## Configuration

Main configuration parameters in `config/config.yaml`:

```yaml
data:
  batch_size: 64
  num_workers: 4

model:
  channel: 128
  groups_S: [[0,1,2], [3,4,5], ...]  # Spatial groups
  groups_T: [[0,1,2], [3,4,5], ...]  # Temporal groups
  eps: 0.01
  beta: 10.0
  tau: 0.5

train:
  lr: 1e-5
  epochs: 128
  weight_decay: 0.01

loss:
  lambda_vis: 0.1
  lambda_hc: 0.1
  lambda_mpjpe: 1.0
```

