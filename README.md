# Pose Estimator

An independent PyTorch implementation of top-down human pose estimation with [SimCC](https://arxiv.org/abs/2107.03332), using a ResNet-50 backbone and an upsampling-free coordinate classification head.

The project covers the complete COCO keypoint pipeline: annotation loading, person crop preprocessing, data augmentation, SimCC target generation, training, flip-test inference, coordinate restoration, and evaluation with the official COCO API.

## Highlights

- ResNet-50 implemented directly in PyTorch, including BasicBlock, Bottleneck, and residual stages
- SimCC head without deconvolution or other upsampling layers
- Independent classification of horizontal and vertical keypoint coordinates
- COCO `train2017` and `val2017` preprocessing with ground-truth person bounding boxes
- Affine crop and coordinate transforms with aspect-ratio correction and bbox expansion
- Random horizontal flip, rotation, and scale augmentation
- Gaussian-smoothed SimCC targets and KL divergence loss
- Flip-test inference with left-right keypoint swapping and coordinate shifting
- COCO Keypoint AP evaluation through `pycocotools`
- Development-time verification against the official SimCC implementation

## Model

The default model uses a ResNet-50 backbone with a `192 x 256` input, expressed as width by height in the configuration.

The final ResNet feature map is passed through a `1 x 1` convolution. Its channels are grouped by keypoint and flattened, then two linear classifiers predict the x and y coordinate distributions independently. With the default SimCC split factor of `2.0`, the output dimensions are:

- x-axis: `192 x 2 = 384` bins per keypoint
- y-axis: `256 x 2 = 512` bins per keypoint
- keypoints: 17 COCO body keypoints

The head operates directly on the low-resolution backbone feature map. It does not use deconvolution, interpolation, or a heatmap upsampling stage.

The repository initializes convolutional weights from a normal distribution and trains the model from scratch. It does not load an ImageNet-pretrained ResNet checkpoint.

## COCO pipeline

### Preprocessing

For each annotated person instance, the dataset code:

1. Loads the ground-truth person bbox and 17 keypoints from COCO annotations.
2. Adjusts the bbox to the model input aspect ratio.
3. Expands the bbox scale by `1.25`.
4. Applies an affine transform to produce a `192 x 256` person crop.
5. Normalizes the crop with ImageNet channel statistics.
6. Transforms the keypoints with the same affine matrix.
7. Generates separate Gaussian-smoothed SimCC targets for the x and y axes.

### Training augmentation

The default training configuration applies:

- horizontal flip with probability `0.5`
- rotation with probability `0.6`, sampled with a factor of 40 degrees and clipped to 80 degrees
- scale jitter sampled with a factor of `0.3` and clipped to `[0.7, 1.3]`

Validation does not use random augmentation.

### Optimization

The default run uses:

- Adam optimizer
- initial learning rate `1e-3`
- 140 epochs
- learning-rate decay by `0.1` at epochs 90 and 120
- KL divergence loss over the x-axis and y-axis distributions
- random seed 42

### Evaluation

Validation uses `pycocotools.COCOeval` with `iouType="keypoints"`. Predicted crop coordinates are transformed back to the original image coordinate system before evaluation.

The repository's full validation path uses:

- COCO `val2017`
- ground-truth person bounding boxes from `person_keypoints_val2017.json`
- bbox score `1.0`
- flip-test averaging
- a one-bin shift of the flipped x-axis output before averaging

Because ground-truth boxes are used, this result evaluates the pose stage rather than a complete detector-plus-pose system.

## Result

| Result | COCO Keypoint AP | Interpretation |
| --- | ---: | --- |
| This implementation | **0.6929** | Measured result produced by this project |
| Officially published/reference SimCC | **0.709** | External reference value, not a score measured by this repository |

The official/reference AP is included as context only. The checked repository establishes this project's use of COCO `val2017`, ground-truth bboxes, `192 x 256` inputs, and flip-test inference. It does not include enough metadata from the official AP 0.709 run to prove that every evaluation and training setting was identical. Initialization, exact head design, label smoothing, bbox handling, preprocessing, scoring, and post-processing can affect AP. The two values should therefore not be interpreted as a controlled apples-to-apples reproduction gap.

No locally reproduced reference score is used in this comparison.

## Verification against the reference implementation

The model and pipeline were developed independently, then checked against artifacts generated from the official SimCC code. The scripts under `script/verify_*.py` compare:

- model parameter shapes and values
- bbox centers, bbox scales, transformed keypoints, and annotation ordering
- normalized inputs and SimCC target tensors
- model x and y outputs and training loss
- validation model parameters and validation loss
- decoded keypoint coordinates
- COCO-format prediction dumps

These scripts are a development verification harness. Several of them reference local debug dumps and checkpoints that are not included in the repository, so they are not an out-of-the-box test suite for a fresh clone.

## Installation

The provided Dockerfile targets NVIDIA CUDA 12.4.1 with cuDNN on Ubuntu 22.04 and installs PyTorch 2.5.1 and torchvision 0.20.1.

```bash
git clone https://github.com/lancifollia/pose.git
cd pose

docker build -t pose-simcc .
docker run --gpus all --rm -it \
  -v "$PWD":/home/dev/pose \
  -v /path/to/coco:/home/dev/data/coco \
  -w /home/dev/pose \
  pose-simcc
```

Inside the container, install the runtime packages used by the project:

```bash
pip3 install \
  einops==0.8.0 \
  loguru==0.7.2 \
  numpy==2.1.3 \
  opencv-python==4.10.0.84 \
  orjson==3.10.12 \
  pycocotools==2.0.8 \
  PyYAML==6.0.2 \
  yacs==0.1.8
```

`requirements.txt` is the original full environment snapshot. It contains development, notebook, CUDA, and Ubuntu-specific packages in addition to the runtime dependencies, so it is less portable than the focused installation above.

## Data preparation

Download COCO 2017 train images, validation images, and train/validation annotations from the [COCO dataset download page](https://cocodataset.org/#download).

The default configuration expects this layout:

```text
/home/dev/data/coco/
├── annotations/
│   ├── person_keypoints_train2017.json
│   └── person_keypoints_val2017.json
└── images/
    ├── train2017/
    │   └── *.jpg
    └── val2017/
        └── *.jpg
```

If the data is stored elsewhere, update `DATA_TRAIN.ANNOTATION_FNAME`, `DATA_TRAIN.IMAGE_DIR`, `DATA_VAL.ANNOTATION_FNAME`, and `DATA_VAL.IMAGE_DIR` in `pose/config.py`.

## Training

Run the default GPU training configuration:

```bash
sh train.sh
```

This executes:

```bash
python3 script/manage.py train configs/resnet50_simcc_upfree.yaml
```

Each run creates a timestamped directory under `result/`. Depending on the configuration, it contains:

- per-epoch checkpoints
- `best_model/best_model.pth`
- `performance_table.md`
- COCO-format prediction dumps under `pred_dump/`

The default configuration uses a batch size of 200 and 24 data-loader workers. Adjust these values in `configs/resnet50_simcc_upfree.yaml` for the available GPU memory and CPU capacity.

## Evaluation

Set `TEST.MODEL_PATH` in `configs/resnet50_test.yaml` to a saved per-epoch checkpoint. The test path expects a checkpoint dictionary containing `model_state_dict`, as produced by this repository's checkpoint-saving code.

Then run:

```bash
sh test_gpu.sh
```

This executes:

```bash
python3 script/manage.py test configs/resnet50_test.yaml
```

The CPU train and test configurations intentionally restrict the dataset to a small list of image IDs. They are useful for smoke tests, but they do not reproduce the reported full-dataset AP.

## Project structure

```text
.
├── configs/
│   ├── resnet50_simcc_upfree.yaml       # default GPU training config
│   ├── resnet50_simcc_upfree_cpu.yaml   # small CPU smoke-test config
│   ├── resnet50_test.yaml               # full GPU evaluation config
│   └── resnet50_test_cpu.yaml           # small CPU evaluation config
├── data/                                # placeholder for local data
├── notebooks/                           # COCO, transform, evaluation, and debugging studies
├── pose/
│   ├── config.py                        # default configuration and data paths
│   ├── core/
│   │   ├── function.py                  # training, inference, decoding, and validation
│   │   └── loss.py                      # KLDiscretLoss
│   ├── data/
│   │   ├── dataset.py                   # COCO loading, augmentation, targets, and evaluation
│   │   ├── datatype.py                  # dataset and keypoint data structures
│   │   └── transforms.py                # affine and flip transforms
│   └── model/
│       └── resnet.py                    # ResNet backbone and upsampling-free SimCC head
├── script/
│   ├── manage.py                        # train and test entry point
│   └── verify_*.py                      # reference implementation comparison tools
├── Dockerfile
├── requirements.txt
├── train.sh
└── test_gpu.sh
```

## References

- [SimCC: a Simple Coordinate Classification Perspective for Human Pose Estimation](https://arxiv.org/abs/2107.03332)
- [Official SimCC implementation](https://github.com/leeyegy/SimCC)
- [COCO Keypoint Evaluation](https://cocodataset.org/#keypoints-eval)


