# ComputerVision-SSL-Assignment2

Self-Supervised Visual Representation Learning with SimCLR.

## Overview

This project compares supervised ResNet18 training from scratch with SimCLR self-supervised pretraining followed by supervised fine-tuning, regularization, horizontal-flip test-time augmentation (TTA), and Mixup.

**No external pretrained weights or additional training images were used.**

## Dataset

- 2,400 provided training images across 16 classes.
- Stratified split: 2,040 training and 360 validation images, using random state 42.
- 400 separate test images.
- Input resolution: 128 × 128 pixels.
- RGB normalization: mean 0.5 and standard deviation 0.5 per channel.

## Results

| Method | Validation accuracy | Test accuracy |
|---|---:|---:|
| Supervised ResNet18 | 65.56% | 68.00% |
| SimCLR + fine-tuning | 70.00% | 70.50% |
| Improved regularized SSL | 71.94% | 75.00% |
| Improved SSL + horizontal-flip TTA | 72.22% | **75.25%** |
| Improved SSL + Mixup | **73.33%** | 73.75% |

**Highest observed test accuracy:** 75.25%, using `improved_best.pth` with horizontal-flip TTA.

**Validation-selected configuration:** Mixup, using `mixup_best.pth`, with 73.33% validation and 73.75% test accuracy.

These are distinct outcomes. The highest test score does not determine the validation-selected configuration. For the improved checkpoint, TTA was chosen over ordinary inference based on validation accuracy.

## Repository contents

| File | Purpose |
|---|---|
| `train.py` | Training sequence extracted from the original notebook |
| `evaluate.py` | Saved-checkpoint evaluation, including optional TTA |
| `models.py` | ResNet18 classifier, SimCLR model, and NT-Xent loss |
| `hyperparameters.json` | Documented training settings |
| `requirements.txt` | Required dependencies |
| `requirements-tested.txt` | Direct dependency versions used for local verification |
| `verify_submission.py` | Dataset and reported test-result verification |
| `SSL_Assignment.ipynb` | Original experiment notebook |

`hyperparameters.json` documents the constants embedded in `train.py`; the script does not load this JSON as a configuration input.

## Trained checkpoints

[Download trained model checkpoints](https://drive.google.com/drive/folders/1sJOkwIEhKA7CqWO6cSUNOR3pai4WZOhq?usp=sharing)

Checkpoint files:

- `baseline_best.pth`: supervised baseline.
- `ssl_encoder.pth`: pretrained encoder only.
- `ssl_best.pth`: initial SSL classifier.
- `improved_best.pth`: improved classifier; use with TTA for the 75.25% result.
- `mixup_best.pth`: validation-selected Mixup classifier.

Download `improved_best.pth` and `mixup_best.pth` into a local `checkpoints/` folder. TTA uses the same improved checkpoint; no separate TTA checkpoint is needed.

The Google Drive folder must allow grader downloads without requesting access. Checkpoints are hosted externally rather than committed to GitHub.

## Environment setup

Local evaluation was verified using **Windows, Python 3.12.14, and CPU inference**.

From the repository folder, install dependencies:

```bash
python -m pip install -r requirements-tested.txt
```

Alternatively, `requirements.txt` lists the general dependencies. GPU training requires a compatible PyTorch/CUDA installation.

The tested dependency versions describe the local verification environment. The original Colab training environment was not version-pinned.

## Dataset preparation

Extract the provided archives so that these paths directly contain the class folders:

```text
data/
  train/
    train/
      Bedroom/
      Coast/
      ... remaining class folders
  test/
    test/
      Bedroom/
      Coast/
      ... remaining class folders
checkpoints/
  improved_best.pth
  mixup_best.pth
```

Preserve the original class folder names and image filenames. Evaluation checks class ordering against the checkpoint metadata.

In PowerShell, archives in Downloads can be extracted with:

```powershell
Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\train.zip" -DestinationPath ".\data\train"
Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\test.zip" -DestinationPath ".\data\test"
```

## Evaluate saved models

Run these commands from the repository folder using the Python environment where dependencies were installed. No retraining is required.

### Improved SSL with TTA

```bash
python evaluate.py --checkpoint checkpoints/improved_best.pth --data-dir data/test/test --tta --device cpu --output-json evaluation_tta.json
```

Verified result: **301/400 correct = 75.25%**.

TTA averages softmax probabilities from the original image and its horizontal flip before selecting the predicted class.

### Improved SSL without TTA

```bash
python evaluate.py --checkpoint checkpoints/improved_best.pth --data-dir data/test/test --device cpu --output-json evaluation_improved.json
```

Verified result: **300/400 correct = 75.00%**.

### Mixup without TTA

```bash
python evaluate.py --checkpoint checkpoints/mixup_best.pth --data-dir data/test/test --device cpu --output-json evaluation_mixup.json
```

Verified result: **295/400 correct = 73.75%**.

## Verify validation results

The evaluator reconstructs the original stratified validation split from all 2,400 training images.

Improved SSL with TTA:

```bash
python evaluate.py --checkpoint checkpoints/improved_best.pth --data-dir data/train/train --split val --tta --device cpu --output-json validation_tta.json
```

Verified result: **260/360 correct = 72.22%**.

Mixup:

```bash
python evaluate.py --checkpoint checkpoints/mixup_best.pth --data-dir data/train/train --split val --device cpu --output-json validation_mixup.json
```

Verified result: **264/360 correct = 73.33%**.

## Combined test verification

```bash
python verify_submission.py --train-dir data/train/train --test-dir data/test/test --checkpoint-dir checkpoints --device cpu
```

This checks dataset counts and class ordering, loads both classifier checkpoints, evaluates all three test configurations, and compares the measured results with the reported correct-image counts. Results are saved under `verification_results/`.

## Reproduce training

A GPU is recommended. Use a fresh output folder to preserve the original submitted checkpoints:

```bash
python train.py --train-dir data/train/train --test-dir data/test/test --output-dir checkpoints_retrained
```

The training sequence is:

| Stage | Settings |
|---|---|
| Supervised baseline | 15 epochs; cross-entropy; AdamW; learning rate 0.001; weight decay 0.0001 |
| SimCLR pretraining | 30 epochs; batch size 64; NT-Xent temperature 0.5; AdamW; learning rate 0.001; weight decay 0.0001 |
| SSL fine-tuning | 15 epochs; cross-entropy; AdamW; learning rate 0.0003; weight decay 0.0001 |
| Improved fine-tuning | 20 epochs from `ssl_best.pth`; label smoothing 0.1; AdamW; learning rate 0.00005; weight decay 0.001; cosine scheduler with T_max 20 |
| Mixup | 12 epochs per run from `improved_best.pth`; alpha 0.2; AdamW; learning rate 0.00003; weight decay 0.001 |

SimCLR uses two independently augmented views of each training image. Its projection head maps 512-dimensional encoder features through a 512-dimensional hidden layer to 128-dimensional normalized embeddings. NT-Xent encourages agreement between views of the same image relative to other images.

Supervised augmentation uses resizing, random horizontal flips, and random rotations up to 10 degrees. SSL augmentation uses random resized crops, horizontal flips, color jitter, and random grayscale. SSL pretraining uses only the 2,040 training-split images.

Supervised checkpoints are retained according to validation accuracy. Improved and Mixup fine-tuning also consider the initial checkpoint as a candidate.

**Notebook execution detail:** the original notebook contains two consecutive 12-epoch Mixup runs. Each starts from `improved_best.pth`; the second overwrites the first Mixup checkpoint. `train.py` preserves this sequence.

Training saves environment details, split indices, checkpoints, and a measured run summary.

## Original notebook workflow

To use `SSL_Assignment.ipynb` in Google Colab:

1. Enable a GPU.
2. Install the notebook dependencies.
3. Place `train.zip` and `test.zip` in `/content/datasets/`.
4. Run the experiment cells in their documented order.
5. Adapt the Google Drive save location to your account if needed.

## Verification and limitations

Saved-checkpoint evaluation was reproduced locally on Windows using Python 3.12.14 and CPU inference. All three test results and both validation results listed above matched the reported values.

The original full training schedule was not rerun during this local verification. A reduced integration test exercised every training stage, checkpoint saving, and checkpoint reloading.

The notebook seeds Python, NumPy, and PyTorch with 42 once. Later stages share the evolving random state. Repeated cells, library versions, hardware, and nondeterministic operations can change newly trained weights and results. Retraining reproduces the documented procedure but does not guarantee identical historical weights or accuracies.

Use the original saved checkpoints to verify the reported evaluation. Multiple test evaluations and the small validation/test sets limit conclusions about small accuracy differences.

## Conclusion

SimCLR improved test accuracy over supervised training in the reported experiments. Regularized fine-tuning reached 75.00%, and horizontal-flip TTA increased this to 75.25%. Mixup achieved the highest validation accuracy but a lower test accuracy of 73.75%, illustrating that validation and test rankings can differ.
