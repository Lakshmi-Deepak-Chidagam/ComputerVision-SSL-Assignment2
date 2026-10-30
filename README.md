# ComputerVision-SSL-Assignment2
Self-Supervised Visual Representation Learning with SimCLR
# Self-Supervised Visual Representation Learning

## Overview

This project investigates whether self-supervised contrastive learning improves image classification performance with limited labeled data.

We compare supervised ResNet-18 training against SimCLR self-supervised pretraining followed by supervised fine-tuning, additional regularization, Test-Time Augmentation (TTA), and Mixup.

## Dataset

- 2,400 provided training images across 16 scene categories
- 2,040 images used for training
- 360 images reserved for stratified validation
- 400 separate test images
- Input resolution: 128 × 128 pixels

The dataset is organized into class-specific directories.

## Methods

**1. Supervised baseline:** ResNet-18 trained from scratch using cross-entropy loss and AdamW.

**2. SimCLR:** ResNet-18 trained from scratch using two independently augmented views per image and NT-Xent contrastive loss. The projection head maps 512-dimensional encoder features to 128-dimensional contrastive embeddings.

**3. SSL fine-tuning:** The pretrained encoder is transferred to a 16-class classifier and fine-tuned using labeled images.

**4. Regularized fine-tuning:** Additional fine-tuning using label smoothing, AdamW weight decay, and cosine learning-rate scheduling.

**5. Test-Time Augmentation:** Average softmax probabilities from original and horizontally flipped images.

**6. Mixup:** Further supervised fine-tuning using linearly mixed training images and their labels.

No external pretrained weights were used.

## Results

| Method | Validation Accuracy | Test Accuracy |
|---|---:|---:|
| Supervised ResNet-18 | 65.56% | 68.00% |
| SimCLR + Fine-tuning | 70.00% | 70.50% |
| Improved SSL | 71.94% | 75.00% |
| Improved SSL + TTA | 72.22% | 75.25% |
| Improved SSL + Mixup | 73.33% | 73.75% |

## Reproducibility

The `SSL_Assignment.ipynb` notebook contains the data preparation, training, evaluation, and experimental code.

To reproduce the experiments:

1. Open the notebook in Google Colab and enable a GPU.
2. Install the dependencies listed in the notebook.
3. Place `train.zip` and `test.zip` inside `/content/datasets/`.
4. Run the notebook cells in order.
5. The training process generates model checkpoints and evaluates each experiment.

Training uses a fixed random seed of 42 for the initial dataset split and random-number generators.

## Model Checkpoints

The trained checkpoints are stored separately in Google Drive:

- `baseline_best.pth`
- `ssl_encoder.pth`
- `ssl_best.pth`
- `improved_best.pth`
- `mixup_best.pth`

All trained model checkpoints are available here:

[Download Trained Model Checkpoints]( https://drive.google.com/file/d/1HFMCxzKKi7koVulgyj3adaMWXSEkyOsJ/view?usp=sharing)

Best measured test accuracy: 75.25%

Best test-performing configuration:
- Model: ResNet-18
- Pretraining: SimCLR
- Fine-tuning: Regularized supervised fine-tuning
- Inference: Horizontal-flip Test-Time Augmentation
- Checkpoint: improved_best.pth

## Conclusion

SimCLR self-supervised pretraining improved test accuracy compared with supervised-only training. Additional regularized fine-tuning achieved 75.00% test accuracy, while Test-Time Augmentation achieved 75.25%. Mixup improved validation accuracy but did not improve test accuracy over the previous model.

The results demonstrate the potential of contrastive representation learning for limited-data classification while highlighting the importance of training configuration and evaluation methodology.
