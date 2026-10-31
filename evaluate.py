"""Evaluate an original assignment classifier checkpoint without retraining."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models import create_model


@torch.no_grad()
def accuracy(model, loader, tta=False):
    model.eval()
    device = next(model.parameters()).device
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        if tta:
            flipped_logits = model(torch.flip(images, dims=[3]))
            predictions = ((logits.softmax(dim=1) + flipped_logits.softmax(dim=1)) / 2).argmax(dim=1)
        else:
            predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
    if not total:
        raise ValueError('Evaluation dataset is empty.')
    return 100 * correct / total, correct, total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True, help='Folder whose immediate subfolders are class names')
    parser.add_argument('--split', choices=['test', 'val'], default='test', help='For val, --data-dir must contain all 2400 training images')
    parser.add_argument('--tta', action='store_true', help='Average original/flip probabilities exactly as in notebook')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output-json', type=Path)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, dict) or 'model_state_dict' not in checkpoint or 'classes' not in checkpoint:
        parser.error('Use a classifier checkpoint containing model_state_dict and classes, not ssl_encoder.pth.')
    classes = checkpoint['classes']
    if len(classes) != 16:
        parser.error('Checkpoint must contain the original 16 class names.')
    transform = transforms.Compose([
        transforms.Resize((128, 128)), transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    dataset = datasets.ImageFolder(str(args.data_dir), transform=transform)
    if dataset.classes != classes:
        parser.error(f'Class order does not match checkpoint. Expected {classes}; found {dataset.classes}.')
    if args.split == 'val':
        import numpy as np
        from sklearn.model_selection import train_test_split
        if len(dataset) != 2400:
            parser.error('Validation reconstruction requires all 2400 original training images.')
        _, indices = train_test_split(np.arange(len(dataset)), test_size=0.15, stratify=dataset.targets, random_state=42)
        dataset = Subset(dataset, indices)
    if args.split == 'test' and len(dataset) != 400:
        parser.error('Expected the original 400-image test set. Check --data-dir.')
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    model = create_model(args.device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    value, correct, total = accuracy(model, loader, args.tta)
    result = {'checkpoint': str(args.checkpoint), 'split': args.split, 'tta': args.tta,
              'correct': correct, 'total': total, 'accuracy_percent': value, 'classes': classes}
    print(json.dumps(result, indent=2))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
