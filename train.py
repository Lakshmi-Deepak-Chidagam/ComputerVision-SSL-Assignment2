"""Notebook-faithful training sequence; see README for reproducibility limits."""
import argparse
import json
import platform
from pathlib import Path
import torch
import torchvision
from models import create_model as create_classifier, SimCLR, nt_xent
from evaluate import accuracy

def main():
    import torch  # Notebook imports below make torch local to this function.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-dir', type=Path, required=True)
    parser.add_argument('--test-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('checkpoints'))
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    names = ['baseline_best.pth', 'ssl_encoder.pth', 'ssl_best.pth', 'improved_best.pth', 'mixup_best.pth']
    if any((args.output_dir / name).exists() for name in names):
        parser.error('Output already contains checkpoints. Choose a fresh --output-dir to preserve them.')
    def create_model():
        return create_classifier(args.device)
    def evaluate(model, loader):
        return accuracy(model, loader)[0]
    def evaluate_tta(model, loader):
        return accuracy(model, loader, tta=True)[0]
    (args.output_dir / 'environment.json').write_text(json.dumps({
        'python': platform.python_version(), 'torch': torch.__version__,
        'torchvision': torchvision.__version__, 'cuda': torch.version.cuda,
        'device': args.device,
    }, indent=2), encoding='utf-8')
    # Original notebook cell 4
    import torch
    import numpy as np
    import random

    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, Subset
    from sklearn.model_selection import train_test_split

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    train_dir = str(args.train_dir)
    test_dir = str(args.test_dir)

    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    full_train = datasets.ImageFolder(train_dir)
    labels = np.array(full_train.targets)

    train_idx, val_idx = train_test_split(
        np.arange(len(labels)),
        test_size=0.15,
        stratify=labels,
        random_state=42
    )

    train_dataset = Subset(
        datasets.ImageFolder(train_dir, transform=train_transform),
        train_idx
    )

    val_dataset = Subset(
        datasets.ImageFolder(train_dir, transform=eval_transform),
        val_idx
    )

    test_dataset = datasets.ImageFolder(
        test_dir, transform=eval_transform
    )

    assert full_train.classes == test_dataset.classes, \
        "Train and test class names do not match!"

    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True,
        num_workers=2, pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=64, shuffle=False,
        num_workers=2, pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, batch_size=64, shuffle=False,
        num_workers=2, pin_memory=True
    )

    print("Classes:", full_train.classes)
    print("Number of classes:", len(full_train.classes))
    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))
    print("Test samples:", len(test_dataset))
    assert len(full_train.classes) == 16, "Expected 16 classes"
    assert len(full_train) == 2400, "Expected 2400 training images"
    assert len(test_dataset) == 400, "Expected 400 test images"

    # Original notebook cell 5
    import torch.nn as nn
    import torch.optim as optim
    from torchvision.models import resnet18
    from tqdm.auto import tqdm
    import copy

    device = torch.device(args.device)



    baseline = create_model()

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        baseline.parameters(),
        lr=0.001,
        weight_decay=1e-4
    )

    best_val = -1
    best_weights = None

    for epoch in range(15):
        baseline.train()
        running_loss = 0

        for images, labels in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/15"
        ):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = baseline(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)

        val_acc = evaluate(baseline, val_loader)

        print(
            f"Epoch {epoch+1}: "
            f"Loss={running_loss/len(train_dataset):.4f}, "
            f"Val Accuracy={val_acc:.2f}%"
        )

        if val_acc > best_val:
            best_val = val_acc
            best_weights = copy.deepcopy(baseline.state_dict())

    baseline.load_state_dict(best_weights)

    torch.save({
        "model_state_dict": baseline.state_dict(),
        "classes": full_train.classes,
        "validation_accuracy": best_val
    }, str(args.output_dir / "baseline_best.pth"))

    print("Best validation accuracy:", best_val)
    print("Baseline test accuracy:", evaluate(baseline, test_loader))

    # Original notebook cell 6
    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset, batch_size=64, shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset, batch_size=64, shuffle=False,
        num_workers=0
    )

    # Original notebook cell 7
    from torchvision import transforms, datasets
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image

    ssl_transform = transforms.Compose([
        transforms.RandomResizedCrop(128, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    class SimCLRDataset(Dataset):
        def __init__(self, imagefolder, indices, transform):
            self.samples = [imagefolder.samples[i] for i in indices]
            self.transform = transform

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, index):
            path, _ = self.samples[index]
            with Image.open(path) as img:
                image = img.convert("RGB")
                return self.transform(image), self.transform(image)

    ssl_dataset = SimCLRDataset(full_train, train_idx, ssl_transform)

    ssl_loader = DataLoader(
        ssl_dataset, batch_size=64, shuffle=True,
        drop_last=True, num_workers=0
    )

    print("SSL images:", len(ssl_dataset))

    # Original notebook cell 8
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision.models import resnet18



    simclr = SimCLR().to(device)
    ssl_optimizer = torch.optim.AdamW(
        simclr.parameters(), lr=0.001, weight_decay=1e-4
    )

    # Original notebook cell 9
    from tqdm.auto import tqdm

    ssl_losses = []

    for epoch in range(30):
        simclr.train()
        total_loss = 0

        for view1, view2 in tqdm(
            ssl_loader, desc=f"SSL Epoch {epoch+1}/30"
        ):
            view1 = view1.to(device)
            view2 = view2.to(device)

            ssl_optimizer.zero_grad()

            z1 = simclr(view1)
            z2 = simclr(view2)

            loss = nt_xent(z1, z2)
            loss.backward()
            ssl_optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(ssl_loader)
        ssl_losses.append(avg_loss)
        print(f"SSL Epoch {epoch+1}: Loss={avg_loss:.4f}")

    torch.save(simclr.encoder.state_dict(), str(args.output_dir / "ssl_encoder.pth"))
    print("SSL pretraining completed!")

    # Original notebook cell 10
    import copy

    ssl_classifier = create_model()

    encoder_weights = torch.load(
        str(args.output_dir / "ssl_encoder.pth"),
        map_location=device,
        weights_only=True
    )

    ssl_classifier.load_state_dict(encoder_weights, strict=False)

    optimizer = torch.optim.AdamW(
        ssl_classifier.parameters(),
        lr=0.0003,
        weight_decay=1e-4
    )

    criterion = nn.CrossEntropyLoss()
    best_ssl_val = -1
    best_ssl_weights = None

    for epoch in range(15):
        ssl_classifier.train()

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = ssl_classifier(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        val_acc = evaluate(ssl_classifier, val_loader)
        print(f"Fine-tune Epoch {epoch+1}: Val={val_acc:.2f}%")

        if val_acc > best_ssl_val:
            best_ssl_val = val_acc
            best_ssl_weights = copy.deepcopy(
                ssl_classifier.state_dict()
            )

    ssl_classifier.load_state_dict(best_ssl_weights)

    torch.save({
        "model_state_dict": ssl_classifier.state_dict(),
        "classes": full_train.classes,
        "validation_accuracy": best_ssl_val
    }, str(args.output_dir / "ssl_best.pth"))

    print("SSL best validation accuracy:", best_ssl_val)
    print("SSL test accuracy:", evaluate(ssl_classifier, test_loader))

    # Original notebook cell 11
    import torch
    import torch.nn as nn
    import copy

    improved_model = create_model()

    checkpoint = torch.load(
        str(args.output_dir / "ssl_best.pth"),
        map_location=device,
        weights_only=True
    )

    improved_model.load_state_dict(checkpoint["model_state_dict"])

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = torch.optim.AdamW(
        improved_model.parameters(),
        lr=0.00005,
        weight_decay=0.001
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=20
    )

    best_improved_val = evaluate(improved_model, val_loader)
    best_improved_weights = copy.deepcopy(improved_model.state_dict())

    print("Starting validation accuracy:", best_improved_val)

    for epoch in range(20):
        improved_model.train()

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = improved_model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

        scheduler.step()

        val_acc = evaluate(improved_model, val_loader)

        print(f"Improved Epoch {epoch+1}/20: Val={val_acc:.2f}%")

        if val_acc > best_improved_val:
            best_improved_val = val_acc
            best_improved_weights = copy.deepcopy(
                improved_model.state_dict()
            )

    improved_model.load_state_dict(best_improved_weights)

    torch.save({
        "model_state_dict": improved_model.state_dict(),
        "classes": full_train.classes,
        "validation_accuracy": best_improved_val
    }, str(args.output_dir / "improved_best.pth"))

    print("Improved best validation accuracy:", best_improved_val)
    print("Improved test accuracy:", evaluate(improved_model, test_loader))

    # Original notebook cell 14
    import torch
    import torch.nn.functional as F


    print("Original validation accuracy:",
          evaluate(improved_model, val_loader))

    print("TTA validation accuracy:",
          evaluate_tta(improved_model, val_loader))

    # Original notebook cell 15
    original_val = evaluate(improved_model, val_loader)
    tta_val = evaluate_tta(improved_model, val_loader)

    if tta_val > original_val:
        selected = "TTA"
        final_test = evaluate_tta(improved_model, test_loader)
    else:
        selected = "Original"
        final_test = evaluate(improved_model, test_loader)

    print("Selected inference:", selected)
    print("Selected test accuracy:", final_test)

    # Original notebook cell 16
    import numpy as np
    import torch
    import torch.nn as nn
    import copy

    mixup_model = create_model()

    checkpoint = torch.load(
        str(args.output_dir / "improved_best.pth"),
        map_location=device,
        weights_only=True
    )

    mixup_model.load_state_dict(checkpoint["model_state_dict"])

    optimizer = torch.optim.AdamW(
        mixup_model.parameters(),
        lr=0.00003,
        weight_decay=0.001
    )

    criterion = nn.CrossEntropyLoss()

    def mixup_data(x, y, alpha=0.2):
        lam = np.random.beta(alpha, alpha)
        index = torch.randperm(x.size(0), device=x.device)

        mixed_x = lam * x + (1 - lam) * x[index]

        return mixed_x, y, y[index], lam

    best_mixup_val = evaluate(mixup_model, val_loader)
    best_mixup_weights = copy.deepcopy(mixup_model.state_dict())

    print("Starting validation:", best_mixup_val)

    for epoch in range(12):
        mixup_model.train()

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            mixed, labels_a, labels_b, lam = mixup_data(
                images, labels, alpha=0.2
            )

            optimizer.zero_grad()

            outputs = mixup_model(mixed)

            loss = (
                lam * criterion(outputs, labels_a)
                + (1 - lam) * criterion(outputs, labels_b)
            )

            loss.backward()
            optimizer.step()

        val_acc = evaluate(mixup_model, val_loader)
        print(f"Mixup Epoch {epoch+1}/12: Val={val_acc:.2f}%")

        if val_acc > best_mixup_val:
            best_mixup_val = val_acc
            best_mixup_weights = copy.deepcopy(
                mixup_model.state_dict()
            )

    mixup_model.load_state_dict(best_mixup_weights)

    torch.save({
        "model_state_dict": mixup_model.state_dict(),
        "classes": full_train.classes,
        "validation_accuracy": best_mixup_val
    }, str(args.output_dir / "mixup_best.pth"))

    print("Best Mixup validation accuracy:", best_mixup_val)

    # Original notebook cell 17
    import numpy as np
    import torch
    import torch.nn as nn
    import copy

    mixup_model = create_model()

    checkpoint = torch.load(
        str(args.output_dir / "improved_best.pth"),
        map_location=device,
        weights_only=True
    )

    mixup_model.load_state_dict(checkpoint["model_state_dict"])

    optimizer = torch.optim.AdamW(
        mixup_model.parameters(),
        lr=0.00003,
        weight_decay=0.001
    )

    criterion = nn.CrossEntropyLoss()

    def mixup_data(x, y, alpha=0.2):
        lam = np.random.beta(alpha, alpha)
        index = torch.randperm(x.size(0), device=x.device)

        mixed_x = lam * x + (1 - lam) * x[index]

        return mixed_x, y, y[index], lam

    best_mixup_val = evaluate(mixup_model, val_loader)
    best_mixup_weights = copy.deepcopy(mixup_model.state_dict())

    print("Starting validation:", best_mixup_val)

    for epoch in range(12):
        mixup_model.train()

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            mixed, labels_a, labels_b, lam = mixup_data(
                images, labels, alpha=0.2
            )

            optimizer.zero_grad()

            outputs = mixup_model(mixed)

            loss = (
                lam * criterion(outputs, labels_a)
                + (1 - lam) * criterion(outputs, labels_b)
            )

            loss.backward()
            optimizer.step()

        val_acc = evaluate(mixup_model, val_loader)
        print(f"Mixup Epoch {epoch+1}/12: Val={val_acc:.2f}%")

        if val_acc > best_mixup_val:
            best_mixup_val = val_acc
            best_mixup_weights = copy.deepcopy(
                mixup_model.state_dict()
            )

    mixup_model.load_state_dict(best_mixup_weights)

    torch.save({
        "model_state_dict": mixup_model.state_dict(),
        "classes": full_train.classes,
        "validation_accuracy": best_mixup_val
    }, str(args.output_dir / "mixup_best.pth"))

    print("Best Mixup validation accuracy:", best_mixup_val)

    # Original notebook cell 18
    original_val = evaluate(mixup_model, val_loader)
    tta_val = evaluate_tta(mixup_model, val_loader)

    print("Mixup original validation:", original_val)
    print("Mixup TTA validation:", tta_val)

    # Original notebook cell 19
    mixup_test_accuracy = evaluate(mixup_model, test_loader)

    print(f"Mixup validation accuracy: {best_mixup_val:.2f}%")
    print(f"Mixup test accuracy: {mixup_test_accuracy:.2f}%")
    (args.output_dir / 'run_summary.json').write_text(json.dumps({
        'classes': full_train.classes,
        'train_indices': train_idx.tolist(), 'validation_indices': val_idx.tolist(),
        'baseline_val': best_val, 'ssl_val': best_ssl_val,
        'improved_val': best_improved_val, 'mixup_val': best_mixup_val,
        'improved_inference_selected_on_validation': selected,
        'improved_selected_test': final_test, 'mixup_test': mixup_test_accuracy,
    }, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
