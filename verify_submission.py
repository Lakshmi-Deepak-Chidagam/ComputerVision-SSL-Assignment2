"""Check the provided dataset and original checkpoints before submission."""
import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch
import torchvision
from torchvision.datasets import ImageFolder
from models import create_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-dir', type=Path, required=True)
    parser.add_argument('--test-dir', type=Path, required=True)
    parser.add_argument('--checkpoint-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('verification_results'))
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    train, test = ImageFolder(args.train_dir), ImageFolder(args.test_dir)
    if len(train) != 2400 or len(test) != 400 or len(train.classes) != 16 or train.classes != test.classes:
        raise ValueError('Expected original 2400/400-image datasets with identical 16-class ordering.')
    print('PASS: dataset counts and class order', flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evaluations = [
        ('improved_best.pth', False, 300),
        ('improved_best.pth', True, 301),
        ('mixup_best.pth', False, 295),
    ]
    results = []
    for name, tta, expected_correct in evaluations:
        checkpoint = args.checkpoint_dir / name
        if not checkpoint.is_file():
            raise FileNotFoundError(f'Download the ORIGINAL checkpoint first: {checkpoint}')
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if saved['classes'] != train.classes:
            raise ValueError(f'Checkpoint class order mismatch: {name}')
        create_model().load_state_dict(saved['model_state_dict'], strict=True)
        result_file = args.output_dir / (checkpoint.stem + ('_tta' if tta else '') + '.json')
        command = [sys.executable, str(Path(__file__).with_name('evaluate.py')),
                   '--checkpoint', str(checkpoint), '--data-dir', str(args.test_dir),
                   '--device', args.device, '--output-json', str(result_file)]
        if tta:
            command.append('--tta')
        subprocess.run(command, check=True)
        result = json.loads(result_file.read_text(encoding='utf-8'))
        result['matches_reported_result'] = result['correct'] == expected_correct and result['total'] == 400
        results.append(result)
        print(('PASS' if result['matches_reported_result'] else 'MISMATCH') +
              f": {name}, TTA={tta}, measured {result['accuracy_percent']:.2f}%", flush=True)
    summary = {'python': platform.python_version(), 'torch': torch.__version__,
               'torchvision': torchvision.__version__, 'device': args.device,
               'results': results, 'all_reported_test_results_match': all(r['matches_reported_result'] for r in results)}
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    if not summary['all_reported_test_results_match']:
        raise SystemExit('Verification found an accuracy mismatch. Check the original weights, data and environment before submission.')
    print('PASS: all three reported test results reproduced from the saved checkpoints.', flush=True)


if __name__ == '__main__':
    main()
