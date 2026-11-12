"""Fixed TTA comparison on the held-out Run G twin; no training or Kaggle calls."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from src.data import build_val_loader, build_test_loader
from src.train import load_model
from src.models import count_params
from src.predict import write_submission
from audit_helpers import audit_data, digest

MODES = ('none', 'flip_prob', 'flip_logit', 'shift_prob')
SUITE = {'clean': (), 'brightness_low': (('brightness', .8),),
         'brightness_high': (('brightness', 1.2),), 'blur': (('blur', .6),)}

@torch.inference_mode()
def predict(model, loader, device, labeled):
    result = {m: [] for m in MODES}
    labels = []
    model.eval()
    for batch in loader:
        x, y = batch if labeled else (batch, None)
        x = x.to(device)
        flipped = x.flip(3)
        z, zf = model(x).float(), model(flipped).float()
        p, pf = z.softmax(1), zf.softmax(1)
        result['none'].append(p.cpu().numpy())
        result['flip_prob'].append(((p + pf) / 2).cpu().numpy())
        result['flip_logit'].append(((z + zf) / 2).softmax(1).cpu().numpy())
        # Same reflect-padded +/-2-pixel cardinal shifts as src.train.tta_probs.
        total = p + pf
        for view in (x, flipped):
            pad = torch.nn.functional.pad(view, (2, 2, 2, 2), mode='reflect')
            for dy, dx in ((2, 0), (-2, 0), (0, 2), (0, -2)):
                total = total + model(pad[:, :, 2+dy:34+dy, 2+dx:34+dx]).float().softmax(1)
        result['shift_prob'].append((total / 10).cpu().numpy())
        if labeled:
            labels.append(y.numpy())
    return {k: np.concatenate(v) for k, v in result.items()}, np.concatenate(labels) if labeled else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    run = root / 'submissions/Abhinav_Run-G'
    args.data_dir = args.data_dir.expanduser().resolve()
    assert torch.cuda.is_available(), 'Use an HPC GPU session'
    device = torch.device('cuda')
    config = json.loads((run/'twin/config.json').read_text())
    assert config['val_size'] == 5000 and config['split_seed'] == 0
    hashes = {name: digest(run/name) for name in ('model.pt', 'twin/model.pt')}
    args.out.mkdir(parents=True, exist_ok=False)
    audit_data(args.data_dir, 'train')
    model = load_model(run/'twin/model.pt', device)
    assert count_params(model) == 4980392
    report = {'checkpoint_sha256': hashes, 'torch_version': torch.__version__, 'validation': {},
              'caveat': 'Reused validation split; fixed screen is not proof of improved hidden-test accuracy.'}
    scores = {}
    for name, corruption in SUITE.items():
        loader = build_val_loader(args.data_dir, val_size=5000, seed=0, corruption=corruption,
                                  batch_size=256, num_workers=2)
        probs, labels = predict(model, loader, device, True)
        np.savez_compressed(args.out/f'{name}_predictions.npz', labels=labels, **probs)
        baseline = probs['flip_prob'].argmax(1)
        rows = {}
        for mode, values in probs.items():
            pred = values.argmax(1)
            rescued = int(((baseline != labels) & (pred == labels)).sum())
            harmed = int(((baseline == labels) & (pred != labels)).sum())
            rows[mode] = {'accuracy': float((pred == labels).mean()), 'rescued': rescued,
                          'harmed': harmed, 'net_correct': rescued-harmed,
                          'changed': int((pred != baseline).sum())}
        report['validation'][name] = rows
        scores[name] = rows
        print(name, json.dumps(rows, indent=2), flush=True)
        del loader
    # Predeclared practical screen: >=5 more clean examples, no worse mean on the three
    # sensitivity checks. These thresholds are not a statistical significance claim.
    eligible = [m for m in MODES if m != 'flip_prob' and scores['clean'][m]['net_correct'] >= 5
                and sum(scores[s][m]['net_correct'] for s in SUITE if s != 'clean') >= 0]
    selected = max(eligible, key=lambda m: scores['clean'][m]['accuracy']) if eligible else None
    report['selected_mode'] = selected
    report['candidate_file'] = None
    if selected:
        del model
        torch.cuda.empty_cache()
        audit_data(args.data_dir, 'test')
        model = load_model(run/'model.pt', device)
        assert count_params(model) == 4980392
        loader, ids = build_test_loader(args.data_dir, batch_size=500, num_workers=2)
        probs, _ = predict(model, loader, device, False)
        np.savez_compressed(args.out/'test_predictions.npz', ids=ids, **probs)
        original = pd.read_csv(run/'submission.csv')
        assert np.array_equal(ids, original.id.to_numpy())
        mismatches = int((probs['flip_prob'].argmax(1) != original.label.to_numpy()).sum())
        report['baseline_mismatches'] = mismatches
        (args.out/'tta_report.json').write_text(json.dumps(report, indent=2))
        assert mismatches == 0, 'Original submission not reproduced. Do not submit.'
        candidate = args.out/f'candidate_{selected}.csv'
        write_submission(ids, probs[selected].argmax(1), candidate, args.data_dir/'sample_submission.csv')
        report['candidate_file'] = str(candidate)
        report['test_changed'] = int((probs[selected].argmax(1) != original.label.to_numpy()).sum())
    for name, before in hashes.items():
        assert digest(run/name) == before
    (args.out/'tta_report.json').write_text(json.dumps(report, indent=2))
    print('Candidate:', report['candidate_file'] or 'NONE: no method passed validation', flush=True)
    return report

if __name__ == '__main__':
    main()
