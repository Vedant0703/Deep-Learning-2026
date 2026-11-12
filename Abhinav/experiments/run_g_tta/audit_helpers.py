from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from PIL import Image
from src.data import load_split

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def audit_data(root, split):
    # Check the complete cache, including image content, against the authoritative CSV/PNGs.
    images, labels, ids = load_split(root, split)
    frame = pd.read_csv(root / ('train_labels.csv' if split == 'train' else 'test.csv'))
    expected_n = 50000 if split == 'train' else 2000
    assert len(frame) == expected_n and frame.id.is_unique
    assert images.shape == (expected_n, 32, 32, 3) and images.dtype == np.uint8
    assert np.array_equal(ids, frame.id.to_numpy()), 'Cache ID order mismatch; move stale cache aside and rerun'
    if split == 'train':
        assert np.array_equal(labels, frame.label.to_numpy()), 'Cache labels mismatch'
    for i, filename in enumerate(frame.filename):
        with Image.open(root / f'{split}_images' / filename) as image:
            assert np.array_equal(images[i], np.asarray(image.convert('RGB'))), f'Cache mismatch at {split} row {i}'
    print(f'Full {split} CSV/PNG/cache audit passed ({expected_n} images)', flush=True)
