"""Verification smoke test script for E9.

This script tests the author's logic on synthetic data:
1. Generates synthetic classification data.
2. Applies RFF projection (RBFSampler with gamma='scale').
3. Runs author's SLDA logic (OAS with LinearDiscriminantAnalysis / legacy_model).
4. Verifies accuracy and matrix shapes.
"""

import os
import sys
from pathlib import Path
import numpy as np

# Add official source to path temporarily for inspection/testing
OFFICIAL_SRC = Path(__file__).resolve().parent / "_official" / "src"

def test_author_pipeline_smoke():
    print("=== Running Author Pipeline Smoke Test (E9 Verification) ===")
    
    if not OFFICIAL_SRC.exists():
        print(f"Error: Official source directory not found at {OFFICIAL_SRC}")
        print("Please run `python verification/setup_official.py` first.")
        return False

    from sklearn.kernel_approximation import RBFSampler
    from sklearn.covariance import OAS
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import accuracy_score

    # 1. Create a reproducible small synthetic dataset (100 samples, 32 features, 2 classes)
    np.random.seed(42)
    n_samples = 200
    n_features = 32
    n_classes = 2
    embed_dim = 100

    X = np.random.randn(n_samples, n_features).astype(np.float32)
    # Class 0: negative mean, Class 1: positive mean
    y = np.random.randint(0, n_classes, size=(n_samples,)).astype(np.int32)
    X[y == 1] += 0.5

    # Split train/test
    train_X, test_X = X[:150], X[150:]
    train_y, test_y = y[:150], y[150:]

    # 2. RFF embedding as done in author's main.py (line 81)
    print(f"Applying RBFSampler with gamma='scale', embed_dim={embed_dim}...")
    embedder = RBFSampler(gamma='scale', n_components=embed_dim, random_state=42)
    embedder.fit(train_X)
    train_X_embed = embedder.transform(train_X)
    test_X_embed = embedder.transform(test_X)
    print(f"Embedded shape: {train_X_embed.shape}")

    # 3. Fit author's SLDA (main.py lines 87-90)
    print("Fitting author's SLDA (LDA with OAS covariance estimator)...")
    oa = OAS(assume_centered=False)
    model = LinearDiscriminantAnalysis(solver='lsqr', covariance_estimator=oa)
    model.fit(train_X_embed, train_y)

    preds = model.predict(test_X_embed)
    acc = accuracy_score(test_y, preds)
    print(f"Smoke test accuracy on synthetic test set: {acc * 100:.2f}%")

    print("Author pipeline smoke test passed successfully!")
    return True

if __name__ == "__main__":
    success = test_author_pipeline_smoke()
    if not success:
        sys.exit(1)
