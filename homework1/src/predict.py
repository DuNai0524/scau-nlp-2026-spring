"""Generate predictions on test set and create submission file."""

import os

import numpy as np
import pandas as pd

from .model import SoftmaxRegression


def predict_and_save(
    model: SoftmaxRegression,
    X_test: np.ndarray,
    test_df: pd.DataFrame,
    label_encoder,
    label_to_c_numerical: dict[str, int],
    output_path: str = "submission.csv",
) -> str:
    """Predict on test set and save submission CSV."""
    preds = model.predict(X_test)
    categories = label_encoder.inverse_transform(preds)
    c_numerical = [label_to_c_numerical[label] for label in categories]

    submission = pd.DataFrame({
        "ID": test_df["ID"],
        "c_numerical": c_numerical,
    })
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path} ({len(submission)} rows)")
    return output_path
