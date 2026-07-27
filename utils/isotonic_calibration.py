"""Post-hoc isotonic calibration utilities for multiclass probabilities."""

import numpy as np
from sklearn.isotonic import IsotonicRegression


class OneVsRestIsotonicCalibrator:
    """Calibrate each class probability independently, then renormalize.

    The calibrator is deliberately fitted only with labelled in-distribution
    examples.  This is the usual post-hoc classification-calibration setting
    and avoids using OOD validation labels to tune a test-time detector.
    """

    def __init__(self):
        self.models = []
        self.num_classes = None

    def fit(self, probabilities, targets):
        probabilities = np.asarray(probabilities, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.int64)
        if probabilities.ndim != 2:
            raise ValueError("probabilities must have shape [num_examples, num_classes]")
        if probabilities.shape[0] != targets.shape[0]:
            raise ValueError("probabilities and targets must contain the same number of examples")

        self.num_classes = probabilities.shape[1]
        self.models = []
        for class_index in range(self.num_classes):
            binary_targets = (targets == class_index).astype(np.int64)
            # IsotonicRegression cannot fit a constant target.  Retaining its
            # empirical constant is the appropriate calibrated prediction.
            if binary_targets.min() == binary_targets.max():
                self.models.append(float(binary_targets[0]))
                continue
            model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            model.fit(probabilities[:, class_index], binary_targets)
            self.models.append(model)
        return self

    def predict_proba(self, probabilities):
        if self.num_classes is None:
            raise RuntimeError("Call fit before predict_proba")
        probabilities = np.asarray(probabilities, dtype=np.float64)
        if probabilities.ndim != 2 or probabilities.shape[1] != self.num_classes:
            raise ValueError("probabilities have an incompatible shape")

        calibrated = np.empty_like(probabilities, dtype=np.float64)
        for class_index, model in enumerate(self.models):
            if isinstance(model, float):
                calibrated[:, class_index] = model
            else:
                calibrated[:, class_index] = model.predict(probabilities[:, class_index])

        row_sums = calibrated.sum(axis=1, keepdims=True)
        zero_rows = row_sums[:, 0] <= 1e-12
        if np.any(zero_rows):
            calibrated[zero_rows] = probabilities[zero_rows]
            row_sums = calibrated.sum(axis=1, keepdims=True)
        return calibrated / np.clip(row_sums, 1e-12, None)


def expected_calibration_error(probabilities, targets, num_bins=15):
    """Top-label expected calibration error for multiclass probabilities."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correctness = (predictions == targets).astype(np.float64)

    ece = 0.0
    for lower in np.linspace(0.0, 1.0, num_bins, endpoint=False):
        upper = lower + 1.0 / num_bins
        in_bin = (confidences >= lower) & (
            confidences <= upper if upper == 1.0 else confidences < upper
        )
        if np.any(in_bin):
            ece += np.abs(correctness[in_bin].mean() - confidences[in_bin].mean()) * in_bin.mean()
    return float(ece)
