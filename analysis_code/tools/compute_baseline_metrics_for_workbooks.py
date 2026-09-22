"""Compute stage-by-region No-change and PLR baselines for the summary workbooks.

This helper mirrors the sequence construction used by the existing Bern and
GRAPE LSTM evaluators.  It is intentionally post-hoc: no model is retrained
and no split assignment is changed.  Both baselines are evaluated on the same
held-out patient-eye windows as the LSTM summaries.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from notebooks2.run_grape_cluster_lstm_iop_stage_region import (  # noqa: E402
    _build_sequences,
    _read_grape_aligned,
)
from notebooks2.stratify_bern_region_lstm import (  # noqa: E402
    _build_sequence_data,
    _build_test_eye_strata,
    _load_bern_frame,
)


REGION_ORDER = [
    "Overall",
    "Supero_Nasal",
    "Supero_Temporal",
    "Macular",
    "Infero_Nasal",
    "Infero_Temporal",
    "Temporal",
]
REGION_MAPPING = {
    "Supero_Nasal": [1, 14, 15, 21, 24, 28, 34, 38, 42, 46, 50, 53, 56],
    "Supero_Temporal": [0, 5, 20, 30, 33, 37, 45, 49, 55],
    "Macular": [4, 7, 10, 13, 16, 17, 18, 19, 32],
    "Infero_Nasal": [2, 11, 12, 22, 25, 29, 35, 39, 43, 47, 51, 54, 57],
    "Infero_Temporal": [3, 8, 23, 31, 36, 40, 48, 52, 58],
    "Temporal": [6, 9, 26, 27, 41, 44],
}
STAGE_ORDER = ["All", "Early", "Moderate", "Severe"]


def _to_db(sequence_scaled: np.ndarray, scaler) -> np.ndarray:
    return np.asarray(sequence_scaled, dtype=np.float32) * scaler.scale_ + scaler.mean_


def _last_two_plr(sequence_db: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Pointwise linear extrapolation using the last two visits.

    The horizon is the last observed interval, matching the existing GRAPE
    baseline experiment.  If a degenerate interval is encountered, the
    no-change prediction is used for that sample.
    """

    sequence_db = np.asarray(sequence_db, dtype=np.float32)
    times = np.asarray(times, dtype=np.float32)
    if sequence_db.shape[0] < 2:
        return sequence_db[-1].copy()
    dt = max(float(times[-1] - times[-2]), 1e-6)
    if not np.isfinite(dt) or dt <= 0:
        return sequence_db[-1].copy()
    # Keep the arithmetic in the same form as the notebook baseline
    # (slope times the extrapolated interval), rather than algebraically
    # simplifying it.  This preserves the exact floating-point convention
    # used by the previously reported GRAPE baseline values.
    slope = (sequence_db[-1] - sequence_db[-2]) / dt
    future_time = float(times[-1] + (times[-1] - times[-2]))
    return sequence_db[-1] + slope * (future_time - float(times[-1]))


def _evaluate(data: dict, strata, dataset: str) -> dict:
    test = data["test"]
    scaler = data["scaler"]
    eyes = np.asarray(test["eyes"], dtype=str)
    targets = np.asarray(test["y_raw"], dtype=np.float32)

    last_predictions = []
    plr_predictions = []
    for sequence_scaled, times in zip(test["x"], test["t"]):
        sequence_db = _to_db(sequence_scaled, scaler)
        last_predictions.append(sequence_db[-1].copy())
        plr_predictions.append(_last_two_plr(sequence_db, times))
    last_predictions = np.asarray(last_predictions, dtype=np.float32)
    plr_predictions = np.asarray(plr_predictions, dtype=np.float32)

    stage_by_eye = dict(zip(strata["eye_id"], strata["severity"]))
    point_sets = {"Overall": np.arange(59, dtype=int)}
    point_sets.update({name: np.asarray(idx, dtype=int) for name, idx in REGION_MAPPING.items()})

    result = {}
    for stage in STAGE_ORDER:
        stage_mask = (
            np.ones(len(eyes), dtype=bool)
            if stage == "All"
            else np.asarray([stage_by_eye[eye] == stage for eye in eyes], dtype=bool)
        )
        for region in REGION_ORDER:
            points = point_sets[region]
            for baseline_name, predictions in (
                ("No-change classifier", last_predictions),
                ("PLR", plr_predictions),
            ):
                sample_errors = np.abs(predictions[stage_mask][:, points] - targets[stage_mask][:, points]).mean(axis=1)
                key = f"{dataset}|{stage}|{region}|{baseline_name}"
                if len(sample_errors) == 0:
                    result[key] = {
                        "dataset": dataset,
                        "stage": stage,
                        "region": region,
                        "baseline": baseline_name,
                        "n_patient_eyes": 0,
                        "n_prediction_samples": 0,
                        "sample_mae_db": None,
                        "sample_sd_db": None,
                    }
                else:
                    result[key] = {
                        "dataset": dataset,
                        "stage": stage,
                        "region": region,
                        "baseline": baseline_name,
                        "n_patient_eyes": int(np.unique(eyes[stage_mask]).size),
                        "n_prediction_samples": int(len(sample_errors)),
                        "sample_mae_db": float(sample_errors.mean()),
                        "sample_sd_db": float(sample_errors.std(ddof=1)) if len(sample_errors) > 1 else None,
                    }
    return result


def main() -> None:
    bern_data = _build_sequence_data(_load_bern_frame())
    bern_strata = _build_test_eye_strata(bern_data)

    grape_data = _build_sequences(_read_grape_aligned())
    grape_strata = grape_data["strata"]

    payload = {
        "protocol": {
            "history_lengths": [2, 3, 4],
            "window_mode": "consecutive",
            "split_seed": 42,
            "target_time": "one equal-step horizon using the last observed interval",
            "no_change": "predict the last observed VF for every point",
            "plr": "pointwise linear extrapolation in dB using the last two visits",
            "metric": "sample-weighted regional MAE and sample SD in dB",
        },
        "test_counts": {
            "Bern": {
                "patient_eyes": int(np.unique(bern_data["test"]["eyes"]).size),
                "prediction_samples": int(len(bern_data["test"]["y_raw"])),
            },
            "GRAPE": {
                "patient_eyes": int(np.unique(grape_data["test"]["eyes"]).size),
                "prediction_samples": int(len(grape_data["test"]["y_raw"])),
            },
        },
        "metrics": {
            **_evaluate(bern_data, bern_strata, "Bern"),
            **_evaluate(grape_data, grape_strata, "GRAPE"),
        },
    }
    out = Path("/private/tmp/baseline_metrics_for_workbooks.json")
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(out)
    print(json.dumps(payload["test_counts"], indent=2))


if __name__ == "__main__":
    main()
