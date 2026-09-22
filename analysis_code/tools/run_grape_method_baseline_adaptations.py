#!/usr/bin/env python
"""Adapted published-method baselines on the native GRAPE 59-point task.

This script is deliberately separate from the thesis and from the main GRAPE
notebook.  It reproduces the current consecutive k=2,3,4 GRAPE protocol
(including the fixed patient-eye split and train-target standardisation), then
adds two external-method baselines:

* Marta-style AE forecasting: a 59-dimensional fully-connected AE with
  sinusoidal time encodings, initialised by one-to-one AE pretraining and then
  trained to forecast the next VF from the two most recent historical VFs.
* Nature-style VAE latent forecasting: a 59-dimensional vector VAE followed by
  per-latent linear/quadratic temporal regression and decoding.

The Marta thesis uses Bern/Octopus G-pattern visual fields, which are aligned
with the native 59-point representation used here.  The Nature paper uses an
image-like 52-point HFA 24-2 representation.  Both implementations are
therefore labelled as study-level adaptations rather than exact replications
of the original end-to-end protocols.

The script never uses test outcomes for model selection.  The existing
patient-eye split is reconstructed exactly from the notebook: pooled
consecutive k=2,3,4 samples, RNG seed 42, 85% train/validation versus 15%
test, then 10% of the train/validation eyes for validation.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BERN_PATH = PROJECT_ROOT / "data" / "Bern" / "bern_59_master_dataset_splitIDs.csv"
GRAPE_PATH = PROJECT_ROOT / "GRAPE" / "VF and clinical information.xlsx"
COORD_PATH = PROJECT_ROOT / "GRAPE" / "GRAPE coordinate.xlsx"

REGION_ORDER = [
    "Overall",
    "Supero_Nasal",
    "Supero_Temporal",
    "Macular",
    "Infero_Nasal",
    "Infero_Temporal",
    "Temporal",
]

# Zero-based indices in the Bern-aligned 59-point order used throughout the
# existing GRAPE notebook and thesis tables.
REGION_MAPPING = {
    "Supero_Nasal": [1, 14, 15, 21, 24, 28, 34, 38, 42, 46, 50, 53, 56],
    "Supero_Temporal": [0, 5, 20, 30, 33, 37, 45, 49, 55],
    "Macular": [4, 7, 10, 13, 16, 17, 18, 19, 32],
    "Infero_Nasal": [2, 11, 12, 22, 25, 29, 35, 39, 43, 47, 51, 54, 57],
    "Infero_Temporal": [3, 8, 23, 31, 36, 40, 48, 52, 58],
    "Temporal": [6, 9, 26, 27, 41, 44],
}


@dataclass
class RunConfig:
    seeds: tuple[int, ...] = (0,)
    latent_dims: tuple[int, ...] = (8, 16)
    ae_pretrain_epochs: int = 40
    marta_epochs: int = 60
    vae_epochs: int = 60
    batch_size: int = 64
    ae_lr: float = 1e-3
    marta_lr: float = 1e-4
    vae_lr: float = 1e-3
    mmd_weight: float = 0.1
    use_actual_target_time: bool = True
    output_dir: str = "results/paper_method_baselines_grape"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fit_bern_norm_models() -> dict[int, LinearRegression]:
    bern = pd.read_csv(BERN_PATH, low_memory=False)
    age_col = "age_at_exam"
    if age_col not in bern.columns:
        raise KeyError(f"Missing {age_col!r} in {BERN_PATH}")

    models: dict[int, LinearRegression] = {}
    for i in range(1, 60):
        col = f"Norm_{i}"
        if col not in bern.columns:
            raise KeyError(f"Missing {col!r} in {BERN_PATH}")
        sub = bern[[age_col, col]].copy()
        sub[age_col] = pd.to_numeric(sub[age_col], errors="coerce")
        sub[col] = pd.to_numeric(sub[col], errors="coerce")
        sub = sub.dropna()
        model = LinearRegression()
        model.fit(sub[[age_col]].to_numpy(), sub[col].to_numpy())
        models[i] = model
    return models


def _flatten_baseline_columns(columns: Iterable[object]) -> list[str]:
    return [
        "_".join(
            str(x).strip()
            for x in col
            if str(x).strip() and str(x).strip().lower() != "nan"
        )
        for col in columns
    ]


def load_aligned_grape() -> tuple[pd.DataFrame, list[str]]:
    """Recreate the native Bern-aligned GRAPE TD table from the notebook."""

    norm_models = fit_bern_norm_models()

    baseline = pd.read_excel(GRAPE_PATH, sheet_name="Baseline", header=[0, 1])
    baseline.columns = _flatten_baseline_columns(baseline.columns)
    baseline = baseline.rename(
        columns={
            "Subject Number_Unnamed: 0_level_1": "Subject Number",
            "Laterality_Unnamed: 1_level_1": "Laterality",
            "Age_Unnamed: 2_level_1": "Age",
            "Gender_Unnamed: 3_level_1": "Gender",
            "IOP_Unnamed: 4_level_1": "IOP",
            "CCT_Unnamed: 5_level_1": "CCT",
            "Total Visits_Unnamed: 6_level_1": "Total Visits",
            "Category of Glaucoma_Unnamed: 10_level_1": "Category of Glaucoma",
            "Corresponding CFP_Unnamed: 16_level_1": "Corresponding CFP",
            "Acquisition Device_Unnamed: 17_level_1": "Acquisition Device",
            "Resolusion_Unnamed: 18_level_1": "Resolution",
        }
    )

    followup = pd.read_excel(GRAPE_PATH, sheet_name="Follow-up", header=1)
    followup.columns = [str(c).strip() for c in followup.columns]
    followup = followup.rename(
        columns={
            "Unnamed: 0": "Subject Number",
            "Unnamed: 1": "Laterality",
            "Unnamed: 2": "Visit Number",
            "Unnamed: 3": "Interval Years",
            "Unnamed: 4": "IOP",
            "Unnamed: 5": "Corresponding CFP",
            "Unnamed: 6": "Acquisition Device",
            "Unnamed: 7": "Resolution",
        }
    )
    numeric_vf_cols = sorted(
        [c for c in followup.columns if str(c).isdigit()], key=lambda c: int(c)
    )
    followup = followup.rename({c: f"VF_{c}" for c in numeric_vf_cols}, axis=1)

    age_lookup = baseline[["Subject Number", "Laterality", "Age"]].drop_duplicates()
    df = followup.merge(age_lookup, on=["Subject Number", "Laterality"], how="left")

    blindspot_cols = ["VF_21", "VF_32"]
    missing = [c for c in blindspot_cols if c not in df.columns]
    if missing:
        raise KeyError(f"GRAPE blind-spot columns not found: {missing}")
    df = df.drop(columns=blindspot_cols).copy()
    remaining_vf_cols = sorted(
        [c for c in df.columns if c.startswith("VF_")],
        key=lambda c: int(c.split("_")[1]),
    )
    df[remaining_vf_cols] = (
        df[remaining_vf_cols]
        .apply(pd.to_numeric, errors="coerce")
        .replace(-1, np.nan)
    )

    coord = pd.read_excel(COORD_PATH)
    coord = coord[~coord["ids Grape"].isin([21, 32])].copy()
    coord["bern_idx"] = pd.to_numeric(coord["Ids G"], errors="raise") + 1
    coord = coord.sort_values("bern_idx").reset_index(drop=True)
    if len(coord) != 59 or coord["bern_idx"].tolist() != list(range(1, 60)):
        raise ValueError("Unexpected GRAPE coordinate mapping")

    sens_data = {}
    for grape_id, bern_id in zip(coord["ids Grape"], coord["bern_idx"]):
        src = f"VF_{int(grape_id)}"
        if src not in df.columns:
            raise KeyError(f"Missing mapped source column {src}")
        sens_data[f"Sens_{int(bern_id)}"] = df[src].to_numpy()
    df = pd.concat([df, pd.DataFrame(sens_data, index=df.index)], axis=1)

    age_values = pd.to_numeric(df["Age"], errors="coerce")
    pred_age = age_values.to_numpy(dtype=float).reshape(-1, 1)
    # LinearRegression cannot predict NaN ages.  Age is complete for the rows
    # used by the current notebook, but keep this explicit guard.
    if np.any(~np.isfinite(pred_age)):
        raise ValueError("Missing baseline age after GRAPE merge")
    norm_data = {}
    td_data = {}
    for i in range(1, 60):
        norm_values = norm_models[i].predict(pred_age)
        norm_data[f"Norm_{i}"] = norm_values
        td_data[f"TD_{i}"] = pd.to_numeric(df[f"Sens_{i}"], errors="coerce").to_numpy() - norm_values
    df = pd.concat(
        [df, pd.DataFrame(norm_data, index=df.index), pd.DataFrame(td_data, index=df.index)],
        axis=1,
    )

    df["eye_id"] = (
        df["Subject Number"].astype(str).str.strip()
        + "_"
        + df["Laterality"].astype(str).str.strip()
    )
    df["patient_uid"] = df["Subject Number"].astype(str).str.strip()
    df["Eye"] = df["Laterality"].astype(str).str.strip()
    df["VisitN"] = pd.to_numeric(df["Visit Number"], errors="coerce")
    df["Time_from_Baseline"] = pd.to_numeric(df["Interval Years"], errors="coerce")
    df["Age"] = age_values
    df = df.dropna(subset=["eye_id", "VisitN", "Time_from_Baseline", "Age"])
    df["VisitN"] = df["VisitN"].astype(int)
    df["Time_from_Baseline"] = df["Time_from_Baseline"].astype(float)
    td_cols = [f"TD_{i}" for i in range(1, 60)]
    df[td_cols] = df[td_cols].apply(pd.to_numeric, errors="coerce")
    return df.sort_values(["eye_id", "VisitN"]).reset_index(drop=True), td_cols


def build_samples(df: pd.DataFrame, td_cols: list[str], k: int) -> list[dict]:
    samples: list[dict] = []
    for eye_id, group in df.groupby("eye_id", sort=True):
        group = group.sort_values(["Time_from_Baseline", "VisitN"]).reset_index(drop=True)
        if len(group) < k + 1:
            continue
        for start in range(0, len(group) - k):
            hist = group.iloc[start : start + k]
            target = group.iloc[start + k]
            hist_td = hist[td_cols].to_numpy(dtype=np.float32)
            target_td = target[td_cols].to_numpy(dtype=np.float32)
            hist_times = hist["Time_from_Baseline"].to_numpy(dtype=np.float32)
            target_time = float(target["Time_from_Baseline"])
            hist_ages = hist["Age"].to_numpy(dtype=np.float32) + hist_times
            target_age = float(target["Age"] + target_time)
            if not (
                np.all(np.isfinite(hist_td))
                and np.all(np.isfinite(target_td))
                and np.all(np.isfinite(hist_times))
                and np.isfinite(target_time)
                and np.all(np.diff(hist_times) >= 0)
            ):
                continue
            samples.append(
                {
                    "eye_id": eye_id,
                    "k": k,
                    "X": hist_td,
                    "Y": target_td,
                    "times": hist_times,
                    "target_time": target_time,
                    "ages": hist_ages,
                    "target_age": target_age,
                    "visit_numbers": hist["VisitN"].astype(int).tolist()
                    + [int(target["VisitN"])],
                }
            )
    return samples


def make_fixed_split(samples: list[dict]) -> dict[str, list[dict]]:
    unique_eyes = np.unique(np.asarray([s["eye_id"] for s in samples], dtype=str))
    rng = np.random.default_rng(42)
    rng.shuffle(unique_eyes)
    n_trainval = int(0.85 * len(unique_eyes))
    trainval_eyes = unique_eyes[:n_trainval]
    test_eyes = unique_eyes[n_trainval:]
    rng.shuffle(trainval_eyes)
    n_val = int(0.10 * len(trainval_eyes))
    val_eyes = set(trainval_eyes[:n_val].tolist())
    train_eyes = set(trainval_eyes[n_val:].tolist())
    test_eyes_set = set(test_eyes.tolist())
    out = {
        "train": [s for s in samples if s["eye_id"] in train_eyes],
        "val": [s for s in samples if s["eye_id"] in val_eyes],
        "test": [s for s in samples if s["eye_id"] in test_eyes_set],
    }
    return out


def split_counts(split: dict[str, list[dict]]) -> dict[str, object]:
    return {
        name: {
            "samples": len(items),
            "eyes": len({s["eye_id"] for s in items}),
            "by_k": {
                str(k): sum(int(s["k"] == k) for s in items) for k in (2, 3, 4)
            },
        }
        for name, items in split.items()
    }


def fit_scaler(split: dict[str, list[dict]]) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(np.stack([s["Y"] for s in split["train"]]))
    return scaler


def standardize_samples(items: Sequence[dict], scaler: StandardScaler) -> list[dict]:
    result = []
    for s in items:
        result.append(
            {
                **s,
                "X_s": scaler.transform(s["X"]).astype(np.float32),
                "Y_s": scaler.transform(s["Y"][None, :])[0].astype(np.float32),
            }
        )
    return result


def make_visit_matrix(df: pd.DataFrame, td_cols: list[str], eyes: set[str]) -> np.ndarray:
    sub = df[df["eye_id"].isin(eyes)][td_cols].dropna()
    if sub.empty:
        raise ValueError("No complete training visit vectors available")
    return sub.to_numpy(dtype=np.float32)


def sin_time_encoding(t: torch.Tensor, dim: int) -> torch.Tensor:
    if dim % 2:
        raise ValueError("Time-encoding dimension must be even")
    was_vector = t.ndim == 1
    if t.ndim == 1:
        t = t[:, None]
    i = torch.arange(dim // 2, device=t.device, dtype=t.dtype)
    angles = t[..., None] / (10000.0 ** (2 * i / dim))
    pe = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    return pe.squeeze(1) if was_vector else pe


class MartaEncoder(nn.Module):
    def __init__(self, input_dim: int = 59, latent_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MartaDecoder(nn.Module):
    def __init__(self, latent_dim: int = 32, output_dim: int = 59):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class MartaCore(nn.Module):
    def __init__(self, input_dim: int = 59, latent_dim: int = 32):
        super().__init__()
        self.encoder = MartaEncoder(input_dim, latent_dim)
        self.decoder = MartaDecoder(latent_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class MartaForecast(nn.Module):
    """Marta thesis architecture implemented on the native 59-point task."""

    def __init__(self, input_dim: int = 59, latent_dim: int = 32):
        super().__init__()
        self.encoder = MartaEncoder(input_dim, latent_dim)
        self.decoder = MartaDecoder(latent_dim, input_dim)
        self.fc = nn.Sequential(
            nn.Linear(latent_dim * 3, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        self.latent_dim = latent_dim

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        t1: torch.Tensor,
        t2: torch.Tensor,
        target_time: torch.Tensor,
    ) -> torch.Tensor:
        z1 = self.encoder(x1) + sin_time_encoding(t1, self.latent_dim)
        z2 = self.encoder(x2) + sin_time_encoding(t2, self.latent_dim)
        target_pe = sin_time_encoding(target_time, self.latent_dim)
        z = self.fc(torch.cat([z1, z2, target_pe], dim=-1))
        return self.decoder(z)


class VectorVAE(nn.Module):
    def __init__(self, input_dim: int = 59, latent_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.mu = nn.Linear(64, latent_dim)
        self.logvar = nn.Linear(64, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )
        self.latent_dim = latent_dim

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def rbf_mmd(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Small-batch RBF MMD used as the generalized-VAE regularizer."""

    def kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        d2 = torch.cdist(a, b).pow(2)
        # The median bandwidth is detached so it does not create unstable
        # gradients through a batch statistic.
        positive = d2.detach()[d2.detach() > 0]
        bandwidth = torch.median(positive) if positive.numel() else torch.tensor(1.0, device=a.device)
        bandwidth = bandwidth.clamp_min(1e-4)
        return torch.exp(-d2 / (2.0 * bandwidth))

    return kernel(x, x).mean() + kernel(y, y).mean() - 2.0 * kernel(x, y).mean()


def make_marta_tensors(items: Sequence[dict], use_actual_target_time: bool) -> tuple[torch.Tensor, ...]:
    x1 = np.stack([s["X_s"][-2] for s in items]).astype(np.float32)
    x2 = np.stack([s["X_s"][-1] for s in items]).astype(np.float32)
    times = np.stack([s["ages"][-2:] for s in items]).astype(np.float32)
    if use_actual_target_time:
        target_t = np.asarray([s["target_age"] for s in items], dtype=np.float32)
    else:
        # A fair no-future-time variant: extrapolate by the last observed gap.
        target_t = np.asarray(
            [s["ages"][-1] + (s["times"][-1] - s["times"][-2]) for s in items],
            dtype=np.float32,
        )
    y = np.stack([s["Y_s"] for s in items]).astype(np.float32)
    return tuple(torch.from_numpy(a) for a in (x1, x2, times[:, 0], times[:, 1], target_t, y))


def mae_rows(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(pred - true), axis=1)


def summarise_predictions(
    pred: np.ndarray,
    true: np.ndarray,
    model: str,
    seed: int,
    split_name: str,
) -> pd.DataFrame:
    error = np.abs(pred - true)
    rows = []
    for region in REGION_ORDER:
        idx = np.arange(59) if region == "Overall" else np.asarray(REGION_MAPPING[region])
        sample_mae = error[:, idx].mean(axis=1)
        rows.append(
            {
                "Model": model,
                "seed": int(seed),
                "split": split_name,
                "Region": region,
                "MAE_mean_db": float(sample_mae.mean()),
                "MAE_SD_db": float(sample_mae.std(ddof=1)) if len(sample_mae) > 1 else 0.0,
                "n_samples": int(len(sample_mae)),
            }
        )
    return pd.DataFrame(rows)


def inverse_predictions(pred_s: np.ndarray, true_s: np.ndarray, scaler: StandardScaler) -> tuple[np.ndarray, np.ndarray]:
    return scaler.inverse_transform(pred_s), scaler.inverse_transform(true_s)


def train_marta(
    split_s: dict[str, list[dict]],
    train_visits_s: np.ndarray,
    val_visits_s: np.ndarray,
    scaler: StandardScaler,
    config: RunConfig,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, float]]:
    set_seed(seed + 1000)
    visit_train = DataLoader(
        TensorDataset(torch.from_numpy(train_visits_s)),
        batch_size=config.batch_size,
        shuffle=True,
    )
    visit_val = DataLoader(
        TensorDataset(torch.from_numpy(val_visits_s)),
        batch_size=config.batch_size,
        shuffle=False,
    )
    core = MartaCore().to(device)
    opt = torch.optim.Adam(core.parameters(), lr=config.ae_lr)
    criterion = nn.MSELoss()
    best_val = float("inf")
    best_core_state = copy.deepcopy(core.state_dict())
    for epoch in range(1, config.ae_pretrain_epochs + 1):
        core.train()
        for (xb,) in visit_train:
            xb = xb.to(device)
            opt.zero_grad()
            loss = criterion(core(xb), xb)
            loss.backward()
            opt.step()
        core.eval()
        val_loss = 0.0
        n = 0
        with torch.no_grad():
            for (xb,) in visit_val:
                xb = xb.to(device)
                val_loss += float(criterion(core(xb), xb).item()) * len(xb)
                n += len(xb)
        val_loss /= max(n, 1)
        if val_loss < best_val:
            best_val = val_loss
            best_core_state = copy.deepcopy(core.state_dict())

    model = MartaForecast().to(device)
    model.encoder.load_state_dict({k.removeprefix("encoder."): v for k, v in best_core_state.items() if k.startswith("encoder.")})
    model.decoder.load_state_dict({k.removeprefix("decoder."): v for k, v in best_core_state.items() if k.startswith("decoder.")})
    opt = torch.optim.Adam(model.parameters(), lr=config.marta_lr)
    train_t = make_marta_tensors(split_s["train"], config.use_actual_target_time)
    val_t = make_marta_tensors(split_s["val"], config.use_actual_target_time)
    train_loader = DataLoader(TensorDataset(*train_t), batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(*val_t), batch_size=config.batch_size, shuffle=False)
    std = torch.from_numpy(scaler.scale_.astype(np.float32)).to(device)
    best_val_mae = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    for epoch in range(1, config.marta_epochs + 1):
        model.train()
        for x1, x2, t1, t2, tt, y in train_loader:
            x1, x2, t1, t2, tt, y = (v.to(device) for v in (x1, x2, t1, t2, tt, y))
            opt.zero_grad()
            loss = criterion(model(x1, x2, t1, t2, tt), y)
            loss.backward()
            opt.step()
        model.eval()
        val_error = []
        with torch.no_grad():
            for x1, x2, t1, t2, tt, y in val_loader:
                x1, x2, t1, t2, tt, y = (v.to(device) for v in (x1, x2, t1, t2, tt, y))
                p = model(x1, x2, t1, t2, tt)
                val_error.append((torch.abs(p - y) * std).mean(dim=1).cpu().numpy())
        val_mae = float(np.concatenate(val_error).mean())
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
    model.load_state_dict(best_state)
    test_t = make_marta_tensors(split_s["test"], config.use_actual_target_time)
    test_loader = DataLoader(TensorDataset(*test_t), batch_size=config.batch_size, shuffle=False)
    pred_s = []
    true_s = []
    model.eval()
    with torch.no_grad():
        for x1, x2, t1, t2, tt, y in test_loader:
            x1, x2, t1, t2, tt = (v.to(device) for v in (x1, x2, t1, t2, tt))
            pred_s.append(model(x1, x2, t1, t2, tt).cpu().numpy())
            true_s.append(y.numpy())
    return np.concatenate(pred_s), {"best_val_mae_db": best_val_mae, "best_epoch": best_epoch, "ae_val_mse": best_val}


def encode_means(model: VectorVAE, items: Sequence[dict], device: torch.device) -> list[np.ndarray]:
    model.eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for s in items:
            x = torch.from_numpy(s["X_s"]).to(device)
            mu, _ = model.encode(x)
            out.append(mu.cpu().numpy())
    return out


def forecast_latents(
    model: VectorVAE,
    items: Sequence[dict],
    degree: int,
    use_actual_target_time: bool,
    device: torch.device,
) -> np.ndarray:
    latents = encode_means(model, items, device)
    predicted = []
    for s, z_hist in zip(items, latents):
        t = s["times"].astype(float)
        if use_actual_target_time:
            target_t = float(s["target_time"])
        else:
            target_t = float(t[-1] + (t[-1] - t[-2]))
        z_out = np.zeros(z_hist.shape[1], dtype=np.float32)
        deg = min(int(degree), len(t) - 1)
        for j in range(z_hist.shape[1]):
            if deg <= 0:
                z_out[j] = z_hist[-1, j]
            else:
                coeff = np.polyfit(t, z_hist[:, j], deg=deg)
                z_out[j] = np.polyval(coeff, target_t)
        with torch.no_grad():
            decoded = model.decode(torch.from_numpy(z_out[None, :]).to(device)).cpu().numpy()[0]
        predicted.append(decoded)
    return np.stack(predicted).astype(np.float32)


def mmd_vae_loss(model: VectorVAE, xb: torch.Tensor, mmd_weight: float) -> tuple[torch.Tensor, torch.Tensor]:
    recon, mu, logvar = model(xb)
    # The generalized VAE regularises a sampled latent distribution rather
    # than only the posterior mean.
    z = model.reparameterize(mu, logvar)
    prior = torch.randn_like(z)
    mmd = rbf_mmd(z, prior)
    return nn.functional.mse_loss(recon, xb) + mmd_weight * mmd, recon


def train_vae(
    train_visits_s: np.ndarray,
    val_visits_s: np.ndarray,
    latent_dim: int,
    config: RunConfig,
    seed: int,
    device: torch.device,
) -> tuple[VectorVAE, dict[str, float]]:
    set_seed(seed + 2000 + latent_dim)
    model = VectorVAE(latent_dim=latent_dim).to(device)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(train_visits_s)), batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(val_visits_s)), batch_size=config.batch_size, shuffle=False)
    opt = torch.optim.Adam(model.parameters(), lr=config.vae_lr)
    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    for _ in range(config.vae_epochs):
        model.train()
        for (xb,) in train_loader:
            xb = xb.to(device)
            opt.zero_grad()
            loss, _ = mmd_vae_loss(model, xb, config.mmd_weight)
            loss.backward()
            opt.step()
        model.eval()
        val_mse = 0.0
        n = 0
        with torch.no_grad():
            for (xb,) in val_loader:
                xb = xb.to(device)
                mu, _ = model.encode(xb)
                recon = model.decode(mu)
                val_mse += float(nn.functional.mse_loss(recon, xb).item()) * len(xb)
                n += len(xb)
        val_mse /= max(n, 1)
        if val_mse < best_val:
            best_val = val_mse
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, {"reconstruction_val_mse": best_val}


def select_vae_degree(
    model: VectorVAE,
    split_s: dict[str, list[dict]],
    scaler: StandardScaler,
    use_actual_target_time: bool,
    device: torch.device,
) -> tuple[int, pd.DataFrame]:
    true_val = np.stack([s["Y"] for s in split_s["val"]])
    rows = []
    for degree in (1, 2):
        pred_s = forecast_latents(model, split_s["val"], degree, use_actual_target_time, device)
        pred, true = inverse_predictions(pred_s, np.stack([s["Y_s"] for s in split_s["val"]]), scaler)
        value = float(mae_rows(pred, true).mean())
        rows.append({"degree": degree, "val_mae_db": value})
    selection = pd.DataFrame(rows)
    best_degree = int(selection.sort_values(["val_mae_db", "degree"]).iloc[0]["degree"])
    return best_degree, selection


def run(config: RunConfig) -> tuple[pd.DataFrame, dict[str, object]]:
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df, td_cols = load_aligned_grape()
    pooled = sum((build_samples(df, td_cols, k) for k in (2, 3, 4)), [])
    split = make_fixed_split(pooled)
    scaler = fit_scaler(split)
    split_s = {name: standardize_samples(items, scaler) for name, items in split.items()}
    train_eyes = {s["eye_id"] for s in split["train"]}
    val_eyes = {s["eye_id"] for s in split["val"]}
    train_visits_s = scaler.transform(make_visit_matrix(df, td_cols, train_eyes)).astype(np.float32)
    val_visits_s = scaler.transform(make_visit_matrix(df, td_cols, val_eyes)).astype(np.float32)

    out_dir = PROJECT_ROOT / config.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = split_counts(split)
    if counts["test"]["samples"] != 123:
        raise AssertionError(f"Expected 123 test samples under the current protocol, got {counts['test']['samples']}")

    all_rows = []
    candidate_rows = []
    diagnostics: list[dict[str, object]] = []
    # Deterministic baselines are evaluated once and copied into each seed
    # block for convenient downstream tables.
    y_test = np.stack([s["Y"] for s in split["test"]])
    x_last = np.stack([s["X"][-1] for s in split["test"]])
    nochange = summarise_predictions(x_last, y_test, "No-change classifier", 0, "test")
    all_rows.append(nochange)

    # Existing equal-step temporal LR, kept only as a protocol sanity check.
    plr = []
    for s in split["test"]:
        x = s["X"]
        t = s["times"]
        if len(t) < 2 or t[-1] <= t[-2]:
            plr.append(x[-1])
        else:
            plr.append(x[-1] + (x[-1] - x[-2]) / (t[-1] - t[-2]) * (t[-1] - t[-2]))
    all_rows.append(summarise_predictions(np.stack(plr), y_test, "Temporal LR (equal-step)", 0, "test"))

    for seed in config.seeds:
        print(f"\n=== GRAPE paper-method baselines | seed={seed} | device={device} ===")
        pred_s, marta_diag = train_marta(
            split_s,
            train_visits_s,
            val_visits_s,
            scaler,
            config,
            seed,
            device,
        )
        pred, true = inverse_predictions(pred_s, np.stack([s["Y_s"] for s in split_s["test"]]), scaler)
        all_rows.append(summarise_predictions(pred, true, "Marta-style AE + PE", seed, "test"))
        diagnostics.append({"model": "Marta-style AE + PE", "seed": seed, **marta_diag})
        print(f"Marta-style AE + PE: {mae_rows(pred, true).mean():.4f} dB")

        vae_candidates = []
        for latent_dim in config.latent_dims:
            vae, vae_diag = train_vae(train_visits_s, val_visits_s, latent_dim, config, seed, device)
            degree, degree_df = select_vae_degree(
                vae,
                split_s,
                scaler,
                config.use_actual_target_time,
                device,
            )
            val_pred_s = forecast_latents(vae, split_s["val"], degree, config.use_actual_target_time, device)
            val_true_s = np.stack([s["Y_s"] for s in split_s["val"]])
            val_pred, val_true = inverse_predictions(val_pred_s, val_true_s, scaler)
            val_forecast_mae = float(mae_rows(val_pred, val_true).mean())
            pred_s = forecast_latents(vae, split_s["test"], degree, config.use_actual_target_time, device)
            pred, true = inverse_predictions(pred_s, np.stack([s["Y_s"] for s in split_s["test"]]), scaler)
            label = f"Nature-style VAE latent regression (z={latent_dim}, degree={degree})"
            candidate_summary = summarise_predictions(pred, true, label, seed, "test").assign(
                validation_forecast_mae_db=val_forecast_mae,
                validation_selected=False,
            )
            candidate_rows.append(candidate_summary)
            vae_candidates.append(
                {
                    "latent_dim": latent_dim,
                    "degree": degree,
                    "pred": pred,
                    "true": true,
                    "val_forecast_mae": val_forecast_mae,
                    "label": label,
                    "candidate_summary": candidate_summary,
                    "vae_diag": vae_diag,
                    "degree_selection": degree_df.to_dict(orient="records"),
                }
            )
            print(
                f"{label}: test={mae_rows(pred, true).mean():.4f} dB; "
                f"val={val_forecast_mae:.4f} dB"
            )

        selected = min(
            vae_candidates,
            key=lambda candidate: (candidate["val_forecast_mae"], candidate["latent_dim"]),
        )
        # Keep the result label identical across seeds.  The selected latent
        # dimension is recorded in the manifest and candidate table, while
        # the aggregate table should represent one validation-selected
        # method rather than separate post-hoc z=8/z=16 groups.
        selected_label = "Nature-style VAE latent regression (validation-selected)"
        selected_summary = summarise_predictions(
            selected["pred"], selected["true"], selected_label, seed, "test"
        )
        all_rows.append(selected_summary)
        for candidate in vae_candidates:
            candidate["validation_selected"] = candidate is selected
            candidate["candidate_summary"]["validation_selected"] = candidate is selected
            diagnostics.append(
                {
                    "model": candidate["label"],
                    "seed": seed,
                    "latent_dim": candidate["latent_dim"],
                    "selected_degree": candidate["degree"],
                    "validation_forecast_mae_db": candidate["val_forecast_mae"],
                    "validation_selected": candidate is selected,
                    **candidate["vae_diag"],
                    "degree_selection": candidate["degree_selection"],
                }
            )
        print(
            f"Selected Nature-style candidate: z={selected['latent_dim']}, "
            f"degree={selected['degree']}; "
            f"test={mae_rows(selected['pred'], selected['true']).mean():.4f} dB"
        )

    summary = pd.concat(all_rows, ignore_index=True)
    candidate_summary = pd.concat(candidate_rows, ignore_index=True)
    candidate_summary.to_csv(out_dir / "candidate_summary_long.csv", index=False)
    summary.to_csv(out_dir / "summary_long.csv", index=False)
    summary["MAE_plus_minus_SD_db"] = summary.apply(
        lambda r: f"{r['MAE_mean_db']:.2f} ± {r['MAE_SD_db']:.2f}", axis=1
    )
    summary.to_csv(out_dir / "summary_long_formatted.csv", index=False)
    seed_aggregate = (
        summary.groupby(["Model", "split", "Region"], as_index=False)
        .agg(
            MAE_mean_across_seeds_db=("MAE_mean_db", "mean"),
            MAE_seed_SD_db=("MAE_mean_db", "std"),
            mean_sample_SD_db=("MAE_SD_db", "mean"),
            n_seeds=("seed", "nunique"),
            n_samples=("n_samples", "first"),
        )
    )
    seed_aggregate["MAE_seed_SD_db"] = seed_aggregate["MAE_seed_SD_db"].fillna(0.0)
    seed_aggregate["MAE_mean_plus_minus_seed_SD_db"] = seed_aggregate.apply(
        lambda r: f"{r['MAE_mean_across_seeds_db']:.2f} ± {r['MAE_seed_SD_db']:.2f}",
        axis=1,
    )
    seed_aggregate.to_csv(out_dir / "summary_seed_aggregate.csv", index=False)
    # A paper-style table uses the test-sample SD for the ± column, matching
    # the existing GRAPE baseline tables.  Seed-to-seed SD remains available
    # separately in ``summary_seed_aggregate.csv``.
    paper_style = seed_aggregate.copy()
    paper_style["MAE_plus_minus_test_SD_db"] = paper_style.apply(
        lambda r: f"{r['MAE_mean_across_seeds_db']:.2f} ± {r['mean_sample_SD_db']:.2f}",
        axis=1,
    )
    paper_style.to_csv(out_dir / "summary_paper_style.csv", index=False)
    manifest = {
        "protocol": {
            "dataset": "GRAPE",
            "representation": "native 59-point Bern-aligned total deviation",
            "history_lengths": [2, 3, 4],
            "window_mode": "consecutive",
            "split_seed": 42,
            "standardisation": "training targets only",
            "target_time": "actual target visit time" if config.use_actual_target_time else "last observed gap extrapolation",
        },
        "counts": counts,
        "train_visit_vectors": int(len(train_visits_s)),
        "val_visit_vectors": int(len(val_visits_s)),
        "config": asdict(config),
        "device": str(device),
        "diagnostics": diagnostics,
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return summary, manifest


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--latent-dims", nargs="+", type=int, default=[8, 16])
    parser.add_argument("--ae-pretrain-epochs", type=int, default=40)
    parser.add_argument("--marta-epochs", type=int, default=60)
    parser.add_argument("--vae-epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--mmd-weight", type=float, default=0.1)
    parser.add_argument("--no-actual-target-time", action="store_true")
    parser.add_argument("--output-dir", default="results/paper_method_baselines_grape")
    args = parser.parse_args()
    return RunConfig(
        seeds=tuple(args.seeds),
        latent_dims=tuple(args.latent_dims),
        ae_pretrain_epochs=args.ae_pretrain_epochs,
        marta_epochs=args.marta_epochs,
        vae_epochs=args.vae_epochs,
        batch_size=args.batch_size,
        mmd_weight=args.mmd_weight,
        use_actual_target_time=not args.no_actual_target_time,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    cfg = parse_args()
    result, _ = run(cfg)
    print("\n=== Overall test summary ===")
    print(
        result[result["Region"].eq("Overall")][
            ["Model", "seed", "MAE_mean_db", "MAE_SD_db", "n_samples"]
        ].to_string(index=False)
    )
