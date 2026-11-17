# -*- coding: utf-8 -*-
"""
Impute missing DO in the Yellow River Basin with a three-stage evaluation.

Train the development model on 2021–2023. Use the 2024 validation set for early stopping and epoch selection. Reserve 2025 for independent evaluation.

After selecting the epoch count, refit the final model using observed DO from 2021–2024. Impute missing DO with that model. The 2025 observations are reserved for testing.

The BiLSTM-Attention excludes Event_Label, SD, SWE and WT from its input features. Keep Event_Label for grouped outputs and preserve the original WT values without forward filling or zero filling.
"""

import os
import glob
import copy
import random
import sys
import warnings
from pathlib import Path
from shutil import copy2

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[3]

warnings.filterwarnings("ignore")




# ===================== V1 model assets and provenance =====================
V1_INPUT_MANIFEST = []


def V1_记录输入文件(files):
    """Record files in the original glob order used for training."""
    import hashlib
    records = []
    for index, path in enumerate(files):
        with open(path, "rb") as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
        records.append({"order": index, "path": os.path.abspath(path),
                        "size_bytes": os.path.getsize(path), "sha256": digest})
    return records


def V1_保存最终模型(model, sx, sy, features, target, best_epoch, best_val_loss, files):
    """Serialize the fitted final model and scalers without refitting or initializing a new random model."""
    import hashlib
    import json
    import platform
    from datetime import datetime, timezone
    import joblib
    import sklearn

    asset_dir = Path(OUT_ROOT) / "Saved_Models_V1"
    asset_dir.mkdir(parents=True, exist_ok=True)
    weights = asset_dir / f"{target}_BiLSTM_Attention_V1.pth"
    sx_path = asset_dir / f"{target}_sx_final_V1.joblib"
    sy_path = asset_dir / f"{target}_sy_final_V1.joblib"
    # The CPU copy is only for saving. The active model stays on its current device and retains its train or eval state.
    saved_state = {key: value.detach().cpu().clone()
                   for key, value in model.state_dict().items()}
    torch.save(saved_state, weights)
    joblib.dump(sx, sx_path)
    joblib.dump(sy, sy_path)

    def digest(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    source = Path(__file__).resolve()
    baseline = source.with_name("LSTM.py")
    source_copy = asset_dir / "V1_training_source.py"
    source_copy.write_bytes(source.read_bytes())
    config = {
        "sequence_length": SEQ_LEN, "input_size": len(features),
        "hidden_size": model.lstm.hidden_size,
        "num_layers": model.lstm.num_layers,
        "bidirectional": model.lstm.bidirectional,
        "dropout": model.drop.p,
        "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
        "batch_size": BATCH_SIZE, "prediction_batch_size": PRED_BATCH_SIZE,
        "learning_rate": LR, "weight_decay": WEIGHT_DECAY,
        "optimizer": "Adam", "loss": "MSELoss", "shuffle": True,
        "seed": 42,
        "development_training_years": list(TRAIN_YEARS),
        "validation_year": VAL_YEAR, "test_year": TEST_YEAR,
        "final_training_years": list(FINAL_TRAIN_YEARS),
        "selected_final_epochs": int(best_epoch),
        "best_validation_loss": float(best_val_loss),
        "base_features": list(BASE_FEATURES), "features": list(features),
        "lag_policy": "Original DO shift(1/2/3), forward fill then leading zero; static, not recursive",
        "model_class": type(model).__name__,
    }
    environment = {
        "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "numpy": np.__version__,
        "pandas": pd.__version__, "sklearn": sklearn.__version__,
        "torch": str(torch.__version__), "joblib": joblib.__version__,
        "device": str(DEVICE), "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }
    manifest = {
        "schema_version": 1, "version": "V1", "target": target,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_origin": "V1 rerun of LSTM.py with artifact saving and a separate output root",
        "verified_original_model": False,
        "verification_status": "Pending reload prediction check and comparison with historical V13 outputs",
        "continuous_gap_validation": "Not performed by this training script",
        "save_stage": "Immediately after final training, before evaluation and imputation exports",
        "weights": str(weights.resolve()), "weights_sha256": digest(weights),
        "sx": str(sx_path.resolve()), "sx_sha256": digest(sx_path),
        "sy": str(sy_path.resolve()), "sy_sha256": digest(sy_path),
        "weights_format": "Plain CPU state_dict; use BiLSTMAttention with the recorded configuration",
        "scaler_types": {"sx": type(sx).__name__, "sy": type(sy).__name__},
        "scaler_feature_counts": {"sx": int(sx.n_features_in_), "sy": int(sy.n_features_in_)},
        "features": list(features), "training_years": list(FINAL_TRAIN_YEARS),
        "station_file_order": list(files), "input_files": V1_INPUT_MANIFEST,
        "input_directory": INPUT_DIR, "output_root": OUT_ROOT,
        "source_script": str(source), "source_sha256": digest(source),
        "source_snapshot": str(source_copy.resolve()),
        "baseline_script": str(baseline), "baseline_sha256": digest(baseline),
        "configuration": config, "environment": environment,
    }
    (asset_dir / f"{target}_Model_Manifest_V1.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[V1] 最终权重、两个标准化器及来源配置已保存：{asset_dir}")
    print("[V1] 当前仅保存本轮资产，尚未与历史V13结果比较，也未执行连续缺测验证。")

# ===================== End of V1 model asset export =====================


# ===================== 1. Configuration =====================
def 固定随机种子(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


固定随机种子(42)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"🚀 当前计算设备: {DEVICE}")


# =========================================================
# Input path
# =========================================================

# LSTM_Y directory produced by feature preprocessing
INPUT_DIR = str(ROOT / 'Data/LSTM_Y')


# =========================================================
# Output path
# =========================================================

OUT_ROOT = str(ROOT / 'Result/BiLstm_Att_V1')

IMPUTED_DIR = os.path.join(
    OUT_ROOT,
    "Imputed_Data"
)

REPORT_DIR = os.path.join(
    OUT_ROOT,
    "Reports"
)

os.makedirs(
    IMPUTED_DIR,
    exist_ok=True
)

os.makedirs(
    REPORT_DIR,
    exist_ok=True
)


# =========================================================
# Loss history directory
# =========================================================

LOSS_DIR = os.path.join(
    REPORT_DIR,
    "Loss_History"
)

os.makedirs(
    LOSS_DIR,
    exist_ok=True
)


# =========================================================
# Year-based splits
# =========================================================

# Development training set
TRAIN_YEARS = [
    2021,
    2022,
    2023
]

# Validation set
VAL_YEAR = 2024

# Independent test set
TEST_YEAR = 2025

# Years used to refit the final model
FINAL_TRAIN_YEARS = [
    2021,
    2022,
    2023,
    2024
]


# =========================================================
# Model parameters
# =========================================================

# Window length
SEQ_LEN = 30

# Maximum epochs
MAX_EPOCHS = 80

# Early-stopping patience
PATIENCE = 10

# Training batch size
BATCH_SIZE = 64

# Prediction batch size
PRED_BATCH_SIZE = 2048

# Learning rate
LR = 5e-4

# Weight decay
WEIGHT_DECAY = 1e-4


# =========================================================
# Input features
# =========================================================

# ---------------------
# 1. Meteorological features
# ---------------------
WEATHER = [
    "Tmax",
    "SSR",
    "TP_M",
    "TP_M_sum3",
    "WS"
]


# ---------------------
# 2. Hydrological features
# ---------------------
HYDRO = [
    "WL",
    "Q",
    "Q_mean7",
    "Ss"
]


# ---------------------
# 3. Dynamic environmental features
#
# WT and Event_Label are excluded from model inputs.
# Missing-feature handling therefore leaves the original WT column intact.
# ---------------------
DYNAMIC = [
    "EVI",
    "LandCover_Class"
]


# ---------------------
# 4. Time features
# ---------------------
TIME = [
    "month",
    "season",
    "dayofyear"
]


# ---------------------
# 5. Spatial features
# ---------------------
SPACE = [
    "Region_Code"
]


# ---------------------
# Prediction target
# ---------------------
TARGETS = [
    "DO"
]


# ---------------------
# Base input features
# ---------------------
BASE_FEATURES = (
    WEATHER
    + HYDRO
    + DYNAMIC
    + TIME
    + SPACE
)


# =========================================================
# Event_Label is excluded from the model
# and used to group the final output by event type.
# =========================================================

EVENT_MAP = {
    1: "Flood",
    2: "Heatwave",
    3: "Direct_Compound",
    4: "H2F_Compound",
    5: "F2H_Compound",
}


# ===================== 2. Model architecture =====================
class BiLSTMAttention(nn.Module):
    """
    Predict daily DO with a bidirectional LSTM and attention.
    """

    def __init__(
        self,
        input_size,
        hidden_size=128,
        dropout=0.3
    ):
        super().__init__()

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Attention
        self.att = nn.Sequential(
            nn.Linear(
                hidden_size * 2,
                hidden_size
            ),
            nn.Tanh(),
            nn.Linear(
                hidden_size,
                1
            ),
            nn.Softmax(dim=1),
        )

        # Dropout
        self.drop = nn.Dropout(
            dropout
        )

        # Output layer
        self.fc = nn.Linear(
            hidden_size * 2,
            1
        )

    def forward(self, x):

        # LSTM outputs
        h, _ = self.lstm(x)

        # Attention weights
        w = self.att(h)

        # Weighted context vector
        context = torch.sum(
            w * h,
            dim=1
        )

        # Predict DO.
        return self.fc(
            self.drop(context)
        )


# ===================== 3. Shared utilities =====================
def 读表(path):
    """
    Read CSV files encoded in UTF-8 or GBK.
    """

    try:
        return pd.read_csv(
            path,
            encoding="utf-8-sig"
        )

    except UnicodeDecodeError:
        return pd.read_csv(
            path,
            encoding="gbk"
        )


def 安全文件名(x):
    """
    Prevent slashes and backslashes in station names from breaking paths.
    """

    return (
        str(x)
        .replace("/", "_")
        .replace("\\", "_")
    )


def calc_nse(pred, obs):
    """
    Nash-Sutcliffe Efficiency
    """

    pred = np.asarray(
        pred,
        dtype=float
    )

    obs = np.asarray(
        obs,
        dtype=float
    )

    denom = np.sum(
        (obs - obs.mean()) ** 2
    )

    if denom == 0:
        return np.nan

    return (
        1
        - np.sum(
            (obs - pred) ** 2
        )
        / denom
    )


def 站点信息(df, file_stem):
    """
    Get station metadata.
    """

    if "Region_Code" in df.columns:
        region_code = df[
            "Region_Code"
        ].iloc[0]
    else:
        region_code = 2

    region = {
        1: "UR",
        2: "MR",
        3: "LR"
    }.get(
        region_code,
        "Unknown"
    )

    station = file_stem

    if (
        "WQS_name" in df.columns
        and not df[
            "WQS_name"
        ].isna().all()
    ):
        station = str(
            df[
                "WQS_name"
            ].iloc[0]
        )

    if "WQS_Lon" in df.columns:
        lon = df[
            "WQS_Lon"
        ].iloc[0]

    elif "lon" in df.columns:
        lon = df[
            "lon"
        ].iloc[0]

    else:
        lon = np.nan

    if "WQS_Lat" in df.columns:
        lat = df[
            "WQS_Lat"
        ].iloc[0]

    elif "lat" in df.columns:
        lat = df[
            "lat"
        ].iloc[0]

    else:
        lat = np.nan

    return (
        station,
        region,
        lon,
        lat
    )


def 可信度标签(
    dataset,
    nse,
    n,
    months
):
    """
    Flag confidence for reporting while preserving the imputed values.
    """

    if (
        pd.isna(nse)
        or pd.isna(n)
    ):
        return "Not_evaluated"

    # Independent test in 2025
    if "Test" in dataset:

        if (
            n < 50
            or months < 3
            or nse < 0
        ):
            return "Low"

        if (
            n < 100
            or months < 6
            or nse < 0.5
        ):
            return "Medium"

        return "High"

    # Train / Validation
    if nse < 0:
        return "Low"

    if nse < 0.5:
        return "Medium"

    return "High"


# ===================== 4. Prepare features and build sequences =====================
def 准备特征(
    dfs,
    target
):
    """
    Add lagged DO and handle missing input features.

    Preserve missing values in the DO target. Keep the original Event_Label and WT columns in the DataFrame, outside the model input. Leave WT values as recorded. Use DO_lag1, DO_lag2 and DO_lag3 as model inputs.
    """

    features = (
        BASE_FEATURES
        + [
            f"{target}_lag{i}"
            for i in [1, 2, 3]
        ]
    )

    out = []

    for df in dfs:

        df = df.copy()

        # ---------------------
        # Check the date column.
        # ---------------------
        if "Date" not in df.columns:
            raise ValueError(
                "输入文件缺少 Date 列。"
            )

        # ---------------------
        # Check the target column.
        # ---------------------
        if target not in df.columns:
            raise ValueError(
                f"输入文件缺少目标列: {target}"
            )

        # Parse dates.
        df["Date"] = pd.to_datetime(
            df["Date"]
        )

        # Sort by date.
        df = (
            df
            .sort_values("Date")
            .reset_index(drop=True)
        )

        # ---------------------
        # Lagged DO features
        # ---------------------
        for lag in [1, 2, 3]:

            df[
                f"{target}_lag{lag}"
            ] = (
                df[target]
                .shift(lag)
            )

        # ---------------------
        # Handle missing model inputs.
        #
        # Apply this step only to the input features.
        # Leave the DO target column intact.
        # ---------------------
        for col in features:

            if col not in df.columns:
                df[col] = 0

            df[col] = (
                pd.to_numeric(
                    df[col],
                    errors="coerce"
                )
                .ffill()
                .fillna(0)
            )

        out.append(df)

    return (
        out,
        features
    )


def 拟合标准化器(
    dfs,
    years,
    features,
    target
):
    """
    Fit the X and y scalers using the specified years.
    """

    x_blocks = []
    y_blocks = []

    for df in dfs:

        mask = (
            df["Date"]
            .dt.year
            .isin(years)
        )

        if mask.sum() == 0:
            continue

        # Input variables
        x_blocks.append(
            df.loc[
                mask,
                features
            ].values
        )

        # Target variable
        y = (
            df.loc[
                mask,
                target
            ]
            .dropna()
            .values
            .reshape(-1, 1)
        )

        if len(y) > 0:
            y_blocks.append(y)

    if (
        not x_blocks
        or not y_blocks
    ):
        return None, None

    sx = StandardScaler()
    sy = StandardScaler()

    sx.fit(
        np.vstack(
            x_blocks
        )
    )

    sy.fit(
        np.vstack(
            y_blocks
        )
    )

    return (
        sx,
        sy
    )


def 构建样本(
    dfs,
    years,
    features,
    target,
    sx,
    sy
):
    """
    Build 30-day input windows.

    Require target days in the specified years. Allow the input window to cross year boundaries so early-year targets can be retained.
    """

    years = set(years)

    X_list = []
    y_list = []

    for df in dfs:

        if len(df) <= SEQ_LEN:
            continue

        # Standardize input variables.
        X = sx.transform(
            df[
                features
            ].values
        )

        # DO
        y = df[
            target
        ].values

        # Years
        yy = (
            df["Date"]
            .dt.year
            .values
        )

        # Start forming samples on day 31.
        for end in range(
            SEQ_LEN,
            len(df)
        ):

            # The target day must be in the specified years.
            if yy[end] not in years:
                continue

            # Require an observed DO target.
            if pd.isna(
                y[end]
            ):
                continue

            # Use the previous 30 days as the input window.
            X_list.append(
                X[
                    end - SEQ_LEN:end
                ]
            )

            # Use DO on the target day.
            y_list.append(
                y[end]
            )

    if not X_list:
        return None, None

    X_arr = np.asarray(
        X_list,
        dtype=np.float32
    )

    y_arr = (
        sy
        .transform(
            np.asarray(
                y_list
            ).reshape(-1, 1)
        )
        .astype(
            np.float32
        )
    )

    return (
        X_arr,
        y_arr
    )


# ===================== 5. Training, prediction and evaluation =====================
def 训练模型(
    X_train,
    y_train,
    input_size,
    X_val=None,
    y_val=None,
    fixed_epochs=None
):
    """
    Fit the model with early stopping when fixed_epochs is None.

    When fixed_epochs is provided, fit the final model for exactly that many epochs.
    """

    # ---------------------
    # DataLoader
    # ---------------------
    loader = DataLoader(
        TensorDataset(
            torch.tensor(
                X_train
            ),
            torch.tensor(
                y_train
            )
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    # ---------------------
    # Model
    # ---------------------
    model = BiLSTMAttention(
        input_size
    ).to(
        DEVICE
    )

    # ---------------------
    # Loss function
    # ---------------------
    loss_fn = nn.MSELoss()

    # ---------------------
    # Adam optimizer
    # ---------------------
    opt = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    # ---------------------
    # Check whether a validation set is available.
    # ---------------------
    use_val = (
        X_val is not None
        and y_val is not None
        and len(X_val) > 0
    )

    if use_val:

        Xv = torch.tensor(
            X_val,
            dtype=torch.float32
        ).to(
            DEVICE
        )

        yv = torch.tensor(
            y_val,
            dtype=torch.float32
        ).to(
            DEVICE
        )

    # ---------------------
    # Training epochs
    # ---------------------
    if fixed_epochs is not None:
        epochs = int(
            fixed_epochs
        )
    else:
        epochs = MAX_EPOCHS

    # ---------------------
    # Initialize early stopping.
    # ---------------------
    best_score = np.inf
    best_epoch = 1

    best_state = copy.deepcopy(
        model.state_dict()
    )

    wait = 0

    # ---------------------
    # Loss history
    # ---------------------
    loss_history = []

    # =====================================================
    # Training
    # =====================================================
    for epoch in range(
        1,
        max(1, epochs) + 1
    ):

        model.train()

        total_loss = 0.0

        # ---------------------
        # mini-batch
        # ---------------------
        for bx, by in loader:

            bx = bx.to(
                DEVICE
            )

            by = by.to(
                DEVICE
            )

            # Reset gradients.
            opt.zero_grad()

            # Forward pass
            pred = model(
                bx
            )

            # Calculate the loss.
            loss = loss_fn(
                pred,
                by
            )

            # Backpropagate.
            loss.backward()

            # Update parameters.
            opt.step()

            total_loss += (
                loss.item()
                * len(bx)
            )

        # ---------------------
        # Average training loss
        # ---------------------
        train_loss = (
            total_loss
            / len(
                loader.dataset
            )
        )

        # =================================================
        # Final fit with a fixed epoch count
        # =================================================
        if fixed_epochs is not None:

            loss_history.append(
                {
                    "Epoch": epoch,
                    "Train_Loss": float(
                        train_loss
                    ),
                    "Validation_Loss": np.nan,
                }
            )

            continue

        # =================================================
        # Development validation
        # =================================================
        model.eval()

        with torch.no_grad():

            if use_val:

                score = loss_fn(
                    model(Xv),
                    yv
                ).item()

            else:

                score = train_loss

        # Record loss.
        loss_history.append(
            {
                "Epoch": epoch,
                "Train_Loss": float(
                    train_loss
                ),
                "Validation_Loss": float(
                    score
                ),
            }
        )

        # ---------------------
        # Save the best model.
        # ---------------------
        if score < best_score:

            best_score = score

            best_epoch = epoch

            best_state = copy.deepcopy(
                model.state_dict()
            )

            wait = 0

        else:

            wait += 1

            # -----------------
            # Early stopping
            # -----------------
            if wait >= PATIENCE:

                print(
                    f"      ⏱️ 早停触发："
                    f"Epoch {epoch}，"
                    f"最佳轮次 = {best_epoch}"
                )

                break

    # =====================================================
    # Restore the best model.
    # =====================================================
    if fixed_epochs is None:

        model.load_state_dict(
            best_state
        )

    else:

        best_epoch = max(
            1,
            epochs
        )

        best_score = np.nan

    return (
        model,
        best_epoch,
        best_score,
        loss_history
    )


def 批量预测(
    model,
    X_seq,
    sy
):
    """
    Predict in batches and convert back to the original DO units.
    """

    model.eval()

    loader = DataLoader(
        TensorDataset(
            torch.tensor(
                X_seq,
                dtype=torch.float32
            )
        ),
        batch_size=PRED_BATCH_SIZE
    )

    preds = []

    with torch.no_grad():

        for (bx,) in loader:

            pred = model(
                bx.to(
                    DEVICE
                )
            )

            preds.append(
                pred
                .cpu()
                .numpy()
            )

    pred = (
        sy
        .inverse_transform(
            np.vstack(
                preds
            )
        )
        .flatten()
    )

    # Do not allow negative DO predictions.
    return np.clip(
        pred,
        a_min=0,
        a_max=None
    )


def 评估单站(
    model,
    df,
    years,
    label,
    features,
    target,
    sx,
    sy,
    file_stem,
    region
):
    """
    Evaluate one station for the specified years.
    """

    years = set(
        years
    )

    X = sx.transform(
        df[
            features
        ].values
    )

    y = df[
        target
    ].values

    yy = (
        df["Date"]
        .dt.year
        .values
    )

    mm = (
        df["Date"]
        .dt.month
        .values
    )

    seqs = []
    obs = []
    months = []

    # =====================================================
    # Build evaluation windows.
    # =====================================================
    for end in range(
        SEQ_LEN,
        len(df)
    ):

        if yy[end] not in years:
            continue

        if pd.isna(
            y[end]
        ):
            continue

        seqs.append(
            X[
                end - SEQ_LEN:end
            ]
        )

        obs.append(
            y[end]
        )

        months.append(
            mm[end]
        )

    # =====================================================
    # No available samples
    # =====================================================
    if not seqs:

        return {
            "Dataset": label,
            "Sample_Count": 0,
            "Observed_DO_STD": np.nan,
            "Covered_Months": 0,
            "MAE": np.nan,
            "RMSE": np.nan,
            "NSE": np.nan,
        }, []

    obs = np.asarray(
        obs,
        dtype=float
    )

    pred = 批量预测(
        model,
        np.asarray(
            seqs,
            dtype=np.float32
        ),
        sy
    )

    # =====================================================
    # Performance metrics
    # =====================================================
    metrics = {

        "Dataset":
            label,

        "Sample_Count":
            int(
                len(obs)
            ),

        "Observed_DO_STD":
            round(
                float(
                    np.std(
                        obs,
                        ddof=1
                    )
                ),
                3
            )
            if len(obs) > 1
            else 0.0,

        "Covered_Months":
            int(
                len(
                    set(months)
                )
            ),

        "MAE":
            round(
                float(
                    mean_absolute_error(
                        obs,
                        pred
                    )
                ),
                3
            ),

        "RMSE":
            round(
                float(
                    np.sqrt(
                        mean_squared_error(
                            obs,
                            pred
                        )
                    )
                ),
                3
            ),

        "NSE":
            round(
                float(
                    calc_nse(
                        pred,
                        obs
                    )
                ),
                3
            ),
    }

    # =====================================================
    # Data for scatter plots
    # =====================================================
    scatter = [

        {
            "Station": file_stem,
            "Region": region,

            "Observed DO":
                round(
                    float(o),
                    3
                ),

            "Simulated DO":
                round(
                    float(p),
                    3
                ),

            "Dataset": label,
        }

        for o, p
        in zip(
            obs,
            pred
        )
    ]

    return (
        metrics,
        scatter
    )


def 填补缺失(
    dfs,
    features,
    target,
    model,
    sx,
    sy
):
    """
    Use the final model to fill the original DO gaps.
    """

    out = []

    for df in dfs:

        df = df.copy()

        # Keep the original DO values.
        df[
            "Original_DO"
        ] = df[
            target
        ]

        # Record model-filled DO.
        df[
            "Imputed_DO"
        ] = np.nan

        # Find the positions originally missing.
        miss = df[
            target
        ].isna()

        # Return early if there are no missing values.
        if miss.sum() == 0:

            out.append(
                df
            )

            continue

        # ---------------------
        # Standardize inputs.
        # ---------------------
        X = sx.transform(
            df[
                features
            ].values
        )

        # ---------------------
        # Pad the initial 30-day window.
        # ---------------------
        pad = np.repeat(
            X[0].reshape(
                1,
                -1
            ),
            SEQ_LEN,
            axis=0
        )

        X_pad = np.vstack(
            [
                pad,
                X
            ]
        )

        # ---------------------
        # Build a 30-day window for each day.
        # ---------------------
        seqs = np.asarray(

            [
                X_pad[
                    i:i + SEQ_LEN
                ]

                for i
                in range(
                    len(df)
                )
            ],

            dtype=np.float32
        )

        # ---------------------
        # Predict DO.
        # ---------------------
        pred = np.round(

            批量预测(
                model,
                seqs,
                sy
            ),

            3
        )

        # ---------------------
        # Fill only originally missing positions.
        # ---------------------
        df.loc[
            miss,
            target
        ] = pred[
            miss
        ]

        df.loc[
            miss,
            "Imputed_DO"
        ] = pred[
            miss
        ]

        out.append(
            df
        )

    return out


# ===================== 6. Workflow for one target variable =====================
def 运行一个目标(
    dfs,
    files,
    target
):

    # =====================================================
    # Prepare features.
    # =====================================================
    dfs, features = 准备特征(
        dfs,
        target
    )

    print(
        "\n"
        "========================================================="
    )

    print(
        f"🎯 当前目标变量：{target}"
    )

    print(
        f"📌 基础输入特征数：{len(BASE_FEATURES)}"
    )

    print(
        f"📌 DO 滞后特征数：3"
    )

    print(
        f"📌 最终输入特征数：{len(features)}"
    )

    print(
        "📌 Event_Label：不参与模型训练"
    )

    print(
        "📌 WT：不参与模型训练且不执行缺失值填充"
    )

    print(
        "========================================================="
    )

    # =====================================================
    # 1. Development phase
    # Train on 2021–2023.
    # Validate on 2024.
    # =====================================================

    sx_dev, sy_dev = 拟合标准化器(
        dfs,
        TRAIN_YEARS,
        features,
        target
    )

    if sx_dev is None:

        print(
            f"❌ {target}: "
            "无法拟合开发阶段标准化器。"
        )

        return (
            dfs,
            [],
            [],
            None
        )

    # ---------------------
    # Training samples
    # ---------------------
    X_tr, y_tr = 构建样本(
        dfs,
        TRAIN_YEARS,
        features,
        target,
        sx_dev,
        sy_dev
    )

    # ---------------------
    # 2024 validation samples
    # ---------------------
    X_val, y_val = 构建样本(
        dfs,
        [VAL_YEAR],
        features,
        target,
        sx_dev,
        sy_dev
    )

    if X_tr is None:

        print(
            f"❌ {target}: "
            "训练样本为空。"
        )

        return (
            dfs,
            [],
            [],
            None
        )

    print(
        f"\n⚙️ 开发阶段 {target}: "
        f"Train(21-23)={len(X_tr)}, "
        f"Val(24)="
        f"{0 if X_val is None else len(X_val)}"
    )

    # ---------------------
    # Train the development model.
    # ---------------------
    (
        dev_model,
        best_epoch,
        best_val_loss,
        dev_loss_history
    ) = 训练模型(
        X_tr,
        y_tr,
        len(features),
        X_val,
        y_val
    )

    print(
        f"   ✅ 2024 验证选择最佳轮次: "
        f"{best_epoch}"
    )

    # =====================================================
    # Save development loss.
    # =====================================================

    dev_loss_df = pd.DataFrame(
        dev_loss_history
    )

    dev_loss_df.insert(
        0,
        "Target_Feature",
        target
    )

    dev_loss_df[
        "Best_Epoch_Selected_By_Val24"
    ] = best_epoch

    dev_loss_df.to_csv(

        os.path.join(
            LOSS_DIR,
            f"{target}_Development_Loss_History.csv"
        ),

        index=False,
        encoding="utf-8-sig"
    )

    # =====================================================
    # 2. Final fit
    #
    # Fix the epoch count at the best value.
    # Retrain on 2021–2024.
    # =====================================================

    sx_final, sy_final = 拟合标准化器(
        dfs,
        FINAL_TRAIN_YEARS,
        features,
        target
    )

    if sx_final is None:

        print(
            f"❌ {target}: "
            "无法拟合最终阶段标准化器。"
        )

        return (
            dfs,
            [],
            [],
            None
        )

    X_final, y_final = 构建样本(
        dfs,
        FINAL_TRAIN_YEARS,
        features,
        target,
        sx_final,
        sy_final
    )

    if X_final is None:

        print(
            f"❌ {target}: "
            "最终训练样本为空。"
        )

        return (
            dfs,
            [],
            [],
            None
        )

    print(
        f"⚙️ 最终模型 {target}: "
        f"FinalTrain(21-24)="
        f"{len(X_final)}, "
        f"固定轮次={best_epoch}"
    )

    (
        final_model,
        _,
        _,
        final_loss_history
    ) = 训练模型(
        X_final,
        y_final,
        len(features),
        fixed_epochs=best_epoch
    )

    # V1 saves the final assets and reuses the active model for evaluation and gap filling.
    V1_保存最终模型(
        final_model, sx_final, sy_final, features, target,
        best_epoch, best_val_loss, files
    )

    # =====================================================
    # Save the final retraining loss.
    # =====================================================

    final_loss_df = pd.DataFrame(
        final_loss_history
    )

    final_loss_df.insert(
        0,
        "Target_Feature",
        target
    )

    final_loss_df.to_csv(

        os.path.join(
            LOSS_DIR,
            f"{target}_Final_Training_Loss_History.csv"
        ),

        index=False,
        encoding="utf-8-sig"
    )

    # =====================================================
    # 3. Evaluate by station
    # =====================================================

    eval_rows = []
    scatter_rows = []

    for i, df in enumerate(
        dfs
    ):

        file_stem = os.path.splitext(
            files[i]
        )[0]

        (
            station,
            region,
            lon,
            lat
        ) = 站点信息(
            df,
            file_stem
        )

        # Require a minimum number of original DO observations.
        if (
            df[target]
            .notna()
            .sum()
            < 30
        ):
            continue

        miss_rate = round(
            float(
                df[target]
                .isna()
                .mean()
            ),
            3
        )

        # =================================================
        # Evaluate the three data splits.
        # =================================================

        eval_plan = [

            (
                "Development",
                "Train(21-23)",
                TRAIN_YEARS,
                dev_model,
                sx_dev,
                sy_dev
            ),

            (
                "Development",
                "Val(24)",
                [VAL_YEAR],
                dev_model,
                sx_dev,
                sy_dev
            ),

            (
                "FinalFixedEpoch",
                "Test(25)",
                [TEST_YEAR],
                final_model,
                sx_final,
                sy_final
            ),

        ]

        for (
            phase,
            label,
            years,
            model,
            sx,
            sy
        ) in eval_plan:

            m, sc = 评估单站(
                model,
                df,
                years,
                label,
                features,
                target,
                sx,
                sy,
                file_stem,
                region
            )

            scatter_rows.extend(
                sc
            )

            eval_rows.append(
                [
                    file_stem,
                    station,
                    lon,
                    lat,
                    region,
                    target,
                    miss_rate,
                    phase,

                    m[
                        "Dataset"
                    ],

                    m[
                        "Sample_Count"
                    ],

                    m[
                        "Observed_DO_STD"
                    ],

                    m[
                        "Covered_Months"
                    ],

                    m[
                        "MAE"
                    ],

                    m[
                        "RMSE"
                    ],

                    m[
                        "NSE"
                    ],

                    可信度标签(
                        m[
                            "Dataset"
                        ],
                        m[
                            "NSE"
                        ],
                        m[
                            "Sample_Count"
                        ],
                        m[
                            "Covered_Months"
                        ]
                    ),

                    best_epoch,
                ]
            )

    # =====================================================
    # 4. Fill missing DO with the final model.
    # =====================================================

    imputed_dfs = 填补缺失(
        dfs,
        features,
        target,
        final_model,
        sx_final,
        sy_final
    )

    # =====================================================
    # Model metadata
    # =====================================================

    info = {

        "Target_Feature":
            target,

        "Train_Years_Development":
            ",".join(
                map(
                    str,
                    TRAIN_YEARS
                )
            ),

        "Validation_Year":
            VAL_YEAR,

        "Test_Year":
            TEST_YEAR,

        "Final_Training_Years":
            ",".join(
                map(
                    str,
                    FINAL_TRAIN_YEARS
                )
            ),

        "Best_Epoch_Selected_By_Val24":
            best_epoch,

        "Best_Val_Loss":
            best_val_loss,

        "Sequence_Length":
            SEQ_LEN,

        "Max_Epochs":
            MAX_EPOCHS,

        "Patience":
            PATIENCE,

        "Batch_Size":
            BATCH_SIZE,

        "Learning_Rate":
            LR,

        "Weight_Decay":
            WEIGHT_DECAY,

        # Record the number of input features used in this version.
        "Base_Feature_Count":
            len(
                BASE_FEATURES
            ),

        "Total_Input_Feature_Count":
            len(
                features
            ),

        "Event_Label_Used_As_Input":
            False,

        "WT_Used_As_Input":
            False,

        "WT_Imputed_By_Script":
            False,
    }

    return (
        imputed_dfs,
        eval_rows,
        scatter_rows,
        info
    )


# ===================== 7. Export results =====================
def 汇总站点填补文件(by_site_dir, written_files):
    """Validate the 66 station-level daily files and collect them unchanged in the YR directory for downstream event analysis."""
    root = Path(__file__).resolve().parents[3]
    mapping = pd.read_csv(root / "Config/Study_Station_Mapping.csv")
    if len(mapping) != 66 or mapping.WQS_name.nunique() != 66 or mapping.Matched_HS_Name.nunique() != 42:
        raise ValueError("站点清单应包含66个水质站和42个匹配水文站")
    study_dates = set(pd.date_range("2021-01-01", "2025-12-31"))
    sources = []
    for station in mapping.itertuples():
        source = by_site_dir / station.Region / f"{安全文件名(station.WQS_name)}_Imputed.csv"
        if source not in written_files:
            raise RuntimeError(f"本轮未生成预期站点填补文件：{source}")
        dates = pd.to_datetime(读表(source)["Date"]).dt.normalize()
        if dates.duplicated().any() or set(dates) != study_dates:
            raise ValueError(f"填补日表应覆盖全部研究日期，且每天只有一条记录：{source}")
        sources.append(source)

    output_dir = by_site_dir / "YR"
    output_dir.mkdir(parents=True, exist_ok=True)
    for source in sources:
        copy2(source, output_dir / source.name)
    print(f"📌 已汇总 {len(sources)} 站填补文件：{output_dir}")


def 输出填补数据(
    final_df
):
    """
    Export completed data for all stations, individual station files and the YR collection, plus regional means by event type.

    Use the retained Event_Label column for grouping.
    """

    # =====================================================
    # Drop temporary modeling columns.
    # =====================================================

    drop_cols = [
        c
        for c in final_df.columns
        if "_lag" in c
    ]

    drop_cols += [
        "Q_mean7",
        "month",
        "season",
        "dayofyear",
        "Region_Code",
        "TP_M_sum3"
    ]

    final_df = final_df.drop(
        columns=[
            c
            for c in drop_cols
            if c in final_df.columns
        ]
    )

    # =====================================================
    # 1. Combine all stations.
    # =====================================================

    full_path = os.path.join(
        IMPUTED_DIR,
        "All_Sites_LSTM_Imputed_Full.csv"
    )

    final_df.to_csv(
        full_path,
        index=False,
        encoding="utf-8-sig"
    )

    # =====================================================
    # 2. Export by station.
    # =====================================================

    by_site_dir = Path(IMPUTED_DIR) / "By_Site_Imputed"
    by_site_dir.mkdir(parents=True, exist_ok=True)
    written_files = set()

    if "WQS_name" in final_df.columns:

        for (
            site,
            sdf
        ) in final_df.groupby(
            "WQS_name"
        ):

            region = "Unknown"

            if (
                "Region" in sdf.columns
                and not sdf[
                    "Region"
                ].isna().all()
            ):

                region = str(
                    sdf[
                        "Region"
                    ].iloc[0]
                ).strip()

            site_dir = by_site_dir / region
            site_dir.mkdir(parents=True, exist_ok=True)
            site_file = site_dir / f"{安全文件名(site)}_Imputed.csv"
            sdf.to_csv(site_file, index=False, encoding="utf-8-sig")
            written_files.add(site_file)

    # =====================================================
    # 3. Export regional and station means by event type.
    #
    # Order of operations
    # Event_Label is excluded from model inputs,
    # and the original column is retained
    # so the output can still be grouped by event type.
    # =====================================================

    event_dir = os.path.join(
        IMPUTED_DIR,
        "By_Event_Region_Averages"
    )

    os.makedirs(
        event_dir,
        exist_ok=True
    )

    if "Event_Label" in final_df.columns:

        mean_cols = [
            "T",
            "Tmax",
            "Tmin",
            "SP",
            "SSR",
            "TP_M",
            "WS",
            "Td",
            "SD",
            "SWE",
            "WL",
            "Q",
            "Ss",
            "EVI",
            "DO"
        ]

        mean_cols = [
            c
            for c in mean_cols
            if c in final_df.columns
        ]

        for (
            ev_label,
            ev_name
        ) in EVENT_MAP.items():

            ev_df = final_df[
                final_df[
                    "Event_Label"
                ] == ev_label
            ].copy()

            if ev_df.empty:
                continue

            group_cols = [
                "Date",
                "Region"
            ]

            for c in [
                "WQS_name",
                "WQS_Lon",
                "WQS_Lat"
            ]:

                if c in ev_df.columns:
                    group_cols.append(
                        c
                    )

            out = (
                ev_df[
                    group_cols
                    + mean_cols
                ]
                .groupby(
                    group_cols
                )
                .mean(
                    numeric_only=True
                )
                .reset_index()
            )

            out.to_csv(

                os.path.join(
                    event_dir,
                    f"Regional_Avg_"
                    f"{ev_label}_"
                    f"{ev_name}.csv"
                ),

                index=False,
                encoding="utf-8-sig"
            )

    汇总站点填补文件(by_site_dir, written_files)
    return final_df


def 输出评估报告(
    eval_rows,
    scatter_rows,
    model_info
):
    """
    Write the model performance report.
    """

    if not eval_rows:
        return

    # =====================================================
    # Detailed evaluation table
    # =====================================================

    eval_cols = [

        "Station_Filename",
        "Station_Name",

        "WQS_Lon",
        "WQS_Lat",

        "Region",

        "Target_Feature",

        "Original_Missing_Rate",

        "Model_Phase",

        "Dataset",

        "Sample_Count",

        "Observed_DO_STD",

        "Covered_Months",

        "MAE",
        "RMSE",
        "NSE",

        "Reliability_Flag",

        "Best_Epoch_Selected_By_Val24",
    ]

    eval_df = pd.DataFrame(
        eval_rows,
        columns=eval_cols
    )

    eval_df.to_csv(

        os.path.join(
            REPORT_DIR,
            "Evaluation_Details_Report.csv"
        ),

        index=False,
        encoding="utf-8-sig"
    )

    # =====================================================
    # Model metadata
    # =====================================================

    if model_info:

        pd.DataFrame(
            model_info
        ).to_csv(

            os.path.join(
                REPORT_DIR,
                "Model_Info_Report.csv"
            ),

            index=False,
            encoding="utf-8-sig"
        )

    # =====================================================
    # 2025 independent test results
    # =====================================================

    test_df = eval_df[
        eval_df[
            "Dataset"
        ] == "Test(25)"
    ].copy()

    test_valid = test_df.dropna(
        subset=[
            "NSE"
        ]
    )

    summary = []

    for target in TARGETS:

        fdf = test_valid[
            test_valid[
                "Target_Feature"
            ] == target
        ]

        if fdf.empty:
            continue

        summary.append(
            {
                "Target_Feature":
                    target,

                "Evaluated_Stations_Count":
                    len(fdf),

                "Mean_Test_NSE":
                    round(
                        fdf["NSE"].mean(),
                        3
                    ),

                "Median_Test_NSE":
                    round(
                        fdf["NSE"].median(),
                        3
                    ),

                "Max_Test_NSE":
                    round(
                        fdf["NSE"].max(),
                        3
                    ),

                "Min_Test_NSE":
                    round(
                        fdf["NSE"].min(),
                        3
                    ),

                "Good_Fit_Ratio(Test_NSE>=0.5)":
                    f"{fdf['NSE'].ge(0.5).mean() * 100:.1f}%",

                "Negative_NSE_Count":
                    int(
                        fdf["NSE"]
                            .lt(0)
                            .sum()
                    ),

                "High_Reliability_Count":
                    int(
                        (
                                fdf["Reliability_Flag"]
                                == "High"
                        ).sum()
                    ),

                "Medium_Reliability_Count":
                    int(
                        (
                                fdf["Reliability_Flag"]
                                == "Medium"
                        ).sum()
                    ),

                "Low_Reliability_Count":
                    int(
                        (
                                fdf["Reliability_Flag"]
                                == "Low"
                        ).sum()
                    ),

                "Mean_Test_MAE":
                    round(
                        fdf["MAE"].mean(),
                        4
                    ),

                "Median_Test_MAE":
                    round(
                        fdf["MAE"].median(),
                        4
                    ),

                "Mean_Test_RMSE":
                    round(
                        fdf["RMSE"].mean(),
                        4
                    ),

                "Median_Test_RMSE":
                    round(
                        fdf["RMSE"].median(),
                        4
                    ),

                "Mean_Test_Sample_Count":
                    round(
                        fdf["Sample_Count"].mean(),
                        1
                    ),

                "Median_Test_Sample_Count":
                    round(
                        fdf["Sample_Count"].median(),
                        1
                    ),

                "Mean_Test_Covered_Months":
                    round(
                        fdf["Covered_Months"].mean(),
                        1
                    ),

                "Median_Test_Covered_Months":
                    round(
                        fdf["Covered_Months"].median(),
                        1
                    ),
            }
        )

    pd.DataFrame(
        summary
    ).to_csv(

        os.path.join(
            REPORT_DIR,
            "Evaluation_Summary_Report.csv"
        ),

        index=False,
        encoding="utf-8-sig"
    )

    # =====================================================
    # Stations with NSE below zero
    # =====================================================

    test_df[
        test_df[
            "NSE"
        ] < 0
    ].to_csv(

        os.path.join(
            REPORT_DIR,
            "Evaluation_Negative_NSE_Report.csv"
        ),

        index=False,
        encoding="utf-8-sig"
    )

    # =====================================================
    # Low-confidence stations
    # =====================================================

    test_df[
        test_df[
            "Reliability_Flag"
        ] == "Low"
    ].to_csv(

        os.path.join(
            REPORT_DIR,
            "Evaluation_Low_Confidence_Report.csv"
        ),

        index=False,
        encoding="utf-8-sig"
    )

    # =====================================================
    # Station-level scatter-plot data
    # =====================================================

    scatter_df = pd.DataFrame(
        scatter_rows
    )

    if scatter_df.empty:
        return

    scatter_dir = os.path.join(
        REPORT_DIR,
        "By_Site_Scatter"
    )

    for (
        site,
        sdf
    ) in scatter_df.groupby(
        "Station"
    ):

        sdf = sdf.copy()

        reg = str(
            sdf[
                "Region"
            ].iloc[0]
        ).strip()

        out_dir = os.path.join(
            scatter_dir,
            reg
        )

        os.makedirs(
            out_dir,
            exist_ok=True
        )

        site_m = eval_df[
            eval_df[
                "Station_Filename"
            ] == site
        ]

        # ---------------------
        # Extract metrics.
        # ---------------------
        def get_m(
            dataset,
            metric
        ):

            temp = site_m[
                site_m[
                    "Dataset"
                ] == dataset
            ]

            if temp.empty:
                return np.nan

            return temp[
                metric
            ].iloc[0]

        # ---------------------
        # Create additional columns.
        # ---------------------
        for c in [

            "Train_NSE",
            "Train_MAE",

            "Val_NSE",
            "Val_MAE",

            "Test_NSE",
            "Test_MAE",

            "Test_Sample_Count",
            "Test_Covered_Months"

        ]:

            sdf[c] = np.nan

        sdf[
            "Test_Reliability_Flag"
        ] = ""

        # ---------------------
        # Write performance metrics in the first row.
        # ---------------------
        if not sdf.empty:

            first = sdf.index[0]

            sdf.loc[
                first,
                "Train_NSE"
            ] = get_m(
                "Train(21-23)",
                "NSE"
            )

            sdf.loc[
                first,
                "Train_MAE"
            ] = get_m(
                "Train(21-23)",
                "MAE"
            )

            sdf.loc[
                first,
                "Val_NSE"
            ] = get_m(
                "Val(24)",
                "NSE"
            )

            sdf.loc[
                first,
                "Val_MAE"
            ] = get_m(
                "Val(24)",
                "MAE"
            )

            sdf.loc[
                first,
                "Test_NSE"
            ] = get_m(
                "Test(25)",
                "NSE"
            )

            sdf.loc[
                first,
                "Test_MAE"
            ] = get_m(
                "Test(25)",
                "MAE"
            )

            sdf.loc[
                first,
                "Test_Sample_Count"
            ] = get_m(
                "Test(25)",
                "Sample_Count"
            )

            sdf.loc[
                first,
                "Test_Covered_Months"
            ] = get_m(
                "Test(25)",
                "Covered_Months"
            )

            sdf.loc[
                first,
                "Test_Reliability_Flag"
            ] = get_m(
                "Test(25)",
                "Reliability_Flag"
            )

        # ---------------------
        # Export results.
        # ---------------------
        sdf.to_csv(

            os.path.join(
                out_dir,
                f"{安全文件名(site)}_Scatter.csv"
            ),

            index=False,
            encoding="utf-8-sig"
        )


# ===================== 8. Main workflow =====================
def main():

    # =====================================================
    # Find all station CSV files.
    # =====================================================

    files = glob.glob(
        os.path.join(
            INPUT_DIR,
            "**",
            "*.csv"
        ),
        recursive=True
    )

    if not files:

        print(
            f"❌ 未找到输入文件: "
            f"{INPUT_DIR}"
        )

        return

    # Record the input sources in their original glob order.
    global V1_INPUT_MANIFEST
    V1_INPUT_MANIFEST = V1_记录输入文件(files)

    print(
        f"📂 发现 {len(files)} "
        f"个站点文件，开始读取..."
    )

    # =====================================================
    # Load data.
    # =====================================================

    dfs = []
    names = []

    for path in files:

        df = 读表(
            path
        )

        if "Date" in df.columns:

            df[
                "Date"
            ] = pd.to_datetime(
                df[
                    "Date"
                ]
            )

            df = (
                df
                .sort_values(
                    "Date"
                )
                .reset_index(
                    drop=True
                )
            )

        dfs.append(
            df
        )

        names.append(
            os.path.basename(
                path
            )
        )

    # =====================================================
    # Display the current model inputs.
    # =====================================================

    current_features = (
        BASE_FEATURES
        + [
            "DO_lag1",
            "DO_lag2",
            "DO_lag3"
        ]
    )

    print(
        "\n📌 当前 BiLSTM-Attention 输入特征："
    )

    for i, feature in enumerate(
        current_features,
        start=1
    ):

        print(
            f"   {i:02d}. {feature}"
        )

    print(
        f"\n📌 输入特征总数："
        f"{len(current_features)}"
    )

    print(
        "📌 Event_Label 已从模型输入中删除，"
        "但原始列仍保留用于事件结果分类。"
    )

    print(
        "📌 SD 和 SWE 已从 BiLSTM-Attention 输入特征中删除。"
    )

    print(
        "📌 WT 已从模型输入中删除，并保留原始缺失状态。"
    )

    # =====================================================
    # Run the DO model.
    # =====================================================

    all_eval = []
    all_scatter = []
    all_info = []

    for target in TARGETS:

        if not any(
            target in df.columns
            for df in dfs
        ):
            continue

        (
            dfs,
            eval_rows,
            scatter_rows,
            info
        ) = 运行一个目标(
            dfs,
            names,
            target
        )

        all_eval.extend(
            eval_rows
        )

        all_scatter.extend(
            scatter_rows
        )

        if info:

            all_info.append(
                info
            )

    # =====================================================
    # Export results.
    # =====================================================

    print(
        "\n💾 正在输出结果..."
    )

    final_df = pd.concat(
        dfs,
        ignore_index=True
    )

    final_df = 输出填补数据(
        final_df
    )

    输出评估报告(
        all_eval,
        all_scatter,
        all_info
    )

    print(
        "\n🎉 完成！"
    )

    print(
        f"📌 填补数据: "
        f"{IMPUTED_DIR}"
    )

    print(
        f"📌 评估报告: "
        f"{REPORT_DIR}"
    )

    print(
        f"📌 损失历史: "
        f"{LOSS_DIR}"
    )


# ===================== Script entry point =====================
if __name__ == "__main__":
    main()
