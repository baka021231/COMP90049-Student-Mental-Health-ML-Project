from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


RANDOM_STATE = 42
DATA_PATH = Path("final_student_day_table_v01_processed.csv")
OUT_DIR = Path("rq3_outputs")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed: int = RANDOM_STATE) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["student_id", "date"]).reset_index(drop=True).copy()
    df["semester_week"] = ((df["date"] - df["date"].min()).dt.days // 7) + 1

    base_numeric = [
        "anxiety",
        "sleep_score",
        "deep_sleep_minutes",
        "total_steps",
        "sedentary_minutes",
        "lightly_active_minutes",
        "moderately_active_minutes",
        "very_active_minutes",
        "avg_rmssd",
        "avg_low_frequency",
        "avg_high_frequency",
        "avg_oxygen",
        "std_oxygen",
        "STRESS_SCORE",
    ]

    lag_source = ["stress"] + base_numeric
    for col in lag_source:
        df[f"{col}_lag1"] = df.groupby("student_id")[col].shift(1)

    for window in [3, 7]:
        for col in lag_source:
            df[f"{col}_roll{window}_mean"] = df.groupby("student_id")[col].transform(
                lambda s: s.shift(1).rolling(window, min_periods=1).mean()
            )

    return df


def make_figure4(df: pd.DataFrame) -> pd.DataFrame:
    weekly = (
        df.groupby("semester_week")
        .agg(
            n=("stress", "size"),
            mean_stress=("stress", "mean"),
            sd_stress=("stress", "std"),
            week_start=("date", "min"),
            week_end=("date", "max"),
        )
        .reset_index()
    )

    plt.figure(figsize=(9, 5.2))
    plt.plot(
        weekly["semester_week"],
        weekly["mean_stress"],
        marker="o",
        linewidth=2,
        color="#2563eb",
        label="Mean stress",
    )
    plt.fill_between(
        weekly["semester_week"],
        weekly["mean_stress"] - weekly["sd_stress"],
        weekly["mean_stress"] + weekly["sd_stress"],
        color="#93c5fd",
        alpha=0.35,
        label="+/- 1 SD",
    )
    plt.xlabel("Semester week")
    plt.ylabel("Self-reported stress")
    plt.title("Figure 4. Temporal stress trend by semester week")
    plt.xticks(weekly["semester_week"])
    plt.grid(axis="y", alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "figure4_temporal_stress_trend.png", dpi=300)
    plt.close()

    weekly.to_csv(OUT_DIR / "figure4_weekly_stress_summary.csv", index=False)
    return weekly


class TabularTransformer(nn.Module):
    """A compact FT-Transformer-style classifier for tabular feature variants."""

    def __init__(
        self,
        n_numeric: int,
        cat_cardinalities: list[int],
        n_classes: int,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.n_numeric = n_numeric
        self.numeric_weight = nn.Parameter(torch.randn(n_numeric, d_model) * 0.02)
        self.numeric_bias = nn.Parameter(torch.zeros(n_numeric, d_model))
        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, d_model) for cardinality in cat_cardinalities]
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        num_tokens = x_num.unsqueeze(-1) * self.numeric_weight + self.numeric_bias
        tokens = [num_tokens]
        if len(self.cat_embeddings) > 0:
            cat_tokens = [
                emb(x_cat[:, i]).unsqueeze(1)
                for i, emb in enumerate(self.cat_embeddings)
            ]
            tokens.extend(cat_tokens)

        x = torch.cat(tokens, dim=1)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.encoder(x)
        return self.head(x[:, 0])


def prepare_fold_data(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    numeric_cols: list[str],
    categorical_cols: list[str],
):
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    num_imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_train_num = scaler.fit_transform(num_imputer.fit_transform(train_df[numeric_cols]))
    x_test_num = scaler.transform(num_imputer.transform(test_df[numeric_cols]))

    if categorical_cols:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
        )
        x_train_cat = encoder.fit_transform(
            cat_imputer.fit_transform(train_df[categorical_cols])
        ).astype("int64")
        x_test_cat = encoder.transform(
            cat_imputer.transform(test_df[categorical_cols])
        ).astype("int64")
        x_train_cat += 1
        x_test_cat += 1
        cat_cardinalities = [
            len(categories) + 1 for categories in encoder.categories_
        ]
    else:
        x_train_cat = np.zeros((len(train_df), 0), dtype="int64")
        x_test_cat = np.zeros((len(test_df), 0), dtype="int64")
        cat_cardinalities = []

    return (
        x_train_num.astype("float32"),
        x_train_cat,
        x_test_num.astype("float32"),
        x_test_cat,
        cat_cardinalities,
    )


def train_one_fold(
    x_num: np.ndarray,
    x_cat: np.ndarray,
    y: np.ndarray,
    x_test_num: np.ndarray,
    x_test_cat: np.ndarray,
    cat_cardinalities: list[int],
    n_classes: int,
) -> np.ndarray:
    train_idx, val_idx = train_test_split(
        np.arange(len(y)),
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    train_ds = TensorDataset(
        torch.tensor(x_num[train_idx]),
        torch.tensor(x_cat[train_idx], dtype=torch.long),
        torch.tensor(y[train_idx], dtype=torch.long),
    )
    val_num = torch.tensor(x_num[val_idx], device=DEVICE)
    val_cat = torch.tensor(x_cat[val_idx], dtype=torch.long, device=DEVICE)
    val_y = y[val_idx]

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    model = TabularTransformer(
        n_numeric=x_num.shape[1],
        cat_cardinalities=cat_cardinalities,
        n_classes=n_classes,
    ).to(DEVICE)

    class_counts = np.bincount(y[train_idx], minlength=n_classes)
    class_weights = class_counts.sum() / np.maximum(class_counts, 1)
    class_weights = class_weights / class_weights.mean()
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_state = None
    best_val_f1 = -1.0
    patience = 8
    stale_epochs = 0

    for _epoch in range(60):
        model.train()
        for batch_num, batch_cat, batch_y in train_loader:
            batch_num = batch_num.to(DEVICE)
            batch_cat = batch_cat.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()
            loss = criterion(model(batch_num, batch_cat), batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(val_num, val_cat).argmax(dim=1).cpu().numpy()
        val_f1 = f1_score(val_y, val_pred, average="macro")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(
            torch.tensor(x_test_num, device=DEVICE),
            torch.tensor(x_test_cat, dtype=torch.long, device=DEVICE),
        )
    return logits.argmax(dim=1).cpu().numpy()


def evaluate_table5(df: pd.DataFrame) -> pd.DataFrame:
    # Current-day stress is excluded because stress_label is derived from stress.
    # Lagged stress is allowed because it represents previous self-report history.
    base_numeric = [
        "anxiety",
        "sleep_score",
        "deep_sleep_minutes",
        "total_steps",
        "sedentary_minutes",
        "lightly_active_minutes",
        "moderately_active_minutes",
        "very_active_minutes",
        "avg_rmssd",
        "avg_low_frequency",
        "avg_high_frequency",
        "avg_oxygen",
        "std_oxygen",
        "STRESS_SCORE",
    ]
    base_categorical = ["CALCULATION_FAILED"]
    lag_source = ["stress"] + base_numeric

    configs = {
        "No temporal features": base_numeric + base_categorical,
        "Semester week only": base_numeric + base_categorical + ["semester_week"],
        "One-day lag features": base_numeric
        + base_categorical
        + ["semester_week"]
        + [f"{c}_lag1" for c in lag_source],
        "Three-day rolling means": base_numeric
        + base_categorical
        + ["semester_week"]
        + [f"{c}_lag1" for c in lag_source]
        + [f"{c}_roll3_mean" for c in lag_source],
        "Seven-day rolling means": base_numeric
        + base_categorical
        + ["semester_week"]
        + [f"{c}_lag1" for c in lag_source]
        + [f"{c}_roll7_mean" for c in lag_source],
    }

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["stress_label"])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    for config_name, cols in configs.items():
        print(f"Evaluating Transformer: {config_name}")
        numeric_cols = [c for c in cols if c not in base_categorical]
        categorical_cols = [c for c in cols if c in base_categorical]

        fold_scores = []
        for fold, (train_idx, test_idx) in enumerate(cv.split(df[cols], y), start=1):
            seed_everything(RANDOM_STATE + fold)
            (
                x_train_num,
                x_train_cat,
                x_test_num,
                x_test_cat,
                cat_cardinalities,
            ) = prepare_fold_data(
                df, train_idx, test_idx, numeric_cols, categorical_cols
            )
            y_train = y[train_idx]
            y_test = y[test_idx]
            y_pred = train_one_fold(
                x_train_num,
                x_train_cat,
                y_train,
                x_test_num,
                x_test_cat,
                cat_cardinalities,
                n_classes=len(label_encoder.classes_),
            )
            fold_scores.append(f1_score(y_test, y_pred, average="macro"))

        rows.append(
            {
                "feature_configuration": config_name,
                "model": "Tabular Transformer",
                "macro_f1_mean": np.mean(fold_scores),
                "macro_f1_sd": np.std(fold_scores),
                "n_features_before_encoding": len(cols),
            }
        )

    table5 = pd.DataFrame(rows)
    baseline = table5.loc[
        table5["feature_configuration"].eq("No temporal features"), "macro_f1_mean"
    ].iloc[0]
    table5["delta_vs_baseline"] = table5["macro_f1_mean"] - baseline
    table5.to_csv(OUT_DIR / "table5_temporal_feature_variants.csv", index=False)
    return table5


def write_summary(weekly: pd.DataFrame, table5: pd.DataFrame) -> None:
    peak = weekly.loc[weekly["mean_stress"].idxmax()]
    low = weekly.loc[weekly["mean_stress"].idxmin()]
    best = table5.loc[table5["macro_f1_mean"].idxmax()]

    summary = f"""RQ3 temporal analysis summary

Figure 4 trend:
Mean self-reported stress was lowest in week {int(low.semester_week)} \
(M={low.mean_stress:.2f}, SD={low.sd_stress:.2f}) and highest in week \
{int(peak.semester_week)} (M={peak.mean_stress:.2f}, SD={peak.sd_stress:.2f}). \
The trajectory starts at a moderate level, dips in the early semester, rises around \
weeks 5-8, drops again around weeks 9-12, and then climbs toward the end of the \
semester, with the strongest stress levels in weeks 17-19.

Table 5 best configuration:
Using a Tabular Transformer, {best.feature_configuration} achieved the highest \
macro-F1 of {best.macro_f1_mean:.3f}, an improvement of \
{best.delta_vs_baseline:+.3f} over the non-temporal baseline.
"""
    (OUT_DIR / "rq3_summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
    print("Table 5")
    print(table5.round(4).to_string(index=False))


def main() -> None:
    seed_everything()
    OUT_DIR.mkdir(exist_ok=True)
    print(f"Using device: {DEVICE}")
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = add_temporal_features(df)
    weekly = make_figure4(df)
    table5 = evaluate_table5(df)
    write_summary(weekly, table5)


if __name__ == "__main__":
    main()
