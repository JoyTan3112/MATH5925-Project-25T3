# -*- coding: utf-8 -*-
"""
main_final.py  — ICU mortality pipeline + convergence curves (Logit & XGBoost)

功能要点
--------
- 统一输出路径：输出到 ./output，图像到 ./figures
- 数据集默认：full_selected_副本.csv（与本文件在同一目录）
- 自动识别/指定标签列；非数字标签自动映射到 0/1
- 训练/测试划分：按 RecordID 做 patient-level 划分
- Logistic & XGBoost + GridSearchCV（老师精简过的搜索空间）
- 指标：AUROC, AUPRC, Accuracy, Brier, Precision/Recall/F1（正类）
- 图像：ROC, PR, 混淆矩阵、收敛曲线、校准图
- 结果表：metrics_summary.csv，run_manifest.json
- 随机种子统一设置，结果可复现
"""

import argparse
import warnings
from pathlib import Path
import os
import sys
import json
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    log_loss,
    accuracy_score,
    brier_score_loss,
    classification_report,
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, SGDClassifier

# XGBoost optional
try:
    import xgboost as xgb

    HAS_XGB = True
except Exception:
    HAS_XGB = False
    warnings.warn(
        "xgboost is not installed. XGBoost model will be skipped. "
        "Install with `pip install xgboost`."
    )

# ==========
# Seeds
# ==========
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

DEFAULT_DATA = "full_selected_副本.csv"

# =========================
# Paths & Saving Strategy
# =========================
def setup_paths(base_path: Path):
    base_path = Path(base_path).resolve()
    os.makedirs(base_path, exist_ok=True)
    output_dir = base_path / "output"
    fig_dir = base_path / "figures"
    for d in (output_dir, fig_dir):
        d.mkdir(parents=True, exist_ok=True)
    return output_dir, fig_dir


# =========================
# Target Column Detection
# =========================
CANDIDATE_TARGETS = [
    "mortality",
    "death",
    "deceased",
    "y",
    "label",
    "outcome",
    "In-hospital_death",
]


def detect_target_column(df: pd.DataFrame, candidates=CANDIDATE_TARGETS) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    last = df.columns[-1]
    if df[last].nunique() == 2:
        warnings.warn(
            f"No standard target name found; using the last column '{last}' as target (nunique=2)."
        )
        return last
    raise ValueError(f"Target column not found. Tried {candidates}. Please set --target.")


# =========================
# Threshold Utilities
# =========================
def youden_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    fpr, tpr, thresh = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    j_best_idx = int(np.argmax(j_scores))
    if j_best_idx >= len(thresh):
        j_best_idx = len(thresh) - 1
    return float(thresh[j_best_idx])


def apply_threshold(y_prob: np.ndarray, threshold: float) -> np.ndarray:
    return (y_prob >= threshold).astype(int)


# =========================
# Plotting Helpers
# =========================
def plot_roc(y_true, y_prob, title, save_path: Path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUROC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def plot_pr(y_true, y_prob, title, save_path: Path):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    plt.figure()
    plt.plot(recall, precision, label=f"AUPRC = {auprc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def plot_confusion(cm: np.ndarray, title: str, save_path: Path, labels=("Negative", "Positive")):
    plt.figure()
    plt.imshow(cm, interpolation="nearest", aspect="auto")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def plot_convergence_curve(train_losses, val_losses, save_path: Path, title: str):
    epochs = np.arange(1, len(train_losses) + 1)
    plt.figure()
    plt.plot(epochs, train_losses, label="train loss")
    plt.plot(epochs, val_losses, label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def plot_calibration_main(
    y_true,
    y_prob,
    threshold,
    n_bins=10,
    strategy="quantile",
    min_count=10,
    title="Calibration Curve",
    save_path="calibration.png",
):
    """
    绘制校准曲线 + 风险直方图 + Youden 阈值
    支持 quantile / uniform
    """
    # 分箱
    if strategy == "quantile":
        try:
            bins = pd.qcut(y_prob, q=n_bins, duplicates="drop")
        except ValueError:
            print("⚠️ qcut 分箱失败，请减少 n_bins 或使用 uniform。")
            return
    elif strategy == "uniform":
        bins = pd.cut(y_prob, bins=n_bins)
    else:
        raise ValueError("strategy must be 'quantile' or 'uniform'")

    df_cal = pd.DataFrame({"y_true": y_true, "y_prob": y_prob, "bin": bins})
    bin_stats = df_cal.groupby("bin", observed=False).agg(
        mean_predicted_prob=("y_prob", "mean"),
        observed_rate=("y_true", "mean"),
        count=("y_true", "count"),
    ).dropna()

    bin_stats = bin_stats[bin_stats["count"] >= min_count]
    if bin_stats.empty:
        print("⚠️ 有效分箱为空，无法绘制校准图。")
        return

    hist_counts, hist_bins = np.histogram(y_prob, bins=n_bins, range=(0, 1))
    bin_centers = 0.5 * (hist_bins[:-1] + hist_bins[1:])

    fig, ax = plt.subplots(
        2, 1, figsize=(7, 9), gridspec_kw={"height_ratios": [2, 1]}, sharex=True
    )

    # 校准曲线
    ax[0].plot(
        bin_stats["mean_predicted_prob"],
        bin_stats["observed_rate"],
        marker="o",
        linestyle="-",
        color="tab:orange",
        label="Calibration",
    )
    ax[0].plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")
    ax[0].axvline(threshold, color="red", linestyle=":", label=f"Th={threshold:.3f}")
    ax[0].set_ylabel("Observed Event Rate")
    ax[0].set_title(title)
    ax[0].legend()
    ax[0].grid(True, linestyle=":", alpha=0.6)

    # 风险直方图
    ax[1].bar(
        bin_centers,
        hist_counts,
        width=(1 / n_bins) * 0.95,
        edgecolor="black",
        color="steelblue",
        alpha=0.7,
    )
    ax[1].axvline(threshold, color="red", linestyle=":")
    ax[1].set_xlabel("Predicted Probability")
    ax[1].set_ylabel("Count")
    ax[1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


# =========================
# Modeling Routines
# =========================
def evaluate_model(y_true, y_prob, threshold=None):
    """
    计算完整的一套二分类指标：
    - accuracy, AUROC, AUPRC, Brier
    - precision/recall/F1（正类）
    - specificity, NPV
    - TP/FP/TN/FN 及 FPR/FNR/TPR/TNR
    同时返回 (metrics_dict, confusion_matrix)
    """
    if threshold is None:
        threshold = youden_optimal_threshold(y_true, y_prob)

    y_pred = apply_threshold(y_prob, threshold)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    acc = accuracy_score(y_true, y_pred)
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)

    precision_pos = precision_score(y_true, y_pred, zero_division=0)
    recall_pos = recall_score(y_true, y_pred, zero_division=0)
    f1_pos = f1_score(y_true, y_pred, zero_division=0)

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    tpr = recall_pos
    tnr = specificity

    metrics = {
        "threshold": float(threshold),
        "accuracy": float(acc),
        "auroc": float(auroc),
        "auprc": float(auprc),
        "brier": float(brier),
        "precision_pos": float(precision_pos),
        "recall_pos": float(recall_pos),
        "f1_pos": float(f1_pos),
        "specificity": float(specificity),
        "npv": float(npv),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "tpr": float(tpr),
        "tnr": float(tnr),
    }

    cm = np.array([[tn, fp], [fn, tp]])
    return metrics, cm


def logistic_pipeline():
    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "clf",
                LogisticRegression(
                    max_iter=200,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    param_grid = {
        "clf__C": [0.1, 1.0, 3.0, 10.0],
        "clf__penalty": ["l2"],
        "clf__solver": ["liblinear", "lbfgs"],
    }
    return pipe, param_grid


def xgb_pipeline():
    if not HAS_XGB:
        return None, None
    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                xgb.XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                    tree_method="hist",
                ),
            ),
        ]
    )
    param_grid = {
        "clf__n_estimators": [200, 300, 400, 500],
        "clf__max_depth": [3, 5],
        "clf__learning_rate": [0.03, 0.1, 0.2],
        "clf__subsample": [0.8],
        "clf__colsample_bytree": [0.8],
    }
    return pipe, param_grid


def fit_with_cv_grid(estimator, param_grid, X_train, y_train, scoring="roc_auc", n_splits=5):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        estimator,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True,
        return_train_score=False,
    )
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    return grid, best


# =========================
# Convergence helpers
# =========================
def train_logistic_with_convergence(X_train, y_train, X_val, y_val, n_epochs=50):
    classes = np.unique(y_train)
    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        max_iter=1,
        learning_rate="optimal",
        random_state=RANDOM_STATE,
        warm_start=True,
    )

    train_losses = []
    val_losses = []
    for _ in range(n_epochs):
        clf.partial_fit(X_train, y_train, classes=classes)
        y_tr_prob = clf.predict_proba(X_train)[:, 1]
        y_val_prob = clf.predict_proba(X_val)[:, 1]
        train_losses.append(log_loss(y_train, y_tr_prob))
        val_losses.append(log_loss(y_val, y_val_prob))
    return train_losses, val_losses


def train_xgb_with_convergence(X_train, y_train, X_val, y_val, num_boost_round=50):
    if not HAS_XGB:
        return [], []

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        tree_method="hist",
        n_estimators=num_boost_round,
        learning_rate=0.1,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False,
    )
    evals_result = model.evals_result()
    train_losses = evals_result["validation_0"]["logloss"]
    val_losses = evals_result["validation_1"]["logloss"]
    return train_losses, val_losses


# =========================
# Main
# =========================
def run(args):
    base_dir = Path(__file__).resolve().parent
    output_dir, fig_dir = setup_paths(base_dir)

    # ---- Load data ----
    data_path = Path(args.data) if args.data else (base_dir / DEFAULT_DATA)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    try:
        df = pd.read_csv(data_path)
    except UnicodeDecodeError:
        df = pd.read_csv(data_path, encoding="utf-8-sig")

    # ---- Patient-level split by RecordID ----
    if "RecordID" not in df.columns:
        raise ValueError("Column 'RecordID' not found in dataframe, cannot do patient-level split.")

    patient_ids = df["RecordID"].unique()
    train_ids, test_ids = train_test_split(
        patient_ids,
        test_size=args.test_size,
        random_state=RANDOM_STATE,
    )
    train_mask = df["RecordID"].isin(train_ids).values
    test_mask = df["RecordID"].isin(test_ids).values

    sorted_test_ids = np.sort(test_ids)
    print("\n=== Test patient IDs (RecordID, patient-level split) ===")
    print(", ".join(str(x) for x in sorted_test_ids))

    # ---- Target column ----
    target_col = args.target if args.target else detect_target_column(df)

    y_raw = df[target_col]
    if y_raw.dtype.kind not in "biufc":
        if args.positive_label is None:
            pos_tokens = {
                "yes",
                "y",
                "true",
                "t",
                "death",
                "dead",
                "deceased",
                "1",
                "positive",
                "pos",
            }
            y = y_raw.astype(str).str.lower().isin(pos_tokens).astype(int).values
            warnings.warn(
                "Non-numeric target mapped via common tokens. "
                "Use --positive-label to control mapping explicitly."
            )
        else:
            y = (y_raw.astype(str) == str(args.positive_label)).astype(int).values
    else:
        y = y_raw.astype(float)
        uniq = np.unique(y[~np.isnan(y)])
        if len(uniq) == 2 and set(uniq) != {0.0, 1.0}:
            mapping = {float(min(uniq)): 0, float(max(uniq)): 1}
            y = np.vectorize(mapping.get)(y).astype(int)
        else:
            y = (y > 0.5).astype(int)

    # ---- Features (drop ID & target & meta) ----
    drop_cols = {target_col, "RecordID"} | set(args.drop_cols or [])
    for candidate in [
        "patient_id",
        "stay_id",
        "hadm_id",
        "subject_id",
        "icustay_id",
        "timestamp",
        "charttime",
        "time",
    ]:
        if candidate in df.columns:
            drop_cols.add(candidate)

    feature_names = [c for c in df.columns if c not in drop_cols]
    X = df[feature_names].values

    # ---- Train / test split (by patient masks) ----
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    # 训练集内部再划 validation，用于 convergence curve
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=RANDOM_STATE
    )

    # 可选：测试不同随机种子的稳定性（普通 stratified split）
    test_split_stability(df, feature_names, target_col)

    results = []
    artifacts = []

    # ========================
    # Logistic Regression
    # ========================
    logi_pipe, logi_grid = logistic_pipeline()
    logi_gridcv, logi_best = fit_with_cv_grid(
        logi_pipe, logi_grid, X_train, y_train, scoring="roc_auc", n_splits=args.cv_splits
    )

    y_prob_logi = logi_best.predict_proba(X_test)[:, 1]
    threshold_logi = (
        args.threshold if args.threshold is not None else youden_optimal_threshold(y_test, y_prob_logi)
    )
    metrics_logi, cm_logi = evaluate_model(y_test, y_prob_logi, threshold=threshold_logi)

    roc_path_logi = fig_dir / "roc_logistic.png"
    pr_path_logi = fig_dir / "pr_logistic.png"
    cm_path_logi = fig_dir / "cm_logistic.png"
    calib_path_logi = fig_dir / "calibration_logistic.png"

    plot_roc(y_test, y_prob_logi, "ROC - Logistic Regression", roc_path_logi)
    plot_pr(y_test, y_prob_logi, "PR Curve - Logistic Regression", pr_path_logi)
    plot_confusion(
        cm_logi, f"Confusion Matrix - Logistic (thr={metrics_logi['threshold']:.3f})", cm_path_logi
    )
    plot_calibration_main(
        y_true=y_test,
        y_prob=y_prob_logi,
        threshold=threshold_logi,
        n_bins=10,
        strategy="quantile",
        title="Logistic Calibration (Quantile)",
        save_path=calib_path_logi,
    )

    y_pred_logi = apply_threshold(y_prob_logi, threshold_logi)
    report_logi = classification_report(y_test, y_pred_logi, digits=4)
    print("\n=== Classification Report (Logistic Regression) ===")
    print(report_logi)
    report_logi_path = output_dir / "classification_report_logistic.txt"
    with open(report_logi_path, "w") as f:
        f.write(report_logi)

    results.append(
        {
            "model": "LogisticRegression",
            **metrics_logi,
            "best_params": json.dumps(logi_gridcv.best_params_),
            "roc_path": str(roc_path_logi),
            "pr_path": str(pr_path_logi),
            "cm_path": str(cm_path_logi),
            "calibration_path": str(calib_path_logi),
            "report_path": str(report_logi_path),
        }
    )
    artifacts += [roc_path_logi, pr_path_logi, cm_path_logi, calib_path_logi, report_logi_path]

    # Logistic convergence curve
    imp = SimpleImputer(strategy="median")
    X_tr_imp = imp.fit_transform(X_tr)
    X_val_imp = imp.transform(X_val)
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_imp)
    X_val_scaled = scaler.transform(X_val_imp)

    logi_train_losses, logi_val_losses = train_logistic_with_convergence(
        X_tr_scaled, y_tr, X_val_scaled, y_val, n_epochs=50
    )
    logi_conv_path = fig_dir / "logistic_convergence_curve.png"
    plot_convergence_curve(
        logi_train_losses,
        logi_val_losses,
        logi_conv_path,
        title="Convergence Curve - Logistic Regression",
    )
    artifacts.append(logi_conv_path)

    # ========================
    # XGBoost
    # ========================
    if HAS_XGB:
        xgb_pipe, xgb_grid = xgb_pipeline()
        xgb_gridcv, xgb_best = fit_with_cv_grid(
            xgb_pipe, xgb_grid, X_train, y_train, scoring="roc_auc", n_splits=args.cv_splits
        )

        y_prob_xgb = xgb_best.predict_proba(X_test)[:, 1]
        threshold_xgb = (
            args.threshold if args.threshold is not None else youden_optimal_threshold(y_test, y_prob_xgb)
        )
        metrics_xgb, cm_xgb = evaluate_model(y_test, y_prob_xgb, threshold=threshold_xgb)

        roc_path_xgb = fig_dir / "roc_xgboost.png"
        pr_path_xgb = fig_dir / "pr_xgboost.png"
        cm_path_xgb = fig_dir / "cm_xgboost.png"
        calib_path_xgb = fig_dir / "calibration_xgboost.png"

        plot_roc(y_test, y_prob_xgb, "ROC - XGBoost", roc_path_xgb)
        plot_pr(y_test, y_prob_xgb, "PR Curve - XGBoost", pr_path_xgb)
        plot_confusion(
            cm_xgb, f"Confusion Matrix - XGBoost (thr={metrics_xgb['threshold']:.3f})", cm_path_xgb
        )
        plot_calibration_main(
            y_true=y_test,
            y_prob=y_prob_xgb,
            threshold=threshold_xgb,
            n_bins=10,
            strategy="quantile",
            title="XGBoost Calibration (Quantile)",
            save_path=calib_path_xgb,
        )

        y_pred_xgb = apply_threshold(y_prob_xgb, threshold_xgb)
        report_xgb = classification_report(y_test, y_pred_xgb, digits=4)
        print("\n=== Classification Report (XGBoost) ===")
        print(report_xgb)
        report_xgb_path = output_dir / "classification_report_xgboost.txt"
        with open(report_xgb_path, "w") as f:
            f.write(report_xgb)

        # Feature importance (Top 30)
        fi_path_str = ""
        try:
            fitted_xgb = xgb_best.named_steps["clf"]
            importances = fitted_xgb.feature_importances_
            if len(importances) == len(feature_names):
                imp_df = (
                    pd.DataFrame({"feature": feature_names, "importance": importances})
                    .sort_values("importance", ascending=False)
                    .head(30)
                )
                imp_path = fig_dir / "feature_importance_xgb_top30.png"
                plt.figure(figsize=(8, 10))
                plt.barh(imp_df["feature"][::-1], imp_df["importance"][::-1])
                plt.title("XGBoost Feature Importance (Top 30)")
                plt.tight_layout()
                plt.savefig(imp_path, dpi=160)
                plt.close()
                artifacts.append(imp_path)
                fi_path_str = str(imp_path)
        except Exception as e:
            warnings.warn(f"Failed to plot XGBoost feature importance: {e}")

        results.append(
            {
                "model": "XGBoost",
                **metrics_xgb,
                "best_params": json.dumps(xgb_gridcv.best_params_),
                "roc_path": str(roc_path_xgb),
                "pr_path": str(pr_path_xgb),
                "cm_path": str(cm_path_xgb),
                "calibration_path": str(calib_path_xgb),
                "report_path": str(report_xgb_path),
                "feature_importance_path": fi_path_str,
            }
        )
        artifacts += [roc_path_xgb, pr_path_xgb, cm_path_xgb, calib_path_xgb, report_xgb_path]

        # XGBoost convergence
        imp2 = SimpleImputer(strategy="median")
        X_tr_imp2 = imp2.fit_transform(X_tr)
        X_val_imp2 = imp2.transform(X_val)
        xgb_train_losses, xgb_val_losses = train_xgb_with_convergence(
            X_tr_imp2, y_tr, X_val_imp2, y_val, num_boost_round=50
        )
        xgb_conv_path = fig_dir / "xgboost_convergence_curve.png"
        plot_convergence_curve(
            xgb_train_losses,
            xgb_val_losses,
            xgb_conv_path,
            title="Convergence Curve - XGBoost",
        )
        artifacts.append(xgb_conv_path)

    # ========================
    # Save metrics & manifest
    # ========================
    results_df = pd.DataFrame(results)
    metrics_csv = output_dir / "metrics_summary.csv"
    results_df.to_csv(metrics_csv, index=False)

    print("\n=== Metrics Summary ===")
    print(
        results_df[
            [
                "model",
                "accuracy",
                "auroc",
                "auprc",
                "brier",
                "precision_pos",
                "recall_pos",
                "f1_pos",
                "threshold",
            ]
        ]
    )
    print(f"\nSaved metrics table to: {metrics_csv}")
    print(f"Figures & reports saved to: {fig_dir} / {output_dir}")

    manifest = {
        "data_path": str(data_path),
        "target_col": target_col,
        "n_samples": int(df.shape[0]),
        "n_features": int(len(feature_names)),
        "class_balance": {
            "positive_rate": float(np.mean(y)),
            "negative_rate": float(1.0 - np.mean(y)),
        },
        "artifacts": [str(p) for p in artifacts],
        "metrics_csv": str(metrics_csv),
        "notes": "Thresholds based on Youden-J unless overridden by --threshold. Patient-level split by RecordID.",
        "dataset_default": DEFAULT_DATA,
    }
    manifest_path = output_dir / "run_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"Saved manifest to: {manifest_path}")


# =========================
# Split stability test
# =========================
def test_split_stability(df, feature_names, target_col, seeds=[0, 1, 2, 3, 4]):
    """
    用普通 stratified train_test_split 检查不同随机种子下
    Logistic (固定超参) 的 AUROC/AUPRC/Brier 的稳定性
    （只是写在 thesis 里说明 seed 变化影响不大）
    """
    print("\n=== Testing Train/Test Split Stability (multiple seeds) ===\n")

    X = df[feature_names].values
    y_raw = df[target_col].values

    if y_raw.dtype.kind not in "biufc":
        pos_tokens = {"yes", "y", "true", "t", "death", "dead", "1", "positive", "pos"}
        y = pd.Series(y_raw).astype(str).str.lower().isin(pos_tokens).astype(int).values
    else:
        y = (y_raw.astype(float) > 0.5).astype(int)

    logi_model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=200, class_weight="balanced")),
        ]
    )

    results = []

    for sd in seeds:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=sd
        )

        logi_model.fit(X_train, y_train)
        y_prob = logi_model.predict_proba(X_test)[:, 1]

        auroc = roc_auc_score(y_test, y_prob)
        auprc = average_precision_score(y_test, y_prob)
        brier = np.mean((y_test - y_prob) ** 2)

        results.append([sd, auroc, auprc, brier])

    df_res = pd.DataFrame(results, columns=["seed", "AUROC", "AUPRC", "Brier"])
    print(df_res)
    print("\n=== Mean Performance ===")
    print(df_res[["AUROC", "AUPRC", "Brier"]].mean())

    return df_res


# =========================
# CLI
# =========================
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="ICU Mortality Modeling Pipeline + Convergence Curves"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help=f"Path to CSV data (default: ./{DEFAULT_DATA})",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Target column name. If omitted, auto-detect from common names or last binary column.",
    )
    parser.add_argument(
        "--positive-label",
        type=str,
        default=None,
        help="When target is non-numeric, which value counts as positive (e.g., 'Yes' or 'Death').",
    )
    parser.add_argument(
        "--drop-cols",
        type=str,
        nargs="*",
        default=None,
        help="Extra columns to drop from features.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test size proportion (default 0.2).",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Number of CV folds (default 5).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional fixed decision threshold (e.g., 0.121). If omitted, use Youden-J.",
    )
    return parser.parse_args(argv)


# =========================
# Entry
# =========================
if __name__ == "__main__":
    args = parse_args()
    try:
        run(args)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
