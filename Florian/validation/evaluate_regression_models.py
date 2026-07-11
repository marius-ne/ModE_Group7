"""
Evaluate regression models on new test datasets and export results.

This script:
1. Loads 1D regression models (trained on 2, 5, 20, 40 samples) 
2. Loads 2D regression models (trained on 40 samples with discrete prices)
3. Applies them to the corresponding test datasets
4. Computes R² scores and predictions
5. Exports results in the same format as the archive files

For 1D models:
- Test data contains 'ratio' feature (gas_price / electricity_price)
- OPEX values are specific (normalized)
- To convert to absolute OPEX: multiply by electricity price
- The electricity price is inferred from archive validation files
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple

import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score


# ============================================================
# Configuration
# ============================================================

# 1D Model Setup (m·x + b)
MODEL_DIR_1D = Path(__file__).resolve().parents[1] / "validation" / "joblibs"
TEST_DATA_1D = Path(__file__).resolve().parents[2] / "Marius" / "results" / "evaluation_lhs_10_test_1D.csv"
OUTPUT_DIR_1D = Path(__file__).resolve().parents[1] / "validation" / "results_1d_models"
ARCHIVE_DIR = Path(__file__).resolve().parents[1] / "validation" / "archive"

# 2D Model Setup (a·c_g + b·c_el + c)
TEST_DATA_2D = Path(__file__).resolve().parents[2] / "Marius" / "results" / "evaluation_lhs_10_test_2D.csv"
OUTPUT_DIR_2D = Path(__file__).resolve().parents[1] / "validation" / "results_2d_models"

# Training dataset directories and output directories for training-set evaluation
TRAINING_OUTPUT_DIR_1D = Path(__file__).resolve().parents[1] / "validation" / "training_results_1d_models"
TRAINING_OUTPUT_DIR_2D = Path(__file__).resolve().parents[1] / "validation" / "training_results_2d_models"
TRAINING_SAMPLE_PATHS_1D = {
    2: Path(__file__).resolve().parents[2] / "Marius" / "results" / "evaluation_2_training_samples_1D.csv",
    5: Path(__file__).resolve().parents[2] / "Marius" / "results" / "evaluation_5_training_samples_1D.csv",
    20: Path(__file__).resolve().parents[2] / "Marius" / "results" / "evaluation_20_training_samples_1D.csv",
    40: Path(__file__).resolve().parents[2] / "Marius" / "results" / "evaluation_training_samples_1D_angle.csv",
}
TRAINING_SAMPLE_PATH_2D = Path(__file__).resolve().parents[2] / "Marius" / "results" / "evaluation_training_samples_2D.csv"

# Training sample sizes for 1D models
TRAINING_SIZES_1D = [2, 5, 20, 40]

# Model prefixes and target types
OPEX_TARGETS = ["milp", "lp_lower", "lp_upper", "lp_approx"]
TARGET_TO_FULL_NAME = {
    "milp": "opex_milp",
    "lp_lower": "opex_lp_lower",
    "lp_upper": "opex_lp_upper",
    "lp_approx": "opex_lp_approx",
}
FULL_NAME_TO_TARGET = {v: k for k, v in TARGET_TO_FULL_NAME.items()}

MODEL_LABELS = {
    "opex_milp": "MILP",
    "opex_lp_lower": "LP lower",
    "opex_lp_upper": "LP upper",
    "opex_lp_approx": "LP approx",
}

MODEL_ORDER = ["MILP", "LP lower", "LP upper", "LP approx"]


class PredictionResult(NamedTuple):
    """Result of model prediction on test data."""
    y_test: np.ndarray | pd.Series
    y_pred: np.ndarray
    r2: float


# ============================================================
# 1D Model Functions (m·x + b)
# ============================================================

def load_1d_test_data() -> pd.DataFrame:
    """Load 1D test data with ratio feature."""
    df = pd.read_csv(TEST_DATA_1D)
    return df


def load_1d_training_data(training_size: int) -> pd.DataFrame:
    """Load the 1D training sample used for the given training size."""
    if training_size not in TRAINING_SAMPLE_PATHS_1D:
        raise ValueError(f"Unknown 1D training size: {training_size}")
    training_path = TRAINING_SAMPLE_PATHS_1D[training_size]
    if not training_path.exists():
        raise FileNotFoundError(f"1D training sample not found: {training_path}")
    return pd.read_csv(training_path)


def predict_1d_single_model_on_training_data(
    model_path: Path,
    train_df: pd.DataFrame,
    target: str,
) -> PredictionResult:
    """Make predictions on the 1D model's own training sample."""
    model = joblib.load(model_path)
    X = train_df[["ratio"]].values
    y_test = train_df[target].values
    y_pred = model.predict(X)
    r2 = r2_score(y_test, y_pred)
    return PredictionResult(y_test=y_test, y_pred=y_pred, r2=r2)


def infer_price_multiplier_1d(target: str) -> float:
    """
    Infer electricity price multiplier from archive validation files.
    
    The archive files contain:
    - y_test: actual absolute OPEX values
    - y_pred: predicted specific OPEX values from the model
    
    The multiplier is estimated from the relationship between y_test and y_pred.
    """
    # Try to find a training file with this target
    archive_paths = sorted(ARCHIVE_DIR.glob(f"*_train_*_test_ratio_opex_{target}.csv"))
    
    if not archive_paths:
        # Fallback: use typical electricity price
        print(f"    INFO: No archive files found for {target}, using default multiplier 100")
        return 100.0
    
    # Use the first available file to estimate multiplier
    archive_path = archive_paths[0]
    try:
        archive_df = pd.read_csv(archive_path)
        # Average multiplier: y_test / y_pred
        multipliers = archive_df["y_test"] / archive_df["y_pred"]
        median_mult = multipliers.median()
        mean_mult = multipliers.mean()
        print(f"    INFO: Estimated price multiplier from {archive_path.name}: "
              f"median={median_mult:.2f}, mean={mean_mult:.2f}")
        return median_mult
    except Exception as e:
        print(f"    WARNING: Could not infer multiplier from {archive_path.name}: {e}")
        return 100.0


def predict_1d_single_model(
    model_path: Path,
    test_df: pd.DataFrame,
    price_multiplier: float,
    target: str,
) -> PredictionResult:
    """
    Make predictions with a single 1D model.
    
    Args:
        model_path: Path to .joblib model
        test_df: Test dataframe with 'ratio' column
        price_multiplier: Electricity price multiplier to convert specific to absolute OPEX
        target: Target OPEX type (e.g., "opex_milp")
    
    Returns:
        PredictionResult with y_test, y_pred, r2
    """
    model = joblib.load(model_path)
    
    # Extract ratio feature
    X = test_df[["ratio"]].values
    
    # Get true OPEX values (specific, need to multiply to get absolute)
    y_test_specific = test_df[target].values
    y_test_absolute = y_test_specific * price_multiplier
    
    # Make predictions (model returns specific OPEX)
    y_pred_specific = model.predict(X)
    y_pred_absolute = y_pred_specific * price_multiplier
    
    # Calculate R² on absolute values
    r2 = r2_score(y_test_absolute, y_pred_absolute)
    
    return PredictionResult(
        y_test=y_test_absolute,
        y_pred=y_pred_absolute,
        r2=r2,
    )


def evaluate_1d_models():
    """Evaluate all 1D regression models."""
    OUTPUT_DIR_1D.mkdir(parents=True, exist_ok=True)
    
    test_df = load_1d_test_data()
    
    # Collect results for R² summary
    r2_summary_data = {size: {} for size in TRAINING_SIZES_1D}
    
    print(f"\n{'='*70}")
    print("EVALUATING 1D MODELS (m·x + b)")
    print(f"{'='*70}")
    print(f"Test data shape: {test_df.shape}")
    print(f"Ratio range: [{test_df['ratio'].min():.4f}, {test_df['ratio'].max():.4f}]")
    
    for training_size in TRAINING_SIZES_1D:
        print(f"\n--- Training size: {training_size} ---")
        
        for target in OPEX_TARGETS:
            full_target_name = TARGET_TO_FULL_NAME[target]
            
            # Load model
            model_pattern = f"{training_size}_ratio_opex_{target}.joblib"
            model_path = MODEL_DIR_1D / model_pattern
            
            if not model_path.exists():
                print(f"  WARNING: Model not found: {model_path.name}")
                continue
            
            # Infer price multiplier
            price_mult = infer_price_multiplier_1d(target)
            
            # Predict
            result = predict_1d_single_model(
                model_path,
                test_df,
                price_mult,
                full_target_name,
            )
            
            print(f"  {full_target_name:20} R² = {result.r2:.6f}")
            
            # Save predictions CSV
            output_csv = OUTPUT_DIR_1D / f"{training_size}_train_10_test_ratio_opex_{target}.csv"
            pred_df = pd.DataFrame({
                "y_test": result.y_test,
                "y_pred": result.y_pred,
                "r2": result.r2,
            })
            pred_df.to_csv(output_csv, index=False)
            
            # Store for summary
            r2_summary_data[training_size][full_target_name] = result.r2
    
    # Export R² summary
    summary_df = pd.DataFrame(r2_summary_data).T
    summary_df = summary_df[[f"opex_{t}" for t in OPEX_TARGETS]]
    summary_df.index.name = "training_size"
    summary_df.columns = [MODEL_LABELS[col] for col in summary_df.columns]
    summary_df = summary_df[MODEL_ORDER]
    
    summary_path = OUTPUT_DIR_1D / "r2_scores_summary.csv"
    summary_df.to_csv(summary_path)
    print(f"\n[OK] 1D R² summary saved to: {summary_path.name}")


def evaluate_1d_models_on_training_samples():
    """Evaluate all 1D models on their own training samples."""
    TRAINING_OUTPUT_DIR_1D.mkdir(parents=True, exist_ok=True)
    r2_summary_data = {size: {} for size in TRAINING_SIZES_1D}

    print(f"\n{'='*70}")
    print("EVALUATING 1D MODELS ON TRAINING SAMPLES")
    print(f"{'='*70}")

    for training_size in TRAINING_SIZES_1D:
        train_df = load_1d_training_data(training_size)
        print(f"\n--- Training size: {training_size} (training sample shape: {train_df.shape}) ---")

        for target in OPEX_TARGETS:
            full_target_name = TARGET_TO_FULL_NAME[target]
            model_pattern = f"{training_size}_ratio_opex_{target}.joblib"
            model_path = MODEL_DIR_1D / model_pattern
            if not model_path.exists():
                print(f"  WARNING: Model not found: {model_path.name}")
                continue

            result = predict_1d_single_model_on_training_data(model_path, train_df, full_target_name)
            print(f"  {full_target_name:20} R² = {result.r2:.6f}")

            output_csv = TRAINING_OUTPUT_DIR_1D / f"{training_size}_train_ratio_opex_{target}.csv"
            pred_df = pd.DataFrame({
                "y_test": result.y_test,
                "y_pred": result.y_pred,
                "r2": result.r2,
            })
            pred_df.to_csv(output_csv, index=False)
            r2_summary_data[training_size][full_target_name] = result.r2

    summary_df = pd.DataFrame(r2_summary_data).T
    summary_df = summary_df[[f"opex_{t}" for t in OPEX_TARGETS]]
    summary_df.index.name = "training_size"
    summary_df.columns = [MODEL_LABELS[col] for col in summary_df.columns]
    summary_df = summary_df[MODEL_ORDER]

    summary_path = TRAINING_OUTPUT_DIR_1D / "r2_scores_summary.csv"
    summary_df.to_csv(summary_path)
    print(f"\n[OK] 1D training-sample R² summary saved to: {summary_path.name}")


def load_2d_training_data() -> pd.DataFrame:
    """Load the 2D training sample used for the 2D model."""
    if not TRAINING_SAMPLE_PATH_2D.exists():
        raise FileNotFoundError(f"2D training sample not found: {TRAINING_SAMPLE_PATH_2D}")
    return pd.read_csv(TRAINING_SAMPLE_PATH_2D)


def predict_2d_single_model_on_training_data(
    model_path: Path,
    train_df: pd.DataFrame,
    target: str,
) -> PredictionResult:
    """Make predictions on the 2D model's own training sample."""
    model = joblib.load(model_path)
    X = train_df[["gas_price_MWh", "electricity_price_MWh"]].values
    y_test = train_df[target].values
    y_pred = model.predict(X)
    r2 = r2_score(y_test, y_pred)
    return PredictionResult(y_test=y_test, y_pred=y_pred, r2=r2)


def evaluate_2d_models_on_training_samples():
    """Evaluate the 2D discrete-price models on the 2D training sample."""
    TRAINING_OUTPUT_DIR_2D.mkdir(parents=True, exist_ok=True)
    train_df = load_2d_training_data()

    print(f"\n{'='*70}")
    print("EVALUATING 2D MODELS ON TRAINING SAMPLES")
    print(f"{'='*70}")
    print(f"Training data shape: {train_df.shape}")
    print(f"Gas price range: [{train_df['gas_price_MWh'].min():.2f}, {train_df['gas_price_MWh'].max():.2f}]")
    print(f"Electricity price range: [{train_df['electricity_price_MWh'].min():.2f}, {train_df['electricity_price_MWh'].max():.2f}]")

    r2_summary_data: dict[str, float] = {}

    for target in OPEX_TARGETS:
        full_target_name = TARGET_TO_FULL_NAME[target]
        model_path = find_2d_model_path(target)
        if model_path is None:
            print(f"  WARNING: No 2D model found for {target}")
            continue

        result = predict_2d_single_model_on_training_data(model_path, train_df, full_target_name)
        print(f"  {full_target_name:20} R² = {result.r2:.6f}")

        output_csv = TRAINING_OUTPUT_DIR_2D / f"40_train_2d_discrete_opex_{target}.csv"
        pred_df = pd.DataFrame({
            "y_test": result.y_test,
            "y_pred": result.y_pred,
            "r2": result.r2,
        })
        pred_df.to_csv(output_csv, index=False)
        r2_summary_data[full_target_name] = result.r2

    summary_df = pd.DataFrame([r2_summary_data])
    summary_df = summary_df[[f"opex_{t}" for t in OPEX_TARGETS]]
    summary_df.columns = [MODEL_LABELS[col] for col in summary_df.columns]
    summary_df = summary_df[MODEL_ORDER]

    summary_path = TRAINING_OUTPUT_DIR_2D / "r2_scores_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[OK] 2D training-sample R² summary saved to: {summary_path.name}")


def evaluate_1d_models_vs_milp_on_training_samples():
    """Compute R² for every 1D model prediction against actual MILP OPEX on training data."""
    TRAINING_OUTPUT_DIR_1D.mkdir(parents=True, exist_ok=True)
    rows = []

    print(f"\n{'='*70}")
    print("EVALUATING 1D MODELS VS MILP ON TRAINING SAMPLES")
    print(f"{'='*70}")

    for training_size in TRAINING_SIZES_1D:
        train_df = load_1d_training_data(training_size)
        y_actual = train_df["opex_milp"].values
        print(f"\n--- Training size: {training_size} (shape {train_df.shape}) ---")

        for target in OPEX_TARGETS:
            full_target_name = TARGET_TO_FULL_NAME[target]
            model_path = MODEL_DIR_1D / f"{training_size}_ratio_opex_{target}.joblib"
            if not model_path.exists():
                print(f"  WARNING: Model not found: {model_path.name}")
                continue

            model = joblib.load(model_path)
            y_pred = model.predict(train_df[["ratio"]].values)
            r2 = r2_score(y_actual, y_pred)
            print(f"  {full_target_name:20} vs MILP R² = {r2:.6f}")

            output_csv = TRAINING_OUTPUT_DIR_1D / f"{training_size}_train_ratio_opex_{target}_vs_milp.csv"
            pd.DataFrame({
                "y_test_milp": y_actual,
                "y_pred_model": y_pred,
                "r2": r2,
            }).to_csv(output_csv, index=False)

            rows.append({
                "training_size": training_size,
                "model": MODEL_LABELS[full_target_name],
                "r2_vs_milp": r2,
            })

    summary_df = pd.DataFrame(rows)
    if not summary_df.empty:
        summary_df = summary_df.pivot(index="training_size", columns="model", values="r2_vs_milp")
        summary_df = summary_df[MODEL_ORDER]
        summary_path = TRAINING_OUTPUT_DIR_1D / "r2_vs_milp_summary.csv"
        summary_df.to_csv(summary_path)
        print(f"\n[OK] 1D models vs MILP R² summary saved to: {summary_path.name}")
    else:
        print("\nWARNING: No 1D models vs MILP R² results were generated.")


def evaluate_2d_models_vs_milp_on_training_samples():
    """Compute R² for every 2D model prediction against actual MILP OPEX on 2D training data."""
    TRAINING_OUTPUT_DIR_2D.mkdir(parents=True, exist_ok=True)
    train_df = load_2d_training_data()
    y_actual = train_df["opex_milp"].values

    print(f"\n{'='*70}")
    print("EVALUATING 2D MODELS VS MILP ON TRAINING SAMPLES")
    print(f"{'='*70}")
    print(f"Training data shape: {train_df.shape}")

    rows = []

    for target in OPEX_TARGETS:
        full_target_name = TARGET_TO_FULL_NAME[target]
        model_path = find_2d_model_path(target)
        if model_path is None:
            print(f"  WARNING: No 2D model found for {target}")
            continue

        model = joblib.load(model_path)
        y_pred = model.predict(train_df[["gas_price_MWh", "electricity_price_MWh"]].values)
        r2 = r2_score(y_actual, y_pred)
        print(f"  {full_target_name:20} vs MILP R² = {r2:.6f}")

        output_csv = TRAINING_OUTPUT_DIR_2D / f"40_train_2d_discrete_opex_{target}_vs_milp.csv"
        pd.DataFrame({
            "y_test_milp": y_actual,
            "y_pred_model": y_pred,
            "r2": r2,
        }).to_csv(output_csv, index=False)

        rows.append({
            "model": MODEL_LABELS[full_target_name],
            "r2_vs_milp": r2,
        })

    summary_df = pd.DataFrame(rows)
    if not summary_df.empty:
        summary_df = summary_df.set_index("model")["r2_vs_milp"].to_frame().T
        summary_df = summary_df[MODEL_ORDER]
        summary_path = TRAINING_OUTPUT_DIR_2D / "r2_vs_milp_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\n[OK] 2D models vs MILP R² summary saved to: {summary_path.name}")
    else:
        print("\nWARNING: No 2D models vs MILP R² results were generated.")


# ============================================================
# 2D Model Functions (a·c_g + b·c_el + c)
# ============================================================

def load_2d_test_data() -> pd.DataFrame:
    """Load 2D test data with discrete prices."""
    df = pd.read_csv(TEST_DATA_2D)
    return df


def predict_2d_single_model(
    model_path: Path,
    test_df: pd.DataFrame,
    target: str,
) -> PredictionResult:
    """
    Make predictions with a single 2D model.
    
    Args:
        model_path: Path to .joblib model
        test_df: Test dataframe with 'gas_price_MWh' and 'electricity_price_MWh' columns
        target: Target OPEX type (e.g., "opex_milp")
    
    Returns:
        PredictionResult with y_test, y_pred, r2
    """
    model = joblib.load(model_path)
    
    # Extract 2D features
    X = test_df[["gas_price_MWh", "electricity_price_MWh"]].values
    
    # Get true OPEX values (already absolute)
    y_test = test_df[target].values
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate R²
    r2 = r2_score(y_test, y_pred)
    
    return PredictionResult(
        y_test=y_test,
        y_pred=y_pred,
        r2=r2,
    )


def find_2d_model_path(target: str) -> Path | None:
    """Find 2D model path for given target."""
    # Try common naming patterns for 2D models
    patterns = [
        f"2D_40_2d_discrete_absolute_opex_{target}.joblib",
        f"2D_40_2d_discrete_absolute_{target}.joblib",
    ]
    
    for pattern in patterns:
        path = MODEL_DIR_1D / pattern
        if path.exists():
            return path
    
    return None


def evaluate_2d_models():
    """Evaluate 2D regression models with discrete prices."""
    OUTPUT_DIR_2D.mkdir(parents=True, exist_ok=True)
    
    test_df = load_2d_test_data()
    
    print(f"\n{'='*70}")
    print("EVALUATING 2D MODELS (a·c_g + b·c_el + c)")
    print(f"{'='*70}")
    print(f"Test data shape: {test_df.shape}")
    print(f"Gas price range: [{test_df['gas_price_MWh'].min():.2f}, "
          f"{test_df['gas_price_MWh'].max():.2f}]")
    print(f"Electricity price range: [{test_df['electricity_price_MWh'].min():.2f}, "
          f"{test_df['electricity_price_MWh'].max():.2f}]")
    
    r2_summary_data = {}
    
    for target in OPEX_TARGETS:
        full_target_name = TARGET_TO_FULL_NAME[target]
        
        # Find model
        model_path = find_2d_model_path(target)
        
        if model_path is None:
            print(f"  WARNING: No 2D model found for {target}")
            continue
        
        # Predict
        result = predict_2d_single_model(
            model_path,
            test_df,
            full_target_name,
        )
        
        print(f"  {full_target_name:20} R² = {result.r2:.6f}")
        
        # Save predictions CSV
        output_csv = OUTPUT_DIR_2D / f"40_train_10_test_2d_discrete_opex_{target}.csv"
        pred_df = pd.DataFrame({
            "y_test": result.y_test,
            "y_pred": result.y_pred,
            "r2": result.r2,
        })
        pred_df.to_csv(output_csv, index=False)
        
        # Store for summary
        r2_summary_data[full_target_name] = result.r2
    
    # Export R² summary
    summary_df = pd.DataFrame([r2_summary_data])
    summary_df = summary_df[[f"opex_{t}" for t in OPEX_TARGETS]]
    summary_df.columns = [MODEL_LABELS[col] for col in summary_df.columns]
    summary_df = summary_df[MODEL_ORDER]
    
    summary_path = OUTPUT_DIR_2D / "r2_scores_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[OK] 2D R² summary saved to: {summary_path.name}")


# ============================================================
# Comparison and Analysis
# ============================================================

def compare_1d_vs_2d_r2():
    """
    Compare R² scores between 1D and 2D models.
    Create a combined summary if both evaluations completed.
    """
    if not (OUTPUT_DIR_1D / "r2_scores_summary.csv").exists():
        print("  (Skipping comparison: 1D results not available)")
        return
    
    if not (OUTPUT_DIR_2D / "r2_scores_summary.csv").exists():
        print("  (Skipping comparison: 2D results not available)")
        return
    
    print(f"\n{'='*70}")
    print("1D vs 2D MODEL COMPARISON")
    print(f"{'='*70}")
    
    df_1d = pd.read_csv(OUTPUT_DIR_1D / "r2_scores_summary.csv", index_col=0)
    df_2d = pd.read_csv(OUTPUT_DIR_2D / "r2_scores_summary.csv")
    
    print("\n1D Models (Ratio-based, varying training sizes):")
    print(df_1d.to_string())
    
    print("\n2D Models (Discrete prices, 40 samples):")
    print(df_2d.to_string())
    
    # Create combined comparison table: average R² for 1D models vs 2D models
    print("\n\nAVERAGE PERFORMANCE BY MODEL TYPE:")
    print(f"-" * 60)
    
    avg_1d = df_1d.mean()
    avg_2d = df_2d.iloc[0]
    
    comparison_df = pd.DataFrame({
        "1D (Ratio-based)": avg_1d,
        "2D (Discrete prices)": avg_2d,
        "Difference (2D - 1D)": avg_2d - avg_1d,
    })
    print(comparison_df.to_string())
    
    # Save combined comparison
    comparison_path = OUTPUT_DIR_1D.parent / "model_comparison_summary.csv"
    comparison_df.to_csv(comparison_path)
    print(f"\n[OK] Comparison summary saved to: {comparison_path.name}")


# ============================================================
# Main
# ============================================================

def main():
    """Run full evaluation pipeline."""
    print("\n" + "="*70)
    print("REGRESSION MODEL EVALUATION")
    print("="*70)
    
    # Check that test data files exist
    if not TEST_DATA_1D.exists():
        raise FileNotFoundError(f"1D test data not found: {TEST_DATA_1D}")
    if not TEST_DATA_2D.exists():
        raise FileNotFoundError(f"2D test data not found: {TEST_DATA_2D}")
    
    # Evaluate both model types on test data
    try:
        evaluate_1d_models()
    except Exception as e:
        print(f"ERROR in 1D evaluation: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        evaluate_2d_models()
    except Exception as e:
        print(f"ERROR in 2D evaluation: {e}")
        import traceback
        traceback.print_exc()

    # Evaluate both model types on their own training samples
    try:
        evaluate_1d_models_on_training_samples()
    except Exception as e:
        print(f"ERROR in 1D training-sample evaluation: {e}")
        import traceback
        traceback.print_exc()

    try:
        evaluate_2d_models_on_training_samples()
    except Exception as e:
        print(f"ERROR in 2D training-sample evaluation: {e}")
        import traceback
        traceback.print_exc()

    # Evaluate model predictions against actual MILP OPEX on training data
    try:
        evaluate_1d_models_vs_milp_on_training_samples()
    except Exception as e:
        print(f"ERROR in 1D models vs MILP training evaluation: {e}")
        import traceback
        traceback.print_exc()

    try:
        evaluate_2d_models_vs_milp_on_training_samples()
    except Exception as e:
        print(f"ERROR in 2D models vs MILP training evaluation: {e}")
        import traceback
        traceback.print_exc()
    
    # Compare results
    compare_1d_vs_2d_r2()
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"1D Results saved to:  {OUTPUT_DIR_1D}")
    print(f"2D Results saved to:  {OUTPUT_DIR_2D}")


if __name__ == "__main__":
    main()
