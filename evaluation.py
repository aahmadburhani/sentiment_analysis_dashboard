import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt


def evaluate(y_true, y_pred, model_name="model"):
    """
    Evaluates model predictions against true labels and returns a dictionary of metrics.

    Args:
        y_true (pd.Series or list): Series containing true labels.
        y_pred (pd.Series or list): Series containing predicted labels.
        model_name (str): Name of the model for the output dictionary.

    Returns:
        dict: Dictionary containing accuracy, precision, recall, F1, and count of evaluated samples.
              Returns NaN for metrics if predictions are missing.
    """
    # Convert to Series safely
    y_true = pd.Series(y_true)
    y_pred = pd.Series(y_pred)

    # Handle missing predictions
    if y_pred is None or y_pred.isnull().all():
        return {
            "model": model_name,
            "accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "n": 0,
        }

    mask = y_pred.notnull()
    y_true_f = y_true[mask]
    y_pred_f = y_pred[mask]

    if len(y_true_f) == 0:
        return {
            "model": model_name,
            "accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "n": 0,
        }

    # Compute metrics
    acc = accuracy_score(y_true_f, y_pred_f)
    try:
        # Use macro average to handle multiple sentiment classes
        p, r, f1, _ = precision_recall_fscore_support(
            y_true_f, y_pred_f, average="macro", zero_division=0
        )
    except ValueError:
        p, r, f1 = np.nan, np.nan, np.nan

    metrics = {
        "model": model_name,
        "accuracy": acc,
        "precision": p,
        "recall": r,
        "f1": f1,
        "n": len(y_true_f),
    }

    # Optional debug printout
    print(f"\n📈 Evaluation for {model_name}:")
    print(
        f"  Accuracy: {acc:.3f} | Precision: {p:.3f} | Recall: {r:.3f} | F1: {f1:.3f} | Samples: {len(y_true_f)}"
    )

    return metrics


def plot_confusion_matrix(y_true, y_pred, model_name="model", ax=None):
    """
    Plots the confusion matrix for the given true and predicted labels.

    Args:
        y_true (pd.Series): Series containing true labels.
        y_pred (pd.Series): Series containing predicted labels.
        model_name (str): Name of the model for the plot title.
        ax (matplotlib.axes.Axes, optional): Axes object to plot on. Defaults to None.
    """
    y_true = pd.Series(y_true)
    y_pred = pd.Series(y_pred)

    if y_pred is None or y_pred.isnull().all():
        if ax:
            ax.text(
                0.5, 0.5, f"{model_name} predictions not available", ha="center", va="center"
            )
            ax.set_axis_off()
        else:
            print(f"Cannot plot confusion matrix for {model_name}: predictions are missing.")
        return

    mask = y_pred.notnull()
    y_true_f = y_true[mask]
    y_pred_f = y_pred[mask]

    if len(y_true_f) == 0:
        if ax:
            ax.text(
                0.5, 0.5, f"{model_name} predictions not available", ha="center", va="center"
            )
            ax.set_axis_off()
        else:
            print(f"Cannot plot confusion matrix for {model_name}: no valid predictions to evaluate.")
        return

    labels = sorted(list(set(y_true_f).union(y_pred_f)))
    cm = confusion_matrix(y_true_f, y_pred_f, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    if ax:
        disp.plot(ax=ax)
        ax.set_title(f"{model_name} Confusion Matrix")
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(ax=ax)
        ax.set_title(f"{model_name} Confusion Matrix")
        plt.show()


print("✅ Loaded improved evaluation.py with macro-averaged metrics and safe handling.")
