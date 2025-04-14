import matplotlib.pyplot as plt
import numpy as np

def plot_age_distribution(ages, bins=50):
    """Plot histogram of age distribution"""
    plt.figure(figsize=(10, 6))
    plt.hist(ages, bins=bins)
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.savefig("../../outputs/figures/age_dist.png")
    plt.close()

def plot_predictions(y_true, y_pred):
    """Scatter plot of true vs predicted ages"""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.3)
    plt.plot([0, 100], [0, 100], 'r--')  # Perfect prediction line
    plt.title("True vs Predicted Age")
    plt.xlabel("True Age")
    plt.ylabel("Predicted Age")
    plt.savefig("../../outputs/figures/predictions.png")
    plt.close()