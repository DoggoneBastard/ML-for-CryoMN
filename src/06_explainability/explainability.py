#!/usr/bin/env python3
"""
CryoMN Model Explainability Module

Generates comprehensive visualizations to explain the GP model predictions:
- Feature importance bar chart
- SHAP values analysis
- Partial dependence plots (PDPs)
- 2D contour plots (ingredient interactions)
- Acquisition function landscape
- GP uncertainty visualization

Author: CryoMN ML Project
Date: 2026-01-27
"""

import pandas as pd
import numpy as np
import os
import json
import pickle
import warnings
from typing import Tuple, Dict, List, Optional
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Optional imports with graceful fallback
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Note: seaborn not installed. Using matplotlib defaults.")

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set style for all plots
if HAS_SEABORN:
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        try:
            plt.style.use('seaborn-whitegrid')
        except OSError:
            pass
    sns.set_palette("husl")
else:
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3


# =============================================================================
# CONFIGURATION
# =============================================================================

class ExplainabilityConfig:
    """Configuration for explainability visualizations."""
    # General settings
    figsize_small = (8, 6)
    figsize_medium = (10, 8)
    figsize_large = (14, 10)
    dpi = 150
    
    # PDP settings
    n_pdp_points = 50
    n_top_features_pdp = 8
    
    # Contour settings
    n_contour_points = 30
    n_top_pairs = 3
    
    # SHAP settings
    n_shap_samples = 100  # Background samples for SHAP
    
    # Color settings
    cmap_viability = 'RdYlGn'
    cmap_uncertainty = 'YlOrRd'
    cmap_ei = 'viridis'


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_model_and_data(project_root: str) -> Tuple[GaussianProcessRegressor, 
                                                      StandardScaler, 
                                                      List[str], 
                                                      pd.DataFrame,
                                                      pd.DataFrame]:
    """
    Load the trained GP model, scaler, and data.
    
    Returns:
        Tuple of (gp_model, scaler, feature_names, data_df, importance_df)
    """
    model_dir = os.path.join(project_root, 'models')
    data_path = os.path.join(project_root, 'data', 'processed', 'parsed_formulations.csv')
    
    # Load model
    with open(os.path.join(model_dir, 'gp_model.pkl'), 'rb') as f:
        gp = pickle.load(f)
    with open(os.path.join(model_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(model_dir, 'model_metadata.json'), 'r') as f:
        metadata = json.load(f)
    
    feature_names = metadata['feature_names']
    
    # Load data
    df = pd.read_csv(data_path)
    df = df[df['viability_percent'] <= 100].copy()
    
    # Load feature importance
    importance_df = pd.read_csv(os.path.join(model_dir, 'feature_importance.csv'))
    
    return gp, scaler, feature_names, df, importance_df


def clean_feature_name(name: str) -> str:
    """Clean feature name for display."""
    return name.replace('_M', '').replace('_', ' ').title()


# =============================================================================
# 1. FEATURE IMPORTANCE BAR CHART
# =============================================================================

def plot_feature_importance(importance_df: pd.DataFrame, output_dir: str,
                            config: ExplainabilityConfig = None):
    """
    Create a horizontal bar chart of feature importance.
    
    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        output_dir: Directory to save the plot
        config: Visualization configuration
    """
    config = config or ExplainabilityConfig()
    
    # Sort by importance
    df = importance_df.sort_values('importance', ascending=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=config.figsize_medium)
    
    # Color gradient based on importance
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(df)))
    
    # Create horizontal bar chart
    bars = ax.barh(range(len(df)), df['importance'], color=colors)
    
    # Customize
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([clean_feature_name(f) for f in df['feature']])
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title('Feature Importance for Cell Viability Prediction', fontsize=14, fontweight='bold')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, df['importance'])):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center', fontsize=9)
    
    ax.set_xlim(0, df['importance'].max() * 1.15)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'feature_importance.png')
    plt.savefig(output_path, dpi=config.dpi, bbox_inches='tight', transparent=True)
    plt.close()
    
    print(f"  ✓ Feature importance chart saved: {output_path}")


# =============================================================================
# 2. SHAP VALUES ANALYSIS
# =============================================================================

def compute_shap_values(gp: GaussianProcessRegressor, scaler: StandardScaler,
                        X: np.ndarray, feature_names: List[str],
                        config: ExplainabilityConfig = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute SHAP values using KernelExplainer (GP-compatible).
    
    Returns:
        Tuple of (shap_values, background_data)
    """
    config = config or ExplainabilityConfig()
    
    # Use a subset for background
    np.random.seed(42)
    n_samples = min(config.n_shap_samples, len(X))
    bg_idx = np.random.choice(len(X), n_samples, replace=False)
    X_background = X[bg_idx]
    
    # Define prediction function
    def predict_fn(X_raw):
        X_scaled = scaler.transform(X_raw)
        return gp.predict(X_scaled)
    
    try:
        import shap
        
        # Create explainer
        explainer = shap.KernelExplainer(predict_fn, X_background)
        
        # Compute SHAP values for a subset of samples
        n_explain = min(100, len(X))
        explain_idx = np.random.choice(len(X), n_explain, replace=False)
        X_explain = X[explain_idx]
        
        shap_values = explainer.shap_values(X_explain, silent=True)
        
        return shap_values, X_explain
    
    except ImportError:
        print("  ⚠ SHAP library not installed. Skipping SHAP analysis.")
        return None, None


def plot_shap_summary(shap_values: np.ndarray, X_explain: np.ndarray,
                      feature_names: List[str], output_dir: str,
                      config: ExplainabilityConfig = None):
    """Create SHAP summary beeswarm plot."""
    config = config or ExplainabilityConfig()
    
    try:
        import shap
        
        # Clean feature names for display
        clean_names = [clean_feature_name(f) for f in feature_names]
        
        # Create summary plot
        fig, ax = plt.subplots(figsize=config.figsize_medium)
        shap.summary_plot(shap_values, X_explain, feature_names=clean_names,
                          show=False, plot_size=None)
        plt.title('SHAP Summary: Feature Impact on Viability', fontsize=14, fontweight='bold')
        
        output_path = os.path.join(output_dir, 'shap_summary.png')
        plt.savefig(output_path, dpi=config.dpi, bbox_inches='tight', transparent=True)
        plt.close()
        
        print(f"  ✓ SHAP summary plot saved: {output_path}")
        
        # Create SHAP importance bar plot
        fig, ax = plt.subplots(figsize=config.figsize_small)
        shap.summary_plot(shap_values, X_explain, feature_names=clean_names,
                          plot_type="bar", show=False, plot_size=None)
        plt.title('SHAP Feature Importance', fontsize=14, fontweight='bold')
        
        output_path = os.path.join(output_dir, 'shap_importance.png')
        plt.savefig(output_path, dpi=config.dpi, bbox_inches='tight', transparent=True)
        plt.close()
        
        print(f"  ✓ SHAP importance plot saved: {output_path}")
        
    except Exception as e:
        print(f"  ⚠ Error creating SHAP plots: {e}")


# =============================================================================
# 3. PARTIAL DEPENDENCE PLOTS (PDPs)
# =============================================================================

def plot_partial_dependence(gp: GaussianProcessRegressor, scaler: StandardScaler,
                            X: np.ndarray, feature_names: List[str],
                            importance_df: pd.DataFrame, output_dir: str,
                            config: ExplainabilityConfig = None):
    """
    Create partial dependence plots for top features.
    Shows how predicted viability changes with each ingredient concentration.
    """
    config = config or ExplainabilityConfig()
    
    # Get top features by importance
    top_features = importance_df.nlargest(config.n_top_features_pdp, 'importance')['feature'].tolist()
    
    # Get feature indices
    feature_indices = {name: i for i, name in enumerate(feature_names)}
    
    # Create figure with subplots
    n_cols = 2
    n_rows = (len(top_features) + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
    axes = axes.flatten()
    
    for idx, feature in enumerate(top_features):
        ax = axes[idx]
        feat_idx = feature_indices.get(feature + '_M', feature_indices.get(feature, None))
        
        if feat_idx is None:
            continue
        
        # Create feature range
        feat_min = X[:, feat_idx].min()
        feat_max = X[:, feat_idx].max()
        if feat_max - feat_min < 1e-6:
            feat_max = feat_min + 1
        
        feat_values = np.linspace(feat_min, feat_max, config.n_pdp_points)
        
        # Compute predictions for each feature value (averaging over other features)
        pdp_means = []
        pdp_stds = []
        
        X_mean = X.mean(axis=0)
        
        for val in feat_values:
            X_temp = X_mean.copy()
            X_temp[feat_idx] = val
            X_scaled = scaler.transform(X_temp.reshape(1, -1))
            mean, std = gp.predict(X_scaled, return_std=True)
            pdp_means.append(mean[0])
            pdp_stds.append(std[0])
        
        pdp_means = np.array(pdp_means)
        pdp_stds = np.array(pdp_stds)
        
        # Plot
        ax.plot(feat_values, pdp_means, 'b-', linewidth=2, label='Mean Prediction')
        ax.fill_between(feat_values, 
                        pdp_means - 1.96 * pdp_stds,
                        pdp_means + 1.96 * pdp_stds,
                        alpha=0.3, color='blue', label='95% CI')
        
        ax.set_xlabel(f'{clean_feature_name(feature)} (M)', fontsize=10)
        ax.set_ylabel('Predicted Viability (%)', fontsize=10)
        ax.set_title(f'PDP: {clean_feature_name(feature)}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(top_features), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Partial Dependence Plots: Effect of Individual Ingredients', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'partial_dependence_plots.png')
    plt.savefig(output_path, dpi=config.dpi, bbox_inches='tight', transparent=True)
    plt.close()
    
    print(f"  ✓ Partial dependence plots saved: {output_path}")


# =============================================================================
# 4. 2D CONTOUR PLOTS (INGREDIENT INTERACTIONS)
# =============================================================================

def plot_interaction_contours(gp: GaussianProcessRegressor, scaler: StandardScaler,
                              X: np.ndarray, feature_names: List[str],
                              importance_df: pd.DataFrame, output_dir: str,
                              config: ExplainabilityConfig = None):
    """
    Create 2D contour plots showing interactions between top ingredient pairs.
    """
    config = config or ExplainabilityConfig()
    
    # Get top features
    top_features = importance_df.nlargest(4, 'importance')['feature'].tolist()
    
    # Get feature indices (handle _M suffix)
    feature_indices = {}
    for name in feature_names:
        clean = name.replace('_M', '')
        feature_indices[clean] = feature_names.index(name)
    
    # Generate pairs
    pairs = []
    for i in range(len(top_features)):
        for j in range(i + 1, len(top_features)):
            pairs.append((top_features[i], top_features[j]))
    
    pairs = pairs[:config.n_top_pairs]
    
    # Create figure
    fig, axes = plt.subplots(1, len(pairs), figsize=(6 * len(pairs), 5))
    if len(pairs) == 1:
        axes = [axes]
    
    X_mean = X.mean(axis=0)
    
    for idx, (feat1, feat2) in enumerate(pairs):
        ax = axes[idx]
        
        idx1 = feature_indices.get(feat1)
        idx2 = feature_indices.get(feat2)
        
        if idx1 is None or idx2 is None:
            continue
        
        # Create grid
        x1_range = np.linspace(X[:, idx1].min(), X[:, idx1].max(), config.n_contour_points)
        x2_range = np.linspace(X[:, idx2].min(), X[:, idx2].max(), config.n_contour_points)
        X1, X2 = np.meshgrid(x1_range, x2_range)
        
        # Compute predictions over grid
        Z = np.zeros_like(X1)
        for i in range(X1.shape[0]):
            for j in range(X1.shape[1]):
                X_temp = X_mean.copy()
                X_temp[idx1] = X1[i, j]
                X_temp[idx2] = X2[i, j]
                X_scaled = scaler.transform(X_temp.reshape(1, -1))
                Z[i, j] = gp.predict(X_scaled)[0]
        
        # Plot contour
        contour = ax.contourf(X1, X2, Z, levels=20, cmap=config.cmap_viability)
        plt.colorbar(contour, ax=ax, label='Predicted Viability (%)')
        
        # Add contour lines
        ax.contour(X1, X2, Z, levels=10, colors='white', alpha=0.3, linewidths=0.5)
        
        ax.set_xlabel(f'{clean_feature_name(feat1)} (M)', fontsize=10)
        ax.set_ylabel(f'{clean_feature_name(feat2)} (M)', fontsize=10)
        ax.set_title(f'{clean_feature_name(feat1)} × {clean_feature_name(feat2)}', 
                     fontsize=11, fontweight='bold')
    
    plt.suptitle('Ingredient Interaction Effects on Cell Viability', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'interaction_contours.png')
    plt.savefig(output_path, dpi=config.dpi, bbox_inches='tight', transparent=True)
    plt.close()
    
    print(f"  ✓ Interaction contour plots saved: {output_path}")


# =============================================================================
# 5. ACQUISITION FUNCTION LANDSCAPE
# =============================================================================

def expected_improvement(mean: np.ndarray, std: np.ndarray, 
                         y_best: float, xi: float = 0.01) -> np.ndarray:
    """Calculate Expected Improvement."""
    with np.errstate(divide='warn'):
        z = (mean - y_best - xi) / std
        ei = (mean - y_best - xi) * norm.cdf(z) + std * norm.pdf(z)
        ei[std < 1e-9] = 0.0
    return ei


def plot_acquisition_landscape(gp: GaussianProcessRegressor, scaler: StandardScaler,
                               X: np.ndarray, y: np.ndarray, feature_names: List[str],
                               importance_df: pd.DataFrame, output_dir: str,
                               config: ExplainabilityConfig = None):
    """
    Visualize the Expected Improvement acquisition function landscape.
    """
    config = config or ExplainabilityConfig()
    
    # Get top 2 features for 2D visualization
    top_features = importance_df.nlargest(2, 'importance')['feature'].tolist()
    
    # Get feature indices
    feature_indices = {}
    for name in feature_names:
        clean = name.replace('_M', '')
        feature_indices[clean] = feature_names.index(name)
    
    feat1, feat2 = top_features[0], top_features[1]
    idx1 = feature_indices.get(feat1)
    idx2 = feature_indices.get(feat2)
    
    X_mean = X.mean(axis=0)
    y_best = y.max()
    
    # Create grid
    n_points = config.n_contour_points
    x1_range = np.linspace(X[:, idx1].min(), X[:, idx1].max(), n_points)
    x2_range = np.linspace(X[:, idx2].min(), X[:, idx2].max(), n_points)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    
    # Compute predictions and EI
    Z_mean = np.zeros_like(X1)
    Z_std = np.zeros_like(X1)
    Z_ei = np.zeros_like(X1)
    
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            X_temp = X_mean.copy()
            X_temp[idx1] = X1[i, j]
            X_temp[idx2] = X2[i, j]
            X_scaled = scaler.transform(X_temp.reshape(1, -1))
            mean, std = gp.predict(X_scaled, return_std=True)
            Z_mean[i, j] = mean[0]
            Z_std[i, j] = std[0]
            Z_ei[i, j] = expected_improvement(mean, std, y_best)[0]
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: GP Mean
    contour1 = axes[0].contourf(X1, X2, Z_mean, levels=20, cmap=config.cmap_viability)
    plt.colorbar(contour1, ax=axes[0], label='Predicted Viability (%)')
    axes[0].set_xlabel(f'{clean_feature_name(feat1)} (M)')
    axes[0].set_ylabel(f'{clean_feature_name(feat2)} (M)')
    axes[0].set_title('GP Mean Prediction', fontweight='bold')
    
    # Plot 2: GP Uncertainty
    contour2 = axes[1].contourf(X1, X2, Z_std, levels=20, cmap=config.cmap_uncertainty)
    plt.colorbar(contour2, ax=axes[1], label='Uncertainty (std)')
    axes[1].set_xlabel(f'{clean_feature_name(feat1)} (M)')
    axes[1].set_ylabel(f'{clean_feature_name(feat2)} (M)')
    axes[1].set_title('GP Uncertainty', fontweight='bold')
    
    # Plot 3: Expected Improvement
    contour3 = axes[2].contourf(X1, X2, Z_ei, levels=20, cmap=config.cmap_ei)
    plt.colorbar(contour3, ax=axes[2], label='Expected Improvement')
    axes[2].set_xlabel(f'{clean_feature_name(feat1)} (M)')
    axes[2].set_ylabel(f'{clean_feature_name(feat2)} (M)')
    axes[2].set_title('Acquisition Function (EI)', fontweight='bold')
    
    # Mark best observed point
    best_idx = np.argmax(y)
    for ax in axes:
        ax.scatter(X[best_idx, idx1], X[best_idx, idx2], 
                   c='red', s=100, marker='*', edgecolors='white',
                   linewidths=1.5, zorder=5, label='Best Observed')
        ax.legend(loc='upper right', fontsize=8)
    
    plt.suptitle('Acquisition Function Landscape: Exploration vs Exploitation', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'acquisition_landscape.png')
    plt.savefig(output_path, dpi=config.dpi, bbox_inches='tight', transparent=True)
    plt.close()
    
    print(f"  ✓ Acquisition landscape saved: {output_path}")


# =============================================================================
# 6. GP UNCERTAINTY VISUALIZATION
# =============================================================================

def plot_uncertainty_analysis(gp: GaussianProcessRegressor, scaler: StandardScaler,
                              X: np.ndarray, y: np.ndarray, feature_names: List[str],
                              output_dir: str, config: ExplainabilityConfig = None):
    """
    Visualize GP uncertainty across the observed data.
    """
    config = config or ExplainabilityConfig()
    
    # Get predictions with uncertainty for all data points
    X_scaled = scaler.transform(X)
    y_pred, y_std = gp.predict(X_scaled, return_std=True)
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=config.figsize_large)
    
    # Plot 1: Predicted vs Actual with error bars
    ax1 = axes[0, 0]
    scatter = ax1.scatter(y, y_pred, c=y_std, cmap='YlOrRd', 
                          s=50, alpha=0.7, edgecolors='white', linewidths=0.5)
    plt.colorbar(scatter, ax=ax1, label='Uncertainty (std)')
    
    # Add diagonal line
    min_val = min(y.min(), y_pred.min())
    max_val = max(y.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Prediction')
    
    ax1.set_xlabel('Actual Viability (%)', fontsize=10)
    ax1.set_ylabel('Predicted Viability (%)', fontsize=10)
    ax1.set_title('Predicted vs Actual (colored by uncertainty)', fontweight='bold')
    ax1.legend()
    
    # Plot 2: Uncertainty distribution
    ax2 = axes[0, 1]
    ax2.hist(y_std, bins=30, color='coral', edgecolor='white', alpha=0.8)
    ax2.axvline(y_std.mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {y_std.mean():.2f}')
    ax2.set_xlabel('Prediction Uncertainty (std)', fontsize=10)
    ax2.set_ylabel('Frequency', fontsize=10)
    ax2.set_title('Distribution of Model Uncertainty', fontweight='bold')
    ax2.legend()
    
    # Plot 3: Residuals vs Uncertainty
    ax3 = axes[1, 0]
    residuals = y - y_pred
    ax3.scatter(y_std, np.abs(residuals), c=y, cmap='viridis', 
                s=50, alpha=0.7, edgecolors='white', linewidths=0.5)
    ax3.set_xlabel('Prediction Uncertainty (std)', fontsize=10)
    ax3.set_ylabel('Absolute Error (%)', fontsize=10)
    ax3.set_title('Error vs Uncertainty (calibration check)', fontweight='bold')
    
    # Add trend line
    z = np.polyfit(y_std, np.abs(residuals), 1)
    p = np.poly1d(z)
    x_line = np.linspace(y_std.min(), y_std.max(), 100)
    ax3.plot(x_line, p(x_line), 'r--', alpha=0.7, linewidth=2, label='Trend')
    ax3.legend()
    
    # Plot 4: Uncertainty by viability range
    ax4 = axes[1, 1]
    viability_bins = [0, 30, 50, 70, 90, 100]
    bin_labels = ['0-30%', '30-50%', '50-70%', '70-90%', '90-100%']
    bin_uncertainties = []
    
    for i in range(len(viability_bins) - 1):
        mask = (y >= viability_bins[i]) & (y < viability_bins[i+1])
        if mask.sum() > 0:
            bin_uncertainties.append(y_std[mask].mean())
        else:
            bin_uncertainties.append(0)
    
    bars = ax4.bar(bin_labels, bin_uncertainties, color=plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(bin_labels))))
    ax4.set_xlabel('Viability Range', fontsize=10)
    ax4.set_ylabel('Mean Uncertainty (std)', fontsize=10)
    ax4.set_title('Model Uncertainty by Viability Range', fontweight='bold')
    
    # Add value labels
    for bar, val in zip(bars, bin_uncertainties):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                 f'{val:.1f}', ha='center', fontsize=9)
    
    plt.suptitle('GP Model Uncertainty Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'uncertainty_analysis.png')
    plt.savefig(output_path, dpi=config.dpi, bbox_inches='tight', transparent=True)
    plt.close()
    
    print(f"  ✓ Uncertainty analysis saved: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point for explainability visualizations."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    output_dir = os.path.join(project_root, 'results', 'explainability')
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("CryoMN Model Explainability Analysis")
    print("=" * 80)
    
    # Load model and data
    print("\n📊 Loading model and data...")
    gp, scaler, feature_names, df, importance_df = load_model_and_data(project_root)
    
    X = df[feature_names].values
    y = df['viability_percent'].values
    
    print(f"  Model loaded with {len(feature_names)} features")
    print(f"  Data loaded with {len(df)} formulations")
    
    config = ExplainabilityConfig()
    
    # Generate all visualizations
    print("\n📈 Generating visualizations...\n")
    
    # 1. Feature Importance
    print("1️⃣ Feature Importance Bar Chart")
    plot_feature_importance(importance_df, output_dir, config)
    
    # 2. SHAP Analysis
    print("\n2️⃣ SHAP Values Analysis")
    shap_values, X_explain = compute_shap_values(gp, scaler, X, feature_names, config)
    if shap_values is not None:
        plot_shap_summary(shap_values, X_explain, feature_names, output_dir, config)
    
    # 3. Partial Dependence Plots
    print("\n3️⃣ Partial Dependence Plots")
    plot_partial_dependence(gp, scaler, X, feature_names, importance_df, output_dir, config)
    
    # 4. Interaction Contours
    print("\n4️⃣ 2D Interaction Contour Plots")
    plot_interaction_contours(gp, scaler, X, feature_names, importance_df, output_dir, config)
    
    # 5. Acquisition Landscape
    print("\n5️⃣ Acquisition Function Landscape")
    plot_acquisition_landscape(gp, scaler, X, y, feature_names, importance_df, output_dir, config)
    
    # 6. Uncertainty Analysis
    print("\n6️⃣ GP Uncertainty Visualization")
    plot_uncertainty_analysis(gp, scaler, X, y, feature_names, output_dir, config)
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ Explainability Analysis Complete!")
    print("=" * 80)
    print(f"\nAll visualizations saved to: {output_dir}")
    print("\nGenerated files:")
    for f in sorted(os.listdir(output_dir)):
        if f.endswith('.png'):
            print(f"  • {f}")


if __name__ == '__main__':
    main()
