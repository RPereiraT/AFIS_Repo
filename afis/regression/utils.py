"""
Experiment Utilities for AFIS Regression Experiments

Utility functions shared across experiment notebooks:
- Correlation analysis
- Results visualization
- Metrics computation
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


# =============================================================================
# Correlation Analysis
# =============================================================================

def compute_correlation_weights(df):
    """
    Compute normalized absolute correlation weights between features and target.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame where the last column is the target variable.
        
    Returns
    -------
    weights : list
        Normalized absolute correlations (sum to 1).
    """
    corr_matrix = df.corr()
    target_col = corr_matrix.columns[-1]
    
    # Get correlations with target (excluding target's self-correlation)
    correlations = corr_matrix[target_col].values[:-1]
    
    # Take absolute value and normalize
    abs_corr = np.abs(correlations)
    weights = (abs_corr / abs_corr.sum()).tolist()
    
    return weights


def plot_correlation_matrix(df, title="Correlation Matrix"):
    """
    Plot an interactive correlation matrix heatmap.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to compute correlations for.
    title : str
        Plot title.
        
    Returns
    -------
    fig : plotly.graph_objects.Figure
        The correlation heatmap figure.
    """
    corr_matrix = df.corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto='.2f',
        aspect='auto',
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1
    )
    fig.update_layout(
        title=title,
        height=500,
        width=600
    )
    
    return fig


def print_correlation_summary(df):
    """
    Print a summary of feature correlations with the target.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame where the last column is the target variable.
    """
    corr_matrix = df.corr()
    target_col = df.columns[-1]
    weights = compute_correlation_weights(df)
    
    print("Feature correlations with target:")
    for i, (col, weight) in enumerate(zip(df.columns[:-1], weights)):
        corr_val = corr_matrix[target_col].values[i]
        print(f"  {col}: corr = {corr_val:+.4f}, weight = {weight:.4f}")
    print(f"\nSum of weights: {sum(weights):.4f}")


# =============================================================================
# Results Visualization
# =============================================================================

def plot_predictions_vs_actual(y_true, y_pred, title="Predictions vs Actual"):
    """
    Create a scatter plot of predictions vs actual values.
    
    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted values.
    title : str
        Plot title.
        
    Returns
    -------
    fig : plotly.graph_objects.Figure
        The scatter plot figure.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate bounds for perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    
    fig = go.Figure()
    
    # Scatter plot
    fig.add_trace(go.Scatter(
        x=y_true, y=y_pred,
        mode='markers',
        marker=dict(size=6, opacity=0.6),
        name='Predictions'
    ))
    
    # Perfect prediction line
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Perfect'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Actual',
        yaxis_title='Predicted',
        height=500,
        width=600
    )
    
    return fig


def plot_residuals(y_true, y_pred, title="Residuals"):
    """
    Create a residual plot (predicted vs residuals).
    
    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted values.
    title : str
        Plot title.
        
    Returns
    -------
    fig : plotly.graph_objects.Figure
        The residual plot figure.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    residuals = y_true - y_pred
    
    fig = go.Figure()
    
    # Scatter plot
    fig.add_trace(go.Scatter(
        x=y_pred, y=residuals,
        mode='markers',
        marker=dict(size=6, opacity=0.6),
        name='Residuals'
    ))
    
    # Zero line
    fig.add_hline(y=0, line_dash='dash', line_color='red')
    
    fig.update_layout(
        title=title,
        xaxis_title='Predicted',
        yaxis_title='Residual (Actual - Predicted)',
        height=400,
        width=600
    )
    
    return fig


# =============================================================================
# Metrics
# =============================================================================

def compute_metrics(y_true, y_pred):
    """
    Compute regression metrics.
    
    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted values.
        
    Returns
    -------
    metrics : dict
        Dictionary with RMSE, MAE, and R².
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    return {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }


def print_metrics(y_true, y_pred, prefix=""):
    """
    Print regression metrics.
    
    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted values.
    prefix : str
        Optional prefix for the output.
    """
    metrics = compute_metrics(y_true, y_pred)
    
    if prefix:
        print(f"{prefix}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  R²:   {metrics['r2']:.4f}")

