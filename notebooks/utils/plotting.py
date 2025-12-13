"""
Plotting utilities for optimization workshop notebooks.

This module provides helper functions for creating consistent,
professional visualizations following BelkX design guidelines.
"""

import matplotlib.pyplot as plt
import numpy as np

# BelkX color palette
BELKX_BLUE = '#0079C1'
BELKX_GRAY = '#6B7280'
BELKX_LIGHT_GRAY = '#E5E5E5'
BELKX_DARK = '#1F2937'


def setup_belkx_style():
    """
    Configure matplotlib to use BelkX style guidelines.

    Call this at the beginning of each notebook.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Inter', 'Arial', 'Helvetica'],
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.color': BELKX_LIGHT_GRAY,
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        'axes.edgecolor': BELKX_GRAY,
        'axes.linewidth': 1.5,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'legend.framealpha': 0.9,
        'lines.linewidth': 2.5,
    })


def plot_time_series(time, data, labels=None, xlabel='Time', ylabel='Value',
                     title=None, colors=None, save_path=None):
    """
    Plot time series data with BelkX styling.

    Parameters:
    -----------
    time : array-like
        Time axis data
    data : array-like or list of array-like
        Data to plot (can be 2D array or list of 1D arrays)
    labels : list of str, optional
        Legend labels for each data series
    xlabel : str, default 'Time'
        X-axis label
    ylabel : str, default 'Value'
        Y-axis label
    title : str, optional
        Plot title
    colors : list of str, optional
        Colors for each series (defaults to BelkX palette)
    save_path : str, optional
        If provided, save figure to this path

    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

    # Handle single or multiple data series
    if isinstance(data, np.ndarray) and data.ndim == 1:
        data = [data]

    # Default colors
    if colors is None:
        colors = [BELKX_BLUE, BELKX_GRAY, '#F59E0B', '#10B981', '#EF4444']

    # Plot each series
    for i, series in enumerate(data):
        label = labels[i] if labels and i < len(labels) else None
        color = colors[i % len(colors)]
        ax.plot(time, series, color=color, label=label, linewidth=2.5)

    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)

    if title:
        ax.set_title(title, fontsize=16, pad=15)

    if labels:
        ax.legend(loc='best', framealpha=0.9)

    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')

    return fig, ax


# TODO: Add more plotting functions:
# - plot_battery_schedule()
# - plot_trajectory()
# - plot_comparison()
# - plot_cost_breakdown()
