"""
Visualization Utilities for X-FFM Explainability
Author: Sumetee Jirapattarasakul
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
import seaborn as sns


def plot_concept_activations(
    concept_names: List[str],
    concept_values: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Concept Activations"
):
    """
    Plot concept activation values as a bar chart
    
    Args:
        concept_names: List of concept names
        concept_values: Array of concept values [num_concepts]
        save_path: Path to save the figure
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Sort by value
    sorted_indices = np.argsort(concept_values)[::-1]
    sorted_names = [concept_names[i] for i in sorted_indices]
    sorted_values = concept_values[sorted_indices]
    
    # Create color map
    colors = plt.cm.RdYlGn(sorted_values)
    
    # Plot bars
    bars = ax.barh(range(len(sorted_names)), sorted_values, color=colors)
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names)
    ax.set_xlabel('Activation Value', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, sorted_values)):
        ax.text(value + 0.02, i, f'{value:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_counterfactual_comparison(
    original_signal: np.ndarray,
    counterfactual_signal: np.ndarray,
    modality_name: str,
    original_pred: str,
    cf_pred: str,
    save_path: Optional[str] = None
):
    """
    Plot comparison between original and counterfactual signals
    
    Args:
        original_signal: Original signal [signal_length]
        counterfactual_signal: Counterfactual signal [signal_length]
        modality_name: Name of the modality
        original_pred: Original prediction label
        cf_pred: Counterfactual prediction label
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    time = np.arange(len(original_signal))
    diff = counterfactual_signal - original_signal
    
    # Plot original signal
    axes[0].plot(time, original_signal, color='#2E86AB', linewidth=1.5, label='Original')
    axes[0].fill_between(time, original_signal, alpha=0.2, color='#2E86AB')
    axes[0].set_title(f'Original {modality_name} Signal (Prediction: {original_pred})', 
                     fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot counterfactual signal
    axes[1].plot(time, counterfactual_signal, color='#F18F01', linewidth=1.5, label='Counterfactual')
    axes[1].fill_between(time, counterfactual_signal, alpha=0.2, color='#F18F01')
    axes[1].set_title(f'Counterfactual {modality_name} Signal (Prediction: {cf_pred})', 
                     fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Plot difference
    axes[2].plot(time, diff, color='#C73E1D', linewidth=1.5, label='Difference')
    axes[2].fill_between(time, diff, alpha=0.3, color='#C73E1D')
    axes[2].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    axes[2].set_title('Difference (Counterfactual - Original)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('Amplitude Change')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    # Highlight significant changes
    threshold = np.abs(diff).std() * 2
    significant_changes = np.abs(diff) > threshold
    if significant_changes.any():
        axes[2].scatter(time[significant_changes], diff[significant_changes], 
                       color='red', s=30, zorder=5, label='Significant Changes')
        axes[2].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_concept_comparison(
    concept_names: List[str],
    original_concepts: np.ndarray,
    cf_concepts: np.ndarray,
    save_path: Optional[str] = None
):
    """
    Plot comparison of concept activations between original and counterfactual
    
    Args:
        concept_names: List of concept names
        original_concepts: Original concept values [num_concepts]
        cf_concepts: Counterfactual concept values [num_concepts]
        save_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(concept_names))
    width = 0.35
    
    # Plot bars
    bars1 = ax.bar(x - width/2, original_concepts, width, label='Original', 
                   color='#2E86AB', alpha=0.8)
    bars2 = ax.bar(x + width/2, cf_concepts, width, label='Counterfactual', 
                   color='#F18F01', alpha=0.8)
    
    # Customize
    ax.set_ylabel('Concept Activation', fontsize=12)
    ax.set_title('Concept Activation Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(concept_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_cross_modal_attention(
    attention_weights: np.ndarray,
    modality_names: List[str],
    save_path: Optional[str] = None
):
    """
    Plot cross-modal attention weights as a heatmap
    
    Args:
        attention_weights: Attention weights [num_modalities, num_modalities]
        modality_names: List of modality names
        save_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create heatmap
    im = ax.imshow(attention_weights, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks
    ax.set_xticks(np.arange(len(modality_names)))
    ax.set_yticks(np.arange(len(modality_names)))
    ax.set_xticklabels(modality_names)
    ax.set_yticklabels(modality_names)
    
    # Rotate labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention Weight', rotation=270, labelpad=20)
    
    # Add text annotations
    for i in range(len(modality_names)):
        for j in range(len(modality_names)):
            text = ax.text(j, i, f'{attention_weights[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=10)
    
    ax.set_title('Cross-Modal Attention Weights', fontsize=14, fontweight='bold')
    ax.set_xlabel('Key Modality', fontsize=12)
    ax.set_ylabel('Query Modality', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_explainability_dashboard(
    original_signals: Dict[str, np.ndarray],
    cf_signals: Dict[str, np.ndarray],
    concept_names: List[str],
    original_concepts: np.ndarray,
    cf_concepts: np.ndarray,
    original_pred: str,
    cf_pred: str,
    save_path: Optional[str] = None
):
    """
    Create a comprehensive explainability dashboard
    
    Args:
        original_signals: Dictionary of original signals
        cf_signals: Dictionary of counterfactual signals
        concept_names: List of concept names
        original_concepts: Original concept activations
        cf_concepts: Counterfactual concept activations
        original_pred: Original prediction
        cf_pred: Counterfactual prediction
        save_path: Path to save the figure
    """
    num_modalities = len(original_signals)
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, num_modalities + 1, hspace=0.3, wspace=0.3)
    
    # Plot signals for each modality
    for idx, (modality, orig_signal) in enumerate(original_signals.items()):
        cf_signal = cf_signals[modality]
        
        # Original signal
        ax1 = fig.add_subplot(gs[0, idx])
        ax1.plot(orig_signal, color='#2E86AB', linewidth=1)
        ax1.set_title(f'{modality.upper()} - Original', fontsize=10, fontweight='bold')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True, alpha=0.3)
        
        # Counterfactual signal
        ax2 = fig.add_subplot(gs[1, idx])
        ax2.plot(cf_signal, color='#F18F01', linewidth=1)
        ax2.set_title(f'{modality.upper()} - Counterfactual', fontsize=10, fontweight='bold')
        ax2.set_ylabel('Amplitude')
        ax2.grid(True, alpha=0.3)
        
        # Difference
        ax3 = fig.add_subplot(gs[2, idx])
        diff = cf_signal - orig_signal
        ax3.plot(diff, color='#C73E1D', linewidth=1)
        ax3.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        ax3.set_title(f'{modality.upper()} - Difference', fontsize=10, fontweight='bold')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Change')
        ax3.grid(True, alpha=0.3)
    
    # Concept comparison
    ax_concepts = fig.add_subplot(gs[:, -1])
    x = np.arange(len(concept_names))
    width = 0.35
    ax_concepts.barh(x - width/2, original_concepts, width, label='Original', color='#2E86AB')
    ax_concepts.barh(x + width/2, cf_concepts, width, label='Counterfactual', color='#F18F01')
    ax_concepts.set_yticks(x)
    ax_concepts.set_yticklabels(concept_names, fontsize=8)
    ax_concepts.set_xlabel('Activation')
    ax_concepts.set_title('Concept Activations', fontsize=12, fontweight='bold')
    ax_concepts.legend()
    ax_concepts.grid(True, alpha=0.3, axis='x')
    
    # Overall title
    fig.suptitle(f'Explainability Dashboard: {original_pred} → {cf_pred}', 
                fontsize=16, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
