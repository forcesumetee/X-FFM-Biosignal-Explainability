"""
X-FFM Visualization Demo (Without PyTorch)
Author: Sumetee Jirapattarasakul

This script demonstrates X-FFM explainability visualizations.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


def generate_biosignal(length=1000, signal_type='normal'):
    """Generate synthetic biosignal"""
    t = np.linspace(0, 4 * np.pi, length)
    
    if signal_type == 'normal':
        signal = np.sin(1.0 * t) + 0.3 * np.sin(2.0 * t)
    else:  # arrhythmia
        signal = np.sin(1.5 * t) + 0.5 * np.sin(3.0 * t) + 0.2 * np.sin(5.0 * t)
    
    signal += np.random.randn(length) * 0.1
    return (signal - signal.mean()) / (signal.std() + 1e-8)


def main():
    print("=" * 80)
    print("X-FFM: Explainability Visualization Demo")
    print("=" * 80)
    
    os.makedirs('/home/ubuntu/xffm_project/results', exist_ok=True)
    
    # Clinical concepts
    CONCEPT_NAMES = [
        'Regular_Rhythm',
        'Normal_Heart_Rate',
        'Low_Variability',
        'Stable_Amplitude',
        'No_Artifacts'
    ]
    
    print(f"\n[1/5] Generating synthetic biosignals...")
    
    # Generate signals
    ecg_normal = generate_biosignal(1000, 'normal')
    ppg_normal = generate_biosignal(1000, 'normal')
    ecg_cf = ecg_normal + np.random.randn(1000) * 0.3
    ppg_cf = ppg_normal + np.random.randn(1000) * 0.3
    
    print("  ✓ Generated biosignals")
    
    # Simulate concept activations
    print(f"\n[2/5] Simulating concept activations...")
    original_concepts = np.array([0.85, 0.92, 0.78, 0.88, 0.95])
    cf_concepts = np.array([0.45, 0.52, 0.38, 0.48, 0.55])
    
    print("  ✓ Concept activations simulated")
    
    # Plot 1: Concept Activations
    print(f"\n[3/5] Creating concept activation visualization...")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sorted_indices = np.argsort(original_concepts)[::-1]
    sorted_names = [CONCEPT_NAMES[i] for i in sorted_indices]
    sorted_values = original_concepts[sorted_indices]
    colors = plt.cm.RdYlGn(sorted_values)
    
    bars = ax.barh(range(len(sorted_names)), sorted_values, color=colors)
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names)
    ax.set_xlabel('Activation Value', fontsize=12)
    ax.set_title('Concept Activations for Normal Prediction', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3, axis='x')
    
    for i, (bar, value) in enumerate(zip(bars, sorted_values)):
        ax.text(value + 0.02, i, f'{value:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/xffm_project/results/concept_activations.png', dpi=150)
    plt.close()
    print("  ✓ Saved to results/concept_activations.png")
    
    # Plot 2: Counterfactual Comparison
    print(f"\n[4/5] Creating counterfactual comparison...")
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    time = np.arange(len(ecg_normal))
    diff = ecg_cf - ecg_normal
    
    axes[0].plot(time, ecg_normal, color='#2E86AB', linewidth=1.5)
    axes[0].fill_between(time, ecg_normal, alpha=0.2, color='#2E86AB')
    axes[0].set_title('Original ECG Signal (Prediction: Normal)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(time, ecg_cf, color='#F18F01', linewidth=1.5)
    axes[1].fill_between(time, ecg_cf, alpha=0.2, color='#F18F01')
    axes[1].set_title('Counterfactual ECG Signal (Prediction: Arrhythmia)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(time, diff, color='#C73E1D', linewidth=1.5)
    axes[2].fill_between(time, diff, alpha=0.3, color='#C73E1D')
    axes[2].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    axes[2].set_title('Difference (Counterfactual - Original)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('Amplitude Change')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/xffm_project/results/counterfactual_ecg.png', dpi=150)
    plt.close()
    print("  ✓ Saved to results/counterfactual_ecg.png")
    
    # Plot 3: Concept Comparison
    print(f"\n[5/5] Creating concept comparison...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(CONCEPT_NAMES))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, original_concepts, width, label='Original (Normal)', 
                   color='#2E86AB', alpha=0.8)
    bars2 = ax.bar(x + width/2, cf_concepts, width, label='Counterfactual (Arrhythmia)', 
                   color='#F18F01', alpha=0.8)
    
    ax.set_ylabel('Concept Activation', fontsize=12)
    ax.set_title('Concept Activation Comparison: Normal vs Arrhythmia', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(CONCEPT_NAMES, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/xffm_project/results/concept_comparison.png', dpi=150)
    plt.close()
    print("  ✓ Saved to results/concept_comparison.png")
    
    # Plot 4: Comprehensive Dashboard
    print(f"\n[6/6] Creating explainability dashboard...")
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # ECG signals
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(ecg_normal, color='#2E86AB', linewidth=1)
    ax1.set_title('ECG - Original', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(ecg_cf, color='#F18F01', linewidth=1)
    ax2.set_title('ECG - Counterfactual', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Amplitude')
    ax2.grid(True, alpha=0.3)
    
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(ecg_cf - ecg_normal, color='#C73E1D', linewidth=1)
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax3.set_title('ECG - Difference', fontsize=10, fontweight='bold')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Change')
    ax3.grid(True, alpha=0.3)
    
    # PPG signals
    ax4 = fig.add_subplot(gs[0, 1])
    ax4.plot(ppg_normal, color='#2E86AB', linewidth=1)
    ax4.set_title('PPG - Original', fontsize=10, fontweight='bold')
    ax4.set_ylabel('Amplitude')
    ax4.grid(True, alpha=0.3)
    
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(ppg_cf, color='#F18F01', linewidth=1)
    ax5.set_title('PPG - Counterfactual', fontsize=10, fontweight='bold')
    ax5.set_ylabel('Amplitude')
    ax5.grid(True, alpha=0.3)
    
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(ppg_cf - ppg_normal, color='#C73E1D', linewidth=1)
    ax6.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax6.set_title('PPG - Difference', fontsize=10, fontweight='bold')
    ax6.set_xlabel('Time')
    ax6.set_ylabel('Change')
    ax6.grid(True, alpha=0.3)
    
    # Concept comparison
    ax_concepts = fig.add_subplot(gs[:, 2])
    x = np.arange(len(CONCEPT_NAMES))
    width = 0.35
    ax_concepts.barh(x - width/2, original_concepts, width, label='Original', color='#2E86AB')
    ax_concepts.barh(x + width/2, cf_concepts, width, label='Counterfactual', color='#F18F01')
    ax_concepts.set_yticks(x)
    ax_concepts.set_yticklabels(CONCEPT_NAMES, fontsize=8)
    ax_concepts.set_xlabel('Activation')
    ax_concepts.set_title('Concept Activations', fontsize=12, fontweight='bold')
    ax_concepts.legend()
    ax_concepts.grid(True, alpha=0.3, axis='x')
    
    fig.suptitle('X-FFM Explainability Dashboard: Normal → Arrhythmia', 
                fontsize=16, fontweight='bold')
    
    plt.savefig('/home/ubuntu/xffm_project/results/explainability_dashboard.png', dpi=150)
    plt.close()
    print("  ✓ Saved to results/explainability_dashboard.png")
    
    # Summary
    print("\n" + "=" * 80)
    print("✓ Visualization Demo Completed Successfully!")
    print("=" * 80)
    
    print(f"\nKey Findings:")
    print(f"  - Original prediction: Normal")
    print(f"  - Counterfactual prediction: Arrhythmia")
    print(f"  - Top 3 active concepts in original:")
    top_3 = np.argsort(original_concepts)[::-1][:3]
    for idx in top_3:
        print(f"    • {CONCEPT_NAMES[idx]}: {original_concepts[idx]:.3f}")
    
    print(f"\n  - Concept changes:")
    changes = np.abs(cf_concepts - original_concepts)
    top_changed = np.argmax(changes)
    print(f"    • Most changed: {CONCEPT_NAMES[top_changed]} (Δ = {changes[top_changed]:.3f})")
    
    print(f"\nGenerated Files:")
    print(f"  1. results/concept_activations.png")
    print(f"  2. results/counterfactual_ecg.png")
    print(f"  3. results/concept_comparison.png")
    print(f"  4. results/explainability_dashboard.png")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    np.random.seed(42)
    main()
