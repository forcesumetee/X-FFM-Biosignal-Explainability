"""
X-FFM Explainability Demo
Author: Sumetee Jirapattarasakul

This script demonstrates the explainability features of X-FFM including:
1. Concept-based predictions
2. Counterfactual explanations
3. Cross-modal attention visualization
"""

import sys
sys.path.insert(0, '/home/ubuntu/xffm_project')

import torch
import torch.nn as nn
import numpy as np
import os

from models import create_multimodal_encoder, create_concept_bottleneck_model
from explainability import (
    create_counterfactual_generator,
    plot_concept_activations,
    plot_counterfactual_comparison,
    plot_concept_comparison,
    create_explainability_dashboard
)


def generate_synthetic_biosignal(
    signal_length: int = 1000,
    signal_type: str = 'normal',
    seed: Optional[int] = None
) -> np.ndarray:
    """Generate synthetic biosignal"""
    if seed is not None:
        np.random.seed(seed)
    
    t = np.linspace(0, 4 * np.pi, signal_length)
    
    if signal_type == 'normal':
        # Normal rhythm
        signal = np.sin(1.0 * t) + 0.3 * np.sin(2.0 * t)
    elif signal_type == 'arrhythmia':
        # Irregular rhythm
        signal = np.sin(1.5 * t) + 0.5 * np.sin(3.0 * t) + 0.2 * np.sin(5.0 * t)
    else:
        signal = np.sin(t)
    
    # Add noise
    signal += np.random.randn(signal_length) * 0.1
    
    # Normalize
    signal = (signal - signal.mean()) / (signal.std() + 1e-8)
    
    return signal.astype(np.float32)


def main():
    print("=" * 80)
    print("X-FFM: Explainability Demo")
    print("=" * 80)
    
    # Configuration
    SIGNAL_LENGTH = 1000
    NUM_CLASSES = 2  # Normal vs Arrhythmia
    
    # Define clinical concepts
    CONCEPT_NAMES = [
        'Regular_Rhythm',
        'Normal_Heart_Rate',
        'Low_Variability',
        'Stable_Amplitude',
        'No_Artifacts'
    ]
    
    print(f"\nConfiguration:")
    print(f"  - Signal length: {SIGNAL_LENGTH}")
    print(f"  - Number of classes: {NUM_CLASSES} (Normal, Arrhythmia)")
    print(f"  - Number of concepts: {len(CONCEPT_NAMES)}")
    print(f"  - Concepts: {', '.join(CONCEPT_NAMES)}")
    
    # Create results directory
    os.makedirs('/home/ubuntu/xffm_project/results', exist_ok=True)
    
    # Step 1: Create model
    print(f"\n[1/6] Creating X-FFM model...")
    
    modality_configs = {
        'ecg': {
            'signal_length': SIGNAL_LENGTH,
            'in_channels': 1,
            'hidden_dim': 64,
            'num_layers': 3
        },
        'ppg': {
            'signal_length': SIGNAL_LENGTH,
            'in_channels': 1,
            'hidden_dim': 64,
            'num_layers': 3
        }
    }
    
    encoder = create_multimodal_encoder(modality_configs, fusion_dim=256)
    model = create_concept_bottleneck_model(
        encoder=encoder,
        concept_names=CONCEPT_NAMES,
        num_classes=NUM_CLASSES
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Model created with {num_params:,} parameters")
    
    # Step 2: Generate synthetic data
    print(f"\n[2/6] Generating synthetic biosignals...")
    
    # Generate normal signal
    ecg_normal = generate_synthetic_biosignal(SIGNAL_LENGTH, 'normal', seed=42)
    ppg_normal = generate_synthetic_biosignal(SIGNAL_LENGTH, 'normal', seed=43)
    
    signals_normal = {
        'ecg': torch.from_numpy(ecg_normal).unsqueeze(0).unsqueeze(0),
        'ppg': torch.from_numpy(ppg_normal).unsqueeze(0).unsqueeze(0)
    }
    
    print(f"  ✓ Generated normal biosignals")
    
    # Step 3: Forward pass
    print(f"\n[3/6] Running forward pass...")
    
    model.eval()
    with torch.no_grad():
        logits, concepts, attention = model(
            signals_normal,
            return_concepts=True,
            return_attention=True
        )
        pred = logits.argmax(dim=1).item()
        probs = torch.softmax(logits, dim=1)
    
    class_names = ['Normal', 'Arrhythmia']
    print(f"  ✓ Prediction: {class_names[pred]} (confidence: {probs[0, pred].item():.3f})")
    print(f"  ✓ Concepts extracted: {concepts.shape}")
    
    # Step 4: Visualize concepts
    print(f"\n[4/6] Visualizing concept activations...")
    
    concept_values = concepts[0].cpu().numpy()
    plot_concept_activations(
        concept_names=CONCEPT_NAMES,
        concept_values=concept_values,
        save_path='/home/ubuntu/xffm_project/results/concept_activations.png',
        title=f'Concept Activations for {class_names[pred]} Prediction'
    )
    print(f"  ✓ Saved to results/concept_activations.png")
    
    # Step 5: Generate counterfactual
    print(f"\n[5/6] Generating counterfactual explanation...")
    
    cf_generator = create_counterfactual_generator(
        model=model,
        lambda_proximity=1.0,
        lambda_sparsity=0.1,
        lambda_smoothness=0.01
    )
    
    target_class = 1 - pred  # Flip prediction
    cf_signals, cf_info = cf_generator.generate(
        original_signals=signals_normal,
        target_class=target_class,
        num_iterations=300,
        learning_rate=0.01,
        verbose=False
    )
    
    print(f"  ✓ Counterfactual generation:")
    print(f"    - Original prediction: {class_names[cf_info['original_prediction']]}")
    print(f"    - Target class: {class_names[cf_info['target_class']]}")
    print(f"    - Final prediction: {class_names[cf_info['final_prediction']]}")
    print(f"    - Success: {cf_info['success']}")
    print(f"    - Iterations: {cf_info['num_iterations']}")
    
    # Get counterfactual concepts
    with torch.no_grad():
        _, cf_concepts, _ = model(cf_signals, return_concepts=True)
    
    # Step 6: Visualize counterfactual
    print(f"\n[6/6] Creating visualizations...")
    
    # Plot signal comparison for ECG
    plot_counterfactual_comparison(
        original_signal=signals_normal['ecg'].squeeze().cpu().numpy(),
        counterfactual_signal=cf_signals['ecg'].squeeze().cpu().numpy(),
        modality_name='ECG',
        original_pred=class_names[cf_info['original_prediction']],
        cf_pred=class_names[cf_info['final_prediction']],
        save_path='/home/ubuntu/xffm_project/results/counterfactual_ecg.png'
    )
    print(f"  ✓ Saved ECG comparison to results/counterfactual_ecg.png")
    
    # Plot concept comparison
    plot_concept_comparison(
        concept_names=CONCEPT_NAMES,
        original_concepts=concepts[0].cpu().numpy(),
        cf_concepts=cf_concepts[0].cpu().numpy(),
        save_path='/home/ubuntu/xffm_project/results/concept_comparison.png'
    )
    print(f"  ✓ Saved concept comparison to results/concept_comparison.png")
    
    # Create comprehensive dashboard
    create_explainability_dashboard(
        original_signals={
            'ecg': signals_normal['ecg'].squeeze().cpu().numpy(),
            'ppg': signals_normal['ppg'].squeeze().cpu().numpy()
        },
        cf_signals={
            'ecg': cf_signals['ecg'].squeeze().cpu().numpy(),
            'ppg': cf_signals['ppg'].squeeze().cpu().numpy()
        },
        concept_names=CONCEPT_NAMES,
        original_concepts=concepts[0].cpu().numpy(),
        cf_concepts=cf_concepts[0].cpu().numpy(),
        original_pred=class_names[cf_info['original_prediction']],
        cf_pred=class_names[cf_info['final_prediction']],
        save_path='/home/ubuntu/xffm_project/results/explainability_dashboard.png'
    )
    print(f"  ✓ Saved dashboard to results/explainability_dashboard.png")
    
    # Summary
    print("\n" + "=" * 80)
    print("✓ Demo Completed Successfully!")
    print("=" * 80)
    
    print(f"\nKey Findings:")
    print(f"  - Model successfully predicted: {class_names[pred]}")
    print(f"  - Top 3 active concepts:")
    top_3_indices = np.argsort(concept_values)[::-1][:3]
    for idx in top_3_indices:
        print(f"    • {CONCEPT_NAMES[idx]}: {concept_values[idx]:.3f}")
    
    print(f"\n  - Counterfactual explanation generated:")
    print(f"    • Changed prediction from {class_names[cf_info['original_prediction']]} "
          f"to {class_names[cf_info['final_prediction']]}")
    
    # Calculate concept changes
    concept_changes = np.abs(cf_concepts[0].cpu().numpy() - concepts[0].cpu().numpy())
    top_changed_idx = np.argmax(concept_changes)
    print(f"    • Most changed concept: {CONCEPT_NAMES[top_changed_idx]} "
          f"(Δ = {concept_changes[top_changed_idx]:.3f})")
    
    print(f"\nGenerated Files:")
    print(f"  1. results/concept_activations.png")
    print(f"  2. results/counterfactual_ecg.png")
    print(f"  3. results/concept_comparison.png")
    print(f"  4. results/explainability_dashboard.png")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    from typing import Optional
    main()
