"""
Counterfactual Explanation Generator for X-FFM
Author: Sumetee Jirapattarasakul

This module generates counterfactual explanations by finding minimal
changes to input signals that would change the model's prediction.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple, Optional, List
import numpy as np


class CounterfactualGenerator(nn.Module):
    """
    Generates counterfactual explanations for biosignal classifications
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_class: Optional[int] = None,
        lambda_proximity: float = 1.0,
        lambda_sparsity: float = 0.1,
        lambda_smoothness: float = 0.01
    ):
        super().__init__()
        
        self.model = model
        self.target_class = target_class
        self.lambda_proximity = lambda_proximity
        self.lambda_sparsity = lambda_sparsity
        self.lambda_smoothness = lambda_smoothness
        
    def generate(
        self,
        original_signals: Dict[str, torch.Tensor],
        target_class: Optional[int] = None,
        num_iterations: int = 500,
        learning_rate: float = 0.01,
        verbose: bool = False
    ) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """
        Generate counterfactual explanation
        
        Args:
            original_signals: Original input signals
            target_class: Target class for counterfactual (if None, flip prediction)
            num_iterations: Number of optimization iterations
            learning_rate: Learning rate for optimization
            verbose: Whether to print progress
        
        Returns:
            counterfactual_signals: Modified signals
            info: Dictionary containing optimization information
        """
        # Set model to eval mode
        self.model.eval()
        
        # Get original prediction
        with torch.no_grad():
            original_logits, _, _ = self.model(original_signals, return_concepts=True)
            original_pred = original_logits.argmax(dim=1).item()
        
        # Determine target class
        if target_class is None:
            target_class = 1 - original_pred  # Flip prediction
        
        # Initialize counterfactual signals
        cf_signals = {}
        for modality, signal in original_signals.items():
            cf_signals[modality] = signal.clone().detach().requires_grad_(True)
        
        # Optimizer
        optimizer = optim.Adam([cf_signals[m] for m in cf_signals.keys()], lr=learning_rate)
        
        # Optimization loop
        losses_history = []
        
        for iteration in range(num_iterations):
            optimizer.zero_grad()
            
            # Forward pass
            logits, concepts, _ = self.model(cf_signals, return_concepts=True)
            
            # Target class loss (want to maximize probability of target class)
            target_loss = -logits[0, target_class]
            
            # Proximity loss (want to stay close to original)
            proximity_loss = 0
            for modality in cf_signals.keys():
                proximity_loss += torch.norm(
                    cf_signals[modality] - original_signals[modality]
                )
            
            # Sparsity loss (want minimal changes)
            sparsity_loss = 0
            for modality in cf_signals.keys():
                diff = cf_signals[modality] - original_signals[modality]
                sparsity_loss += torch.norm(diff, p=1)
            
            # Smoothness loss (want smooth changes)
            smoothness_loss = 0
            for modality in cf_signals.keys():
                diff = cf_signals[modality] - original_signals[modality]
                # Temporal smoothness
                temporal_diff = diff[:, :, 1:] - diff[:, :, :-1]
                smoothness_loss += torch.norm(temporal_diff)
            
            # Total loss
            total_loss = (
                target_loss +
                self.lambda_proximity * proximity_loss +
                self.lambda_sparsity * sparsity_loss +
                self.lambda_smoothness * smoothness_loss
            )
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Record loss
            losses_history.append(total_loss.item())
            
            # Check if target class is achieved
            with torch.no_grad():
                pred = logits.argmax(dim=1).item()
                if pred == target_class and iteration > 50:
                    if verbose:
                        print(f"Target class achieved at iteration {iteration}")
                    break
            
            if verbose and iteration % 100 == 0:
                print(f"Iteration {iteration}: Loss = {total_loss.item():.4f}, Pred = {pred}")
        
        # Detach counterfactual signals
        cf_signals_detached = {
            modality: signal.detach()
            for modality, signal in cf_signals.items()
        }
        
        # Get final prediction
        with torch.no_grad():
            final_logits, final_concepts, _ = self.model(
                cf_signals_detached,
                return_concepts=True
            )
            final_pred = final_logits.argmax(dim=1).item()
        
        # Prepare info
        info = {
            'original_prediction': original_pred,
            'target_class': target_class,
            'final_prediction': final_pred,
            'num_iterations': iteration + 1,
            'losses_history': losses_history,
            'success': final_pred == target_class
        }
        
        return cf_signals_detached, info
    
    def analyze_changes(
        self,
        original_signals: Dict[str, torch.Tensor],
        counterfactual_signals: Dict[str, torch.Tensor]
    ) -> Dict[str, Dict]:
        """
        Analyze the changes made to generate counterfactual
        
        Args:
            original_signals: Original signals
            counterfactual_signals: Counterfactual signals
        
        Returns:
            analysis: Dictionary containing change analysis for each modality
        """
        analysis = {}
        
        for modality in original_signals.keys():
            orig = original_signals[modality].squeeze().cpu().numpy()
            cf = counterfactual_signals[modality].squeeze().cpu().numpy()
            
            diff = cf - orig
            
            # Calculate statistics
            analysis[modality] = {
                'mean_absolute_change': np.abs(diff).mean(),
                'max_absolute_change': np.abs(diff).max(),
                'l2_distance': np.linalg.norm(diff),
                'percent_changed': (np.abs(diff) > 0.01).mean() * 100,
                'change_locations': np.where(np.abs(diff) > 0.1)[0].tolist()
            }
        
        return analysis


class ConceptCounterfactualGenerator:
    """
    Generate counterfactuals at the concept level
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        
    def generate_concept_counterfactual(
        self,
        original_signals: Dict[str, torch.Tensor],
        target_concepts: Dict[str, float],
        num_iterations: int = 300,
        learning_rate: float = 0.01
    ) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """
        Generate counterfactual by targeting specific concept values
        
        Args:
            original_signals: Original input signals
            target_concepts: Dictionary mapping concept names to target values
            num_iterations: Number of optimization iterations
            learning_rate: Learning rate
        
        Returns:
            counterfactual_signals: Modified signals
            info: Optimization information
        """
        self.model.eval()
        
        # Initialize counterfactual signals
        cf_signals = {}
        for modality, signal in original_signals.items():
            cf_signals[modality] = signal.clone().detach().requires_grad_(True)
        
        # Optimizer
        optimizer = optim.Adam([cf_signals[m] for m in cf_signals.keys()], lr=learning_rate)
        
        # Get concept indices
        concept_indices = {}
        for concept_name in target_concepts.keys():
            if concept_name in self.model.concept_names:
                concept_indices[concept_name] = self.model.concept_names.index(concept_name)
        
        # Optimization loop
        for iteration in range(num_iterations):
            optimizer.zero_grad()
            
            # Forward pass
            _, concepts, _ = self.model(cf_signals, return_concepts=True)
            
            # Concept loss
            concept_loss = 0
            for concept_name, target_value in target_concepts.items():
                if concept_name in concept_indices:
                    idx = concept_indices[concept_name]
                    concept_loss += (concepts[0, idx] - target_value) ** 2
            
            # Proximity loss
            proximity_loss = 0
            for modality in cf_signals.keys():
                proximity_loss += torch.norm(
                    cf_signals[modality] - original_signals[modality]
                )
            
            # Total loss
            total_loss = concept_loss + 0.1 * proximity_loss
            
            # Backward
            total_loss.backward()
            optimizer.step()
        
        # Detach
        cf_signals_detached = {
            modality: signal.detach()
            for modality, signal in cf_signals.items()
        }
        
        # Get final concepts
        with torch.no_grad():
            _, final_concepts, _ = self.model(cf_signals_detached, return_concepts=True)
        
        info = {
            'target_concepts': target_concepts,
            'final_concepts': {
                name: final_concepts[0, concept_indices[name]].item()
                for name in target_concepts.keys()
                if name in concept_indices
            }
        }
        
        return cf_signals_detached, info


def create_counterfactual_generator(
    model: nn.Module,
    **kwargs
) -> CounterfactualGenerator:
    """Factory function to create counterfactual generator"""
    return CounterfactualGenerator(model=model, **kwargs)
