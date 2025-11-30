"""
Concept Bottleneck Model for X-FFM
Author: Sumetee Jirapattarasakul

This module implements a Concept Bottleneck Model that learns
clinically meaningful concepts as an intermediate representation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


class ConceptBottleneckLayer(nn.Module):
    """
    Concept Bottleneck Layer that predicts clinical concepts
    from multimodal features
    """
    
    def __init__(
        self,
        input_dim: int,
        concept_names: List[str],
        concept_dim: int = 64,
        use_sigmoid: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.concept_names = concept_names
        self.num_concepts = len(concept_names)
        self.concept_dim = concept_dim
        self.use_sigmoid = use_sigmoid
        
        # Concept prediction layers
        self.concept_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, concept_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(concept_dim, 1)
            )
            for _ in range(self.num_concepts)
        ])
        
    def forward(
        self,
        features: torch.Tensor,
        return_logits: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            features: Input features [batch_size, input_dim]
            return_logits: Whether to return concept logits
        
        Returns:
            concepts: Concept predictions [batch_size, num_concepts]
            concept_logits: Concept logits [batch_size, num_concepts] (optional)
        """
        batch_size = features.size(0)
        
        # Predict each concept
        concept_logits = []
        for predictor in self.concept_predictors:
            logit = predictor(features)
            concept_logits.append(logit)
        
        # Stack: [batch_size, num_concepts]
        concept_logits = torch.cat(concept_logits, dim=1)
        
        # Apply sigmoid activation
        if self.use_sigmoid:
            concepts = torch.sigmoid(concept_logits)
        else:
            concepts = concept_logits
        
        if return_logits:
            return concepts, concept_logits
        else:
            return concepts, None
    
    def get_concept_importance(
        self,
        features: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate importance of each concept using gradient-based method
        
        Args:
            features: Input features [batch_size, input_dim]
        
        Returns:
            importance: Concept importance scores [batch_size, num_concepts]
        """
        features.requires_grad_(True)
        concepts, _ = self.forward(features)
        
        # Calculate gradients
        importance = []
        for i in range(self.num_concepts):
            grad = torch.autograd.grad(
                outputs=concepts[:, i].sum(),
                inputs=features,
                create_graph=True,
                retain_graph=True
            )[0]
            importance.append(grad.abs().sum(dim=1, keepdim=True))
        
        importance = torch.cat(importance, dim=1)
        return importance


class ConceptBasedClassifier(nn.Module):
    """
    Classifier that makes predictions based on learned concepts
    """
    
    def __init__(
        self,
        num_concepts: int,
        num_classes: int,
        hidden_dim: int = 128,
        use_concept_whitening: bool = True
    ):
        super().__init__()
        
        self.num_concepts = num_concepts
        self.num_classes = num_classes
        self.use_concept_whitening = use_concept_whitening
        
        # Concept whitening (optional)
        if use_concept_whitening:
            self.concept_whitening = nn.BatchNorm1d(num_concepts)
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(num_concepts, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(
        self,
        concepts: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            concepts: Concept predictions [batch_size, num_concepts]
        
        Returns:
            logits: Classification logits [batch_size, num_classes]
        """
        # Apply concept whitening
        if self.use_concept_whitening:
            concepts = self.concept_whitening(concepts)
        
        # Classify
        logits = self.classifier(concepts)
        return logits
    
    def get_concept_weights(self) -> torch.Tensor:
        """
        Get the weights of the first linear layer to understand
        which concepts are important for each class
        
        Returns:
            weights: [num_classes, num_concepts]
        """
        # Get weights from the first linear layer
        first_layer = self.classifier[0]
        weights = first_layer.weight.data  # [hidden_dim, num_concepts]
        
        # Get weights from the last linear layer
        last_layer = self.classifier[-1]
        final_weights = last_layer.weight.data  # [num_classes, hidden_dim//2]
        
        # Approximate contribution (simplified)
        # In practice, you might want to use more sophisticated methods
        return weights.mean(dim=0, keepdim=True).expand(self.num_classes, -1)


class ConceptBottleneckModel(nn.Module):
    """
    Complete Concept Bottleneck Model that combines:
    1. Feature extraction
    2. Concept prediction
    3. Concept-based classification
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        concept_names: List[str],
        num_classes: int,
        concept_dim: int = 64,
        classifier_hidden_dim: int = 128
    ):
        super().__init__()
        
        self.encoder = encoder
        self.concept_names = concept_names
        self.num_concepts = len(concept_names)
        self.num_classes = num_classes
        
        # Get encoder output dimension
        encoder_output_dim = encoder.fusion_dim
        
        # Concept bottleneck layer
        self.concept_layer = ConceptBottleneckLayer(
            input_dim=encoder_output_dim,
            concept_names=concept_names,
            concept_dim=concept_dim
        )
        
        # Concept-based classifier
        self.classifier = ConceptBasedClassifier(
            num_concepts=self.num_concepts,
            num_classes=num_classes,
            hidden_dim=classifier_hidden_dim
        )
        
    def forward(
        self,
        signals: Dict[str, torch.Tensor],
        return_concepts: bool = False,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args:
            signals: Dictionary of input signals
            return_concepts: Whether to return concept predictions
            return_attention: Whether to return attention weights
        
        Returns:
            logits: Classification logits [batch_size, num_classes]
            concepts: Concept predictions [batch_size, num_concepts] (optional)
            attention: Attention weights (optional)
        """
        # Encode multimodal signals
        features, attention = self.encoder(signals, return_attention=return_attention)
        
        # Predict concepts
        concepts, _ = self.concept_layer(features)
        
        # Classify based on concepts
        logits = self.classifier(concepts)
        
        if return_concepts:
            return logits, concepts, attention
        else:
            return logits, None, attention
    
    def intervene_on_concepts(
        self,
        signals: Dict[str, torch.Tensor],
        concept_interventions: Dict[str, float]
    ) -> torch.Tensor:
        """
        Perform concept intervention: manually set certain concepts
        and observe the effect on predictions
        
        Args:
            signals: Dictionary of input signals
            concept_interventions: Dict mapping concept names to values
        
        Returns:
            logits: Classification logits after intervention
        """
        # Encode and predict concepts
        features, _ = self.encoder(signals, return_attention=False)
        concepts, _ = self.concept_layer(features)
        
        # Apply interventions
        for concept_name, value in concept_interventions.items():
            if concept_name in self.concept_names:
                idx = self.concept_names.index(concept_name)
                concepts[:, idx] = value
        
        # Classify with intervened concepts
        logits = self.classifier(concepts)
        return logits


def create_concept_bottleneck_model(
    encoder: nn.Module,
    concept_names: List[str],
    num_classes: int
) -> ConceptBottleneckModel:
    """Factory function to create concept bottleneck model"""
    return ConceptBottleneckModel(
        encoder=encoder,
        concept_names=concept_names,
        num_classes=num_classes
    )
