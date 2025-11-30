"""
Explainability Package for X-FFM
"""

from .counterfactual import (
    CounterfactualGenerator,
    ConceptCounterfactualGenerator,
    create_counterfactual_generator
)

from .visualization import (
    plot_concept_activations,
    plot_counterfactual_comparison,
    plot_concept_comparison,
    plot_cross_modal_attention,
    create_explainability_dashboard
)

__all__ = [
    'CounterfactualGenerator',
    'ConceptCounterfactualGenerator',
    'create_counterfactual_generator',
    'plot_concept_activations',
    'plot_counterfactual_comparison',
    'plot_concept_comparison',
    'plot_cross_modal_attention',
    'create_explainability_dashboard'
]
