# X-FFM: Cross-Modal Interpretability and Explainability for Clinical Decision Support

**Author:** Sumetee Jirapattarasakul  
**Affiliation:** Electrical and Information Engineering

## Overview

X-FFM is an explainable AI framework for multimodal biosignal analysis that combines:
- **Concept Bottleneck Models (CBM):** Learn clinically meaningful concepts
- **Counterfactual Explanations:** Provide actionable insights for clinicians
- **Cross-Modal Attention:** Capture relationships between different biosignal modalities

## Key Features

1. **Interpretable by Design:** Uses concept bottleneck architecture to ensure interpretability
2. **Clinically Relevant:** Learns medical concepts that align with clinical knowledge
3. **Actionable Explanations:** Generates counterfactual explanations for decision support
4. **Multimodal Fusion:** Processes ECG, PPG, and other biosignals simultaneously

## Project Structure

```
xffm_project/
├── models/
│   ├── xffm_architecture.py    # Main X-FFM model
│   ├── concept_bottleneck.py   # Concept Bottleneck Model
│   └── multimodal_encoder.py   # Multimodal signal encoder
├── explainability/
│   ├── counterfactual.py       # Counterfactual explanation generator
│   ├── concept_activation.py   # Concept activation analysis
│   └── visualization.py        # Visualization utilities
├── experiments/
│   ├── train_xffm.py           # Training script
│   └── demo_explainability.py  # Explainability demo
├── results/                    # Generated results and visualizations
└── data/                       # Dataset directory
```

## Installation

```bash
cd xffm_project
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

Run the explainability demo:

```bash
python experiments/demo_explainability.py
```

## Clinical Use Cases

1. **Arrhythmia Detection:** Identify abnormal heart rhythms with concept-based explanations
2. **Stress Assessment:** Analyze stress levels using multimodal biosignals
3. **Sleep Stage Classification:** Classify sleep stages with interpretable features

## Citation

If you use this code in your research, please cite:

```
@article{jirapattarasakul2025xffm,
  title={X-FFM: Cross-Modal Interpretability and Explainability for Clinical Decision Support},
  author={Jirapattarasakul, Sumetee},
  journal={TBD},
  year={2025}
}
```
