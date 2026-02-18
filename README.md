# MedAdapt-SAM: Automatic Prompt-Driven Brain Tumor Segmentation

## Project Overview
This project adapts the Segment Anything Model (SAM) for brain tumor MRI segmentation with automatic prompt generation, achieving high accuracy and clinical usability.

## Dataset Information
- **Dataset**: BraTS 2021
- **Total Images**: 276,267 PNG images
- **Patient Cases**: 1,252
- **Image Types**: FLAIR, RGB, ET, TC, WT masks
- **Location**: `D:\major projrct PNG folder\brats_png`

## Project Structure
```
MedAdapt-SAM/
├── data/
│   ├── dataset_loader.py       # Data loading utilities
│   ├── augmentation.py         # Data augmentation
│   └── preprocessing.py        # Preprocessing pipeline
├── models/
│   ├── unet.py                 # U-Net baseline
│   ├── sam_vanilla.py          # Vanilla SAM
│   ├── sam_adapter.py          # Adapter-based SAM
│   └── prompt_generator.py     # Automatic prompt generation
├── training/
│   ├── train_unet.py           # U-Net training
│   ├── train_sam.py            # SAM training
│   └── train_adapter.py        # Adapter training
├── evaluation/
│   ├── metrics.py              # Dice, IoU, Hausdorff
│   └── evaluator.py            # Model evaluation
├── prompts/
│   ├── box_prompts.py          # Box prompt generation
│   ├── point_prompts.py        # Point prompt generation
│   └── hybrid_prompts.py       # Hybrid strategies
├── uncertainty/
│   ├── uncertainty_maps.py     # Uncertainty estimation
│   └── refinement.py           # Uncertainty-guided refinement
├── explainability/
│   ├── attention_viz.py        # Attention visualization
│   └── prompt_influence.py     # Prompt analysis
├── llm_integration/
│   ├── text_converter.py       # Segmentation to text
│   └── rag_system.py           # RAG for explanations
├── streamlit_app/
│   ├── app.py                  # Main Streamlit app
│   ├── components/             # UI components
│   └── utils.py                # Helper functions
├── configs/
│   └── config.yaml             # Configuration file
├── checkpoints/                # Model checkpoints
├── results/                    # Experiment results
└── requirements.txt            # Dependencies
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Train Baseline Models
```bash
python training/train_unet.py
python training/train_sam.py
```

### 2. Run Prompt Sensitivity Study
```bash
python prompts/evaluate_prompts.py
```

### 3. Train with Automatic Prompts
```bash
python training/train_adapter.py --auto-prompts
```

### 4. Launch Streamlit Demo
```bash
streamlit run streamlit_app/app.py
```

## Key Features

### ✅ Phase 2: Baseline Models
- U-Net implementation
- Vanilla SAM integration
- Adapter-based SAM

### ✅ Phase 3: Prompt Sensitivity
- Box-only prompts
- Point-only prompts
- Hybrid prompt strategies

### ✅ Phase 4: Model Stabilization
- Frozen adapter weights
- Fixed prompt strategy

### ✅ Phase 5: Uncertainty-Guided Refinement
- Uncertainty map generation
- Corrective prompt generation

### ✅ Phase 6-7: Automatic Prompting
- Semi-automatic pipeline
- Fully automatic segmentation

### ✅ Phase 8: Explainable AI
- Attention map visualization
- Prompt influence analysis

### ✅ Phase 9: LLM Integration
- Structured text conversion
- RAG-based explanations

### ✅ Phase 10: Deployment
- Interactive Streamlit interface
- Real-time segmentation demo

## Evaluation Metrics
- **Dice Coefficient**: Overlap measure
- **IoU (Jaccard Index)**: Intersection over Union
- **Hausdorff Distance**: Boundary accuracy
- **95th Percentile HD**: Robust boundary metric

## Authors
Your Name / Team

## License
MIT License
