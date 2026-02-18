# MedAdapt-SAM Quick Start Guide

## 🚀 Getting Started

### 1. Installation

```bash
# Navigate to project directory
cd "D:\project major\MedAdapt-SAM"

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download SAM Checkpoint

Download the SAM checkpoint and place it in the `checkpoints` folder:

```bash
# Create checkpoints directory
mkdir checkpoints

# Download SAM ViT-B checkpoint
# Visit: https://github.com/facebookresearch/segment-anything#model-checkpoints
# Download sam_vit_b_01ec64.pth and place in checkpoints/
```

### 3. Verify Dataset

Your dataset is already prepared at:
```
D:\major projrct PNG folder\brats_png
```

Total images: **276,267 PNG images**
Patient cases: **1,252**

### 4. Train Baseline Model (U-Net)

```bash
# Train U-Net baseline
python training/train_unet.py --config configs/config.yaml --device cuda

# Monitor training with TensorBoard
tensorboard --logdir logs/unet
```

### 5. Train SAM Adapter

```bash
# Train SAM with adapters
python training/train_adapter.py --config configs/config.yaml --device cuda
```

### 6. Evaluate Models

```bash
# Evaluate all models
python evaluation/evaluate_all.py --checkpoint checkpoints/unet_best.pth
```

### 7. Launch Streamlit Demo

```bash
# Run the web application
streamlit run streamlit_app/app.py
```

The app will open at: `http://localhost:8501`

## 📊 Project Phases

### ✅ Phase 1: Dataset Preparation
- Dataset already prepared with 276,267 images
- BraTS 2021 PNG format
- ET, TC, WT masks available

### 🔄 Phase 2: Baseline Models (Current)
- **U-Net**: Implemented ✅
- **Training Script**: Ready ✅
- **Metrics**: Dice, IoU, HD95 ✅

### 🔄 Phase 3: Prompt Sensitivity Study
- Box prompts ✅
- Point prompts ✅
- Hybrid prompts ✅

### 🔄 Phase 4: Model Stabilization
- Adapter freezing implemented ✅
- Best prompt strategy selection pending

### 🔄 Phase 5: Uncertainty-Guided Refinement
- Monte Carlo Dropout ✅
- Deep Ensembles ✅
- Iterative refinement ✅

### 🔄 Phase 6-7: Automatic Prompting
- Semi-automatic pipeline ✅
- Fully automatic generation ✅

### 🔄 Phase 8: Explainable AI
- Attention visualization (pending)
- Prompt influence analysis (pending)

### 🔄 Phase 9: LLM Integration
- RAG system (pending)
- Clinical explanations (pending)

### 🔄 Phase 10: Deployment
- Streamlit app ✅
- Interactive interface ✅

## 🎯 Quick Commands

### Test Dataset Loader
```bash
python data/dataset_loader.py
```

### Test U-Net Model
```bash
python models/unet.py
```

### Test Metrics
```bash
python evaluation/metrics.py
```

### Test Prompt Generation
```bash
python prompts/prompt_generator.py
```

### Test Uncertainty Estimation
```bash
python uncertainty/uncertainty_estimation.py
```

## 📁 Project Structure

```
MedAdapt-SAM/
├── configs/
│   └── config.yaml              # Configuration file
├── data/
│   └── dataset_loader.py        # Dataset loading
├── models/
│   ├── unet.py                  # U-Net baseline
│   ├── sam_adapter.py           # SAM with adapters
│   └── prompt_generator.py      # Automatic prompts
├── training/
│   ├── train_unet.py            # U-Net training
│   └── train_adapter.py         # Adapter training
├── evaluation/
│   ├── metrics.py               # Evaluation metrics
│   └── evaluator.py             # Model evaluation
├── prompts/
│   └── prompt_generator.py      # Prompt generation
├── uncertainty/
│   └── uncertainty_estimation.py # Uncertainty tools
├── streamlit_app/
│   └── app.py                   # Web interface
├── checkpoints/                 # Model checkpoints
├── results/                     # Experiment results
├── logs/                        # Training logs
└── requirements.txt             # Dependencies
```

## 🔧 Configuration

Edit `configs/config.yaml` to customize:
- Dataset paths
- Model architecture
- Training hyperparameters
- Prompt strategies
- Uncertainty settings

## 📈 Monitoring Training

### TensorBoard
```bash
tensorboard --logdir logs
```

### Weights & Biases (Optional)
```bash
# Set up wandb
wandb login
# Training will automatically log to wandb
```

## 🐛 Troubleshooting

### CUDA Out of Memory
- Reduce batch size in `configs/config.yaml`
- Use smaller image size (e.g., 128 instead of 256)

### Dataset Not Found
- Verify path in config: `D:/major projrct PNG folder/brats_png`
- Check that PNG files exist

### SAM Checkpoint Missing
- Download from: https://github.com/facebookresearch/segment-anything
- Place in `checkpoints/` folder

## 📚 Next Steps

1. **Train baseline models** (U-Net, SAM)
2. **Run prompt sensitivity study**
3. **Implement automatic prompting**
4. **Add explainability features**
5. **Integrate LLM for explanations**
6. **Deploy final demo**

## 💡 Tips

- Start with U-Net baseline to verify pipeline
- Use smaller dataset subset for quick testing
- Monitor metrics during training
- Save checkpoints regularly
- Test on validation set before final evaluation

## 📞 Support

For issues or questions:
- Check documentation in README.md
- Review code comments
- Test individual modules

---

**Happy Training! 🧠🚀**
