# MedAdapt-SAM Dashboard - Quick Start Guide

## 🚀 How to Run the Dashboard

### Step 1: Install Streamlit (if not already installed)

```bash
pip install streamlit
```

### Step 2: Navigate to Project Directory

```bash
cd "D:\project major\MedAdapt-SAM"
```

### Step 3: Launch the Dashboard

```bash
streamlit run streamlit_app\app.py
```

The dashboard will automatically open in your default web browser at `http://localhost:8501`

---

## 📋 What You Can Do

### 1. **Upload MRI Scan**
- Click the upload area or drag & drop an image
- Supported formats: PNG, JPG, JPEG

### 2. **Run AI Analysis**
- Click "START AI ANALYSIS" button
- Wait for real-time processing (~3 seconds)

### 3. **View Results**
- **Detection**: Tumor Yes/No
- **Classification**: Tumor type (GBM, LGG, etc.)
- **Staging**: WHO Grade (I-IV)
- **Growth Rate**: High/Moderate/Low
- **Risk Level**: Assessment

### 4. **Explore Metrics**
- Dice Score
- IoU (Jaccard Index)
- Hausdorff Distance (HD & HD95)
- Precision & Recall
- Interactive charts

### 5. **Generate Clinical Report**
- Comprehensive diagnostic report
- Treatment recommendations
- Download as TXT, CSV, or PNG

### 6. **Ask Questions**
- Use the AI chatbot in the sidebar
- Get context-aware medical information

---

## 🎯 Features

✅ **Real-time tumor detection** (Yes/No)  
✅ **Tumor classification** (GBM, LGG, Anaplastic Astrocytoma, etc.)  
✅ **WHO Grade staging** (I-IV)  
✅ **Growth rate estimation**  
✅ **Risk assessment**  
✅ **Complete metrics suite** (Dice, IoU, HD, Precision, Recall)  
✅ **Interactive visualizations**  
✅ **Clinical report generation**  
✅ **Attention heatmaps**  
✅ **RAG-based chatbot**  

---

## 📊 Example Output

```
DETECTION: ✓ Tumor Detected
TYPE: Glioblastoma Multiforme (GBM)
STAGE: WHO Grade IV
CONFIDENCE: 87%

GROWTH RATE: High (>2mm/month)
RISK LEVEL: High

METRICS:
- Dice Score: 0.9124
- IoU: 0.8756
- Precision: 0.9234
- Recall: 0.9012
- Hausdorff: 8.45 mm
```

---

## ⚠️ Note

This is a research and educational tool. All diagnoses should be confirmed by qualified medical professionals.

---

## 📚 Documentation

- **Full Walkthrough**: [dashboard_walkthrough.md](file:///C:/Users/shasw/.gemini/antigravity/brain/218e4650-1f3e-42f2-be6f-a490fc1e8540/dashboard_walkthrough.md)
- **Project Structure**: [project_structure_documentation.md](file:///C:/Users/shasw/.gemini/antigravity/brain/218e4650-1f3e-42f2-be6f-a490fc1e8540/project_structure_documentation.md)

---

**Ready to analyze brain tumors with AI!** 🧠✨
