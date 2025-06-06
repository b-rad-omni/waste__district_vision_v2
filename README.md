# 🚀 Waste‑District‑Vision

ML Vision to help with recycling compliance and approved usage of community trash disposal.

A proof‑of‑concept edge‑inference system that flags non‑compliant items entering a community trash disposal site. The system detects large foreign objects like plastic buckets (5+ gallons), drywall mud buckets, and other items that don't belong in standard waste streams. It aims to improve compliance and reduce contamination in recycling systems.

Built with Ultralytics YOLOv8 as a local tool focused on improving community waste management and promoting proper recycling practices.

This project utilized CVAT for dataset labeling and annotation. More info:
https://github.com/cvat-ai/cvat

**Technology Stack:** YOLOv8, OpenCV, Streamlit, Python

**Maintained by Stafford Solutions, LLC**

[![Project Board](https://img.shields.io/badge/Project%20Board–Kanban-blue)](https://github.com/users/b-rad-omni/projects/2)  
> **Live Project Board** – click the badge above to see To‑Do, In‑Progress, and Done.

This project is currently complete and has delivered a MVP (Minimal Viable Product) to the client. Plans for future iteration and site deployment is in place.

## Model Weights
Due to file size limitations, trained model weights cannot be uploaded to GitHub. If you would like a copy of the trained weights, please contact the maintainer via GitHub.

---

## Features

- Detects plastic buckets 5 gallons or larger (drywall mud buckets, large foreign objects like kiddie pools)
- Saves the events as possible infractions for human review at a later point in time.   
- Uses Streamlit for basic review of infractions with daily, weekly, and monthly highlights. 

---

Examples:
<img src="https://github.com/b-rad-omni/waste__district_vision_v2/blob/db1f7ea32edfcdb1564de6fa8bac43f09977276a/public_still1.jpg" alt="demo still 1" width=30% height=30%>
<img src="https://github.com/b-rad-omni/waste__district_vision_v2/blob/db1f7ea32edfcdb1564de6fa8bac43f09977276a/public_still2.jpg" alt="demo still 2" width=30% height=30%>
<img src="https://github.com/b-rad-omni/waste__district_vision_v2/blob/db1f7ea32edfcdb1564de6fa8bac43f09977276a/public_still3.jpg" alt="demo still 3" width=30% height=30%>


## 🚀 Quickstart

1. **Clone the repo**  
   ```bash
   git clone https://github.com/b-rad-omni/waste__district_vision_v2.git
   cd waste__district_vision_v2
   ```

2. **Install the package**  
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   pip install -e .
   ```

3. **Download or train your YOLOv8 model**  
   - **Pretrained weights**: Contact maintainer for trained model weights
   - **To train** on your own dataset:  
     ```bash
     python scripts/training/train.py --data configs/training/baseline.yaml
     ```

4. **Run inference on video/camera**  
   ```bash
   python scripts/inference/run_inference.py --model models/trained/best.pt --source /path/to/video.mp4
   ```

5. **Launch dashboard** (optional)
   ```bash
   python scripts/run_dashboard.py
   ```

---

## 📂 Project Structure

```
src/                   # Core Python package
├── data/              # Data processing modules
│   ├── collectors/    # Camera, motion detection, frame analysis
│   ├── label_analyzer.py      # Dataset analysis tools
│   ├── dataset_splitter.py    # Stratified dataset splitting
│   └── label_validator.py     # Label validation and cleanup
├── models/            # Model inference and management
├── utils/             # Configuration, storage, registry management
└── dashboard/         # Streamlit web interface

scripts/               # Executable tools and workflows
├── data_collection/   # Data collection pipelines
├── data_preparation/  # Dataset tools (split, validate, analyze)
├── inference/         # Model inference runners
├── training/          # Model training scripts
└── utilities/         # Helper tools and model management

configs/               # Configuration files
├── data_collection/   # Data collection settings
├── tracking/          # Object tracking configurations
└── training/          # Training hyperparameters

models/                # Model storage (gitignored)
├── trained/           # Your trained model weights
├── pretrained/        # Base models
└── production/        # Deployed models

tests/                 # Test suite
requirements.txt       # Python dependencies
pyproject.toml         # Package configuration
```

---

## 🛠️ Available Tools

### Data Preparation
```bash
# Analyze dataset class distribution
python scripts/data_preparation/analyze_labels.py --dataset-root ./datasets

# Validate labels for errors and issues  
python scripts/data_preparation/validate_labels.py --directory ./train --validate

# Split dataset while maintaining class balance
python scripts/data_preparation/split_dataset.py --source ./all_data --output ./splits

# Extract balanced frames from videos
python scripts/data_preparation/extract_balanced_frames.py --input-dir ./videos --output-dir ./frames
```

### Training & Inference
```bash
# Train model with custom configuration
python scripts/training/train.py --config configs/training/baseline.yaml

# Run inference on videos or camera streams
python scripts/inference/run_inference.py --model models/trained/best.pt --source camera

# Export model for deployment
python scripts/utilities/export_model.py --model models/trained/best.pt --format onnx
```

### Data Collection
```bash
# Start data collection pipeline
python scripts/data_collection/main_data_collection_v2.py --config configs/data_collection/default_data_collection.yaml
```

---

## ⚖️ License & Attribution

This project includes Ultralytics YOLOv8 (© Ultralytics) under the GNU Affero General Public License v3.0 (AGPL‑3.0).

- The full text of AGPL‑3.0 is in `LICENSE`.  
- By using or modifying this code (including via a hosted service), you agree to the AGPL‑3.0 terms, which require you to publish your source code under the same license.  
- For a commercial license that removes AGPL‑3.0 obligations, contact Ultralytics for an Enterprise License.

---

## ⚙️ Acknowledgements

- **YOLOv8**: state‑of‑the‑art real‑time object detection & segmentation  
- **Edge Inference**: NVIDIA Jetson Nano / Coral USB TPU optimizations  
- OpenCV
- Streamlit
- Python Libraries

© 2025 Stafford Solutions, LLC
