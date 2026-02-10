# Predict Analytical Axis

Machine Learning workflow for predicting the analytical axis from geometric input.
This repository contains tools for dataset generation, model training, inference, and integration with Grasshopper via a custom plugin.

---

## Overview

This project consists of four main components:

1. Dataset generation using Grasshopper
2. Model training using Python and Jupyter Notebook
3. Model inference/testing using Python
4. Grasshopper integration via a custom plugin

---

## Repository Structure

```text
Predict-Analytical-Axis/
│
├── End-Point_ML_training.ipynb     # Train Machine Learning model
├── Run_Trained_model.py            # Run trained model
│
├── Grasshopper/
│   ├── Generate dataset.gh         # Generate training dataset
│   ├── Case Study 1.gh
│   ├── Case Study 2.gh
│   ├── Case Study 3.gh
│   ├── Case Study 4.gh
│   └── Case Study 5.gh
│
└── README.md
```

Grasshopper plugin source code is located in a separate branch:

**Plug-In-GH branch:**
[https://github.com/larsolavtoppe/Predict-Analytical-Axis/tree/Plug-In-GH](https://github.com/larsolavtoppe/Predict-Analytical-Axis/tree/Plug-In-GH)

---

## Requirements

### Software

* Rhino 8
* Grasshopper
* Python 3.10 or newer
* Visual Studio (required to build Grasshopper plugin)

### Python libraries

Install dependencies:

```bash
pip install numpy pandas scikit-learn torch matplotlib notebook
```

---

## Workflow

### Step 1 — Generate Dataset (Grasshopper)

Open:

```text
Grasshopper/Generate dataset.gh
```

Steps:

1. Open Rhino 8
2. Open Grasshopper
3. Load `Generate dataset.gh`
4. Run the definition
5. Export dataset to file (JSON or CSV)

This dataset will be used for model training.

---

### Step 2 — Train the Model

Open:

```text
End-Point_ML_training.ipynb
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Then:

1. Open the notebook
2. Update dataset path if necessary
3. Run all cells

This will generate a trained model file.

Example output:

```text
trained_model.pkl
```

---

### Step 3 — Run the Trained Model

Run:

```bash
python Run_Trained_model.py
```

This script will:

* Load the trained model
* Load input data
* Predict analytical axis

---

### Step 4 — Use Model in Grasshopper

The Grasshopper plugin must be built from the **Plug-In-GH branch**.

Open:

[https://github.com/larsolavtoppe/Predict-Analytical-Axis/tree/Plug-In-GH](https://github.com/larsolavtoppe/Predict-Analytical-Axis/tree/Plug-In-GH)

Steps:

1. Open the Visual Studio project
2. Build the solution
3. Locate the generated `.gha` file
4. Copy the `.gha` file to:

```text
C:\Users\YOUR_USERNAME\AppData\Roaming\Grasshopper\Libraries\
```

5. Restart Rhino and Grasshopper

The plugin will now be available inside Grasshopper.

---

## Case Studies

Grasshopper definitions for the case studies are located in:

```text
Grasshopper/
```

These files demonstrate:

* Dataset generation
* Model testing
* Integration workflow

---

## Dataset

The dataset is not included in this repository due to its large size.

---

## Typical Workflow Pipeline

```text
Grasshopper → Generate dataset
        ↓
Python → Train model
        ↓
Python → Run trained model
        ↓
Grasshopper → Use model via plugin
```

---

## Notes

* Ensure correct file paths are set in scripts
* Plugin must be rebuilt if model integration changes
* Restart Rhino after installing plugin
