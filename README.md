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

Open the Grasshopper definition:

    Grasshopper/Generate dataset.gh

Before running the script, configure the following parameters in Grasshopper:

- Set the output path where the dataset will be saved  
- Define the cross-section types  
- Define the geometric parameter ranges  
- Set the number of sample points  
- Define the scale range  
- Define the rotation range  

Run the Grasshopper definition to generate the dataset.


---

### Step 2 — Train the Model

Open the training notebook:

    End-Point_ML_training.ipynb

Before running the notebook:

- Set the input file path to the training dataset  
- Set the desired number of points in the point cloud  

Run all cells in the notebook.

The trained model will be saved in the same directory as the input datasets.

---

### Step 3 — Run the Trained Model (Grasshopper)

To run the trained model, use the Grasshopper plugin located in the `Plug-In-GH` branch.
[https://github.com/larsolavtoppe/Predict-Analytical-Axis/tree/Plug-In-GH](https://github.com/larsolavtoppe/Predict-Analytical-Axis/tree/Plug-In-GH)

Save the file:

    Run_Trained_model.py

In the plugin, provide file paths to:

- The trained model file  
- `Run_Trained_model.py`  

Then run the component in Grasshopper to execute the model.

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

The dataset is not included in this repository due to its size.

The dataset can be provided upon request from the authors.


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

