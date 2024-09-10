
# Antimicrobial Resistance Prediction Project

Oscar Nyangiri, Fred Mutisya, Beryl Primrose Gladstone

## Project Overview

This repository contains tools and scripts aimed at predicting antimicrobial resistance (AMR) using machine learning models. The focus is on calculating and interpreting key performance metrics, such as the Number Needed to Predict (NNP) antimicrobial resistance, and evaluating models in the context of healthcare applications.

The repository includes:
1. **Jupyter Notebooks** for interactive analysis and metric computation.
2. **Python Scripts** to automate and perform tasks such as AMR prediction and triage.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Files and Directories](#files-and-directories)
- [Usage](#usage)
- [Assumptions and Limitations](#assumptions-and-limitations)
- [Key Concepts](#key-concepts)
- [Contributing](#contributing)
- [License](#license)
- [source-performance_example] (#source-performance)
---

## Installation

### Prerequisites

To run this project, you need the following software installed:
- Python 3.8+
- Jupyter Notebook
- Required Python libraries, which can be installed with the following command:

```bash
pip install -r requirements.txt
```

### Setting Up the Environment

1. Clone this repository:

   ```bash
   git clone https://github.com/your_username/amr_prediction.git
   ```

2. Navigate to the project directory:

   ```bash
   cd amr_prediction
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## Files and Directories

### `Number needed to Predict AMR.ipynb`
This Jupyter Notebook contains code and analysis for calculating the "Number Needed to Predict" (NNP) for antimicrobial resistance. The NNP metric provides a way to assess the predictive value of models in identifying resistance patterns in bacterial infections. The notebook includes:
- Exploratory Data Analysis (EDA) on AMR datasets.
- Machine learning model training and evaluation.
- Detailed explanations of performance metrics like NNP, sensitivity, specificity, and more.

### `amr_triage.py`
This Python script is designed to automate the triage process for predicting AMR. It includes functions for data preprocessing, model training, and evaluation. The script can be used in command-line workflows or integrated into larger systems.

### `requirements.txt`
This file lists the dependencies required to run the project. You can install them with:

```bash
pip install -r requirements.txt
```

---

## Usage

### Running the Jupyter Notebook

To interact with the notebook, run the following command from the project directory:

```bash
jupyter notebook
```

Once the notebook interface opens, navigate to `Number needed to Predict AMR.ipynb` to start your analysis.

### Running the Python Script

To execute the triage script, run the following command:

```bash
streamlit run amr_triage.py 
```

Replace `<input_data.csv>` with the path to your dataset and `<output_predictions.csv>` with the desired location for the output.

#### Script Options:
- `--input`: Path to the CSV file containing input data.
- `--output`: Path to save the output predictions.
- Additional options and hyperparameters for model tuning can be added based on your needs.

---


### Limitations:
1. **Small Sample Sizes**: For small datasets, model performance might not be accurate, and normal approximation may not hold.
2. **Pooled Variability**: Some metrics like NNP rely on pooled variability assumptions, which may introduce bias in certain datasets.
3. **Imbalanced Data**: The models might perform poorly with highly imbalanced datasets without proper handling of class distribution.

---

## Key Concepts

### Number Needed to Predict (NNP)
The NNP is a metric that quantifies how many patients need to be examined to make one correct prediction about AMR. It is related to the sensitivity and specificity of the predictive model.

### Sensitivity and Specificity
- **Sensitivity**: The true positive rate, or the proportion of actual positives (AMR cases) correctly identified.
- **Specificity**: The true negative rate, or the proportion of actual negatives (non-AMR cases) correctly identified.

---

## Contributing

Contributions are welcome! Please follow the steps below to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add new feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Open a pull request.

---

## source-performance

Comparing screening performance of model across sample sources

<img width="727" alt="image" src="https://github.com/user-attachments/assets/7b57b113-03f5-4dce-9a6c-16774313ebf8">
NND=Number needed to diagnose, NNM=Number needed to Misdiagnose, NNP=Number need to predict
---

## screenshots of tool

<div align="center">
	<img width="50%" alt="image" src="https://github.com/user-attachments/assets/edee1ff9-a55f-4767-9be9-59fd557545b1">
<img width="50%" alt="image" src="https://github.com/user-attachments/assets/f6a8f220-22f3-44ba-bb6f-42b9d363bce2">
</div>

Feel free to contact us with any questions or suggestions!

