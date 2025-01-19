# CustomerSatisfaction-MLOps

This project is designed to streamline the MLOps pipeline for analyzing and improving customer satisfaction. It is based on a Kaggle competition dataset and includes tools for data preprocessing, model training, evaluation, and deployment.

## Project Overview

The **CustomerSatisfaction-MLOps** project automates the steps involved in developing, training, and deploying machine learning models for customer satisfaction analysis. It is designed to handle various stages of the MLOps lifecycle, including data preprocessing, model training, evaluation, and deployment.

## Prerequisites

Ensure you have the following installed on your system:

- Python 3.7 or higher
- pip (Python package manager)

## Installation

1. Clone the repository:
    
    ```
    git clone <repository-url>
    cd CustomerSatisfaction-MLOps
    ```
    
2. Install the required dependencies:
    
    ```
    pip install -r requirements.txt
    ```
    

## Running the Project

### 1. Execute the Pipeline

To run the machine learning pipeline, use the following command:

```
python scripts/run_pipeline.py
```

This script handles the entire process, including data preprocessing, model training, and evaluation.

### 2. Setup the Project

If you need to set up the project environment, you can run:

```
python setup.py
```

This script will configure the necessary environment variables and dependencies for the project.

## Project Structure

```
CustomerSatisfaction-MLOps/
├── src/
│   ├── __init__.py           # Package initialization
│   ├── data_processing.py    # Data preprocessing functions
│   ├── model_evaluation.py   # Model evaluation utilities
│   └── models.py             # Model definitions
│
├── scripts/
│   ├── __init__.py           # Package initialization
│   ├── clean_data.py         # Data cleaning script
│   ├── config.py             # Configuration settings
│   ├── evaluation.py         # Evaluation script
│   ├── ingest_data.py        # Data ingestion script
│   ├── model_train.py        # Model training script
│   ├── models/               # Directory for model artifacts
│   │   └── __init__.py       # Package initialization
│   └── run_pipeline.py       # Main script to execute the ML pipeline
│
├── pipelines/
│   ├── __init__.py           # Package initialization
│   └── training_pipeline.py  # Training pipeline definition
│
├── data/
│   ├── raw/                  # Raw dataset directory
│   │   └── olist_customers_dataset.csv
│
├── requirements.txt          # Python dependencies
├── setup.py                  # Project setup script
└── README.md                 # Project documentation (this file)
```

## Dataset

The project uses the **Olist Customers Dataset** provided in a Kaggle competition. Make sure to download the dataset from Kaggle and place it in the `data/raw/` directory.
