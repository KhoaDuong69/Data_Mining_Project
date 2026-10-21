# Data Mining Project – Frequent Itemset Mining and Association Rules

## 1. Project Overview

This project analyzes the relationships between eating habits and food preferences using survey data. It applies frequent itemset mining and association rule mining to discover common patterns and potential food combinations.

## 2. Project Objectives

- Collect and preprocess eating habit survey data.
- Transform data into a suitable format for mining.
- Apply and compare frequent itemset mining algorithms.
- Analyze relationships between attributes using association rules.
- Visualize results through a Streamlit application.

## 3. Dataset

- **Dataset:** `food_coded.csv`
- **Source:** [Food Choices Dataset](https://www.kaggle.com/datasets/borapajo/food-choices)
- **Format:** CSV
- **Content:** Survey data about eating habits, food choices, and cuisine preferences.

The data is cleaned and transformed before mining, including handling missing values and removing duplicate records.

## 4. Algorithms

The project implements and compares three frequent itemset mining algorithms:

- **Apriori:** Generates and evaluates candidate itemsets.
- **FP-Growth:** Uses an FP-Tree structure to reduce candidate generation.
- **IT-Tree:** Mines frequent itemsets using transaction ID (TID) sets.

Association rules are evaluated using:

- **Support:** Measures how frequently an itemset occurs.
- **Confidence:** Measures the reliability of a rule.
- **Lift:** Measures the strength of the relationship between items.

## 5. Technologies

- Python
- Pandas and NumPy
- MLxtend
- Jupyter Notebook
- Streamlit
- GitHub

## 6. Project Structure

```text
Data_Mining_Project/
└── Data_Mining_Project/
    └── THKTDL/
        ├── Data/
        │   ├── food_coded.csv
        │   ├── food_association_ready.csv
        │   ├── frequent_itemsets_apriori.csv
        │   ├── frequent_itemsets_fpgrowth.csv
        │   ├── frequent_itemsets_ittree.csv
        │   └── ...
        ├── Source_Code/
        │   ├── TienXuLyDL.py
        │   ├── Khai_thác_tập_phổ_biến_.ipynb
        │   ├── app.py
        │   └── pipeline.py
        ├── README.md
        └── requirements.txt
```

- `Data/`: Contains the original dataset and mining results.
- `Source_Code/`: Contains preprocessing, mining, application, and pipeline files.
- `requirements.txt`: Lists the required Python libraries.

## 7. Installation and Usage

### Step 1: Install dependencies

Open the project folder in VS Code and run the following command in the terminal:

```bash
python -m pip install -r requirements.txt
```

### Step 2: Run the project

```bash
python Source_Code/pipeline.py
```

The pipeline automatically:

1. Preprocesses the original dataset.
2. Executes the frequent itemset mining notebook.
3. Launches the Streamlit application after the previous steps complete successfully.

### Step 3: Open the application

Open the Local URL displayed in the terminal, usually:

`http://localhost:8501`

## 8. Outputs

- Preprocessed dataset.
- Frequent itemsets from Apriori, FP-Growth, and IT-Tree.
- Algorithm comparison results.
- Association rules and evaluation metrics.
- Interactive Streamlit interface.

Output files are stored in the `Data/` directory.

## 9. Applications

The discovered patterns help identify relationships between eating habits and food preferences. These insights may support food preference analysis and potential food combination recommendations.

10. Project Information

Project: Data Mining Project

Course: Data Mining - Mr. Dinh Nguyen Trong Nghia

Students:

Duong Tien Anh Khoa

Le Tan Tai

Nguyen Thanh Nhan

Repository: KhoaDuong69/Data_Mining_Project
- **Repository:** [KhoaDuong69/Data_Mining_Project](https://github.com/KhoaDuong69/Data_Mining_Project)
