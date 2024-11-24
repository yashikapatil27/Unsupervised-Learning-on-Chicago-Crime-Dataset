# **Chicago Crime Data Clustering**

This project analyzes crime data from the **Chicago Data Portal** to identify patterns and trends using unsupervised clustering techniques. By employing methods such as **K-Modes** and **Spectral Clustering**, the objective is to uncover natural clusters in the dataset and understand the key features contributing to them. The dataset originally contains millions of records across 21 categorical features, of which 100,000 data points are utilized in this project for analysis.

## Skills and Approaches

- **Machine Learning Pipeline**: Structured workflow for data preprocessing, feature selection, clustering (K-Modes and Spectral Clustering), and evaluation.
- **Python**: Data manipulation, cleaning, and feature engineering using pandas, numpy, and scikit-learn.
- **Unsupervised Learning**: K-Modes and Spectral Clustering for clustering categorical data.
- **Data Preprocessing**: One-hot encoding, entropy-based feature selection
- **Visualization**: Insights through pie charts, heatmaps, and trend plots using matplotlib and seaborn.
- **Cluster Analysis**: Evaluation using Normalized Mutual Information (NMI).

---

## **File Structure**
```bash
├── main.py # Main script to execute the entire pipeline 
├── eda.py # Exploratory Data Analysis (EDA) 
├── preprocessing.py # Data Cleaning and Transformation 
├── feature_selection.py # Feature Selection (Entropy-based) 
├── feature_engineering.py # Feature Engineering 
├── clustering.py # Clustering: K-Modes, Hierarchical, and Spectral 
├── visualization.py # Visualization of clustering results 
└── Report.pdf # Documentation
```

---

## **Overview**

### **1. Introduction**
The Chicago crime dataset contains records from 2001 to date and is used for predictive and analytical purposes. This project uniquely focuses on unsupervised learning to identify natural clusters in the data. The dataset includes categorical features such as crime types, locations, and timestamps.  

Key Highlights:
- **Dataset Size**: 100k records, 21 features  
- **Algorithms Used**: K-Modes, Spectral Clustering  
- **Evaluation Metric**: Normalized Mutual Information (NMI)  

---

### **2. Methods**

#### **2.1 Exploratory Data Analysis (EDA)**
- Conducted analysis of trends in crimes over time, by type, day, and location.
- Key findings include:
  - **Most common crime**: Theft  
  - **Most unsafe day**: Friday  
  - **Most crimes occur in**: Streets  
  - **Crime count trend**: Decreasing over time  

#### **2.2 Preprocessing**
- Handled categorical features using one-hot encoding.
- Converted numeric types to categorical where applicable.
- Removed redundant features such as `Latitude` and `Longitude`.

#### **2.3 Feature Selection**
- Entropy-based method used to retain features with lower entropy (high relevance).
- Selected features: `IUCR`, `Primary Type`, `Location Description`, `Arrest`, etc.

#### **2.4 Clustering**
1. **K-Modes Clustering**:  
   - Designed for categorical data.  
   - Used 31 clusters based on unique values in `Primary Type`.  
   - Achieved **NMI score: 0.5766**.  

2. **Spectral Clustering**:  
   - Utilized cosine similarity for one-hot encoded data.  
   - Generated 8 clusters without requiring pre-defined `k`.  
   - Achieved **NMI score: 0.7032** (better than K-Modes).  

#### **2.5 Visualization**
- Visualized clusters with pie charts for major crime types like Theft, Battery, and Narcotics.

---

### **3. Key Results**
- **Best Algorithm**: Spectral Clustering (based on NMI score).  
- **Feature Importance**: `Primary Type` had the highest correlation with cluster labels.  
- Spectral Clustering effectively identified meaningful clusters in complex, categorical datasets.  

---

## **How to Use**

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repo/chicago-crime-clustering.git
   cd chicago-crime-clustering
   ```
2. **Add the Dataset**:
   - Download the dataset from [Chicago Data Portal](https://data.cityofchicago.org/).
   - Save it as `dataset.csv` in the project directory.

3. **Run the Project**:
Execute the entire pipeline by running:
```bash
python3 main.py
```

## **Future Scope**
- Extend the project to include real-time crime data streams for dynamic clustering.
- Explore more sophisticated feature engineering techniques for better clustering performance.

## **References**
1. Chicago Data Portal: [data.cityofchicago.org](https://data.cityofchicago.org/)  
2. Huang, Z. (1997). *Clustering large data sets with mixed numeric and categorical values.*  
3. Scikit-learn Documentation: [scikit-learn.org](https://scikit-learn.org/)  
