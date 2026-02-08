# Disaster Relief Priority Prediction System

## 📌 Overview
This project is a machine learning–based decision support system designed to assist disaster management authorities in prioritizing disaster events for effective relief planning.  
It analyzes historical disaster data to predict the severity level of a disaster and supports informed allocation of emergency resources.

The system follows a modular and scalable architecture, making it suitable for real-world disaster response planning and future extensions such as AI-based decision explanations.

---

## 🎯 Project Objectives
- Design and develop a robust software system for disaster relief planning  
- Analyze historical disaster data to identify critical patterns  
- Implement machine learning models for disaster priority prediction  
- Integrate explainable AI / LLM-based decision support (future scope)  
- Ensure modular, maintainable, and scalable system architecture  
- Provide accurate and timely allocation recommendations  

---

## 🗂 Dataset
- **Source:** EM-DAT (International Disaster Database)
- **Format:** Excel (`emdat.xlsx`)
- **Key attributes used:**
  - Total deaths
  - Injuries
  - Population affected
  - Economic damage
  - Disaster year and type

---

## 🧠 Methodology

### 1. Data Preprocessing
- Selected relevant disaster attributes
- Handled missing values using domain-standard techniques
- Removed incomplete categorical records

### 2. Feature Engineering
- Log transformation to reduce skewness
- Computed a weighted **severity score** using:
  - Deaths
  - Injuries
  - Affected population
  - Economic damage
- Converted severity scores into three classes:
  - **Low**
  - **Medium**
  - **High**

### 3. Machine Learning Model
- Multi-class classification model
- Trained on engineered severity features
- Achieved **~96% accuracy** on validation data
- Model persisted using `joblib`

---

## 📊 Results
- High accuracy and balanced precision/recall across all severity classes
- Reliable prediction of disaster priority levels
- Suitable for decision-support use cases

---

