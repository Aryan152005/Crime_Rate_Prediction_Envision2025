# Crime Rate Prediction

# Round 1

[balanced_synthetic_crime_data_new.csv](.\Dataset\balanced_synthetic_crime_data_new.csv)

## Problem Statement

The objective of this project is to work with public datasets of crime records to predict crime rates in specific areas and times. The goal is to provide actionable insights to improve safety and enable efficient allocation of resources for law enforcement and community welfare. By analyzing historical crime data, we aim to identify patterns and trends that can guide predictive modeling for crime rates.

## About the Dataset

This dataset has been compiled using web scraping and manual data collection from the [National Crime Records Bureau (NCRB)](https://www.ncrb.gov.in/crime-in-india-year-wise.html?year=2022&keyword=). It represents crime statistics across 19 cities in India from 1996 to 2024.

### Dataset Details

The dataset includes 513 rows, with exactly 27 records for each city. Each row represents a year of crime data for a specific city. The following columns are included:

### Columns Description

1. **Year**:

   - Type: Integer
   - Range: 1996 to 2024
   - Description: The year for which the crime data is recorded.

2. **City**:

   - Type: Categorical
   - Description: The name of the city for which the data is collected. The dataset includes 19 cities.

3. **Population (in Lakhs) (2011)+**:

   - Type: Integer
   - Description: The population of the city as per the 2011 census. This column has been mapped consistently for each city.

4. **Murder**:

   - Type: Integer
   - Description: The total number of murders recorded in the respective year and city.

5. **Kidnapping**:

   - Type: Integer
   - Description: The total number of kidnapping cases recorded.

6. **Crime against women**:

   - Type: Integer
   - Description: The total number of crimes committed against women.

7. **Crime against children**:

   - Type: Integer
   - Description: The total number of crimes committed against children.

8. **Crime Committed by Juveniles**:

   - Type: Integer
   - Description: The total number of crimes committed by juveniles.

9. **Crime against Senior Citizen**:

   - Type: Integer
   - Description: The total number of crimes committed against senior citizens.

10. **Crime against SC**:

    - Type: Integer
    - Description: The total number of crimes committed against Scheduled Castes (SC).

11. **Crime against ST**:

    - Type: Integer
    - Description: The total number of crimes committed against Scheduled Tribes (ST).

12. **Economic Offences**:

    - Type: Integer
    - Description: The total number of economic offences recorded.

13. **Cyber Crimes**:
    - Type: Integer
    - Description: The total number of cybercrimes recorded.

### Dataset Generation Details

- **Source**: Data was obtained from NCRB’s official reports and manually curated to ensure accuracy and completeness.
- **Synthetic Data Expansion**: To align with the project’s requirements of 513 rows, the dataset was expanded using synthetic data generation techniques. Key considerations included:
  - Balancing the dataset with an equal number of records (27) for each city.
  - Ensuring unique years for each city within the range of 1996 to 2024.
  - Preserving the statistical characteristics of the original data, such as mean and standard deviation.

### Data Quality Assurance

- **Validation**: Numerical columns were ensured to have non-negative integers, consistent with real-world crime data.
- **Consistency**: Population values were fixed for each city based on the 2011 census data.
- **Sorting**: The dataset is sorted by city and year to maintain clarity and facilitate analysis.

## Applications of the Dataset

1. **Crime Rate Prediction**:

   - Predict the likelihood of various crimes occurring in specific areas and years.
   - Analyze temporal trends and city-wise patterns.

2. **Resource Allocation**:

   - Help authorities allocate resources like police personnel and surveillance systems more effectively.

3. **Safety Measures**:

   - Provide actionable insights for local governments to implement targeted safety measures.

4. **Public Awareness**:
   - Educate communities about crime trends and promote preventive actions.

## Event Details

### Description

Envision is a team-based competition where participants solve real-world challenges using machine learning and data science. The event includes stages like data collection, preprocessing, model building, and solution presentation, focusing on innovation, technical skills, and collaboration.

---

---

---

# Round 2

[formated_new_dataset.csv](.\Dataset\formated_new_dataset.csv)

# Code Workflow

## 1. Data Loading

### Libraries Used

- **pandas**: For data manipulation and analysis.
- **matplotlib**: For data visualization.
- **sklearn.preprocessing.LabelEncoder**: To encode categorical variables into numerical format.

### Objective

The first step involves loading the dataset to inspect its structure, verify data integrity, and ensure that all required columns are present for analysis. The dataset contains crime statistics for 19 cities across multiple years (1996-2024).

### Insights

- The dataset consists of **513 rows** with complete and non-null values.
- Columns include city names, years, population figures, and counts for various crime categories.
- Population figures are consistent with the 2011 census data, ensuring reliability.
- Each city has 27 records corresponding to the years under consideration.

## 2. Visualization

### Purpose

The purpose of visualizations is to understand the distribution and trends of various crime types across different cities. This step provides insights into city-wise crime statistics and helps identify patterns for further analysis.

### Method

Horizontal bar plots are created for each crime category. Each graph compares the number of crimes reported for all 19 cities in the dataset, providing a clear visual representation.

#### Graph Details

Some graphs are provided below

1. **City vs Murder Cases**

   - **Column Used**: `Murder`
   - **Meaning**: Represents the number of murders reported in each city.
   - **Labels**:
     - **X-axis**: Number of murder cases.
     - **Y-axis**: City names.
   - **Insight**: Cities with high murder rates can be identified easily.

   ![City vs Murder Cases](.\Images\CityvsMurderCases.png)

2. **City vs Kidnapping Cases**

   - **Column Used**: `Kidnapping`
   - **Meaning**: Number of reported kidnappings.
   - **Labels**:
     - **X-axis**: Number of kidnapping cases.
     - **Y-axis**: City names.
   - **Insight**: Highlights cities where kidnapping cases are most prevalent.

   ![City vs Kidnapping Cases](Images\CityvsKidnappingCases.png)

3. **City vs Crime Against Women**

   - **Column Used**: `Crime against women`
   - **Meaning**: Total crimes reported against women.
   - **Labels**:
     - **X-axis**: Number of cases.
     - **Y-axis**: City names.
   - **Insight**: Identifies cities with high rates of gender-based crimes.

   ![City vs Crime Against Women](Images\CityvsCrimeAgainstWomen.png)

4. **City vs Economic Offenses**

   - **Column Used**: `Economic Offences`
   - **Meaning**: Cases of financial frauds and related crimes.
   - **Labels**:
     - **X-axis**: Number of economic offenses.
     - **Y-axis**: City names.
   - **Insight**: Helps pinpoint urban areas with high financial crime activity.

   ![City vs Economic Offenses](Images\CityvsEconomicOffenses.png)
   
5. **City vs Cyber Crimes**
   - **Column Used**: `Cyber Crimes`
   - **Meaning**: Cybersecurity-related incidents.
   - **Labels**:
     - **X-axis**: Number of cybercrimes.
     - **Y-axis**: City names.
   - **Insight**: Shows cities with significant digital crime rates.

   ![City vs Cyber Crimes](Images\CityvsCyberCrimes.png)

#### PowerBI Visualization
   
   ![Power](Images\Envision_CRPDashboard_page-0001.jpg)

  1. **Tree Map**: Crime Against Women, Children, SC, and Senior Citizens

   A Tree Map is used to visualize the distribution of various types of crimes (e.g., against women, children, Scheduled Castes (SC), and senior citizens) across different categories or regions. Each segment represents a specific crime type, with the size of the block indicating the volume or frequency of incidents.
2. **World Map**: Population by City

   A World Map visualizes population data for cities around the globe. Each city is marked on the map, with bubble sizes or color intensities indicating population density or size.
3. **Donut Chart**: Year by City

   A Donut Chart displays the proportion of data for different years within specific cities. Each segment represents a year, with cities as categories in a hierarchical or grouped view.
4. **Stacked Bar Chart**: Cybercrimes by City

   A Stacked Bar Chart showcases the volume of cybercrimes in various cities, with different segments representing specific types of cybercrimes or demographic groups involved. Bars are stacked to show the total, with clear differentiation between subcategories.


   <video width="480" height="256" controls>
   <source src="Images\PowerBi.mp4" type="video/mp4">
   </video>

#### General Observations

- Cities with large populations tend to report higher numbers across all crime types.
- Cybercrime cases have shown an increasing trend in metropolitan cities.



## 3. Data Transformation

### Steps

1. **Reshape the Dataset**:

   - **Objective**: Organize the data into a long format, where each row represents a specific crime type for a city in a given year.
   - **Columns Added**:
     - `Type`: Indicates the crime category (e.g., Murder, Kidnapping).
     - `Number of Cases`: The raw count of reported incidents for the respective type.

2. **Compute Normalized Crime Rates**:
   - **Objective**: Calculate crime rates per lakh of population to provide a normalized comparison.
   - **Formula Used**:
     ```
     Crime Rate = Number of Cases / Population (in Lakhs)
     ```
   - **Reasoning**: Normalizing by population accounts for size disparities among cities and enables fair comparisons.

### Output

The transformed dataset includes the following columns:

- `Year`: Year of observation.
- `City`: Name of the city.
- `Population (in Lakhs)`: Census population data for normalization.
- `Type`: Type of crime.
- `Crime Rate`: Normalized crime rate.

## 4. Encoding

### Goal

Convert categorical data into numerical values to make the dataset compatible with machine learning algorithms.

### Implementation

1. **City Encoding**:

   - Used `LabelEncoder` to convert city names into numeric labels.
   - Saved the mappings in `City_Mapping.txt` for reference.
   - Example Mapping:
     - Ahmedabad: 0
     - Surat: 18

2. **Crime Type Encoding**:
   - Converted crime categories into numerical codes.
   - Saved the mappings in `Type_Mapping.txt` for transparency.
   - Example Mapping:
     - Murder: 9
     - Cyber Crimes: 6

### Importance

- Encoding ensures that categorical data can be fed into predictive models without loss of information.
- Saved mappings allow easy interpretation of model outputs.

### Summary

This workflow transforms the raw dataset into an analytical and machine-learning-ready format. It includes essential preprocessing steps, such as visualization, normalization, and encoding, to enhance the dataset's usability and insights.

## Further steps

### **Model Planning**

The modeling phase involves preparing the data for machine learning and implementing predictive algorithms to estimate crime rates. Here are the steps in-depth:


### **1. Dataset Splitting**

- **Why**: To evaluate model performance effectively, the dataset is split into training and testing subsets. The training data is used to build the model, while the testing data assesses its predictive accuracy.
  
- **Method**:
  - Features (`X`): Includes `Year`, `City`, `Population (in Lakhs)`, and `Type` (encoded).
  - Target (`y`): The `Crime Rate` column.
  - **Train-Test Split**: `80:20` ratio using `train_test_split` from `sklearn`.

- **Code**:
  ```python
  x = new_dataset[new_dataset.columns[0:4]].values  # Selecting Year, City, Population, and Type
  y = new_dataset['Crime Rate'].values             # Selecting the target variable

  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
  ```

- **Insights**:
  - Ensures robust model evaluation by testing on unseen data.
  - Prevents overfitting by exposing the model to diverse data during training.


### **2. Models to Implement**

Multiple machine learning models are considered to predict the crime rate. Each model is evaluated based on metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R² score.

#### **Support Vector Machine (SVM)**

- **Why**: SVM is useful for capturing non-linear relationships between features and the target variable.
- **Evaluation**:
  - MAE
  - MSE
  - R²
  
- **Insight**: SVM might not be ideal for this dataset due to the complexity of relationships between features.

---

#### **K-Nearest Neighbors (KNN)**

- **Why**: KNN predicts crime rates based on the nearest data points in the feature space.
- **Evaluation**:
  - MAE
  - MSE
  - R²
  
- **Insight**: Performs better than SVM, but struggles with larger variations in the dataset.

---

#### **Decision Tree Regressor**

- **Why**: Decision trees handle non-linear relationships and feature interactions effectively.
- **Evaluation**:
  - MAE
  - MSE
  - R²
  
- **Insight**: Provides a significant improvement, suggesting strong potential for predictive accuracy.

---

#### **Random Forest Regressor**

- **Why**: Combines multiple decision trees to improve accuracy and reduce overfitting.
- **Evaluation**:
  - MAE
  - MSE
  - R²
  
- **Insight**: May Outperform other models, indicating robustness in predicting crime rates.

---

#### **Neural Network (MLP Regressor)**

- **Why**: Captures complex patterns through layers of neurons.
- **Evaluation**:
  - MAE
  - MSE
  - R²
  
- **Insight**:  May Perform poorly, likely due to limited data or insufficient tuning.

---

### **3. Best Model Selection**

- **Random Forest Regressor** is selected as the best-performing model based on its:
  - High R² score.
  - Low error metrics (MAE and MSE).

---

### **4. Model Saving**

- **Why**: Save the trained model for future use and deployment.
- **Method**: Serialize the model using Python’s `pickle` library.
- **Code**:
  ```python
  import pickle
  with open("Model/model.pkl", 'wb') as file:
      pickle.dump(model4, file)
  ```

- **Insight**: Ensures that the trained model can be reused without retraining.

---

### **Next Steps**

1. **Hyperparameter Tuning**:
   - Use Grid Search or Random Search to fine-tune the Random Forest Regressor for optimal performance.

2. **Feature Engineering**:
   - Explore new features, such as socioeconomic factors, to enhance model accuracy.

3. **Scenario Simulations**:
   - Test the model under various population growth and crime trend scenarios.

4. **Visualization and Reporting**:
   - Generate detailed reports and dashboards to present insights to stakeholders.

This detailed model planning sets the foundation for accurate and actionable crime rate predictions.


---
---
---

## Team Members

1. **Arya Hotey**:

   - **GitHub**: [Arya Hotey GitHub](https://github.com/Arya202004)
   - **LinkedIn**: [Arya Hotey LinkedIn](https://in.linkedin.com/in/arya-hotey-aab5b32a7)

2. **Aryan Paratakke**:

   - **GitHub**: [Aryan Paratakke GitHub](https://github.com/Aryan152005/)
   - **LinkedIn**: [Aryan Paratakke LinkedIn](https://in.linkedin.com/in/aryan-paratakke-43b879276)

3. **Nishtha Kashyap**:
   - **GitHub**: [Nishtha Kashyap GitHub](https://github.com/nishtha932005)
   - **LinkedIn**: [Nishtha Kashyap LinkedIn](https://in.linkedin.com/in/nishtha-kashyap-0b6846293)

## Acknowledgment

This dataset has been meticulously prepared using publicly available information from NCRB. Any synthetic data generated is for research purposes and should not be treated as official statistics.

For more information about the data source, visit the [NCRB official website](https://www.ncrb.gov.in/crime-in-india-year-wise.html?year=2022&keyword=).
