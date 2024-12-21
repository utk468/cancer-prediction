import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import pickle


df=pd.read_csv('cancer.csv')
print(df.head())
print(df.info())


print(df.duplicated().sum())

#not much imbalance 
print(df['Level'].value_counts())



# Calculate unique values for each column and sort them
unique_values = df.nunique().sort_values()
plt.figure(figsize=(10, 6))
unique_values.plot(kind='barh', color='darkblue')  # Dark blue color for bars
plt.title('Number of Unique Values per Column')
plt.xlabel('Number of Unique Values')
plt.ylabel('Columns')
plt.tight_layout()
plt.show()



class_counts = df['Level'].value_counts()
colors = ['#00008B', '#FFA500', '#000000']
plt.bar(class_counts.index, class_counts.values, color=colors)
plt.xlabel('Class', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Distribution of Classes', fontsize=14)
plt.show()



class_counts = df['Level'].value_counts()
colors = ['#00008B', '#FFA500', '#000000']
plt.figure(figsize=(7, 7))
plt.pie(class_counts.values, labels=class_counts.index, colors=colors, autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Classes', fontsize=14)
plt.show()


# Smoking – Strongly linked to lung cancer and other types of cancers such as throat, mouth, and bladder.
# Passive Smoker – Secondhand smoke exposure is also a significant risk for lung cancer.
# Genetic Risk – Family history of cancer plays a crucial role, especially in cancers like breast, ovarian, and colon cancers.
# Alcohol Use – Linked to liver, esophageal, breast, and other cancers.
# Occupational Hazards – Exposure to carcinogens (like asbestos, benzene, etc.) in workplaces is a risk factor.
# Obesity – Associated with an increased risk of several cancers including breast, colorectal, and endometrial cancers.
# Air Pollution – Known to contribute to lung cancer risks.
# Chronic Lung Disease – Conditions like chronic obstructive pulmonary disease (COPD) can elevate lung cancer risks.



# Assuming your dataframe is named 'df'
# Strip any extra spaces from column names
df.columns = df.columns.str.strip()

# Calculate average values per class
average_values = df.groupby('Level')[['Smoking', 'Passive Smoker', 'Genetic Risk', 'Alcohol use', 'OccuPational Hazards', 'Obesity', 'Air Pollution', 'chronic Lung Disease']].mean().reset_index()

# Melt the dataframe for easier plotting
average_values_melted = average_values.melt(id_vars='Level',  # Replace 'class' with actual column name 'Level'
                                            value_vars=['Smoking', 'Passive Smoker', 'Genetic Risk', 'Alcohol use', 'OccuPational Hazards', 'Obesity', 'Air Pollution', 'chronic Lung Disease'], 
                                            var_name='Band', value_name='Average')

# Create a grouped bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Level', y='Average', hue='Band', data=average_values_melted, palette='Blues')

# Set plot title and labels
plt.title('Average of Smoking, Passive Smoker, Genetic Risk, Alcohol use, Occupational Hazards, Obesity, Air Pollution, Chronic Lung Disease for Each Class', fontsize=14)
plt.xlabel('Class', fontsize=12)
plt.ylabel('Average Value', fontsize=12)

# Show plot
plt.tight_layout()
plt.show()



#distribution of alcohol use  with respect to level of cancers
plt.figure(figsize=(10, 6))

# Plot the distribution of 'alpha' for each class using a histogram
for class_name in df['Level'].unique():
    sns.histplot(df[df['Level'] == class_name]['Alcohol use'], label=f'level {class_name}', kde=True, bins=30, alpha=0.6)

# Set the title and labels
plt.title('Distribution of Alcohol use  for Each level', fontsize=14)
plt.xlabel('Alcohol', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend(title='Class')

# Display the plot
plt.tight_layout()
plt.show()




#distribution of smoking with respect to level of cancers
plt.figure(figsize=(10, 6))

# Plot the distribution of 'alpha' for each class using a histogram
for class_name in df['Level'].unique():
    sns.histplot(df[df['Level'] == class_name]['Smoking'], label=f'level {class_name}', kde=True, bins=30, alpha=0.6)

# Set the title and labels
plt.title('Distribution of Smoking  for Each level', fontsize=14)
plt.xlabel('Smoking', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend(title='Class')

# Display the plot
plt.tight_layout()
plt.show()





#distribution of passive smoker  with respect to level of cancers
plt.figure(figsize=(10, 6))

# Plot the distribution of 'alpha' for each class using a histogram
for class_name in df['Level'].unique():
    sns.histplot(df[df['Level'] == class_name]['Passive Smoker'], label=f'level {class_name}', kde=True, bins=30, alpha=0.6)

# Set the title and labels
plt.title('Distribution of Alcohol use  for Each level', fontsize=14)
plt.xlabel('passive smoker', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend(title='Class')

# Display the plot
plt.tight_layout()
plt.show()




#distribution of genetic risk  with respect to level of cancers
plt.figure(figsize=(10, 6))

# Plot the distribution of 'Genetic Risk' for each class using a histogram
for class_name in df['Level'].unique():
    sns.histplot(df[df['Level'] == class_name]['Genetic Risk'], label=f'level {class_name}', kde=True, bins=30, alpha=0.6)

# Set the title and labels
plt.title('Distribution of genetic risk  for Each level', fontsize=14)
plt.xlabel('Genetic Risk', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend(title='Class')

# Display the plot
plt.tight_layout()
plt.show()





#distribution of Air Pollution  with respect to level of cancers
plt.figure(figsize=(10, 6))

# Plot the distribution of 'Air Pollution' for each class using a histogram
for class_name in df['Level'].unique():
    sns.histplot(df[df['Level'] == class_name]['Air Pollution'], label=f'level {class_name}', kde=True, bins=30, alpha=0.6)

# Set the title and labels
plt.title('Distribution of Air Pollution use  for Each level', fontsize=14)
plt.xlabel('Air Pollution', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend(title='Class')

# Display the plot
plt.tight_layout()
plt.show()





#distribution of chronic Lung Disease  with respect to level of cancers
plt.figure(figsize=(10, 6))

# Plot the distribution of 'chronic Lung Disease' for each class using a histogram
for class_name in df['Level'].unique():
    sns.histplot(df[df['Level'] == class_name]['chronic Lung Disease'], label=f'level {class_name}', kde=True, bins=30, alpha=0.6)

# Set the title and labels
plt.title('Distribution of chronic Lung Disease use  for Each level', fontsize=14)
plt.xlabel('chronic Lung Disease', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend(title='Class')

# Display the plot
plt.tight_layout()
plt.show()





#distribution of OccuPational Hazards  with respect to level of cancers
plt.figure(figsize=(10, 6))

# Plot the distribution of 'OccuPational Hazards' for each class using a histogram
for class_name in df['Level'].unique():
    sns.histplot(df[df['Level'] == class_name]['OccuPational Hazards'], label=f'level {class_name}', kde=True, bins=30, alpha=0.6)

# Set the title and labels
plt.title('Distribution of OccuPational Hazards use  for Each level', fontsize=14)
plt.xlabel('OccuPational Hazards', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend(title='Class')

# Display the plot
plt.tight_layout()
plt.show()





#distribution of Obesity  with respect to level of cancers
plt.figure(figsize=(10, 6))

# Plot the distribution of 'Obesity' for each class using a histogram
for class_name in df['Level'].unique():
    sns.histplot(df[df['Level'] == class_name]['Obesity'], label=f'level {class_name}', kde=True, bins=30, alpha=0.6)

# Set the title and labels
plt.title('Distribution of Obesity use  for Each level', fontsize=14)
plt.xlabel('Obesity', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend(title='Class')

# Display the plot
plt.tight_layout()
plt.show()










#scatter plot smoker and passive smoker with respect of 3 classes 
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Smoking', y='Passive Smoker', hue='Level', palette='Set2', s=70, alpha=0.8)
plt.title('Scatter Plot of Smoker vs Passive Smoker by Class', fontsize=14)
plt.xlabel('Smoking', fontsize=12)
plt.ylabel('Passive Smoker', fontsize=12)
plt.legend(title='Class')
plt.tight_layout()
plt.show()

# Drop the 'Patient Id' column
df = df.drop(columns=['Patient Id'])



# Step 1: Encode the 'class' column (without modifying original df)
df_copy = df.copy()
label_encoder = LabelEncoder()
df_copy['Level'] = label_encoder.fit_transform(df_copy['Level'])
numeric_df = df_copy.select_dtypes(include=[int, float])
corr_matrix = numeric_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', linewidths=0.5, cbar=True)
plt.title('Correlation Heatmap ', fontsize=16)
plt.tight_layout()
plt.show()


label_encoder = LabelEncoder()
df['Level'] = label_encoder.fit_transform(df['Level'])

# finding outlier
def find_outliers_iqr(dataframe):
    outlier_indices = {}

    for column in dataframe.select_dtypes(include=[np.number]).columns:
        Q1 = dataframe[column].quantile(0.25)
        Q3 = dataframe[column].quantile(0.75)
        IQR = Q3 - Q1


        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR


        outlier_indices[column] = dataframe[(dataframe[column] < lower_bound) | (dataframe[column] > upper_bound)].index.tolist()

    return outlier_indices


outliers = find_outliers_iqr(df)


for column, indices in outliers.items():
    if indices:
        print(f"\nOutliers for column '{column}':")
        print(f"  Outlier indices: {indices}")
    else:
        print(f"\nNo outliers found for column '{column}'.")


print(df.shape)        


#graph with outlier

sns.set(style="whitegrid")

melted_df = df.melt(var_name='Feature', value_name='Value', value_vars=df.select_dtypes(include=[np.number]).columns)

plt.figure(figsize=(12, 8))
sns.boxplot(x='Feature', y='Value', data=melted_df)
plt.title('Box Plots of Numeric Features')
plt.xticks(rotation=45)
plt.xlabel('Features')
plt.ylabel('Values')
plt.show()




# remove outlier
def remove_outliers_iqr(dataframe):
    cleaned_df = dataframe.copy()
    for column in cleaned_df.select_dtypes(include=[np.number]).columns:
        Q1 = cleaned_df[column].quantile(0.25)
        Q3 = cleaned_df[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR


        cleaned_df = cleaned_df[(cleaned_df[column] >= lower_bound) & (cleaned_df[column] <= upper_bound)]

    return cleaned_df


df_cleaned = remove_outliers_iqr(df)


print(f"Original dataset shape: {df.shape}")
print(f"Cleaned dataset shape: {df_cleaned.shape}")



#graph without outlier
sns.set(style="whitegrid")

melted_df = df_cleaned.melt(var_name='Feature', value_name='Value', value_vars=df.select_dtypes(include=[np.number]).columns)

plt.figure(figsize=(12, 8))
sns.boxplot(x='Feature', y='Value', data=melted_df)
plt.title('Box Plots of Numeric Features')
plt.xticks(rotation=45)
plt.xlabel('Features')
plt.ylabel('Values')
plt.show()



X = df_cleaned.drop(columns=['Level'])  
y = df_cleaned['Level']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



pipeline_rf = Pipeline([
    ('scaler', StandardScaler()),        # StandardScaler for scaling features
    ('rf_classifier', RandomForestClassifier(class_weight='balanced', random_state=42) ) # Random Forest Classifier
])

pipeline_dt = Pipeline([
    ('scaler', StandardScaler()),        # Scaling features (for consistency, even if not necessary for DT)
    ('dt_classifier', DecisionTreeClassifier(random_state=42))  # Decision Tree Classifier
])

pipeline_xgb = Pipeline([
    ('scaler', StandardScaler()),        # Scaling features (important for XGBoost)
    ('xgb_classifier', XGBClassifier(random_state=42))  # XGBoost Classifier
])

pipeline_svc = Pipeline([
    ('scaler', StandardScaler()),        # Scaling features (important for SVC)
    ('svc_classifier', SVC(random_state=42))  # Support Vector Classifier
])


pipelines = [pipeline_rf, pipeline_dt, pipeline_xgb,pipeline_svc]
pipe_dict = {0: "RandomForest", 1: "DecisionTree", 2: "XGBoost",3:"SVC"}


cv_results = []
for i, pipe in enumerate(pipelines):
    cv_score = cross_val_score(pipe, X_train, y_train, scoring="accuracy", cv=10)  # Use 'accuracy' for classification tasks
    cv_results.append(cv_score)
    print(f"{pipe_dict[i]}: {cv_score.mean()} ± {cv_score.std()}")  # Print mean and std for better evaluation



pipeline_rf.fit(X_train, y_train)


y_pred = pipeline_rf.predict(X_test)


report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
print(report)



# Save the trained model
with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(pipeline_rf, f)