# Importing necessary libraries
import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, classification_report

# Load the dataset
df_smart_grid = pd.read_csv('C:\Users\heyam\OneDrive\Desktop\data science project\smart_grid_data.csv')

# Convert 'Timestamp' to datetime format
df_smart_grid['Timestamp'] = pd.to_datetime(df_smart_grid['Timestamp'])

# Create new time-based features
df_smart_grid['Hour'] = df_smart_grid['Timestamp'].dt.hour
df_smart_grid['Day'] = df_smart_grid['Timestamp'].dt.day
df_smart_grid['Day_of_Week'] = df_smart_grid['Timestamp'].dt.dayofweek  # Monday=0, Sunday=6

# Create interaction terms between weather features and generation data
df_smart_grid['Temp_Wind_Interaction'] = df_smart_grid['Temperature (C)'] * df_smart_grid['Wind Speed (m/s)']
df_smart_grid['Solar_Wind_Generation'] = df_smart_grid['Solar Generation (kW)'] + df_smart_grid['Wind Generation (kW)']

# Create lag features for energy consumption (this helps with forecasting)
df_smart_grid['Lag_1_Energy_Consumption'] = df_smart_grid['Energy Consumption (kWh)'].shift(1)
df_smart_grid['Lag_24_Energy_Consumption'] = df_smart_grid['Energy Consumption (kWh)'].shift(24)  # Energy usage 1 day ago

# Drop the first rows with NaN values due to lag features
df_smart_grid = df_smart_grid.dropna()

# Data Exploration: Summary statistics of the dataset
print(df_smart_grid.describe())

# Visualizing a few key metrics - Energy Consumption, Solar Generation, Wind Generation, and Grid Faults
plt.figure(figsize=(12, 8))

# Plot 1: Energy Consumption over Time
plt.subplot(3, 1, 1)
plt.plot(df_smart_grid['Timestamp'], df_smart_grid['Energy Consumption (kWh)'], label="Energy Consumption", color='b')
plt.title('Energy Consumption Over Time')
plt.xlabel('Time')
plt.ylabel('kWh')

# Plot 2: Solar and Wind Generation over Time
plt.subplot(3, 1, 2)
plt.plot(df_smart_grid['Timestamp'], df_smart_grid['Solar Generation (kW)'], label="Solar Generation", color='orange')
plt.plot(df_smart_grid['Timestamp'], df_smart_grid['Wind Generation (kW)'], label="Wind Generation", color='green')
plt.title('Solar and Wind Generation Over Time')
plt.xlabel('Time')
plt.ylabel('kW')
plt.legend()

# Plot 3: Grid Faults over Time
plt.subplot(3, 1, 3)
plt.plot(df_smart_grid['Timestamp'], df_smart_grid['Grid Fault (1=Fault)'], label="Grid Fault", color='r')
plt.title('Grid Fault Events Over Time')
plt.xlabel('Time')
plt.ylabel('Fault (1=True)')
plt.tight_layout()
plt.show()

# ------------------ Load Forecasting using Decision Tree Regressor ------------------ #

# Define target and features for load forecasting
target = df_smart_grid['Energy Consumption (kWh)']
features = df_smart_grid[['Hour', 'Day_of_Week', 'Temperature (C)', 'Wind Speed (m/s)', 
                          'Solar Radiation (W/m^2)', 'Lag_1_Energy_Consumption', 'Lag_24_Energy_Consumption']]

# Split into training and test data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize and train the Decision Tree model
dt_regressor = DecisionTreeRegressor(random_state=42)
dt_regressor.fit(X_train, y_train)

# Predict energy consumption on the test data
y_pred = dt_regressor.predict(X_test)

# Calculate error (Mean Absolute Error)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error for Load Forecasting: {mae}")

# ------------------ Fault Detection using Random Forest Classifier ------------------ #

# Define target (Grid Fault) and features for fault detection
target_fault = df_smart_grid['Grid Fault (1=Fault)']
features_fault = df_smart_grid[['Hour', 'Day_of_Week', 'Temperature (C)', 'Wind Speed (m/s)', 
                                'Solar Generation (kW)', 'Wind Generation (kW)', 'Energy Consumption (kWh)']]

# Split the data into training and test sets
X_train_fault, X_test_fault, y_train_fault, y_test_fault = train_test_split(features_fault, target_fault, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train_fault, y_train_fault)

# Predict on test data
y_pred_fault = rf_classifier.predict(X_test_fault)

# Evaluate the model
print("Classification Report for Fault Detection:")
print(classification_report(y_test_fault, y_pred_fault))
