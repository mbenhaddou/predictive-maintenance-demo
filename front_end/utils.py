import pandas as pd
import random

# Load sensor data
sensors_df = pd.read_csv('../Dataset/sensors.csv')

# Load motor specifications
motors_df = pd.read_csv('../Dataset/motor_specifications.csv')

# Check if 'Motor_ID' column exists in sensors_df
if 'Motor_ID' not in sensors_df.columns:
    # If not, we'll need to create a Motor_ID column
    # For demonstration, we'll assign a unique ID to each sensor
    sensors_df['Motor_ID'] = range(1, len(sensors_df) + 1)

# Extract unique Motor IDs
motor_ids = sensors_df['Motor_ID'].unique()

# Get the list of Brands from motor_specifications.csv
brands = motors_df['Brand'].unique()

# Create a mapping from Motor_ID to a random Brand
motor_brand_mapping = {motor_id: random.choice(brands) for motor_id in motor_ids}

# Map the assigned Brands back to the sensors DataFrame
sensors_df['Brand'] = sensors_df['Motor_ID'].map(motor_brand_mapping)

# Save the updated sensors DataFrame to a new CSV file
sensors_df.to_csv('sensors_with_brands.csv', index=False)

print("New CSV file 'sensors_with_brands.csv' has been created with assigned Brands.")
