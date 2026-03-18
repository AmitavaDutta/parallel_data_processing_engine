import pandas as pd
import os

# 1. SETTINGS
DATA_DIR = "climate_data"
OUTPUT_FILE = "global_temperature_comparison.csv"
# 2. GET ALL FILES
# In the current directory we have a folder with name climate_data which contains csv files for each city. We will read all those files and merge them.
all_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
# 3. MERGING LOGIC
merged_df = None

for file in all_files:
    city_name = file.replace('.csv', '')
    file_path = os.path.join(DATA_DIR, file)
    
    # Read the individual city data
    df = pd.read_csv(file_path)
    
    # Rename 'Temp_C' to the City Name to create the unique column
    df = df[['Date', 'Temp_C']].rename(columns={'Temp_C': city_name})
    
    if merged_df is None:
        merged_df = df
    else:
        # Merge on Date (Outer join ensures we don't lose days if one city is missing a record)
        merged_df = pd.merge(merged_df, df, on='Date', how='outer')

# 4. CLEAN UP AND SAVE
# Sort by date so the timeline is chronological
merged_df['Date'] = pd.to_datetime(merged_df['Date'])
merged_df = merged_df.sort_values('Date')

merged_df.to_csv(OUTPUT_FILE, index=False)

print(f"Success! Merged {len(all_files)} cities into {OUTPUT_FILE}")
print(merged_df.head())