import pandas as pd
import geopandas as gpd
import json
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
from datetime import datetime

# Load the benchmark questions
with open('/home/ubuntu/upload/Pasted Content.txt', 'r') as f:
    benchmark_questions = json.load(f)

# Print the questions for reference
print("Benchmark Questions:")
for q in benchmark_questions:
    print(f"Question {q['question_id']}: {q['question']}")
    print(f"  Constraints: {q['constraints']}")
    print()

# Load the processed data
sector_3 = gpd.read_file('/home/ubuntu/site_selection_benchmark/sector_3.geojson')
vendors_sector3 = pd.read_csv('/home/ubuntu/site_selection_benchmark/vendors_sector3.csv')
customers_sector3 = pd.read_csv('/home/ubuntu/site_selection_benchmark/customers_sector3.csv')

# Print basic statistics about the loaded data
print(f"Sector 3 communities: {len(sector_3)}")
print(f"Vendors in Sector 3: {len(vendors_sector3)}")
print(f"Customers in Sector 3: {len(customers_sector3)}")
print(f"Coffee shops in Sector 3: {len(vendors_sector3[vendors_sector3['main_cuisine'] == 'coffee'])}")

# Extract unique cuisines for reference
cuisines = vendors_sector3['main_cuisine'].unique()
print(f"Unique cuisines in Sector 3: {len(cuisines)}")
print(cuisines[:20])  # Print first 20 cuisines

# Check for temporal data availability
if 'order_time' in vendors_sector3.columns:
    # Convert order_time to datetime
    vendors_sector3['order_datetime'] = pd.to_datetime(vendors_sector3['order_time'])
    vendors_sector3['hour'] = vendors_sector3['order_datetime'].dt.hour
    vendors_sector3['day_of_week'] = vendors_sector3['order_datetime'].dt.dayofweek
    
    # Check peak hours (12-2 PM)
    peak_hours = vendors_sector3[(vendors_sector3['hour'] >= 12) & (vendors_sector3['hour'] < 14)]
    print(f"Orders during peak hours (12-2 PM): {len(peak_hours)}")
    
    # Check weekends (5=Saturday, 6=Sunday)
    weekends = vendors_sector3[(vendors_sector3['day_of_week'] == 5) | (vendors_sector3['day_of_week'] == 6)]
    print(f"Orders on weekends: {len(weekends)}")

# Check for economic data availability
if 'gmv_amount_lc' in vendors_sector3.columns:
    print(f"Average order value: {vendors_sector3['gmv_amount_lc'].mean():.2f} AED")
    print(f"Max order value: {vendors_sector3['gmv_amount_lc'].max():.2f} AED")
    print(f"Min order value: {vendors_sector3['gmv_amount_lc'].min():.2f} AED")

# Save the extracted questions for further processing
with open('/home/ubuntu/site_selection_benchmark/extracted_questions.json', 'w') as f:
    json.dump(benchmark_questions, f, indent=2)

print("\nQuestions extracted and saved to extracted_questions.json")
