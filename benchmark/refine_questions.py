import pandas as pd
import geopandas as gpd
import json
from shapely.geometry import Point, Polygon, shape
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from shapely.ops import transform
import pyproj
from functools import partial

# Load the benchmark questions
with open('/home/ubuntu/upload/Pasted Content.txt', 'r') as f:
    benchmark_questions = json.load(f)

# Load the processed data
sector_3 = gpd.read_file('/home/ubuntu/site_selection_benchmark/sector_3.geojson')
vendors_df = pd.read_csv('/home/ubuntu/site_selection_benchmark/vendors_sector3.csv')
customers_df = pd.read_csv('/home/ubuntu/site_selection_benchmark/customers_sector3.csv')

# Convert string representation of geometry back to geometry objects
def wkt_to_geometry(wkt_str):
    if pd.isna(wkt_str):
        return None
    try:
        from shapely import wkt
        return wkt.loads(wkt_str)
    except:
        return None

# Convert vendor and customer locations to geometry
vendors_df['vendor_geometry'] = vendors_df['vendor_geometry'].apply(wkt_to_geometry)
customers_df['customer_geometry'] = customers_df['customer_geometry'].apply(wkt_to_geometry)

# Create GeoDataFrames
vendors_gdf = gpd.GeoDataFrame(vendors_df, geometry='vendor_geometry', crs="EPSG:4326")
customers_gdf = gpd.GeoDataFrame(customers_df, geometry='customer_geometry', crs="EPSG:4326")

# Function to create buffer in kilometers
def create_buffer_km(point, km_radius):
    # Create a projection transformer from WGS84 to a local UTM projection
    project = partial(
        pyproj.transform,
        pyproj.Proj('EPSG:4326'),  # source coordinate system (WGS84)
        pyproj.Proj(proj='utm', zone=42, datum='WGS84')  # UTM zone for Dubai
    )
    
    # Transform point to UTM
    point_utm = transform(project, point)
    
    # Create buffer in meters
    buffer_utm = point_utm.buffer(km_radius * 1000)
    
    # Transform back to WGS84
    project_inverse = partial(
        pyproj.transform,
        pyproj.Proj(proj='utm', zone=42, datum='WGS84'),  # UTM zone for Dubai
        pyproj.Proj('EPSG:4326')  # target coordinate system (WGS84)
    )
    
    buffer_wgs84 = transform(project_inverse, buffer_utm)
    return buffer_wgs84

# Refined questions with data availability assessment
refined_questions = []

for q in benchmark_questions:
    question_id = q['question_id']
    question_text = q['question']
    constraints = q['constraints']
    
    # Create a refined question object
    refined_q = {
        'question_id': question_id,
        'original_question': question_text,
        'constraints': constraints,
        'data_availability': {},
        'refinement_notes': [],
        'refined_question': question_text,
        'calculation_approach': []
    }
    
    # Check data availability for each constraint type
    if question_id == 1:
        # Coffee shops and customer radius analysis
        refined_q['data_availability']['coffee_shops'] = True
        refined_q['data_availability']['customer_locations'] = True
        refined_q['data_availability']['radius_analysis'] = True
        refined_q['refinement_notes'].append("Data available for coffee shop locations and customer locations")
        refined_q['refinement_notes'].append("Can perform radius analysis using buffer operations")
        refined_q['calculation_approach'].append("1. Create a grid of potential sites across Sector 3")
        refined_q['calculation_approach'].append("2. For each site, create 0.7km and 1km buffers")
        refined_q['calculation_approach'].append("3. Count coffee shops within 0.7km buffer")
        refined_q['calculation_approach'].append("4. Count unique customers within 1km buffer")
        refined_q['calculation_approach'].append("5. Filter sites meeting both criteria")
    
    elif question_id == 2:
        # Daily order volume during peak hours
        if 'order_time' in vendors_df.columns:
            refined_q['data_availability']['temporal_data'] = True
            refined_q['data_availability']['peak_hours'] = True
            refined_q['refinement_notes'].append("Temporal data available for peak hour analysis")
            refined_q['calculation_approach'].append("1. Create a grid of potential sites across Sector 3")
            refined_q['calculation_approach'].append("2. Filter orders to peak hours (12-2 PM)")
            refined_q['calculation_approach'].append("3. Group by date to get daily volumes")
            refined_q['calculation_approach'].append("4. Identify sites with ≥150 daily orders during peak hours")
        else:
            refined_q['data_availability']['temporal_data'] = False
            refined_q['refinement_notes'].append("Limited temporal data for peak hour analysis")
            refined_q['refined_question'] = "Identify sites in Dubai Sector 3 that achieve a high order volume based on Talabat sample data."
    
    elif question_id == 3:
        # Average order value and daily GMV
        if 'gmv_amount_lc' in vendors_df.columns:
            refined_q['data_availability']['economic_data'] = True
            refined_q['refinement_notes'].append("Economic data available for order value and GMV analysis")
            refined_q['calculation_approach'].append("1. Create a grid of potential sites across Sector 3")
            refined_q['calculation_approach'].append("2. Calculate average order value for each site")
            refined_q['calculation_approach'].append("3. Calculate daily GMV for each site")
            refined_q['calculation_approach'].append("4. Filter sites with avg order value > AED 50 and daily GMV > AED 10,000")
        else:
            refined_q['data_availability']['economic_data'] = False
            refined_q['refinement_notes'].append("Limited economic data for GMV analysis")
    
    elif question_id == 4:
        # Competing restaurants and unique customers
        refined_q['data_availability']['restaurant_data'] = True
        refined_q['data_availability']['customer_data'] = True
        refined_q['refinement_notes'].append("Data available for restaurant competition and customer analysis")
        refined_q['calculation_approach'].append("1. Create a grid of potential sites across Sector 3")
        refined_q['calculation_approach'].append("2. For each site, create 1km buffer")
        refined_q['calculation_approach'].append("3. Count competing restaurants within buffer")
        refined_q['calculation_approach'].append("4. Count unique customers within buffer")
        refined_q['calculation_approach'].append("5. Filter sites with <2 competing restaurants and >200 unique customers")
    
    elif question_id == 5:
        # Weekend orders and customer satisfaction
        if 'order_time' in vendors_df.columns:
            refined_q['data_availability']['weekend_data'] = True
            refined_q['refinement_notes'].append("Temporal data available for weekend analysis")
            
            if 'customer_satisfaction' not in vendors_df.columns:
                refined_q['data_availability']['satisfaction_data'] = False
                refined_q['refinement_notes'].append("Customer satisfaction score not directly available")
                refined_q['refined_question'] = "Find sites in Dubai Sector 3 that record at least 80 orders per day on weekends (Saturday & Sunday) as per Talabat sample data."
                refined_q['calculation_approach'].append("1. Create a grid of potential sites across Sector 3")
                refined_q['calculation_approach'].append("2. Filter orders to weekends (Saturday & Sunday)")
                refined_q['calculation_approach'].append("3. Group by date to get daily weekend order volumes")
                refined_q['calculation_approach'].append("4. Identify sites with ≥80 daily orders on weekends")
            else:
                refined_q['data_availability']['satisfaction_data'] = True
        else:
            refined_q['data_availability']['weekend_data'] = False
            refined_q['refinement_notes'].append("Limited temporal data for weekend analysis")
    
    elif question_id == 6:
        # Customer demographics and repeat rate
        if 'customer_age' not in customers_df.columns:
            refined_q['data_availability']['demographic_data'] = False
            refined_q['refinement_notes'].append("Customer age data not available")
            
            # Check if we can calculate repeat customer rate
            if 'account_id' in customers_df.columns:
                refined_q['data_availability']['repeat_customer_data'] = True
                refined_q['refinement_notes'].append("Can calculate repeat customer rate using account_id")
                refined_q['refined_question'] = "Select sites in Dubai Sector 3 with a repeat customer rate of over 30%, derived from Talabat sample data."
                refined_q['calculation_approach'].append("1. Create a grid of potential sites across Sector 3")
                refined_q['calculation_approach'].append("2. For each site, calculate repeat customer rate using account_id")
                refined_q['calculation_approach'].append("3. Filter sites with repeat customer rate >30%")
            else:
                refined_q['data_availability']['repeat_customer_data'] = False
                refined_q['refinement_notes'].append("Limited data for repeat customer analysis")
        else:
            refined_q['data_availability']['demographic_data'] = True
    
    elif question_id == 7:
        # Transport hubs and ROI
        refined_q['data_availability']['transport_hub_data'] = False
        refined_q['refinement_notes'].append("Transport hub location data not available")
        
        if 'gmv_amount_lc' in vendors_df.columns:
            refined_q['data_availability']['economic_data'] = True
            refined_q['refinement_notes'].append("Economic data available but insufficient for ROI calculation")
            refined_q['refined_question'] = "Identify sites in Dubai Sector 3 that have high economic potential based on Talabat order data."
            refined_q['calculation_approach'].append("1. Create a grid of potential sites across Sector 3")
            refined_q['calculation_approach'].append("2. Calculate average GMV for each site")
            refined_q['calculation_approach'].append("3. Identify sites with highest economic potential")
        else:
            refined_q['data_availability']['economic_data'] = False
    
    elif question_id == 8:
        # High-demand periods
        if 'order_time' in vendors_df.columns:
            refined_q['data_availability']['temporal_data'] = True
            refined_q['refinement_notes'].append("Temporal data available for demand pattern analysis")
            refined_q['calculation_approach'].append("1. Create a grid of potential sites across Sector 3")
            refined_q['calculation_approach'].append("2. Analyze hourly order patterns for each site")
            refined_q['calculation_approach'].append("3. Identify sites with ≥3 distinct peak periods per day")
            refined_q['calculation_approach'].append("4. Perform spatial clustering of high-demand sites")
        else:
            refined_q['data_availability']['temporal_data'] = False
            refined_q['refinement_notes'].append("Limited temporal data for demand pattern analysis")
    
    elif question_id == 9:
        # Ramadan operations
        refined_q['data_availability']['ramadan_data'] = False
        refined_q['refinement_notes'].append("Specific Ramadan period data not identifiable")
        refined_q['refined_question'] = "Select sites in Dubai Sector 3 that operate with extended evening hours and record at least 100 orders per day, according to Talabat data."
        refined_q['calculation_approach'].append("1. Create a grid of potential sites across Sector 3")
        refined_q['calculation_approach'].append("2. Analyze evening hour operations (after 6 PM)")
        refined_q['calculation_approach'].append("3. Calculate daily order volumes")
        refined_q['calculation_approach'].append("4. Identify sites with extended evening operations and ≥100 daily orders")
    
    elif question_id == 10:
        # Order type distribution
        if 'order_type' not in vendors_df.columns:
            refined_q['data_availability']['order_type_data'] = False
            refined_q['refinement_notes'].append("Order type distribution data not directly available")
            
            if 'gmv_amount_lc' in vendors_df.columns:
                refined_q['data_availability']['economic_data'] = True
                refined_q['refinement_notes'].append("Economic data available for order value analysis")
                refined_q['refined_question'] = "Determine sites in Dubai Sector 3 that have an average order value of at least AED 45, as per Talabat sample data."
                refined_q['calculation_approach'].append("1. Create a grid of potential sites across Sector 3")
                refined_q['calculation_approach'].append("2. Calculate average order value for each site")
                refined_q['calculation_approach'].append("3. Filter sites with average order value ≥ AED 45")
            else:
                refined_q['data_availability']['economic_data'] = False
        else:
            refined_q['data_availability']['order_type_data'] = True
    
    refined_questions.append(refined_q)

# Save the refined questions
with open('/home/ubuntu/site_selection_benchmark/refined_questions.json', 'w') as f:
    json.dump(refined_questions, f, indent=2)

# Print summary of refinements
print("Question Refinement Summary:")
for q in refined_questions:
    print(f"Question {q['question_id']}:")
    print(f"  Original: {q['original_question']}")
    if q['original_question'] != q['refined_question']:
        print(f"  Refined: {q['refined_question']}")
    print(f"  Data availability: {', '.join([k for k, v in q['data_availability'].items() if v])}")
    if len([k for k, v in q['data_availability'].items() if not v]) > 0:
        print(f"  Missing data: {', '.join([k for k, v in q['data_availability'].items() if not v])}")
    print()

print("Refined questions saved to refined_questions.json")
