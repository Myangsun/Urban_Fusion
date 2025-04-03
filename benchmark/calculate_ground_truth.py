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
import folium
from folium.plugins import HeatMap
import os

# Load the refined questions
with open('/home/ubuntu/site_selection_benchmark/refined_questions.json', 'r') as f:
    refined_questions = json.load(f)

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

# Convert order_time to datetime if it exists
if 'order_time' in vendors_gdf.columns:
    vendors_gdf['order_datetime'] = pd.to_datetime(vendors_gdf['order_time'])
    vendors_gdf['hour'] = vendors_gdf['order_datetime'].dt.hour
    vendors_gdf['day_of_week'] = vendors_gdf['order_datetime'].dt.dayofweek
    vendors_gdf['date'] = vendors_gdf['order_datetime'].dt.date

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

# Create a grid of potential sites across Sector 3
def create_site_grid(sector_gdf, grid_size_km=0.5):
    # Get the bounds of the sector
    bounds = sector_gdf.total_bounds
    
    # Create a grid of points
    x_min, y_min, x_max, y_max = bounds
    
    # Convert grid size from km to degrees (approximate)
    # 1 degree of latitude is approximately 111 km
    # 1 degree of longitude varies with latitude, at Dubai's latitude (~25°N) it's about 101 km
    grid_size_lat = grid_size_km / 111.0
    grid_size_lon = grid_size_km / 101.0
    
    # Create grid points
    x_coords = np.arange(x_min, x_max, grid_size_lon)
    y_coords = np.arange(y_min, y_max, grid_size_lat)
    
    points = []
    for x in x_coords:
        for y in y_coords:
            points.append(Point(x, y))
    
    # Create GeoDataFrame
    grid_gdf = gpd.GeoDataFrame(geometry=points, crs="EPSG:4326")
    
    # Filter points to only those within the sector
    grid_in_sector = gpd.sjoin(grid_gdf, sector_gdf, predicate='within')
    
    return grid_in_sector

# Create a directory for results
results_dir = '/home/ubuntu/site_selection_benchmark/results'
os.makedirs(results_dir, exist_ok=True)

# Create a grid of potential sites
print("Creating grid of potential sites across Sector 3...")
site_grid = create_site_grid(sector_3)
print(f"Created {len(site_grid)} potential sites")

# Dictionary to store ground truth answers
ground_truth_answers = {}

# Process each question
for question in refined_questions:
    question_id = question['question_id']
    refined_question = question['refined_question']
    print(f"\nProcessing Question {question_id}: {refined_question}")
    
    # Create a directory for this question's results
    question_dir = os.path.join(results_dir, f"question_{question_id}")
    os.makedirs(question_dir, exist_ok=True)
    
    # Initialize answer object
    answer = {
        'question_id': question_id,
        'refined_question': refined_question,
        'sites_meeting_criteria': [],
        'analysis_summary': {},
        'visualization_files': []
    }
    
    # Question 1: Coffee shops and customer radius
    if question_id == 1:
        print("Analyzing coffee shops and customer density...")
        
        # Identify coffee shops
        coffee_shops = vendors_gdf[vendors_gdf['main_cuisine'] == 'coffee']
        
        # Results for each site
        site_results = []
        
        # Process each potential site
        for idx, site in site_grid.iterrows():
            site_point = site.geometry
            
            # Create buffers
            buffer_0_7km = create_buffer_km(site_point, 0.7)
            buffer_1km = create_buffer_km(site_point, 1.0)
            
            # Count coffee shops within 0.7km
            coffee_shops_within = sum(coffee_shops.geometry.intersects(buffer_0_7km))
            
            # Count unique customers within 1km
            customers_within = customers_gdf[customers_gdf.geometry.intersects(buffer_1km)]
            unique_customers = customers_within['account_id'].nunique()
            
            # Store results
            site_results.append({
                'site_id': idx,
                'longitude': site_point.x,
                'latitude': site_point.y,
                'coffee_shops_within_0_7km': coffee_shops_within,
                'unique_customers_within_1km': unique_customers,
                'meets_criteria': coffee_shops_within >= 4 and unique_customers > 100
            })
        
        # Convert to DataFrame
        results_df = pd.DataFrame(site_results)
        
        # Filter sites meeting criteria
        sites_meeting_criteria = results_df[results_df['meets_criteria']]
        
        # Save results
        results_df.to_csv(os.path.join(question_dir, 'site_analysis.csv'), index=False)
        sites_meeting_criteria.to_csv(os.path.join(question_dir, 'sites_meeting_criteria.csv'), index=False)
        
        # Create visualization
        m = folium.Map(location=[25.24, 55.30], zoom_start=12)
        
        # Add Sector 3 boundary
        for idx, row in sector_3.iterrows():
            folium.GeoJson(row['geometry'], 
                          style_function=lambda x: {'fillColor': 'blue', 'color': 'blue', 'fillOpacity': 0.1}).add_to(m)
        
        # Add sites meeting criteria
        for idx, row in sites_meeting_criteria.iterrows():
            popup_text = f"Site ID: {row['site_id']}<br>Coffee shops: {row['coffee_shops_within_0_7km']}<br>Unique customers: {row['unique_customers_within_1km']}"
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=popup_text,
                icon=folium.Icon(color='green', icon='check', prefix='fa')
            ).add_to(m)
        
        # Save map
        map_file = os.path.join(question_dir, 'sites_map.html')
        m.save(map_file)
        
        # Update answer
        answer['sites_meeting_criteria'] = sites_meeting_criteria[['site_id', 'longitude', 'latitude']].to_dict('records')
        answer['analysis_summary'] = {
            'total_sites_analyzed': len(results_df),
            'sites_meeting_criteria': len(sites_meeting_criteria),
            'avg_coffee_shops_per_site': results_df['coffee_shops_within_0_7km'].mean(),
            'avg_unique_customers_per_site': results_df['unique_customers_within_1km'].mean()
        }
        answer['visualization_files'] = [map_file]
    
    # Question 2: Daily order volume during peak hours
    elif question_id == 2:
        print("Analyzing daily order volume during peak hours...")
        
        # Filter orders to peak hours (12-2 PM)
        peak_hours_orders = vendors_gdf[(vendors_gdf['hour'] >= 12) & (vendors_gdf['hour'] < 14)]
        
        # Group by date to get daily volumes
        daily_volumes = peak_hours_orders.groupby(['date', 'vendor_id']).size().reset_index(name='order_count')
        
        # Find vendors with ≥150 daily orders during peak hours
        high_volume_vendors = daily_volumes[daily_volumes['order_count'] >= 150]
        
        # Get unique vendor locations
        high_volume_locations = vendors_gdf[vendors_gdf['vendor_id'].isin(high_volume_vendors['vendor_id'].unique())]
        
        # Results for each site
        site_results = []
        
        # Process each potential site
        for idx, site in site_grid.iterrows():
            site_point = site.geometry
            
            # Create 1km buffer
            buffer_1km = create_buffer_km(site_point, 1.0)
            
            # Count high volume vendors within buffer
            high_volume_within = sum(high_volume_locations.geometry.intersects(buffer_1km))
            
            # Store results
            site_results.append({
                'site_id': idx,
                'longitude': site_point.x,
                'latitude': site_point.y,
                'high_volume_vendors_within_1km': high_volume_within,
                'meets_criteria': high_volume_within > 0
            })
        
        # Convert to DataFrame
        results_df = pd.DataFrame(site_results)
        
        # Filter sites meeting criteria
        sites_meeting_criteria = results_df[results_df['meets_criteria']]
        
        # Save results
        results_df.to_csv(os.path.join(question_dir, 'site_analysis.csv'), index=False)
        sites_meeting_criteria.to_csv(os.path.join(question_dir, 'sites_meeting_criteria.csv'), index=False)
        
        # Create visualization
        m = folium.Map(location=[25.24, 55.30], zoom_start=12)
        
        # Add Sector 3 boundary
        for idx, row in sector_3.iterrows():
            folium.GeoJson(row['geometry'], 
                          style_function=lambda x: {'fillColor': 'blue', 'color': 'blue', 'fillOpacity': 0.1}).add_to(m)
        
        # Add sites meeting criteria
        for idx, row in sites_meeting_criteria.iterrows():
            popup_text = f"Site ID: {row['site_id']}<br>High volume vendors: {row['high_volume_vendors_within_1km']}"
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=popup_text,
                icon=folium.Icon(color='green', icon='check', prefix='fa')
            ).add_to(m)
        
        # Save map
        map_file = os.path.join(question_dir, 'sites_map.html')
        m.save(map_file)
        
        # Update answer
        answer['sites_meeting_criteria'] = sites_meeting_criteria[['site_id', 'longitude', 'latitude']].to_dict('records')
        answer['analysis_summary'] = {
            'total_sites_analyzed': len(results_df),
            'sites_meeting_criteria': len(sites_meeting_criteria),
            'total_high_volume_vendors': len(high_volume_locations),
            'avg_daily_peak_hour_orders': daily_volumes['order_count'].mean()
        }
        answer['visualization_files'] = [map_file]
    
    # Question 3: Average order value and daily GMV
    elif question_id == 3:
        print("Analyzing average order value and daily GMV...")
        
        # Calculate daily GMV for each vendor
        daily_gmv = vendors_gdf.groupby(['date', 'vendor_id'])['gmv_amount_lc'].sum().reset_index()
        
        # Calculate average order value for each vendor
        avg_order_value = vendors_gdf.groupby('vendor_id')['gmv_amount_lc'].mean().reset_index()
        
        # Find vendors with avg order value > AED 50
        high_value_vendors = avg_order_value[avg_order_value['gmv_amount_lc'] > 50]
        
        # Find vendors with daily GMV > AED 10,000
        high_gmv_vendors = daily_gmv[daily_gmv['gmv_amount_lc'] > 10000]
        
        # Get unique vendor locations meeting both criteria
        high_value_ids = set(high_value_vendors['vendor_id'])
        high_gmv_ids = set(high_gmv_vendors['vendor_id'])
        meeting_both_criteria = high_value_ids.intersection(high_gmv_ids)
        
        high_performing_locations = vendors_gdf[vendors_gdf['vendor_id'].isin(meeting_both_criteria)]
        
        # Results for each site
        site_results = []
        
        # Process each potential site
        for idx, site in site_grid.iterrows():
            site_point = site.geometry
            
            # Create 1km buffer
            buffer_1km = create_buffer_km(site_point, 1.0)
            
            # Count high performing vendors within buffer
            high_performing_within = sum(high_performing_locations.geometry.intersects(buffer_1km))
            
            # Store results
            site_results.append({
                'site_id': idx,
                'longitude': site_point.x,
                'latitude': site_point.y,
                'high_performing_vendors_within_1km': high_performing_within,
                'meets_criteria': high_performing_within > 0
            })
        
        # Convert to DataFrame
        results_df = pd.DataFrame(site_results)
        
        # Filter sites meeting criteria
        sites_meeting_criteria = results_df[results_df['meets_criteria']]
        
        # Save results
        results_df.to_csv(os.path.join(question_dir, 'site_analysis.csv'), index=False)
        sites_meeting_criteria.to_csv(os.path.join(question_dir, 'sites_meeting_criteria.csv'), index=False)
        
        # Create visualization
        m = folium.Map(location=[25.24, 55.30], zoom_start=12)
        
        # Add Sector 3 boundary
        for idx, row in sector_3.iterrows():
            folium.GeoJson(row['geometry'], 
                          style_function=lambda x: {'fillColor': 'blue', 'color': 'blue', 'fillOpacity': 0.1}).add_to(m)
        
        # Add sites meeting criteria
        for idx, row in sites_meeting_criteria.iterrows():
            popup_text = f"Site ID: {row['site_id']}<br>High performing vendors: {row['high_performing_vendors_within_1km']}"
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=popup_text,
                icon=folium.Icon(color='green', icon='check', prefix='fa')
            ).add_to(m)
        
        # Save map
        map_file = os.path.join(question_dir, 'sites_map.html')
        m.save(map_file)
        
        # Update answer
        answer['sites_meeting_criteria'] = sites_meeting_criteria[['site_id', 'longitude', 'latitude']].to_dict('records')
        answer['analysis_summary'] = {
            'total_sites_analyzed': len(results_df),
            'sites_meeting_criteria': len(sites_meeting_criteria),
            'vendors_with_high_avg_order_value': len(high_value_vendors),
            'vendors_with_high_daily_gmv': len(high_gmv_vendors),
            'vendors_meeting_both_criteria': len(meeting_both_criteria)
        }
        answer['visualization_files'] = [map_file]
    
    # Question 4: Competing restaurants and unique customers
    elif question_id == 4:
        print("Analyzing competing restaurants and unique customers...")
        
        # Results for each site
        site_results = []
        
        # Process each potential site
        for idx, site in site_grid.iterrows():
            site_point = site.geometry
            
            # Create 1km buffer
            buffer_1km = create_buffer_km(site_point, 1.0)
            
            #
(Content truncated due to size limit. Use line ranges to read in chunks)