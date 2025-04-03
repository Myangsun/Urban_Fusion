"""
GIS Analysis Functions for Urban Fusion Agent

This module contains core GIS analysis functions for processing spatial data and constraints
in the site selection benchmark. These functions handle various types of spatial operations,
including buffer analysis, distance calculations, and spatial joins.
"""

import os
import json
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, Polygon, shape
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import contextily as ctx
from datetime import datetime
import re
from typing import Dict, List, Any, Optional, Tuple, Set
import io

# File paths
GEOJSON_PATH = 'upload/dubai_sector3_sites.geojson'
CSV_PATH = 'upload/talabat_sample.csv'
BENCHMARK_PATH = 'upload/site_selection_benchmark.json'
OUTPUT_DIR = 'output'

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================
# Data Loading Functions
# =============================================


def load_geojson_data(file_path: str = GEOJSON_PATH):
    try:
        sites_gdf = gpd.read_file(f"GeoJSON:{file_path}")
        print(f"Successfully loaded {len(sites_gdf)} sites.")
        return sites_gdf
    except Exception as e:
        print(f"Error loading GeoJSON with GeoPandas: {str(e)}")
        return None


def load_csv_data(file_path: str = CSV_PATH) -> pd.DataFrame:
    """
    Load and preprocess CSV data, extracting coordinates from POINT geometries.

    Args:
        file_path: Path to the CSV file

    Returns:
        DataFrame with preprocessed data
    """
    try:
        # Load CSV data
        orders_df = pd.read_csv(file_path)

        # Extract vendor coordinates from POINT geometry string
        orders_df['vendor_longitude'] = orders_df['vendor_location'].str.extract(
            r'POINT\(([0-9.]+)\s')
        orders_df['vendor_latitude'] = orders_df['vendor_location'].str.extract(
            r'POINT\([0-9.]+\s([0-9.]+)\)')

        # Convert to numeric
        orders_df['vendor_longitude'] = pd.to_numeric(
            orders_df['vendor_longitude'])
        orders_df['vendor_latitude'] = pd.to_numeric(
            orders_df['vendor_latitude'])

        # Extract customer coordinates from POINT geometry string
        orders_df['customer_longitude'] = orders_df['customer_selected_location'].str.extract(
            r'POINT\(([0-9.]+)\s')
        orders_df['customer_latitude'] = orders_df['customer_selected_location'].str.extract(
            r'POINT\([0-9.]+\s([0-9.]+)\)')

        # Convert to numeric
        orders_df['customer_longitude'] = pd.to_numeric(
            orders_df['customer_longitude'])
        orders_df['customer_latitude'] = pd.to_numeric(
            orders_df['customer_latitude'])

        # Rename columns to match expected names in the code
        orders_df = orders_df.rename(columns={
            'main_cuisine': 'cuisine_type',
            'gmv_amount_lc': 'basket_value',
            'account_id': 'customer_id'
        })

        return orders_df

    except Exception as e:
        print(f"Error loading CSV data: {str(e)}")
        return None


def load_benchmark_data(file_path=BENCHMARK_PATH):
    with open(file_path, 'r') as f:
        data = json.load(f)
    # Ensure data is always a dict with 'questions' key
    if isinstance(data, list):
        return {'questions': data}
    return data

# =============================================
# Geospatial Analysis Functions
# =============================================


def create_buffer(gdf: gpd.GeoDataFrame, distance_km: float) -> gpd.GeoDataFrame:
    """
    Create buffer around geometries in a GeoDataFrame.

    Args:
        gdf: GeoDataFrame with geometries
        distance_km: Buffer distance in kilometers

    Returns:
        GeoDataFrame with buffer geometries
    """
    # Convert to projected CRS for accurate distance calculations
    gdf_projected = gdf.to_crs("EPSG:3857")

    # Create buffer (convert km to meters)
    distance_m = distance_km * 1000
    gdf_projected['buffer'] = gdf_projected.geometry.buffer(distance_m)

    # Convert back to original CRS
    gdf_buffered = gdf_projected.to_crs(gdf.crs)

    return gdf_buffered


def points_within_buffer(points_gdf: gpd.GeoDataFrame, buffer_gdf: gpd.GeoDataFrame, buffer_column: str = 'buffer') -> gpd.GeoDataFrame:
    """
    Find points within buffer geometries.

    Args:
        points_gdf: GeoDataFrame with point geometries
        buffer_gdf: GeoDataFrame with buffer geometries
        buffer_column: Name of the column containing buffer geometries

    Returns:
        GeoDataFrame with points that fall within the buffer
    """
    # Ensure both GeoDataFrames have the same CRS
    if points_gdf.crs != buffer_gdf.crs:
        points_gdf = points_gdf.to_crs(buffer_gdf.crs)

    # Create a spatial join
    joined = gpd.sjoin(points_gdf, buffer_gdf[[
                       buffer_column, 'geometry']], how='inner', predicate='within')

    return joined


def count_points_within_buffer(points_gdf: gpd.GeoDataFrame, buffer_geometry: Any, filter_column: Optional[str] = None, filter_value: Optional[Any] = None) -> int:
    """
    Count points within a buffer geometry, optionally filtered by a column value.

    Args:
        points_gdf: GeoDataFrame with point geometries
        buffer_geometry: Buffer geometry to check against
        filter_column: Optional column name to filter by
        filter_value: Optional value to filter by

    Returns:
        Count of points within the buffer
    """
    # Filter points by buffer
    points_within = points_gdf[points_gdf.geometry.within(buffer_geometry)]

    # Apply additional filter if specified
    if filter_column is not None and filter_value is not None:
        points_within = points_within[points_within[filter_column]
                                      == filter_value]

    return len(points_within)


def count_unique_values_within_buffer(points_gdf: gpd.GeoDataFrame, buffer_geometry: Any, value_column: str) -> int:
    """
    Count unique values of a column for points within a buffer geometry.

    Args:
        points_gdf: GeoDataFrame with point geometries
        buffer_geometry: Buffer geometry to check against
        value_column: Column name to count unique values from

    Returns:
        Count of unique values within the buffer
    """
    # Filter points by buffer
    points_within = points_gdf[points_gdf.geometry.within(buffer_geometry)]

    # Count unique values
    unique_count = points_within[value_column].nunique()

    return unique_count


def distance_between_points(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """
    Calculate the distance between two points in kilometers.

    Args:
        point1: Tuple of (longitude, latitude) for the first point
        point2: Tuple of (longitude, latitude) for the second point

    Returns:
        Distance in kilometers
    """
    # Create Point objects
    p1 = Point(point1)
    p2 = Point(point2)

    # Create GeoDataFrame with points
    points_gdf = gpd.GeoDataFrame(geometry=[p1, p2], crs="EPSG:4326")

    # Convert to projected CRS for accurate distance calculations
    points_projected = points_gdf.to_crs("EPSG:3857")

    # Calculate distance in meters
    distance_m = points_projected.iloc[0].geometry.distance(
        points_projected.iloc[1].geometry)

    # Convert to kilometers
    distance_km = distance_m / 1000

    return distance_km


def find_nearest_features(target_gdf: gpd.GeoDataFrame, source_gdf: gpd.GeoDataFrame, k: int = 1) -> pd.DataFrame:
    """
    Find the k nearest features from source_gdf for each feature in target_gdf.

    Args:
        target_gdf: GeoDataFrame with target geometries
        source_gdf: GeoDataFrame with source geometries
        k: Number of nearest features to find

    Returns:
        DataFrame with target_id, source_id, and distance columns
    """
    # Ensure both GeoDataFrames have the same CRS
    if target_gdf.crs != source_gdf.crs:
        source_gdf = source_gdf.to_crs(target_gdf.crs)

    # Convert to projected CRS for accurate distance calculations
    target_projected = target_gdf.to_crs("EPSG:3857")
    source_projected = source_gdf.to_crs("EPSG:3857")

    # Create empty list to store results
    results = []

    # For each target feature, find the k nearest source features
    for target_idx, target_row in target_projected.iterrows():
        target_geom = target_row.geometry

        # Calculate distances to all source features
        distances = source_projected.geometry.distance(target_geom)

        # Get the k nearest features
        nearest_indices = distances.nsmallest(k).index

        # Add results to list
        for source_idx in nearest_indices:
            source_row = source_projected.loc[source_idx]
            distance_m = target_geom.distance(source_row.geometry)
            distance_km = distance_m / 1000

            results.append({
                'target_id': target_gdf.iloc[target_idx]['site_id'],
                'source_id': source_gdf.iloc[source_idx]['site_id'] if 'site_id' in source_gdf.columns else source_idx,
                'distance_km': distance_km
            })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    return results_df

# =============================================
# Constraint Processing Functions
# =============================================


def parse_geospatial_constraint(constraint_text: str) -> Dict[str, Any]:
    """
    Parse a geospatial constraint from text.

    Args:
        constraint_text: Text description of the constraint

    Returns:
        Dictionary with parsed constraint parameters
    """
    constraint = {}

    # Extract location
    location_match = re.search(r'Dubai\s+Sector\s+\d+', constraint_text)
    if location_match:
        constraint['location'] = location_match.group(0)

    # Extract radius values
    radius_matches = re.findall(r'(\d+(?:\.\d+)?)\s*km', constraint_text)
    if radius_matches:
        constraint['radius_km'] = [float(r) for r in radius_matches]

    # Extract target features
    if 'coffee shops' in constraint_text.lower():
        constraint['target'] = 'coffee shops'
    elif 'burger' in constraint_text.lower():
        constraint['target'] = 'burger restaurants'

    # Extract comparison operators
    if '<=' in constraint_text or '≤' in constraint_text:
        constraint['comparison'] = '<='
    elif '>=' in constraint_text or '≥' in constraint_text:
        constraint['comparison'] = '>='
    elif '<' in constraint_text:
        constraint['comparison'] = '<'
    elif '>' in constraint_text:
        constraint['comparison'] = '>'

    # Extract threshold values
    threshold_match = re.search(r'([<>]=?|≤|≥)\s*(\d+)', constraint_text)
    if threshold_match:
        constraint['threshold'] = int(threshold_match.group(2))

    return constraint


def parse_temporal_constraint(constraint_text: str) -> Dict[str, Any]:
    """
    Parse a temporal constraint from text.

    Args:
        constraint_text: Text description of the constraint

    Returns:
        Dictionary with parsed constraint parameters
    """
    constraint = {}

    # Extract time periods
    time_period_match = re.search(
        r'(\d+(?:\s*[AP]M)?\s*-\s*\d+(?:\s*[AP]M)?)', constraint_text)
    if time_period_match:
        constraint['time_period'] = time_period_match.group(1)
        constraint['constraint_type'] = 'time_of_day'

    # Extract days of week
    day_match = re.search(
        r'(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)(?:\s*-\s*(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday))?', constraint_text)
    if day_match:
        constraint['day_period'] = day_match.group(0)
        constraint['constraint_type'] = 'day_of_week'

    # Extract months or seasons
    month_match = re.search(
        r'(January|February|March|April|May|June|July|August|September|October|November|December)(?:\s*-\s*(January|February|March|April|May|June|July|August|September|October|November|December))?', constraint_text)
    if month_match:
        constraint['month_period'] = month_match.group(0)
        constraint['constraint_type'] = 'month'

    # Extract threshold values
    threshold_match = re.search(r'([<>]=?|≤|≥)\s*(\d+)', constraint_text)
    if threshold_match:
        constraint['threshold'] = int(threshold_match.group(2))

    return constraint


def parse_economic_constraint(constraint_text: str) -> Dict[str, Any]:
    """
    Parse an economic constraint from text.

    Args:
        constraint_text: Text description of the constraint

    Returns:
        Dictionary with parsed constraint parameters
    """
    constraint = {}

    # Extract metric type
    if 'average' in constraint_text.lower():
        constraint['metric'] = 'average'
    elif 'total' in constraint_text.lower():
        constraint['metric'] = 'total'
    elif 'minimum' in constraint_text.lower():
        constraint['metric'] = 'minimum'

    # Extract value type
    if 'order value' in constraint_text.lower():
        constraint['value_type'] = 'order_value'
    elif 'delivery fee' in constraint_text.lower():
        constraint['value_type'] = 'delivery_fee'
    elif 'service fee' in constraint_text.lower():
        constraint['value_type'] = 'service_fee'

    # Extract comparison operators
    if '<=' in constraint_text or '≤' in constraint_text:
        constraint['comparison'] = '<='
    elif '>=' in constraint_text or '≥' in constraint_text:
        constraint['comparison'] = '>='
    elif '<' in constraint_text:
        constraint['comparison'] = '<'
    elif '>' in constraint_text:
        constraint['comparison'] = '>'

    # Extract threshold values
    threshold_match = re.search(
        r'([<>]=?|≤|≥)\s*(\d+(?:\.\d+)?)', constraint_text)
    if threshold_match:
        constraint['threshold'] = float(threshold_match.group(2))

    # Extract currency
    currency_match = re.search(
        r'(\d+(?:\.\d+)?)\s*([A-Z]{3})', constraint_text)
    if currency_match:
        constraint['currency'] = currency_match.group(2)

    return constraint


def parse_market_constraint(constraint_text: str) -> Dict[str, Any]:
    """
    Parse a market constraint from text.

    Args:
        constraint_text: Text description of the constraint

    Returns:
        Dictionary with parsed constraint parameters
    """
    constraint = {}

    # Extract target features
    if 'coffee shops' in constraint_text.lower():
        constraint['target'] = 'coffee shops'
    elif 'burger' in constraint_text.lower():
        constraint['target'] = 'burger restaurants'
    elif 'customer' in constraint_text.lower():
        constraint['target'] = 'customers'

    # Extract metric type
    if 'count' in constraint_text.lower():
        constraint['metric'] = 'count'
    elif 'diversity' in constraint_text.lower():
        constraint['metric'] = 'diversity'
    elif 'ratio' in constraint_text.lower():
        constraint['metric'] = 'ratio'

    # Extract comparison operators
    if '<=' in constraint_text or '≤' in constraint_text:
        constraint['comparison'] = '<='
    elif '>=' in constraint_text or '≥' in constraint_text:
        constraint['comparison'] = '>='
    elif '<' in constraint_text:
        constraint['comparison'] = '<'
    elif '>' in constraint_text:
        constraint['comparison'] = '>'

    # Extract threshold values
    threshold_match = re.search(
        r'([<>]=?|≤|≥)\s*(\d+(?:\.\d+)?)', constraint_text)
    if threshold_match:
        constraint['threshold'] = float(threshold_match.group(2))

    # Extract radius values
    radius_matches = re.findall(r'(\d+(?:\.\d+)?)\s*km', constraint_text)
    if radius_matches:
        constraint['radius_km'] = [float(r) for r in radius_matches]

    return constraint


def process_constraints(question_data: Dict[str, Any], sites_gdf: gpd.GeoDataFrame, orders_df: pd.DataFrame) -> Set[int]:
    """
    Process all constraints for a question and find parcels that satisfy them.

    Args:
        question_data: Dictionary with question data
        sites_gdf: GeoDataFrame with site geometries
        orders_df: DataFrame with order data

    Returns:
        Set of parcel IDs that satisfy all constraints
    """
    if 'site_id' not in sites_gdf.columns:
        sites_gdf = sites_gdf.reset_index()
        sites_gdf['site_id'] = sites_gdf.index + 1

    sites_gdf['site_id'] = sites_gdf['site_id'].astype(int)
    # Convert orders to GeoDataFrames
    vendor_geometry = [Point(xy) for xy in zip(
        orders_df['vendor_longitude'], orders_df['vendor_latitude'])]
    restaurants_gdf = gpd.GeoDataFrame(
        orders_df, geometry=vendor_geometry, crs="EPSG:4326")

    customer_geometry = [Point(xy) for xy in zip(
        orders_df['customer_longitude'], orders_df['customer_latitude'])]
    customers_gdf = gpd.GeoDataFrame(
        orders_df, geometry=customer_geometry, crs="EPSG:4326")

    # Convert to projected CRS for accurate distance calculations
    sites_gdf = sites_gdf.to_crs("EPSG:3857")
    restaurants_gdf = restaurants_gdf.to_crs("EPSG:3857")
    customers_gdf = customers_gdf.to_crs("EPSG:3857")

    # Initialize set of all parcel IDs
    all_parcels = set(sites_gdf['site_id'].astype(int).tolist())

    # Process each constraint
    for constraint_text in question_data.get('constraints', []):
        constraint_text = constraint_text.lower()

        # Determine constraint type
        if 'geospatial' in constraint_text:
            # Parse and process geospatial constraint
            constraint = parse_geospatial_constraint(constraint_text)
            parcels = process_geospatial_constraint(
                constraint, sites_gdf, restaurants_gdf)
        elif 'temporal' in constraint_text:
            # Parse and process temporal constraint
            constraint = parse_temporal_constraint(constraint_text)
            parcels = process_temporal_constraint(
                constraint, sites_gdf, restaurants_gdf)
        elif 'economic' in constraint_text:
            # Parse and process economic constraint
            constraint = parse_economic_constraint(constraint_text)
            parcels = process_economic_constraint(
                constraint, sites_gdf, restaurants_gdf)
        elif 'market' in constraint_text:
            # Parse and process market constraint
            constraint = parse_market_constraint(constraint_text)
            parcels = process_market_constraint(
                constraint, sites_gdf, restaurants_gdf, customers_gdf)
        else:
            # Unknown constraint type, skip
            continue

        # Intersect with current set of parcels
        all_parcels = all_parcels.intersection(parcels)

    return all_parcels


def process_geospatial_constraint(constraint: Dict[str, Any], sites_gdf: gpd.GeoDataFrame, restaurants_gdf: gpd.GeoDataFrame) -> Set[int]:
    """
    Process a geospatial constraint and find parcels that satisfy it.

    Args:
        constraint: Dictionary with constraint parameters
        sites_gdf: GeoDataFrame with site geometries
        restaurants_gdf: GeoDataFrame with restaurant geometries

    Returns:
        Set of parcel IDs that satisfy the constraint
    """
    selected_parcels = set()

    # Get radius value
    radius_km = constraint.get('radius_km', [0.5])[0]

    # Create buffer around each site
    buffer_distance = radius_km * 1000  # Convert km to meters
    sites_gdf['buffer'] = sites_gdf.geometry.buffer(buffer_distance)

    # Process based on target and comparison
    target = constraint.get('target')
    comparison = constraint.get('comparison')
    threshold = constraint.get('threshold', 4)

    for idx, site in sites_gdf.iterrows():
        site_id = int(site['site_id'])
        buffer_geom = site['buffer']

        # Filter restaurants by target
        if target == 'coffee shops':
            filtered_restaurants = restaurants_gdf[restaurants_gdf['cuisine_type'] == 'Coffee']
        elif target == 'burger restaurants':
            filtered_restaurants = restaurants_gdf[restaurants_gdf['cuisine_type'] == 'burgers']
        else:
            filtered_restaurants = restaurants_gdf

        # Count restaurants within buffer
        count = filtered_restaurants[filtered_restaurants.geometry.within(
            buffer_geom)].shape[0]

        # Apply comparison
        if comparison == '<=' and count <= threshold:
            selected_parcels.add(site_id)
        elif comparison == '>=' and count >= threshold:
            selected_parcels.add(site_id)
        elif comparison == '<' and count < threshold:
            selected_parcels.add(site_id)
        elif comparison == '>' and count > threshold:
            selected_parcels.add(site_id)
        elif comparison is None:
            # Default behavior based on target
            if target == 'coffee shops' and count <= 4:
                selected_parcels.add(site_id)
            elif target != 'coffee shops' and count >= 4:
                selected_parcels.add(site_id)

    return selected_parcels


def process_temporal_constraint(constraint: Dict[str, Any], sites_gdf: gpd.GeoDataFrame, restaurants_gdf: gpd.GeoDataFrame) -> Set[int]:
    """
    Process a temporal constraint and find parcels that satisfy it.

    Args:
        constraint: Dictionary with constraint parameters
        sites_gdf: GeoDataFrame with site geometries
        restaurants_gdf: GeoDataFrame with restaurant geometries

    Returns:
        Set of parcel IDs that satisfy the constraint
    """
    selected_parcels = set()

    # Get constraint type
    constraint_type = constraint.get('constraint_type')

    if constraint_type == 'time_of_day':
        # Parse time period
        time_period = constraint.get('time_period', '')
        time_parts = time_period.split('-')
        start_time = time_parts[0].strip()
        end_time = time_parts[1].strip() if len(time_parts) > 1 else start_time

        # Convert to 24-hour format for comparison
        start_hour = int(start_time.split()[
                         0]) + (12 if 'pm' in start_time.lower() and int(start_time.split()[0]) < 12 else 0)
        end_hour = int(end_time.split()[
                       0]) + (12 if 'pm' in end_time.lower() and int(end_time.split()[0]) < 12 else 0)

        # Filter orders by time
        restaurants_df = restaurants_gdf.copy()
        restaurants_df['hour'] = pd.to_datetime(
            restaurants_df['order_time']).dt.hour
        filtered_restaurants = restaurants_df[(
            restaurants_df['hour'] >= start_hour) & (restaurants_df['hour'] <= end_hour)]

        # Convert filtered orders to GeoDataFrame
        filtered_restaurants_gdf = gpd.GeoDataFrame(
            filtered_restaurants, geometry=filtered_restaurants.geometry, crs=restaurants_gdf.crs)

        # Process each site
        threshold = constraint.get('threshold', 50)

        for idx, site in sites_gdf.iterrows():
            site_id = int(site['site_id'])
            buffer_distance = 800  # Default 0.8km buffer in meters
            buffer_geom = site.geometry.buffer(buffer_distance)

            # Count orders within buffer
            count = filtered_restaurants_gdf[filtered_restaurants_gdf.geometry.within(
                buffer_geom)].shape[0]

            # Apply threshold
            if count >= threshold:
                selected_parcels.add(site_id)

    elif constraint_type == 'day_of_week':
        # For demonstration, we'll use a random selection of parcels
        # In a real implementation, we would filter orders by day of week
        selected_parcels = set(sites_gdf.sample(frac=0.3)[
                               'site_id'].astype(int).tolist())

    elif constraint_type == 'month':
        # For demonstration, we'll use a random selection of parcels
        # In a real implementation, we would filter orders by month
        selected_parcels = set(sites_gdf.sample(frac=0.4)[
                               'site_id'].astype(int).tolist())

    return selected_parcels


def process_economic_constraint(constraint: Dict[str, Any], sites_gdf: gpd.GeoDataFrame, restaurants_gdf: gpd.GeoDataFrame) -> Set[int]:
    """
    Process an economic constraint and find parcels that satisfy it.

    Args:
        constraint: Dictionary with constraint parameters
        sites_gdf: GeoDataFrame with site geometries
        restaurants_gdf: GeoDataFrame with restaurant geometries

    Returns:
        Set of parcel IDs that satisfy the constraint
    """
    selected_parcels = set()

    # Get constraint parameters
    metric = constraint.get('metric', 'average')
    value_type = constraint.get('value_type', 'order_value')
    comparison = constraint.get('comparison', '>')
    threshold = constraint.get('threshold', 100)

    # Determine value column based on value type
    if value_type == 'order_value':
        value_column = 'basket_value'
    elif value_type == 'delivery_fee':
        value_column = 'delivery_fee_amount_lc'
    elif value_type == 'service_fee':
        value_column = 'service_fee_amount_lc'
    else:
        value_column = 'basket_value'

    # Process each site
    for idx, site in sites_gdf.iterrows():
        site_id = int(site['site_id'])
        buffer_distance = 800  # Default 0.8km buffer in meters
        buffer_geom = site.geometry.buffer(buffer_distance)

        # Get orders within buffer
        orders_within_buffer = restaurants_gdf[restaurants_gdf.geometry.within(
            buffer_geom)]

        if len(orders_within_buffer) > 0:
            # Calculate metric
            if metric == 'average':
                value = orders_within_buffer[value_column].mean()
            elif metric == 'total':
                value = orders_within_buffer[value_column].sum()
            elif metric == 'minimum':
                value = orders_within_buffer[value_column].min()
            else:
                value = orders_within_buffer[value_column].mean()

            # Apply comparison
            if comparison == '>' and value > threshold:
                selected_parcels.add(site_id)
            elif comparison == '<' and value < threshold:
                selected_parcels.add(site_id)
            elif comparison == '>=' and value >= threshold:
                selected_parcels.add(site_id)
            elif comparison == '<=' and value <= threshold:
                selected_parcels.add(site_id)

    return selected_parcels


def process_market_constraint(constraint: Dict[str, Any], sites_gdf: gpd.GeoDataFrame, restaurants_gdf: gpd.GeoDataFrame, customers_gdf: gpd.GeoDataFrame) -> Set[int]:
    """
    Process a market constraint and find parcels that satisfy it.

    Args:
        constraint: Dictionary with constraint parameters
        sites_gdf: GeoDataFrame with site geometries
        restaurants_gdf: GeoDataFrame with restaurant geometries
        customers_gdf: GeoDataFrame with customer geometries

    Returns:
        Set of parcel IDs that satisfy the constraint
    """
    selected_parcels = set()

    # Get constraint parameters
    target = constraint.get('target')
    metric = constraint.get('metric', 'count')
    comparison = constraint.get('comparison', '>')
    threshold = constraint.get('threshold', 4)
    radius_km = constraint.get('radius_km', [0.5])[
        0] if 'radius_km' in constraint else 0.5

    # Convert radius to meters
    buffer_distance = radius_km * 1000

    # Process each site
    for idx, site in sites_gdf.iterrows():
        site_id = int(site['site_id'])
        buffer_geom = site.geometry.buffer(buffer_distance)

        if target == 'coffee shops':
            # Filter restaurants by coffee shops
            filtered_features = restaurants_gdf[restaurants_gdf['cuisine_type'] == 'Coffee']

            # Count features within buffer
            count = filtered_features[filtered_features.geometry.within(
                buffer_geom)].shape[0]

            # Apply comparison
            if comparison == '>' and count > threshold:
                selected_parcels.add(site_id)
            elif comparison == '<' and count < threshold:
                selected_parcels.add(site_id)
            elif comparison == '>=' and count >= threshold:
                selected_parcels.add(site_id)
            elif comparison == '<=' and count <= threshold:
                selected_parcels.add(site_id)

        elif target == 'burger restaurants':
            # Filter restaurants by burger restaurants
            filtered_features = restaurants_gdf[restaurants_gdf['cuisine_type'] == 'burgers']

            # Count features within buffer
            count = filtered_features[filtered_features.geometry.within(
                buffer_geom)].shape[0]

            # Apply comparison
            if comparison == '>' and count > threshold:
                selected_parcels.add(site_id)
            elif comparison == '<' and count < threshold:
                selected_parcels.add(site_id)
            elif comparison == '>=' and count >= threshold:
                selected_parcels.add(site_id)
            elif comparison == '<=' and count <= threshold:
                selected_parcels.add(site_id)

        elif target == 'customers':
            # Get customers within buffer
            customers_within_buffer = customers_gdf[customers_gdf.geometry.within(
                buffer_geom)]

            if metric == 'count':
                # Count unique customers
                count = customers_within_buffer['customer_id'].nunique()

                # Apply comparison
                if comparison == '>' and count > threshold:
                    selected_parcels.add(site_id)
                elif comparison == '<' and count < threshold:
                    selected_parcels.add(site_id)
                elif comparison == '>=' and count >= threshold:
                    selected_parcels.add(site_id)
                elif comparison == '<=' and count <= threshold:
                    selected_parcels.add(site_id)

            elif metric == 'ratio':
                # Calculate ratio of unique customers to total orders
                unique_customers = customers_within_buffer['customer_id'].nunique(
                )
                total_orders = len(customers_within_buffer)

                if total_orders > 0:
                    ratio = unique_customers / total_orders

                    # Apply comparison
                    if comparison == '>' and ratio > threshold:
                        selected_parcels.add(site_id)
                    elif comparison == '<' and ratio < threshold:
                        selected_parcels.add(site_id)
                    elif comparison == '>=' and ratio >= threshold:
                        selected_parcels.add(site_id)
                    elif comparison == '<=' and ratio <= threshold:
                        selected_parcels.add(site_id)

        else:
            # Default: count all restaurants
            count = restaurants_gdf[restaurants_gdf.geometry.within(
                buffer_geom)].shape[0]

            # Apply comparison
            if comparison == '>' and count > threshold:
                selected_parcels.add(site_id)
            elif comparison == '<' and count < threshold:
                selected_parcels.add(site_id)
            elif comparison == '>=' and count >= threshold:
                selected_parcels.add(site_id)
            elif comparison == '<=' and count <= threshold:
                selected_parcels.add(site_id)

    return selected_parcels

# =============================================
# Visualization Functions
# =============================================


def create_map_visualization(sites_gdf: gpd.GeoDataFrame, selected_parcels: List[int], question_id: int) -> str:
    """
    Create a map visualization of selected parcels.

    Args:
        sites_gdf: GeoDataFrame with site geometries
        selected_parcels: List of selected parcel IDs
        question_id: Question ID for the output filename

    Returns:
        Path to the saved visualization
    """
    try:
        # Ensure site_id is properly set
        if 'site_id' not in sites_gdf.columns:
            sites_gdf = sites_gdf.reset_index()
            sites_gdf['site_id'] = sites_gdf.index + 1

        # Ensure site_id is integer type
        sites_gdf['site_id'] = sites_gdf['site_id'].astype(int)

        # Convert selected_parcels to integers if needed
        selected_parcels = [int(p) for p in selected_parcels]

        # Filter selected sites
        selected_sites = sites_gdf[sites_gdf['site_id'].isin(selected_parcels)]

        # Check if we have any selected sites
        if len(selected_sites) == 0:
            print("Warning: No sites selected for visualization")
            # Create a dummy visualization with a message
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(0.5, 0.5, "No sites selected",
                    ha='center', va='center', fontsize=14)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
        else:
            # Convert to Web Mercator projection for basemap
            sites_gdf_3857 = sites_gdf.to_crs("EPSG:3857")
            selected_sites_3857 = selected_sites.to_crs("EPSG:3857")

            # Compute full extent (bounds) from the complete sites dataset
            xmin, ymin, xmax, ymax = sites_gdf_3857.total_bounds

            # Create a plot
            fig, ax = plt.subplots(figsize=(10, 10))

            # Plot all sites
            # sites_gdf_3857.plot(ax=ax, color='gray', alpha=0.5)

            # Plot selected sites
            selected_sites_3857.plot(
                ax=ax, color='red', edgecolor='black', alpha=1, label="Selected Sites")

            # Set the map extent to include all sites
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_axis_off()

            # Add a satellite basemap
            ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery)

            # Add title and legend
            ax.set_title(f"Selected Sites for Question {question_id}")
            # ax.legend(loc="upper right")

        # Save visualization
        output_path = os.path.join(
            OUTPUT_DIR, f"question{question_id}_map.jpg")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        return output_path

    except Exception as e:
        print(f"Error creating visualization: {str(e)}")
        # Create a simple error visualization
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center',
                    va='center', fontsize=12, wrap=True)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')

            # Save visualization
            output_path = os.path.join(
                OUTPUT_DIR, f"question{question_id}_error_map.jpg")
            plt.savefig(output_path)
            plt.close()

            return output_path
        except:
            return ""


def create_constraint_visualization(constraint_results: Dict[str, List[int]], question_id: int) -> str:
    """
    Create a visualization of constraint satisfaction.

    Args:
        constraint_results: Dictionary mapping constraint types to lists of parcel IDs
        question_id: Question ID for the output filename

    Returns:
        Path to the saved visualization
    """
    try:
        # Create a DataFrame from constraint results
        data = []
        for constraint_type, parcels in constraint_results.items():
            data.append({
                'constraint_type': constraint_type,
                'parcel_count': len(parcels)
            })

        df = pd.DataFrame(data)

        # Create a bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        df.plot(kind='bar', x='constraint_type',
                y='parcel_count', ax=ax, color='skyblue')

        # Add title and labels
        ax.set_title(f"Constraint Satisfaction for Question {question_id}")
        ax.set_xlabel("Constraint Type")
        ax.set_ylabel("Number of Parcels")

        # Add value labels on top of bars
        for i, v in enumerate(df['parcel_count']):
            ax.text(i, v + 0.1, str(v), ha='center')

        # Adjust layout
        plt.tight_layout()

        # Save visualization
        output_path = os.path.join(
            OUTPUT_DIR, f"question{question_id}_constraints.jpg")
        plt.savefig(output_path)
        plt.close()

        return output_path

    except Exception as e:
        print(f"Error creating constraint visualization: {str(e)}")
        return ""

# =============================================
# Evaluation Functions
# =============================================


def evaluate_results(predicted_parcels: List[int], ground_truth_parcels: List[int]) -> Dict[str, float]:
    """
    Evaluate the results against ground truth.

    Args:
        predicted_parcels: List of predicted parcel IDs
        ground_truth_parcels: List of ground truth parcel IDs

    Returns:
        Dictionary with evaluation metrics
    """
    # Convert to sets for easier operations
    predicted_set = set(predicted_parcels)
    ground_truth_set = set(ground_truth_parcels)

    # Calculate metrics
    true_positives = len(predicted_set.intersection(ground_truth_set))
    false_positives = len(predicted_set - ground_truth_set)
    false_negatives = len(ground_truth_set - predicted_set)

    # Calculate precision, recall, and F1 score
    precision = true_positives / \
        len(predicted_set) if len(predicted_set) > 0 else 0
    recall = true_positives / \
        len(ground_truth_set) if len(ground_truth_set) > 0 else 0
    f1_score = 2 * precision * recall / \
        (precision + recall) if (precision + recall) > 0 else 0

    # Calculate accuracy (exact match)
    accuracy = 1.0 if predicted_set == ground_truth_set else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'accuracy': accuracy,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }


def create_evaluation_visualization(evaluation_results: Dict[int, Dict[str, float]]) -> str:
    """
    Create a visualization of evaluation metrics.

    Args:
        evaluation_results: Dictionary mapping question IDs to evaluation metrics

    Returns:
        Path to the saved visualization
    """
    try:
        # Create a DataFrame from evaluation results
        data = []
        for question_id, metrics in evaluation_results.items():
            data.append({
                'question_id': question_id,
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'accuracy': metrics['accuracy']
            })

        df = pd.DataFrame(data)

        # Create a grouped bar chart
        fig, ax = plt.subplots(figsize=(12, 8))

        # Set width of bars
        bar_width = 0.2

        # Set positions of bars on X axis
        r1 = np.arange(len(df))
        r2 = [x + bar_width for x in r1]
        r3 = [x + bar_width for x in r2]
        r4 = [x + bar_width for x in r3]

        # Create bars
        ax.bar(r1, df['precision'], width=bar_width,
               label='Precision', color='skyblue')
        ax.bar(r2, df['recall'], width=bar_width,
               label='Recall', color='lightgreen')
        ax.bar(r3, df['f1_score'], width=bar_width,
               label='F1 Score', color='salmon')
        ax.bar(r4, df['accuracy'], width=bar_width,
               label='Accuracy', color='purple')

        # Add labels and title
        ax.set_xlabel('Question ID')
        ax.set_ylabel('Score')
        ax.set_title('Evaluation Metrics by Question')
        ax.set_xticks([r + bar_width * 1.5 for r in range(len(df))])
        ax.set_xticklabels(df['question_id'])

        # Add legend
        ax.legend()

        # Add grid
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Adjust layout
        plt.tight_layout()

        # Save visualization
        output_path = os.path.join(OUTPUT_DIR, f"evaluation_metrics.jpg")
        plt.savefig(output_path)
        plt.close()

        return output_path

    except Exception as e:
        print(f"Error creating evaluation visualization: {str(e)}")
        return ""

# =============================================
# Main Processing Functions
# =============================================


def process_question(question_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single question and find parcels that satisfy all constraints.

    Args:
        question_data: Dictionary with question data

    Returns:
        Dictionary with question ID, selected parcels, and count
    """
    try:
        # Load data
        sites_gdf = load_geojson_data()

        # Ensure site_id column exists and is properly set
        if 'site_id' not in sites_gdf.columns:
            sites_gdf = sites_gdf.reset_index()
            sites_gdf['site_id'] = sites_gdf.index + 1

        # Ensure site_id is integer type
        sites_gdf['site_id'] = sites_gdf['site_id'].astype(int)

        orders_df = load_csv_data()

        # Process constraints
        selected_parcels = process_constraints(
            question_data, sites_gdf, orders_df)

        # Convert to list of integers
        selected_parcels_list = sorted([int(p) for p in selected_parcels])

        # Create visualization
        question_id = question_data.get('question_id', 0)
        visualization_path = create_map_visualization(
            sites_gdf, selected_parcels_list, question_id)

        return {
            'question_id': question_id,
            'parcels': selected_parcels_list,
            'count': len(selected_parcels_list),
            'visualization_path': visualization_path
        }

    except Exception as e:
        print(f"Error processing question: {str(e)}")
        return {
            'question_id': question_data.get('question_id', 0),
            'parcels': [],
            'count': 0,
            'error': str(e)
        }


def process_all_questions(benchmark_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process all questions in the benchmark.

    Args:
        benchmark_data: Dictionary with benchmark data

    Returns:
        Dictionary with results for all questions
    """
    results = {}
    evaluation = {}

    # Process each question
    for question in benchmark_data.get('questions', []):
        question_id = question.get('question_id', 0)

        # Process question
        result = process_question(question)

        # Evaluate result
        ground_truth = question.get('answer', {}).get('parcels', [])
        metrics = evaluate_results(result['parcels'], ground_truth)

        # Store results and evaluation
        results[question_id] = result
        evaluation[question_id] = metrics

    # Create evaluation visualization
    evaluation_viz_path = create_evaluation_visualization(evaluation)

    return {
        'results': results,
        'evaluation': evaluation,
        'evaluation_visualization': evaluation_viz_path
    }

# =============================================
# Test Functions
# =============================================


def test_geojson_loading():
    """Test GeoJSON loading function."""
    sites_gdf = load_geojson_data()
    if sites_gdf is not None:
        print(f"Successfully loaded {len(sites_gdf)} sites")
        print(f"Site ID column exists: {'site_id' in sites_gdf.columns}")
        print(f"Sample site IDs: {sites_gdf['site_id'].head(5).tolist()}")
    else:
        print("Failed to load GeoJSON data")


def test_csv_loading():
    """Test CSV loading function."""
    orders_df = load_csv_data()
    if orders_df is not None:
        print(f"Successfully loaded {len(orders_df)} orders")
        print(
            f"Sample vendor coordinates: {orders_df[['vendor_longitude', 'vendor_latitude']].head(3).to_dict()}")
        print(
            f"Sample customer coordinates: {orders_df[['customer_longitude', 'customer_latitude']].head(3).to_dict()}")
    else:
        print("Failed to load CSV data")


def test_constraint_parsing():
    """Test constraint parsing functions."""
    # Test geospatial constraint parsing
    geo_constraint = "Geospatial: Dubai Sector 3; 0.5km and 1km radius buffers"
    parsed_geo = parse_geospatial_constraint(geo_constraint)
    print(f"Parsed geospatial constraint: {parsed_geo}")

    # Test market constraint parsing
    market_constraint = "Market: <=4 coffee shops within 0.5km"
    parsed_market = parse_market_constraint(market_constraint)
    print(f"Parsed market constraint: {parsed_market}")


def test_question_processing():
    """Test question processing function."""
    # Load benchmark data
    benchmark_data = load_benchmark_data()

    # Get first question
    question = benchmark_data.get('questions', [])[0]

    # Process question
    result = process_question(question)

    print(f"Question {result['question_id']} result:")
    print(f"Selected parcels: {result['parcels'][:10]}...")
    print(f"Count: {result['count']}")
    print(f"Visualization path: {result.get('visualization_path', '')}")

# =============================================
# Main Function
# =============================================


if __name__ == "__main__":
    # Run tests
    print("Testing GeoJSON loading...")
    test_geojson_loading()

    print("\nTesting CSV loading...")
    test_csv_loading()

    print("\nTesting constraint parsing...")
    test_constraint_parsing()

    print("\nTesting question processing...")
    test_question_processing()
