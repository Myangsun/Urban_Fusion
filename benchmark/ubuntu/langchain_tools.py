"""
LangChain Tools for Urban Fusion Agent

This module implements custom tools for the Urban Fusion agent using LangChain.
These tools provide functionality for map operations, data analysis, and visualization.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel, Field

# Import LangChain components
from langchain.tools import BaseTool

# Import GIS analysis functions
from gis_analysis_functions import (
    load_geojson_data,
    load_csv_data,
    create_buffer,
    points_within_buffer,
    count_points_within_buffer,
    count_unique_values_within_buffer,
    distance_between_points,
    find_nearest_features,
    create_map_visualization
)

# File paths
GEOJSON_PATH = 'upload/dubai_sector3_sites.geojson'
CSV_PATH = 'upload/talabat_sample.csv'
OUTPUT_DIR = 'output'

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)


class MapTool(BaseTool):
    """Tool for performing map operations."""

    name: str = "map_tool"
    description: str = "Tool for performing map operations like creating buffers and finding points within buffers."

    def _run(self, query: str) -> str:
        """
        Run the map tool.

        Args:
            query: Query string with operation and parameters

        Returns:
            Result of the operation
        """
        try:
            # Parse query
            parts = query.split()
            operation = parts[0].lower()

            if operation == "buffer":
                # Example: buffer 0.5km
                distance_km = float(parts[1].replace("km", ""))

                # Load data
                sites_gdf = load_geojson_data(GEOJSON_PATH)

                # Create buffer
                buffered_gdf = create_buffer(sites_gdf, distance_km)

                return f"Created buffer of {distance_km}km around {len(buffered_gdf)} sites"

            elif operation == "count_within":
                # Example: count_within coffee_shops 0.5km
                target = parts[1]
                distance_km = float(parts[2].replace("km", ""))

                # Load data
                sites_gdf = load_geojson_data(GEOJSON_PATH)
                orders_df = load_csv_data(CSV_PATH)

                # Create point geometries for restaurants
                from shapely.geometry import Point
                vendor_geometry = [Point(xy) for xy in zip(
                    orders_df['vendor_longitude'], orders_df['vendor_latitude'])]
                restaurants_gdf = gpd.GeoDataFrame(
                    orders_df, geometry=vendor_geometry, crs="EPSG:4326")

                # Filter restaurants by target
                if target == "coffee_shops":
                    restaurants_gdf = restaurants_gdf[restaurants_gdf['cuisine_type'] == 'Coffee']

                # Create buffer
                buffered_gdf = create_buffer(sites_gdf, distance_km)

                # Count points within buffer
                counts = []
                for idx, site in buffered_gdf.iterrows():
                    count = count_points_within_buffer(
                        restaurants_gdf, site['buffer'])
                    counts.append(count)

                return f"Counted {target} within {distance_km}km of each site. Average count: {np.mean(counts):.2f}"

            else:
                return f"Unknown operation: {operation}"

        except Exception as e:
            return f"Error in map tool: {str(e)}"


class DataAnalysisTool(BaseTool):
    """Tool for performing data analysis."""

    name: str = "data_analysis_tool"
    description: str = "Tool for analyzing data, calculating statistics, and processing constraints."

    def _run(self, query: str) -> str:
        """
        Run the data analysis tool.

        Args:
            query: Query string with operation and parameters

        Returns:
            Result of the operation
        """
        try:
            # Parse query
            parts = query.split()
            operation = parts[0].lower()

            if operation == "count_unique":
                # Example: count_unique customers 1km
                target = parts[1]
                distance_km = float(parts[2].replace("km", ""))

                # Load data
                sites_gdf = load_geojson_data(GEOJSON_PATH)
                orders_df = load_csv_data(CSV_PATH)

                # Create point geometries for customers
                from shapely.geometry import Point
                customer_geometry = [Point(xy) for xy in zip(
                    orders_df['customer_longitude'], orders_df['customer_latitude'])]
                customers_gdf = gpd.GeoDataFrame(
                    orders_df, geometry=customer_geometry, crs="EPSG:4326")

                # Create buffer
                buffered_gdf = create_buffer(sites_gdf, distance_km)

                # Count unique values within buffer
                counts = []
                for idx, site in buffered_gdf.iterrows():
                    count = count_unique_values_within_buffer(
                        customers_gdf, site['buffer'], 'customer_id')
                    counts.append(count)

                return f"Counted unique {target} within {distance_km}km of each site. Average count: {np.mean(counts):.2f}"

            elif operation == "calculate_average":
                # Example: calculate_average order_value 0.8km
                value_type = parts[1]
                distance_km = float(parts[2].replace("km", ""))

                # Load data
                sites_gdf = load_geojson_data(GEOJSON_PATH)
                orders_df = load_csv_data(CSV_PATH)

                # Create point geometries for restaurants
                from shapely.geometry import Point
                vendor_geometry = [Point(xy) for xy in zip(
                    orders_df['vendor_longitude'], orders_df['vendor_latitude'])]
                restaurants_gdf = gpd.GeoDataFrame(
                    orders_df, geometry=vendor_geometry, crs="EPSG:4326")

                # Determine value column
                if value_type == "order_value":
                    value_column = "basket_value"
                elif value_type == "delivery_fee":
                    value_column = "delivery_fee_amount_lc"
                else:
                    value_column = "basket_value"

                # Create buffer
                buffered_gdf = create_buffer(sites_gdf, distance_km)

                # Calculate average values within buffer
                averages = []
                for idx, site in buffered_gdf.iterrows():
                    points_within = points_within_buffer(
                        restaurants_gdf, site, 'buffer')
                    if len(points_within) > 0:
                        average = points_within[value_column].mean()
                        averages.append(average)

                if averages:
                    return f"Calculated average {value_type} within {distance_km}km of each site. Overall average: {np.mean(averages):.2f}"
                else:
                    return f"No data found for calculating average {value_type}"

            else:
                return f"Unknown operation: {operation}"

        except Exception as e:
            return f"Error in data analysis tool: {str(e)}"


class VisualizationTool(BaseTool):
    """Tool for creating visualizations."""

    name: str = "visualization_tool"
    description: str = "Tool for creating visualizations of sites, constraints, and results."

    def _run(self, query: str) -> str:
        """
        Run the visualization tool.

        Args:
            query: Query string with operation and parameters

        Returns:
            Result of the operation
        """
        try:
            # Parse query
            parts = query.split()
            operation = parts[0].lower()

            if operation == "map":
                # Example: map selected_sites 1
                visualization_type = parts[1]
                question_id = int(parts[2])

                if visualization_type == "selected_sites":
                    # Load data
                    sites_gdf = load_geojson_data(GEOJSON_PATH)

                    # For demonstration, select random sites
                    import random
                    selected_parcels = random.sample(
                        list(range(1, len(sites_gdf) + 1)), 10)

                    # Create visualization
                    output_path = create_map_visualization(
                        sites_gdf, selected_parcels, question_id)

                    return f"Created map visualization of selected sites for question {question_id}. Saved to {output_path}"

                else:
                    return f"Unknown visualization type: {visualization_type}"

            else:
                return f"Unknown operation: {operation}"

        except Exception as e:
            return f"Error in visualization tool: {str(e)}"

# Test functions


def test_map_tool():
    """Test the map tool."""
    tool = MapTool()
    result = tool._run("buffer 0.5km")
    print(result)

    result = tool._run("count_within coffee_shops 0.5km")
    print(result)


def test_data_analysis_tool():
    """Test the data analysis tool."""
    tool = DataAnalysisTool()
    result = tool._run("count_unique customers 1km")
    print(result)

    result = tool._run("calculate_average order_value 0.8km")
    print(result)


def test_visualization_tool():
    """Test the visualization tool."""
    tool = VisualizationTool()
    result = tool._run("map selected_sites 1")
    print(result)


if __name__ == "__main__":
    # Run tests
    print("Testing Map Tool...")
    test_map_tool()

    print("\nTesting Data Analysis Tool...")
    test_data_analysis_tool()

    print("\nTesting Visualization Tool...")
    test_visualization_tool()
