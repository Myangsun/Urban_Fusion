"""
UrbanFusion - Data Processor
This file implements data processing utilities for the UrbanFusion system
"""

import os
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
import folium
from typing import List, Dict, Any, Optional, Union
import json

class DataProcessor:
    """Data processor for the UrbanFusion system"""

    def __init__(self, data_dir="data"):
        """Initialize the data processor
        
        Args:
            data_dir: Directory for data storage
        """
        self.data_dir = data_dir
        self.output_dir = os.path.join(data_dir, "output")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load data sources
        self.data_sources = self._load_data_sources()

    def _load_data_sources(self):
        """Load data sources from files
        
        Returns:
            Dictionary of data sources
        """
        data_sources = {}
        
        # Load restaurant data if available
        restaurants_path = os.path.join(self.data_dir, "restaurants.csv")
        if os.path.exists(restaurants_path):
            try:
                data_sources["restaurants"] = pd.read_csv(restaurants_path)
                print(f"Loaded {len(data_sources['restaurants'])} restaurants")
            except Exception as e:
                print(f"Error loading restaurant data: {str(e)}")
        else:
            print(f"Warning: Restaurant data file not found at {restaurants_path}")
        
        # Load district data if available
        districts_path = os.path.join(self.data_dir, "dubai_districts.geojson")
        if os.path.exists(districts_path):
            try:
                data_sources["districts"] = gpd.read_file(districts_path)
                print(f"Loaded {len(data_sources['districts'])} districts")
            except Exception as e:
                print(f"Error loading district data: {str(e)}")
        else:
            print(f"Warning: District data file not found at {districts_path}")
        
        # Load road data if available
        roads_path = os.path.join(self.data_dir, "dubai_roads.geojson")
        if os.path.exists(roads_path):
            try:
                data_sources["roads"] = gpd.read_file(roads_path)
                print(f"Loaded {len(data_sources['roads'])} roads")
            except Exception as e:
                print(f"Error loading road data: {str(e)}")
        else:
            print(f"Warning: Road data file not found at {roads_path}")
        
        # Load demographic data if available
        demographics_path = os.path.join(self.data_dir, "demographics.csv")
        if os.path.exists(demographics_path):
            try:
                data_sources["demographics"] = pd.read_csv(demographics_path)
                print(f"Loaded demographic data for {len(data_sources['demographics'])} districts")
            except Exception as e:
                print(f"Error loading demographic data: {str(e)}")
        else:
            print(f"Warning: Demographic data file not found at {demographics_path}")
        
        return data_sources

    def create_map(self, locations, output_path=None):
        """Create an interactive map with locations
        
        Args:
            locations: List of location dictionaries
            output_path: Path to save the map (optional)
            
        Returns:
            Path to the saved map
        """
        try:
            # Validate input
            if not isinstance(locations, list) or not locations:
                raise ValueError("Locations must be a non-empty list")
            
            # Create a map centered on the first location or Dubai center
            if "latitude" in locations[0] and "longitude" in locations[0]:
                center = [locations[0]["latitude"], locations[0]["longitude"]]
            else:
                # Default to Dubai center
                center = [25.2048, 55.2708]
            
            # Create map
            m = folium.Map(location=center, zoom_start=12)
            
            # Add markers for each location
            for i, loc in enumerate(locations):
                if "latitude" not in loc or "longitude" not in loc:
                    continue
                
                # Get location details
                lat = loc["latitude"]
                lng = loc["longitude"]
                name = loc.get("name", f"Location {i+1}")
                desc = loc.get("description", "")
                color = loc.get("color", "red")
                
                # Create popup content
                popup_content = f"<b>{name}</b><br>{desc}"
                
                # Add marker
                folium.Marker(
                    location=[lat, lng],
                    popup=popup_content,
                    tooltip=name,
                    icon=folium.Icon(color=color)
                ).add_to(m)
            
            # Save map
            if output_path is None:
                output_path = os.path.join(self.output_dir, "locations_map.html")
            
            m.save(output_path)
            
            return output_path
        
        except Exception as e:
            print(f"Error creating map: {str(e)}")
            return None

    def filter_locations(self, locations, constraints):
        """Filter locations based on constraints
        
        Args:
            locations: List of location dictionaries
            constraints: List of constraint dictionaries
            
        Returns:
            Filtered locations and evaluation results
        """
        try:
            # Validate input
            if not isinstance(locations, list):
                raise ValueError("Locations must be a list")
            
            if not isinstance(constraints, list):
                raise ValueError("Constraints must be a list")
            
            # Evaluate each location against constraints
            evaluation_results = []
            
            for location in locations:
                result = {
                    "location": location,
                    "constraints_met": [],
                    "constraints_not_met": [],
                    "score": 0
                }
                
                for constraint in constraints:
                    constraint_result = self._evaluate_constraint(location, constraint)
                    
                    if constraint_result["met"]:
                        result["constraints_met"].append({
                            "constraint": constraint,
                            "details": constraint_result["details"]
                        })
                    else:
                        result["constraints_not_met"].append({
                            "constraint": constraint,
                            "details": constraint_result["details"]
                        })
                
                # Calculate score
                if constraints:
                    result["score"] = len(result["constraints_met"]) / len(constraints) * 100
                
                evaluation_results.append(result)
            
            # Sort by score (descending)
            evaluation_results.sort(key=lambda x: x["score"], reverse=True)
            
            # Extract filtered locations
            filtered_locations = [result["location"] for result in evaluation_results if result["score"] > 0]
            
            return {
                "filtered_locations": filtered_locations,
                "evaluation_results": evaluation_results
            }
        
        except Exception as e:
            print(f"Error filtering locations: {str(e)}")
            return {
                "filtered_locations": [],
                "evaluation_results": []
            }

    def _evaluate_constraint(self, location, constraint):
        """Evaluate a location against a constraint
        
        Args:
            location: Location dictionary
            constraint: Constraint dictionary
            
        Returns:
            Evaluation result
        """
        try:
            # Validate constraint
            if not isinstance(constraint, dict) or "type" not in constraint:
                return {"met": False, "details": "Invalid constraint"}
            
            constraint_type = constraint["type"]
            
            # Handle different constraint types
            if constraint_type == "proximity":
                return self._evaluate_proximity_constraint(location, constraint)
            elif constraint_type == "attribute":
                return self._evaluate_attribute_constraint(location, constraint)
            elif constraint_type == "count":
                return self._evaluate_count_constraint(location, constraint)
            else:
                return {"met": False, "details": f"Unknown constraint type: {constraint_type}"}
        
        except Exception as e:
            return {"met": False, "details": f"Error evaluating constraint: {str(e)}"}

    def _evaluate_proximity_constraint(self, location, constraint):
        """Evaluate a proximity constraint
        
        Args:
            location: Location dictionary
            constraint: Constraint dictionary
            
        Returns:
            Evaluation result
        """
        # Check required parameters
        if "latitude" not in constraint or "longitude" not in constraint or "distance_km" not in constraint:
            return {"met": False, "details": "Missing required parameters"}
        
        # Get location coordinates
        loc_lat = location.get("latitude")
        loc_lng = location.get("longitude")
        
        if loc_lat is None or loc_lng is None:
            return {"met": False, "details": "Location missing coordinates"}
        
        # Get constraint parameters
        constraint_lat = constraint["latitude"]
        constraint_lng = constraint["longitude"]
        max_distance_km = constraint["distance_km"]
        
        # Calculate distance
        distance_km = self._calculate_distance(loc_lat, loc_lng, constraint_lat, constraint_lng)
        
        # Check if constraint is met
        if distance_km <= max_distance_km:
            return {"met": True, "details": f"Distance: {distance_km:.2f} km (max: {max_distance_km} km)"}
        else:
            return {"met": False, "details": f"Distance: {distance_km:.2f} km (max: {max_distance_km} km)"}

    def _evaluate_attribute_constraint(self, location, constraint):
        """Evaluate an attribute constraint
        
        Args:
            location: Location dictionary
            constraint: Constraint dictionary
            
        Returns:
            Evaluation result
        """
        # Check required parameters
        if "attribute" not in constraint or "value" not in constraint:
            return {"met": False, "details": "Missing required parameters"}
        
        # Get constraint parameters
        attribute = constraint["attribute"]
        value = constraint["value"]
        operator = constraint.get("operator", "equals")
        
        # Check if location has the attribute
        if attribute not in location:
            return {"met": False, "details": f"Location missing attribute: {attribute}"}
        
        # Get location attribute value
        loc_value = location[attribute]
        
        # Evaluate based on operator
        if operator == "equals":
            if loc_value == value:
                return {"met": True, "details": f"{attribute}: {loc_value} equals {value}"}
            else:
                return {"met": False, "details": f"{attribute}: {loc_value} does not equal {value}"}
        
        elif operator == "not_equals":
            if loc_value != value:
                return {"met": True, "details": f"{attribute}: {loc_value} not equals {value}"}
            else:
                return {"met": False, "details": f"{attribute}: {loc_value} equals {value}"}
        
        elif operator == "greater_than":
            if loc_value > value:
                return {"met": True, "details": f"{attribute}: {loc_value} > {value}"}
            else:
                return {"met": False, "details": f"{attribute}: {loc_value} <= {value}"}
        
        elif operator == "less_than":
            if loc_value < value:
                return {"met": True, "details": f"{attribute}: {loc_value} < {value}"}
            else:
                return {"met": False, "details": f"{attribute}: {loc_value} >= {value}"}
        
        elif operator == "contains":
            if isinstance(loc_value, str) and isinstance(value, str) and value.lower() in loc_value.lower():
                return {"met": True, "details": f"{attribute}: '{loc_value}' contains '{value}'"}
            else:
                return {"met": False, "details": f"{attribute}: '{loc_value}' does not contain '{value}'"}
        
        else:
            return {"met": False, "details": f"Unknown operator: {operator}"}

    def _evaluate_count_constraint(self, location, constraint):
        """Evaluate a count constraint
        
        Args:
            location: Location dictionary
            constraint: Constraint dictionary
            
        Returns:
            Evaluation result
        """
        # Check required parameters
        if "data_type" not in constraint or "max_count" not in constraint:
            return {"met": False, "details": "Missing required parameters"}
        
        # Get constraint parameters
        data_type = constraint["data_type"]
        max_count = constraint["max_count"]
        filters = constraint.get("filters", {})
        radius_km = constraint.get("radius_km", 1.0)
        
        # Check if data type is available
        if data_type not in self.data_sources:
            return {"met": False, "details": f"Data type not available: {data_type}"}
        
        # Get location coordinates
        loc_lat = location.get("latitude")
        loc_lng = location.get("longitude")
        
        if loc_lat is None or loc_lng is None:
            return {"met": False, "details": "Location missing coordinates"}
        
        # Get data source
        data = self.data_sources[data_type]
        
        # Count items within radius
        count = 0
        
        if data_type == "restaurants":
            # Filter by distance
            for _, row in data.iterrows():
                if "latitude" in row and "longitude" in row:
                    distance_km = self._calculate_distance(loc_lat, loc_lng, row["latitude"], row["longitude"])
                    
                    if distance_km <= radius_km:
                        # Check additional filters
                        match = True
                        for key, value in filters.items():
                            if key in row and row[key] != value:
                                match = False
                                break
                        
                        if match:
                            count += 1
        
        elif data_type in ["districts", "roads"]:
            # For geospatial data
            if isinstance(data, gpd.GeoDataFrame):
                # Create a point for the location
                point = Point(loc_lng, loc_lat)
                
                # Buffer the point by radius
                buffer = point.buffer(radius_km / 111.0)  # Approximate conversion to degrees
                
                # Count intersections
                for _, row in data.iterrows():
                    if row.geometry.intersects(buffer):
                        # Check additional filters
                        match = True
                        for key, value in filters.items():
                            if key in row and row[key] != value:
                                match = False
                                break
                        
                        if match:
                            count += 1
        
        # Check if constraint is met
        if count <= max_count:
            return {"met": True, "details": f"Count: {count} (max: {max_count})"}
        else:
            return {"met": False, "details": f"Count: {count} (max: {max_count})"}

    def _calculate_distance(self, lat1, lng1, lat2, lng2):
        """Calculate distance between two points in kilometers
        
        Args:
            lat1: Latitude of first point
            lng1: Longitude of first point
            lat2: Latitude of second point
            lng2: Longitude of second point
            
        Returns:
            Distance in kilometers
        """
        # Convert to numpy arrays for vectorized operations
        lat1, lng1, lat2, lng2 = map(np.radians, [lat1, lng1, lat2, lng2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlng = lng2 - lng1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371  # Radius of Earth in kilometers
        
        return c * r

    def create_visualization(self, data, visualization_type, output_path=None):
        """Create a visualization
        
        Args:
            data: Data to visualize
            visualization_type: Type of visualization
            output_path: Path to save the visualization (optional)
            
        Returns:
            Path to the saved visualization
        """
        try:
            # Validate input
            if not data:
                raise ValueError("Data is empty")
            
            # Create output path if not provided
            if output_path is None:
                output_path = os.path.join(self.output_dir, f"{visualization_type}_visualization.png")
            
            # Create visualization based on type
            if visualization_type == "heatmap":
                return self._create_heatmap(data, output_path)
            elif visualization_type == "bar_chart":
                return self._create_bar_chart(data, output_path)
            elif visualization_type == "scatter_plot":
                return self._create_scatter_plot(data, output_path)
            else:
                raise ValueError(f"Unknown visualization type: {visualization_type}")
        
        except Exception as e:
            print(f"Error creating visualization: {str(e)}")
            return None

    def _create_heatmap(self, data, output_path):
        """Create a heatmap visualization
        
        Args:
            data: Data to visualize
            output_path: Path to save the visualization
            
        Returns:
            Path to the saved visualization
        """
        # Validate data format
        if not isinstance(data, dict) or "values" not in data or "labels" not in data:
            raise ValueError("Data must be a dictionary with 'values' and 'labels'")
        
        # Extract data
        values = data["values"]
        labels = data["labels"]
        title = data.get("title", "Heatmap")
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        plt.imshow(values, cmap="YlOrRd")
        
        # Add labels
        if "x_labels" in data and "y_labels" in data:
            plt.xticks(range(len(data["x_labels"])), data["x_labels"], rotation=45)
            plt.yticks(range(len(data["y_labels"])), data["y_labels"])
        
        # Add colorbar
        plt.colorbar(label=data.get("colorbar_label", "Value"))
        
        # Add title
        plt.title(title)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        return output_path

    def _create_bar_chart(self, data, output_path):
        """Create a bar chart visualization
        
        Args:
            data: Data to visualize
            output_path: Path to save the visualization
            
        Returns:
            Path to the saved visualization
        """
        # Validate data format
        if not isinstance(data, dict) or "values" not in data or "labels" not in data:
            raise ValueError("Data must be a dictionary with 'values' and 'labels'")
        
        # Extract data
        values = data["values"]
        labels = data["labels"]
        title = data.get("title", "Bar Chart")
        xlabel = data.get("xlabel", "")
        ylabel = data.get("ylabel", "")
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Create bar chart
        plt.bar(labels, values)
        
        # Add labels
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        
        # Rotate x-axis labels if there are many
        if len(labels) > 5:
            plt.xticks(rotation=45)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        return output_path

    def _create_scatter_plot(self, data, output_path):
        """Create a scatter plot visualization
        
        Args:
            data: Data to visualize
            output_path: Path to save the visualization
            
        Returns:
            Path to the saved visualization
        """
        # Validate data format
        if not isinstance(data, dict) or "x" not in data or "y" not in data:
            raise ValueError("Data must be a dictionary with 'x' and 'y'")
        
        # Extract data
        x = data["x"]
        y = data["y"]
        title = data.get("title", "Scatter Plot")
        xlabel = data.get("xlabel", "")
        ylabel = data.get("ylabel", "")
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Create scatter plot
        plt.scatter(x, y)
        
        # Add labels
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        
        # Add trend line if requested
        if data.get("trend_line", False):
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            plt.plot(x, p(x), "r--")
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        return output_path


# Example usage
if __name__ == "__main__":
    processor = DataProcessor()
    
    # Create sample locations
    locations = [
        {"latitude": 25.2048, "longitude": 55.2708, "name": "Downtown Dubai"},
        {"latitude": 25.1972, "longitude": 55.2744, "name": "Business Bay"}
    ]
    
    # Create map
    map_path = processor.create_map(locations)
    print(f"Map saved to: {map_path}")
