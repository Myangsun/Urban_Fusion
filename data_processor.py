"""
UrbanFusion Prototype - Data Processing Module
This file implements the data processing functionality for the UrbanFusion system
"""

import os
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
import folium
from typing import Dict, List, Any, Optional
import json

class DataProcessor:
    """Class for processing and analyzing data in the UrbanFusion system"""
    
    def __init__(self, data_dir="data"):
        """Initialize the data processor"""
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
    def load_csv_data(self, file_path: str) -> pd.DataFrame:
        """Load CSV data from file"""
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            print(f"Error loading CSV data: {e}")
            # Return empty DataFrame with expected columns as fallback
            return pd.DataFrame({
                "vendor_location": [],
                "customer_location": [],
                "order_timestamp": [],
                "cuisine_type": [],
                "delivery_time": [],
                "gmv_amount": [],
                "delivery_fee": [],
                "order_volume": []
            })
    
    def load_gis_data(self, file_path: str) -> gpd.GeoDataFrame:
        """Load GIS data from file"""
        try:
            return gpd.read_file(file_path)
        except Exception as e:
            print(f"Error loading GIS data: {e}")
            # Return empty GeoDataFrame as fallback
            return gpd.GeoDataFrame()
    
    def process_order_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Process order data to extract insights"""
        if df.empty:
            return {"error": "No data available"}
        
        # Convert point strings to coordinates
        # Example format: "POINT (55.2708 25.2048)"
        def extract_coords(point_str):
            try:
                coords = point_str.replace("POINT (", "").replace(")", "").split()
                return float(coords[0]), float(coords[1])
            except:
                return np.nan, np.nan
        
        # Extract vendor coordinates
        df['vendor_lon'], df['vendor_lat'] = zip(*df['vendor_location'].apply(extract_coords))
        
        # Extract customer coordinates
        df['customer_lon'], df['customer_lat'] = zip(*df['customer_location'].apply(extract_coords))
        
        # Convert timestamp to datetime
        df['order_timestamp'] = pd.to_datetime(df['order_timestamp'])
        
        # Aggregate by vendor location
        vendor_stats = df.groupby(['vendor_lon', 'vendor_lat']).agg({
            'order_volume': 'sum',
            'gmv_amount': 'sum',
            'delivery_time': 'mean',
            'delivery_fee': 'mean',
            'cuisine_type': lambda x: x.value_counts().index[0]  # Most common cuisine
        }).reset_index()
        
        # Aggregate by cuisine type
        cuisine_stats = df.groupby('cuisine_type').agg({
            'order_volume': 'sum',
            'gmv_amount': 'sum',
            'delivery_time': 'mean'
        }).reset_index()
        
        return {
            "vendor_stats": vendor_stats.to_dict(orient='records'),
            "cuisine_stats": cuisine_stats.to_dict(orient='records'),
            "total_orders": df.shape[0],
            "total_gmv": df['gmv_amount'].sum(),
            "avg_delivery_time": df['delivery_time'].mean()
        }
    
    def create_heatmap(self, df: pd.DataFrame, output_path: str) -> str:
        """Create a heatmap of order density"""
        if df.empty:
            return "Error: No data available for heatmap"
        
        # Create a map centered on Dubai
        dubai_map = folium.Map(location=[25.2048, 55.2708], zoom_start=12)
        
        # Extract customer coordinates
        if 'customer_lat' not in df.columns or 'customer_lon' not in df.columns:
            # Extract coordinates if not already done
            df['customer_lon'], df['customer_lat'] = zip(*df['customer_location'].apply(
                lambda x: extract_coords(x) if isinstance(x, str) else (np.nan, np.nan)
            ))
        
        # Filter out invalid coordinates
        valid_coords = df[['customer_lat', 'customer_lon']].dropna()
        
        # Add heatmap layer
        if not valid_coords.empty:
            from folium.plugins import HeatMap
            HeatMap(valid_coords.values).add_to(dubai_map)
        
        # Save map to file
        dubai_map.save(output_path)
        
        return output_path
    
    def analyze_constraints(self, locations: List[Dict[str, Any]], constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze locations against constraints"""
        filtered_locations = []
        
        for location in locations:
            # Check if location satisfies all constraints
            satisfies_all = True
            
            # Geospatial constraints
            if "delivery_radius" in constraints:
                # Check if location is within delivery radius of target areas
                # This is a simplified example - real implementation would be more complex
                if "delivery_radius_check" not in location or not location["delivery_radius_check"]:
                    satisfies_all = False
            
            # Economic constraints
            if "max_rent" in constraints and "rent" in location:
                if location["rent"] > constraints["max_rent"]:
                    satisfies_all = False
            
            # Market constraints
            if "max_competition" in constraints and "competition_count" in location:
                if location["competition_count"] > constraints["max_competition"]:
                    satisfies_all = False
            
            # If location satisfies all constraints, add to filtered list
            if satisfies_all:
                filtered_locations.append(location)
        
        return filtered_locations
    
    def calculate_metrics(self, filtered_locations: List[Dict[str, Any]], all_locations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate evaluation metrics"""
        if not all_locations:
            return {"error": "No locations to evaluate"}
        
        metrics = {
            "total_locations": len(all_locations),
            "filtered_locations": len(filtered_locations),
            "pass_rate": len(filtered_locations) / len(all_locations) if all_locations else 0,
        }
        
        # Calculate average scores if available
        if filtered_locations and "score" in filtered_locations[0]:
            metrics["avg_score"] = sum(loc["score"] for loc in filtered_locations) / len(filtered_locations)
        
        return metrics
    
    def generate_visualization(self, data: Dict[str, Any], output_path: str, chart_type: str = "bar") -> str:
        """Generate visualization based on data"""
        plt.figure(figsize=(10, 6))
        
        if chart_type == "bar" and "cuisine_stats" in data:
            cuisine_stats = pd.DataFrame(data["cuisine_stats"])
            cuisine_stats.plot(kind='bar', x='cuisine_type', y='order_volume')
            plt.title('Order Volume by Cuisine Type')
            plt.xlabel('Cuisine Type')
            plt.ylabel('Order Volume')
            
        elif chart_type == "scatter" and "vendor_stats" in data:
            vendor_stats = pd.DataFrame(data["vendor_stats"])
            plt.scatter(vendor_stats['gmv_amount'], vendor_stats['delivery_time'])
            plt.title('Delivery Time vs GMV Amount')
            plt.xlabel('GMV Amount')
            plt.ylabel('Avg Delivery Time (min)')
            
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        return output_path
    
    def create_geospatial_visualization(self, locations: List[Dict[str, Any]], output_path: str) -> str:
        """Create a geospatial visualization of locations"""
        # Create a map centered on Dubai
        dubai_map = folium.Map(location=[25.2048, 55.2708], zoom_start=12)
        
        # Add markers for each location
        for loc in locations:
            if "lat" in loc and "lng" in loc:
                popup_text = f"Score: {loc.get('score', 'N/A')}<br>"
                popup_text += f"Rent: {loc.get('rent', 'N/A')}<br>"
                popup_text += f"Competition: {loc.get('competition_count', 'N/A')}"
                
                folium.Marker(
                    location=[loc["lat"], loc["lng"]],
                    popup=popup_text,
                    icon=folium.Icon(color='green' if loc.get('score', 0) > 0.7 else 'orange')
                ).add_to(dubai_map)
        
        # Save map to file
        dubai_map.save(output_path)
        
        return output_path

# Helper function for extracting coordinates
def extract_coords(point_str):
    try:
        coords = point_str.replace("POINT (", "").replace(")", "").split()
        return float(coords[0]), float(coords[1])
    except:
        return np.nan, np.nan

# Example usage
if __name__ == "__main__":
    processor = DataProcessor()
    
    # Create sample data for testing
    sample_data = pd.DataFrame({
        "vendor_location": ["POINT (55.2708 25.2048)", "POINT (55.3000 25.2200)"],
        "customer_location": ["POINT (55.2800 25.2100)", "POINT (55.3100 25.2300)"],
        "order_timestamp": ["2023-01-01 12:00:00", "2023-01-01 13:00:00"],
        "cuisine_type": ["Italian", "Arabic"],
        "delivery_time": [25, 30],
        "gmv_amount": [100, 150],
        "delivery_fee": [10, 15],
        "order_volume": [1, 1]
    })
    
    # Process the data
    results = processor.process_order_data(sample_data)
    print(json.dumps(results, indent=2))
    
    # Create sample locations for testing
    sample_locations = [
        {"lat": 25.2048, "lng": 55.2708, "rent": 5000, "competition_count": 3, "score": 0.8},
        {"lat": 25.2200, "lng": 55.3000, "rent": 7000, "competition_count": 1, "score": 0.9},
        {"lat": 25.1900, "lng": 55.2600, "rent": 6000, "competition_count": 5, "score": 0.6}
    ]
    
    # Create sample constraints for testing
    sample_constraints = {
        "max_rent": 6000,
        "max_competition": 4
    }
    
    # Analyze constraints
    filtered_locations = processor.analyze_constraints(sample_locations, sample_constraints)
    print(f"Filtered locations: {len(filtered_locations)}")
    
    # Calculate metrics
    metrics = processor.calculate_metrics(filtered_locations, sample_locations)
    print(json.dumps(metrics, indent=2))
