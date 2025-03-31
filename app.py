"""
UrbanFusion Prototype - Main Application
This file implements the main application for the UrbanFusion system
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Import UrbanFusion components
from urban_fusion import UrbanFusionSystem
from data_processor import DataProcessor
from multimodal_database import MultimodalDatabase

# Load environment variables
load_dotenv()

class UrbanFusionApp:
    """Main application class for the UrbanFusion system"""
    
    def __init__(self, data_dir="data"):
        """Initialize the UrbanFusion application"""
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(os.path.join(data_dir, "output"), exist_ok=True)
        
        # Initialize components
        self.system = UrbanFusionSystem()
        self.data_processor = DataProcessor(data_dir=data_dir)
        self.database = MultimodalDatabase(db_path=os.path.join(data_dir, "embeddings"))
        
        # Load sample data
        self._load_sample_data()
    
    def _load_sample_data(self):
        """Load sample data for the application"""
        # Create sample data directory
        sample_dir = os.path.join(self.data_dir, "sample")
        os.makedirs(sample_dir, exist_ok=True)
        
        # Create sample CSV data
        sample_csv_path = os.path.join(sample_dir, "talabat_sample.csv")
        if not os.path.exists(sample_csv_path):
            sample_data = pd.DataFrame({
                "vendor_location": ["POINT (55.2708 25.2048)", "POINT (55.3000 25.2200)", 
                                   "POINT (55.2600 25.1900)", "POINT (55.2900 25.2100)"],
                "customer_location": ["POINT (55.2800 25.2100)", "POINT (55.3100 25.2300)", 
                                     "POINT (55.2650 25.1950)", "POINT (55.2950 25.2150)"],
                "order_timestamp": ["2023-01-01 12:00:00", "2023-01-01 13:00:00", 
                                   "2023-01-01 14:00:00", "2023-01-01 15:00:00"],
                "cuisine_type": ["Italian", "Arabic", "Italian", "Chinese"],
                "delivery_time": [25, 30, 20, 35],
                "gmv_amount": [100, 150, 120, 200],
                "delivery_fee": [10, 15, 12, 18],
                "order_volume": [1, 1, 1, 2]
            })
            sample_data.to_csv(sample_csv_path, index=False)
            print(f"Created sample CSV data at {sample_csv_path}")
        
        # Create sample GeoJSON data
        sample_geojson_path = os.path.join(sample_dir, "dubai_sample.geojson")
        if not os.path.exists(sample_geojson_path):
            sample_geojson = {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"name": "Dubai Mall", "type": "shopping"},
                        "geometry": {"type": "Point", "coordinates": [55.2798, 25.1972]}
                    },
                    {
                        "type": "Feature",
                        "properties": {"name": "Burj Khalifa", "type": "landmark"},
                        "geometry": {"type": "Point", "coordinates": [55.2744, 25.1972]}
                    },
                    {
                        "type": "Feature",
                        "properties": {"name": "Dubai Marina", "type": "residential"},
                        "geometry": {"type": "Point", "coordinates": [55.1384, 25.0806]}
                    }
                ]
            }
            with open(sample_geojson_path, 'w') as f:
                json.dump(sample_geojson, f)
            print(f"Created sample GeoJSON data at {sample_geojson_path}")
    
    def process_query(self, query: str) -> dict:
        """Process a user query through the UrbanFusion system"""
        print(f"Processing query: {query}")
        
        # Process the query through the UrbanFusion system
        result = self.system.process_query(query)
        
        # Save the result to a file
        output_path = os.path.join(self.data_dir, "output", "query_result.json")
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"Query result saved to {output_path}")
        return result
    
    def analyze_data(self, csv_path: str = None) -> dict:
        """Analyze data using the DataProcessor"""
        if csv_path is None:
            csv_path = os.path.join(self.data_dir, "sample", "talabat_sample.csv")
        
        print(f"Analyzing data from {csv_path}")
        
        # Load and process the data
        df = self.data_processor.load_csv_data(csv_path)
        results = self.data_processor.process_order_data(df)
        
        # Create visualizations
        output_dir = os.path.join(self.data_dir, "output")
        
        # Create heatmap
        heatmap_path = os.path.join(output_dir, "order_heatmap.html")
        self.data_processor.create_heatmap(df, heatmap_path)
        
        # Create bar chart
        bar_chart_path = os.path.join(output_dir, "cuisine_orders.png")
        self.data_processor.generate_visualization(results, bar_chart_path, "bar")
        
        # Save the results to a file
        output_path = os.path.join(output_dir, "data_analysis.json")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Data analysis results saved to {output_path}")
        return {
            "analysis_results": results,
            "visualizations": {
                "heatmap": heatmap_path,
                "bar_chart": bar_chart_path
            }
        }
    
    def evaluate_locations(self, locations: list, constraints: dict) -> dict:
        """Evaluate locations against constraints"""
        print(f"Evaluating {len(locations)} locations against constraints")
        
        # Analyze constraints
        filtered_locations = self.data_processor.analyze_constraints(locations, constraints)
        
        # Calculate metrics
        metrics = self.data_processor.calculate_metrics(filtered_locations, locations)
        
        # Create geospatial visualization
        output_dir = os.path.join(self.data_dir, "output")
        map_path = os.path.join(output_dir, "location_map.html")
        self.data_processor.create_geospatial_visualization(filtered_locations, map_path)
        
        # Save the results to a file
        output_path = os.path.join(output_dir, "evaluation_results.json")
        result = {
            "filtered_locations": filtered_locations,
            "metrics": metrics,
            "visualization": map_path
        }
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"Evaluation results saved to {output_path}")
        return result
    
    def run_demo(self):
        """Run a demonstration of the UrbanFusion system"""
        print("Running UrbanFusion demonstration...")
        
        # Step 1: Process a sample query
        query = "Find optimal restaurant locations in Dubai with delivery radius of 3km, near residential areas, with low competition for Italian cuisine"
        query_result = self.process_query(query)
        print("\nQuery processing complete.")
        
        # Step 2: Analyze sample data
        analysis_result = self.analyze_data()
        print("\nData analysis complete.")
        
        # Step 3: Evaluate sample locations
        sample_locations = [
            {"lat": 25.2048, "lng": 55.2708, "rent": 5000, "competition_count": 3, "score": 0.8, "delivery_radius_check": True},
            {"lat": 25.2200, "lng": 55.3000, "rent": 7000, "competition_count": 1, "score": 0.9, "delivery_radius_check": True},
            {"lat": 25.1900, "lng": 55.2600, "rent": 6000, "competition_count": 5, "score": 0.6, "delivery_radius_check": False},
            {"lat": 25.1800, "lng": 55.2500, "rent": 4500, "competition_count": 2, "score": 0.7, "delivery_radius_check": True}
        ]
        
        sample_constraints = {
            "max_rent": 6000,
            "max_competition": 4,
            "delivery_radius": 3
        }
        
        evaluation_result = self.evaluate_locations(sample_locations, sample_constraints)
        print("\nLocation evaluation complete.")
        
        # Step 4: Print summary
        print("\n=== UrbanFusion Demonstration Summary ===")
        print(f"Query: {query}")
        print(f"Query result: {len(query_result)} components processed")
        print(f"Data analysis: {len(analysis_result['analysis_results'])} metrics calculated")
        print(f"Location evaluation: {evaluation_result['metrics']['filtered_locations']} locations passed constraints")
        print(f"Pass rate: {evaluation_result['metrics']['pass_rate']:.2f}")
        print("========================================")
        
        return {
            "query_result": query_result,
            "analysis_result": analysis_result,
            "evaluation_result": evaluation_result
        }

# Example usage
if __name__ == "__main__":
    app = UrbanFusionApp()
    app.run_demo()
