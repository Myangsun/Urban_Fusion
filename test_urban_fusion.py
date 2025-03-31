"""
UrbanFusion Prototype - Testing Module
This file implements testing functionality for the UrbanFusion system
"""

import os
import json
import pandas as pd
import unittest
from dotenv import load_dotenv

# Import UrbanFusion components
from urban_fusion import UrbanFusionSystem
from data_processor import DataProcessor
from multimodal_database import MultimodalDatabase
from app import UrbanFusionApp

# Load environment variables
load_dotenv()

class UrbanFusionTester:
    """Class for testing the UrbanFusion system"""
    
    def __init__(self, data_dir="data"):
        """Initialize the tester"""
        self.data_dir = data_dir
        self.app = UrbanFusionApp(data_dir=data_dir)
        
    def run_tests(self):
        """Run all tests"""
        print("Running UrbanFusion tests...")
        
        # Test components
        self.test_data_processor()
        self.test_multimodal_database()
        self.test_agent_framework()
        
        # Test integration
        self.test_query_processing()
        
        print("All tests completed.")
    
    def test_data_processor(self):
        """Test the DataProcessor component"""
        print("\nTesting DataProcessor...")
        
        # Test data loading
        csv_path = os.path.join(self.data_dir, "sample", "talabat_sample.csv")
        df = self.app.data_processor.load_csv_data(csv_path)
        assert not df.empty, "CSV data loading failed"
        print("✓ Data loading test passed")
        
        # Test data processing
        results = self.app.data_processor.process_order_data(df)
        assert "vendor_stats" in results, "Data processing failed"
        assert "cuisine_stats" in results, "Cuisine statistics missing"
        print("✓ Data processing test passed")
        
        # Test constraint analysis
        sample_locations = [
            {"lat": 25.2048, "lng": 55.2708, "rent": 5000, "competition_count": 3, "score": 0.8, "delivery_radius_check": True},
            {"lat": 25.2200, "lng": 55.3000, "rent": 7000, "competition_count": 1, "score": 0.9, "delivery_radius_check": True}
        ]
        
        sample_constraints = {
            "max_rent": 6000,
            "max_competition": 4
        }
        
        filtered = self.app.data_processor.analyze_constraints(sample_locations, sample_constraints)
        assert len(filtered) == 1, "Constraint analysis failed"
        print("✓ Constraint analysis test passed")
        
        print("DataProcessor tests passed.")
    
    def test_multimodal_database(self):
        """Test the MultimodalDatabase component"""
        print("\nTesting MultimodalDatabase...")
        
        # Test database initialization
        db = self.app.database
        assert "gis" in db.collections, "Database initialization failed"
        
        # Test adding data
        db.add_gis_data("test1", {"name": "Test Point", "coordinates": [55.2708, 25.2048]}, [0.1, 0.2, 0.3])
        assert "test1" in db.collections["gis"], "Adding GIS data failed"
        
        db.add_csv_data("test2", {"value": 42}, [0.4, 0.5, 0.6])
        assert "test2" in db.collections["csv"], "Adding CSV data failed"
        
        # Test search (simplified)
        results = db.search([0.1, 0.2, 0.3], "gis")
        assert len(results) > 0, "Database search failed"
        
        print("MultimodalDatabase tests passed.")
    
    def test_agent_framework(self):
        """Test the agent framework"""
        print("\nTesting agent framework...")
        
        # Test system initialization
        system = self.app.system
        assert system.coordinator is not None, "Coordinator initialization failed"
        assert len(system.specialized_agents) == 4, "Specialized agents initialization failed"
        assert len(system.tools) > 0, "Tools initialization failed"
        
        print("Agent framework tests passed.")
    
    def test_query_processing(self):
        """Test query processing"""
        print("\nTesting query processing...")
        
        # Test simple query
        query = "Find restaurant locations in Dubai near Dubai Mall"
        result = self.app.process_query(query)
        
        assert "query" in result, "Query processing failed"
        assert "coordinator_analysis" in result, "Coordinator analysis missing"
        assert "potential_locations" in result, "Potential locations missing"
        assert "explanation" in result, "Explanation missing"
        
        print("Query processing tests passed.")

# Unit tests
class TestUrbanFusion(unittest.TestCase):
    """Unit tests for the UrbanFusion system"""
    
    def setUp(self):
        """Set up the test environment"""
        self.data_dir = "data"
        self.data_processor = DataProcessor(data_dir=self.data_dir)
        self.database = MultimodalDatabase(db_path=os.path.join(self.data_dir, "embeddings"))
    
    def test_data_processor_constraint_analysis(self):
        """Test constraint analysis in DataProcessor"""
        locations = [
            {"lat": 25.2048, "lng": 55.2708, "rent": 5000, "competition_count": 3, "score": 0.8, "delivery_radius_check": True},
            {"lat": 25.2200, "lng": 55.3000, "rent": 7000, "competition_count": 1, "score": 0.9, "delivery_radius_check": True}
        ]
        
        constraints = {
            "max_rent": 6000,
            "max_competition": 4
        }
        
        filtered = self.data_processor.analyze_constraints(locations, constraints)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["rent"], 5000)
    
    def test_data_processor_metrics_calculation(self):
        """Test metrics calculation in DataProcessor"""
        filtered_locations = [{"score": 0.8}, {"score": 0.9}]
        all_locations = [{"score": 0.8}, {"score": 0.9}, {"score": 0.6}, {"score": 0.7}]
        
        metrics = self.data_processor.calculate_metrics(filtered_locations, all_locations)
        self.assertEqual(metrics["total_locations"], 4)
        self.assertEqual(metrics["filtered_locations"], 2)
        self.assertEqual(metrics["pass_rate"], 0.5)
        self.assertAlmostEqual(metrics["avg_score"], 0.85)
    
    def test_multimodal_database_operations(self):
        """Test MultimodalDatabase operations"""
        # Test adding and retrieving data
        self.database.add_gis_data("test_point", {"name": "Test Point", "coordinates": [55.2708, 25.2048]}, [0.1, 0.2, 0.3])
        self.assertIn("test_point", self.database.collections["gis"])
        
        # Test search
        results = self.database.search([0.1, 0.2, 0.3], "gis")
        self.assertTrue(len(results) > 0)

# Example usage
if __name__ == "__main__":
    # Run functional tests
    tester = UrbanFusionTester()
    tester.run_tests()
    
    # Run unit tests
    unittest.main()
