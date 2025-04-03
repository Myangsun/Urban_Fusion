"""
UrbanFusion - Main Application
This file implements the main application for the UrbanFusion system
"""

import os
import sys
from dotenv import load_dotenv
from urban_fusion import UrbanFusionSystem
import webbrowser

# Load environment variables
load_dotenv()

def main():
    """Main application entry point"""
    print("=" * 80)
    print("UrbanFusion - Urban Site Selection System")
    print("=" * 80)
    
    # Initialize the UrbanFusion system
    print("\nInitializing UrbanFusion system...")
    urban_fusion = UrbanFusionSystem()
    
    # Interactive mode
    print("\nEntering interactive mode. Type 'exit' to quit.")
    print("\nExample query: Find restaurant locations in Dubai with a delivery radius of 3km of Dubai's city center, in residential areas, with less than 2 Italian cuisines in a 5km radius.")
    
    while True:
        # Get user query
        print("\n" + "-" * 80)
        query = input("Enter your query: ")
        
        # Check for exit command
        if query.lower() in ["exit", "quit", "q"]:
            print("Exiting UrbanFusion. Goodbye!")
            break
        
        # Process query
        print("\nProcessing query...")
        result = urban_fusion.process_query(query)
        
        # Display results
        print("\nResults:")
        print(f"Found {len(result['locations'])} locations")
        
        # Display locations
        if result['locations']:
            print("\nRecommended locations:")
            for i, location in enumerate(result['locations']):
                print(f"{i+1}. {location.get('name', 'Unnamed location')} ({location.get('latitude', 'N/A')}, {location.get('longitude', 'N/A')})")
                if 'description' in location:
                    print(f"   {location['description']}")
        else:
            print("\nNo locations found matching your criteria.")
        
        # Open map in browser if available
        if result['map_path'] and os.path.exists(result['map_path']):
            print(f"\nMap saved to: {result['map_path']}")
            print("Opening map in browser...")
            try:
                webbrowser.open('file://' + os.path.abspath(result['map_path']))
            except Exception as e:
                print(f"Error opening map in browser: {str(e)}")
                print(f"You can manually open the map at: {os.path.abspath(result['map_path'])}")

if __name__ == "__main__":
    main()
