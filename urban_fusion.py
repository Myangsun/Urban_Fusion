"""
UrbanFusion - Urban Site Selection System
This file implements the core UrbanFusion system with LangChain agents
"""

import os
import json
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, Polygon
import folium
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Union
import time
import tempfile
from dotenv import load_dotenv

# LangChain imports
from langchain.agents import AgentExecutor
from langchain_openai.agents import create_openai_functions_agent
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool
from langchain_core.messages import AIMessage, HumanMessage

# Import multimodal database
from multimodal_database import MultimodalDatabase

# Load environment variables
load_dotenv()

class UrbanFusionSystem:
    """UrbanFusion system for urban site selection"""

    def __init__(self, data_dir="data", model_name="gpt-4"):
        """Initialize the UrbanFusion system
        
        Args:
            data_dir: Directory for data storage
            model_name: Name of the OpenAI model to use
        """
        self.data_dir = data_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.join(data_dir, "output"), exist_ok=True)
        
        # Initialize LLM
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        
        # Initialize multimodal database
        self.db = MultimodalDatabase(data_dir=data_dir)
        
        # Load data sources
        self.data_sources = self._load_data_sources()
        
        # Initialize tools
        self.tools = self._create_tools()
        
        # Initialize agents
        self.agents = self._create_agents()

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
                
                # Add to multimodal database
                self.db.add_geospatial_data(data_sources["districts"], id_column="name")
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

    def _create_tools(self):
        """Create tools for the agents
        
        Returns:
            List of tools
        """
        tools = [
            Tool(
                name="generate_map",
                func=self._generate_map,
                description="Generate an interactive map with locations. Input should be a list of dictionaries with 'latitude', 'longitude', 'name', and optional 'description' and 'color'."
            ),
            Tool(
                name="query_database",
                func=self._query_database,
                description="Query the database for information. Input should be a dictionary with 'query' (string), 'data_type' (string: 'restaurants', 'districts', 'roads', 'demographics'), and optional 'filters' (dictionary)."
            ),
            Tool(
                name="analyze_constraints",
                func=self._analyze_constraints,
                description="Analyze locations against constraints. Input should be a dictionary with 'locations' (list of location dictionaries) and 'constraints' (list of constraint dictionaries with 'type', 'value', etc.)."
            ),
            Tool(
                name="generate_code",
                func=self._generate_code,
                description="Generate Python code to solve a specific problem. Input should be a string describing the code generation task."
            ),
            Tool(
                name="execute_code",
                func=self._execute_code,
                description="Execute Python code and return the results. Input should be a string containing Python code."
            ),
            Tool(
                name="fetch_satellite_image",
                func=self._fetch_satellite_image,
                description="Fetch a satellite image for a specific location. Input should be a dictionary with 'latitude' and 'longitude', and optional 'zoom' (default 18)."
            )
        ]
        
        return tools

    def _create_agents(self):
        """Create agents for the UrbanFusion system
        
        Returns:
            Dictionary of agent executors
        """
        agents = {}
        
        # Coordinator agent
        coordinator_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are the Coordinator Agent for the UrbanFusion system, an urban site selection tool.
            Your role is to:
            1. Analyze user queries to identify constraints and requirements
            2. Break down complex site selection problems into steps
            3. Coordinate with specialized agents to find suitable locations
            4. Ensure all constraints are properly evaluated
            5. Provide clear explanations of the results
            
            Available data sources:
            - Restaurants: Information about existing restaurants
            - Districts: Geospatial boundaries of Dubai districts
            - Roads: Road network information
            - Demographics: Population and demographic data by district
            - Satellite imagery: Visual data of locations
            
            Constraint categories:
            - Geospatial: Location, proximity, zoning, visibility
            - Temporal: Peak hours, seasonality
            - Economic: Cost, revenue, ROI
            - Market: Competition, demand
            - Demographic: Population density, age groups, income levels
            - Operational: Logistics, access, compliance
            
            Focus on constraint-based site selection rather than optimization.
            """),
            ("human", "{input}")
        ])
        
        coordinator_agent = create_openai_functions_agent(
            self.llm,
            self.tools,
            coordinator_prompt
        )
        
        agents["coordinator"] = AgentExecutor(
            agent=coordinator_agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
        )
        
        # GeoAgent
        geo_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are the GeoAgent for the UrbanFusion system, specializing in geospatial analysis.
            Your role is to:
            1. Analyze geospatial constraints in user queries
            2. Find locations that meet proximity requirements
            3. Evaluate zoning and land use constraints
            4. Calculate distances and spatial relationships
            5. Generate maps and visualizations
            
            Focus on constraint-based site selection rather than optimization.
            """),
            ("human", "{input}")
        ])
        
        geo_agent = create_openai_functions_agent(
            self.llm,
            self.tools,
            geo_prompt
        )
        
        agents["geo"] = AgentExecutor(
            agent=geo_agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
        )
        
        # Constraints Agent
        constraints_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are the Constraints Agent for the UrbanFusion system, specializing in constraint evaluation.
            Your role is to:
            1. Identify all constraints in user queries
            2. Formalize constraints into structured format
            3. Evaluate locations against constraints
            4. Rank locations by constraint satisfaction
            5. Provide detailed constraint analysis
            
            Constraint categories:
            - Geospatial: Location, proximity, zoning, visibility
            - Temporal: Peak hours, seasonality
            - Economic: Cost, revenue, ROI
            - Market: Competition, demand
            - Demographic: Population density, age groups, income levels
            - Operational: Logistics, access, compliance
            
            Focus on constraint-based site selection rather than optimization.
            """),
            ("human", "{input}")
        ])
        
        constraints_agent = create_openai_functions_agent(
            self.llm,
            self.tools,
            constraints_prompt
        )
        
        agents["constraints"] = AgentExecutor(
            agent=constraints_agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
        )
        
        # Evaluation Agent
        evaluation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are the Evaluation Agent for the UrbanFusion system, specializing in location evaluation.
            Your role is to:
            1. Evaluate candidate locations against all constraints
            2. Calculate scores for each location
            3. Rank locations by overall suitability
            4. Provide detailed evaluation reports
            5. Highlight strengths and weaknesses of each location
            
            Focus on constraint-based site selection rather than optimization.
            """),
            ("human", "{input}")
        ])
        
        evaluation_agent = create_openai_functions_agent(
            self.llm,
            self.tools,
            evaluation_prompt
        )
        
        agents["evaluation"] = AgentExecutor(
            agent=evaluation_agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
        )
        
        # Explanation Agent
        explanation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are the Explanation Agent for the UrbanFusion system, specializing in explaining results.
            Your role is to:
            1. Generate clear explanations of site selection results
            2. Explain how constraints were evaluated
            3. Justify location rankings
            4. Provide insights about selected locations
            5. Create visualizations to support explanations
            
            Focus on constraint-based site selection rather than optimization.
            """),
            ("human", "{input}")
        ])
        
        explanation_agent = create_openai_functions_agent(
            self.llm,
            self.tools,
            explanation_prompt
        )
        
        agents["explanation"] = AgentExecutor(
            agent=explanation_agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
        )
        
        return agents

    def process_query(self, query):
        """Process a user query
        
        Args:
            query: User query string
            
        Returns:
            Dictionary with results
        """
        # Step 1: Coordinator agent analyzes the query
        print("Step 1: Analyzing query with Coordinator Agent...")
        coordinator_result = self.agents["coordinator"].invoke({
            "input": f"Analyze this query and identify constraints: {query}"
        })
        
        # Step 2: Constraints agent formalizes constraints
        print("\nStep 2: Formalizing constraints with Constraints Agent...")
        constraints_result = self.agents["constraints"].invoke({
            "input": f"Formalize the constraints from this query: {query}\n\nCoordinator analysis: {coordinator_result['output']}"
        })
        
        # Step 3: GeoAgent finds candidate locations
        print("\nStep 3: Finding candidate locations with GeoAgent...")
        geo_result = self.agents["geo"].invoke({
            "input": f"Find candidate locations for this query: {query}\n\nFormalized constraints: {constraints_result['output']}"
        })
        
        # Step 4: Evaluation agent evaluates locations
        print("\nStep 4: Evaluating locations with Evaluation Agent...")
        evaluation_result = self.agents["evaluation"].invoke({
            "input": f"Evaluate these candidate locations against the constraints: {geo_result['output']}\n\nFormalized constraints: {constraints_result['output']}"
        })
        
        # Step 5: Explanation agent generates explanation
        print("\nStep 5: Generating explanation with Explanation Agent...")
        explanation_result = self.agents["explanation"].invoke({
            "input": f"Generate an explanation for the results of this query: {query}\n\nEvaluation results: {evaluation_result['output']}"
        })
        
        # Extract locations and map path from the results
        locations = self._extract_locations(evaluation_result['output'])
        map_path = self._extract_map_path(explanation_result['output'])
        
        # Return the results
        return {
            "query": query,
            "locations": locations,
            "explanation": explanation_result['output'],
            "map_path": map_path
        }

    def _extract_locations(self, evaluation_output):
        """Extract locations from evaluation output
        
        Args:
            evaluation_output: Output from evaluation agent
            
        Returns:
            List of location dictionaries
        """
        # This is a simple extraction - in a real implementation, this would be more robust
        locations = []
        
        try:
            # Try to find a JSON block in the output
            if "```json" in evaluation_output and "```" in evaluation_output.split("```json")[1]:
                json_str = evaluation_output.split("```json")[1].split("```")[0].strip()
                data = json.loads(json_str)
                if isinstance(data, list):
                    locations = data
                elif isinstance(data, dict) and "locations" in data:
                    locations = data["locations"]
            
            # If no JSON block, try to parse the text
            elif "latitude" in evaluation_output and "longitude" in evaluation_output:
                # Simple regex-like extraction
                lines = evaluation_output.split("\n")
                current_location = {}
                
                for line in lines:
                    if "Location" in line and ":" in line and current_location:
                        locations.append(current_location)
                        current_location = {}
                    
                    if ":" in line:
                        key, value = line.split(":", 1)
                        key = key.strip().lower().replace(" ", "_")
                        value = value.strip()
                        
                        if key in ["latitude", "longitude"]:
                            try:
                                value = float(value)
                            except:
                                continue
                        
                        current_location[key] = value
                
                if current_location:
                    locations.append(current_location)
        except Exception as e:
            print(f"Error extracting locations: {str(e)}")
        
        # Ensure each location has required fields
        valid_locations = []
        for loc in locations:
            if "latitude" in loc and "longitude" in loc:
                if "name" not in loc:
                    loc["name"] = f"Location {len(valid_locations) + 1}"
                valid_locations.append(loc)
        
        return valid_locations

    def _extract_map_path(self, explanation_output):
        """Extract map path from explanation output
        
        Args:
            explanation_output: Output from explanation agent
            
        Returns:
            Path to the map file, or None if not found
        """
        # Look for map path in the output
        if "map saved to:" in explanation_output.lower():
            lines = explanation_output.split("\n")
            for line in lines:
                if "map saved to:" in line.lower():
                    parts = line.split(":", 1)
                    if len(parts) > 1:
                        path = parts[1].strip()
                        if os.path.exists(path):
                            return path
        
        # Check for default map path
        default_path = os.path.join(self.data_dir, "output", "locations_map.html")
        if os.path.exists(default_path):
            return default_path
        
        return None

    def _generate_map(self, locations):
        """Generate an interactive map with locations
        
        Args:
            locations: List of location dictionaries
            
        Returns:
            Path to the saved map
        """
        try:
            # Validate input
            if not isinstance(locations, list):
                return {"error": "Input must be a list of locations"}
            
            # Create a map centered on the first location or Dubai center
            if locations and "latitude" in locations[0] and "longitude" in locations[0]:
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
            map_path = os.path.join(self.data_dir, "output", "locations_map.html")
            m.save(map_path)
            
            return {"result": f"Map saved to: {map_path}", "map_path": map_path}
        
        except Exception as e:
            return {"error": f"Error generating map: {str(e)}"}

    def _query_database(self, query_params):
        """Query the database for information
        
        Args:
            query_params: Dictionary with query parameters
            
        Returns:
            Query results
        """
        try:
            # Validate input
            if not isinstance(query_params, dict):
                return {"error": "Input must be a dictionary with query parameters"}
            
            if "query" not in query_params:
                return {"error": "Query parameter 'query' is required"}
            
            if "data_type" not in query_params:
                return {"error": "Query parameter 'data_type' is required"}
            
            # Extract parameters
            query_text = query_params["query"]
            data_type = query_params["data_type"]
            filters = query_params.get("filters", {})
            
            # Check if data type is available
            if data_type not in self.data_sources and data_type != "satellite_images":
                return {"error": f"Data type '{data_type}' not available. Available types: {list(self.data_sources.keys())} and 'satellite_images'"}
            
            # Handle different data types
            if data_type == "restaurants":
                df = self.data_sources["restaurants"]
                
                # Apply filters
                for key, value in filters.items():
                    if key in df.columns:
                        df = df[df[key] == value]
                
                # Simple text search
                if query_text:
                    df = df[df.apply(lambda row: any(str(query_text).lower() in str(val).lower() for val in row), axis=1)]
                
                # Convert to list of dictionaries
                results = df.head(10).to_dict(orient="records")
                
                return {"results": results, "count": len(results)}
            
            elif data_type == "districts":
                gdf = self.data_sources["districts"]
                
                # Apply filters
                for key, value in filters.items():
                    if key in gdf.columns:
                        gdf = gdf[gdf[key] == value]
                
                # Simple text search
                if query_text:
                    gdf = gdf[gdf.apply(lambda row: any(str(query_text).lower() in str(val).lower() for val in row if isinstance(val, (str, int, float))), axis=1)]
                
                # Convert to list of dictionaries (excluding geometry)
                results = []
                for _, row in gdf.iterrows():
                    result = {col: row[col] for col in gdf.columns if col != "geometry"}
                    result["centroid"] = row["geometry"].centroid.coords[0]
                    results.append(result)
                
                return {"results": results, "count": len(results)}
            
            elif data_type == "roads":
                gdf = self.data_sources["roads"]
                
                # Apply filters
                for key, value in filters.items():
                    if key in gdf.columns:
                        gdf = gdf[gdf[key] == value]
                
                # Simple text search
                if query_text:
                    gdf = gdf[gdf.apply(lambda row: any(str(query_text).lower() in str(val).lower() for val in row if isinstance(val, (str, int, float))), axis=1)]
                
                # Convert to list of dictionaries (excluding geometry)
                results = []
                for _, row in gdf.iterrows():
                    result = {col: row[col] for col in gdf.columns if col != "geometry"}
                    results.append(result)
                
                return {"results": results, "count": len(results)}
            
            elif data_type == "demographics":
                df = self.data_sources["demographics"]
                
                # Apply filters
                for key, value in filters.items():
                    if key in df.columns:
                        df = df[df[key] == value]
                
                # Simple text search
                if query_text:
                    df = df[df.apply(lambda row: any(str(query_text).lower() in str(val).lower() for val in row), axis=1)]
                
                # Convert to list of dictionaries
                results = df.to_dict(orient="records")
                
                return {"results": results, "count": len(results)}
            
            elif data_type == "satellite_images":
                # Use multimodal database to query satellite images
                if "latitude" in filters and "longitude" in filters:
                    location = Point(filters["longitude"], filters["latitude"])
                    radius_km = filters.get("radius_km", 1.0)
                    
                    results = self.db.query_satellite_images(
                        query=query_text,
                        location=location,
                        radius_km=radius_km,
                        n_results=5
                    )
                    
                    # Format results
                    formatted_results = []
                    for result in results:
                        formatted_results.append({
                            "id": result["id"],
                            "description": result.get("text", ""),
                            "image_path": result["image_path"],
                            "metadata": result["metadata"]
                        })
                    
                    return {"results": formatted_results, "count": len(formatted_results)}
                else:
                    results = self.db.query_satellite_images(query=query_text, n_results=5)
                    
                    # Format results
                    formatted_results = []
                    for result in results:
                        formatted_results.append({
                            "id": result["id"],
                            "description": result.get("text", ""),
                            "image_path": result["image_path"],
                            "metadata": result["metadata"]
                        })
                    
                    return {"results": formatted_results, "count": len(formatted_results)}
            
            return {"error": "Unknown data type"}
        
        except Exception as e:
            return {"error": f"Error querying database: {str(e)}"}

    def _analyze_constraints(self, analysis_params):
        """Analyze locations against constraints
        
        Args:
            analysis_params: Dictionary with locations and constraints
            
        Returns:
            Analysis results
        """
        try:
            # Validate input
            if not isinstance(analysis_params, dict):
                return {"error": "Input must be a dictionary with 'locations' and 'constraints'"}
            
            if "locations" not in analysis_params:
                return {"error": "Analysis parameter 'locations' is required"}
            
            if "constraints" not in analysis_params:
                return {"error": "Analysis parameter 'constraints' is required"}
            
            # Extract parameters
            locations = analysis_params["locations"]
            constraints = analysis_params["constraints"]
            
            # Validate locations and constraints
            if not isinstance(locations, list):
                return {"error": "Locations must be a list"}
            
            if not isinstance(constraints, list):
                return {"error": "Constraints must be a list"}
            
            # Analyze each location against each constraint
            results = []
            
            for location in locations:
                location_result = {
                    "location": location,
                    "constraints_met": [],
                    "constraints_not_met": [],
                    "score": 0.0
                }
                
                for constraint in constraints:
                    constraint_result = self._check_constraint(location, constraint)
                    
                    if constraint_result["met"]:
                        location_result["constraints_met"].append({
                            "constraint": constraint,
                            "details": constraint_result["details"]
                        })
                    else:
                        location_result["constraints_not_met"].append({
                            "constraint": constraint,
                            "details": constraint_result["details"]
                        })
                
                # Calculate score (percentage of constraints met)
                if constraints:
                    location_result["score"] = len(location_result["constraints_met"]) / len(constraints) * 100
                
                results.append(location_result)
            
            # Sort results by score (descending)
            results.sort(key=lambda x: x["score"], reverse=True)
            
            return {"results": results}
        
        except Exception as e:
            return {"error": f"Error analyzing constraints: {str(e)}"}

    def _check_constraint(self, location, constraint):
        """Check if a location meets a constraint
        
        Args:
            location: Location dictionary
            constraint: Constraint dictionary
            
        Returns:
            Dictionary with result and details
        """
        try:
            # Validate constraint
            if not isinstance(constraint, dict):
                return {"met": False, "details": "Invalid constraint format"}
            
            if "type" not in constraint:
                return {"met": False, "details": "Constraint missing 'type'"}
            
            # Extract constraint type and parameters
            constraint_type = constraint["type"]
            
            # Handle different constraint types
            if constraint_type == "proximity":
                # Proximity constraint (e.g., within X km of a point)
                if "latitude" not in constraint or "longitude" not in constraint or "distance_km" not in constraint:
                    return {"met": False, "details": "Proximity constraint missing required parameters"}
                
                # Calculate distance
                loc_lat = location.get("latitude")
                loc_lng = location.get("longitude")
                
                if loc_lat is None or loc_lng is None:
                    return {"met": False, "details": "Location missing coordinates"}
                
                constraint_lat = constraint["latitude"]
                constraint_lng = constraint["longitude"]
                max_distance_km = constraint["distance_km"]
                
                # Simple distance calculation (approximate)
                distance_km = self._calculate_distance(loc_lat, loc_lng, constraint_lat, constraint_lng)
                
                if distance_km <= max_distance_km:
                    return {"met": True, "details": f"Distance: {distance_km:.2f} km (max: {max_distance_km} km)"}
                else:
                    return {"met": False, "details": f"Distance: {distance_km:.2f} km (max: {max_distance_km} km)"}
            
            elif constraint_type == "attribute":
                # Attribute constraint (e.g., cuisine type, price level)
                if "attribute" not in constraint or "value" not in constraint:
                    return {"met": False, "details": "Attribute constraint missing required parameters"}
                
                attribute = constraint["attribute"]
                value = constraint["value"]
                operator = constraint.get("operator", "equals")
                
                if attribute not in location:
                    return {"met": False, "details": f"Location missing attribute '{attribute}'"}
                
                loc_value = location[attribute]
                
                # Compare based on operator
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
            
            elif constraint_type == "count":
                # Count constraint (e.g., number of competitors within X km)
                if "data_type" not in constraint or "max_count" not in constraint:
                    return {"met": False, "details": "Count constraint missing required parameters"}
                
                data_type = constraint["data_type"]
                max_count = constraint["max_count"]
                filters = constraint.get("filters", {})
                radius_km = constraint.get("radius_km", 1.0)
                
                # Get location coordinates
                loc_lat = location.get("latitude")
                loc_lng = location.get("longitude")
                
                if loc_lat is None or loc_lng is None:
                    return {"met": False, "details": "Location missing coordinates"}
                
                # Query database for nearby items
                query_result = self._query_database({
                    "query": "",
                    "data_type": data_type,
                    "filters": {**filters, "latitude": loc_lat, "longitude": loc_lng, "radius_km": radius_km}
                })
                
                if "error" in query_result:
                    return {"met": False, "details": f"Error querying database: {query_result['error']}"}
                
                count = query_result.get("count", 0)
                
                if count <= max_count:
                    return {"met": True, "details": f"Count: {count} (max: {max_count})"}
                else:
                    return {"met": False, "details": f"Count: {count} (max: {max_count})"}
            
            elif constraint_type == "satellite_image":
                # Satellite image constraint (e.g., green space visible)
                if "query" not in constraint:
                    return {"met": False, "details": "Satellite image constraint missing 'query'"}
                
                # Get location coordinates
                loc_lat = location.get("latitude")
                loc_lng = location.get("longitude")
                
                if loc_lat is None or loc_lng is None:
                    return {"met": False, "details": "Location missing coordinates"}
                
                # Query satellite images
                query_result = self._query_database({
                    "query": constraint["query"],
                    "data_type": "satellite_images",
                    "filters": {"latitude": loc_lat, "longitude": loc_lng, "radius_km": constraint.get("radius_km", 0.5)}
                })
                
                if "error" in query_result:
                    return {"met": False, "details": f"Error querying satellite images: {query_result['error']}"}
                
                count = query_result.get("count", 0)
                
                if count > 0:
                    return {"met": True, "details": f"Found {count} matching satellite images"}
                else:
                    return {"met": False, "details": "No matching satellite images found"}
            
            else:
                return {"met": False, "details": f"Unknown constraint type: {constraint_type}"}
        
        except Exception as e:
            return {"met": False, "details": f"Error checking constraint: {str(e)}"}

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

    def _generate_code(self, task_description):
        """Generate Python code to solve a specific problem
        
        Args:
            task_description: Description of the code generation task
            
        Returns:
            Generated code
        """
        try:
            # Create a prompt for code generation
            prompt = f"""
            Generate Python code to solve the following task:
            
            {task_description}
            
            The code should be self-contained and executable. Use pandas, geopandas, numpy, matplotlib, and folium libraries as needed.
            Only return the Python code without any explanations or markdown formatting.
            """
            
            # Use the LLM to generate code
            messages = [
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            # Extract code from response
            code = response.content
            
            # Clean up the code (remove markdown code blocks if present)
            if "```python" in code and "```" in code.split("```python")[1]:
                code = code.split("```python")[1].split("```")[0].strip()
            elif "```" in code:
                code = code.split("```")[1].split("```")[0].strip()
            
            return {"code": code}
        
        except Exception as e:
            return {"error": f"Error generating code: {str(e)}"}

    def _execute_code(self, code):
        """Execute Python code and return the results
        
        Args:
            code: Python code to execute
            
        Returns:
            Execution results
        """
        try:
            # Create a temporary file for the code
            with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
                temp_file_path = temp_file.name
                temp_file.write(code.encode())
            
            # Create a temporary file for output
            output_file_path = temp_file_path + ".out"
            
            # Execute the code with timeout
            command = f"python {temp_file_path} > {output_file_path} 2>&1"
            
            # Set environment variables
            env = os.environ.copy()
            env["PYTHONPATH"] = os.getcwd()
            env["DATA_DIR"] = self.data_dir
            
            # Execute with timeout
            import subprocess
            process = subprocess.Popen(command, shell=True, env=env)
            
            # Wait for process to complete with timeout
            try:
                process.wait(timeout=30)  # 30 second timeout
            except subprocess.TimeoutExpired:
                process.kill()
                return {"error": "Code execution timed out (30 seconds)"}
            
            # Read output
            with open(output_file_path, "r") as output_file:
                output = output_file.read()
            
            # Clean up temporary files
            os.unlink(temp_file_path)
            os.unlink(output_file_path)
            
            # Check for output files
            output_files = []
            for filename in os.listdir(self.data_dir):
                if filename.startswith("output_") and os.path.isfile(os.path.join(self.data_dir, filename)):
                    output_files.append(os.path.join(self.data_dir, filename))
            
            return {"output": output, "output_files": output_files}
        
        except Exception as e:
            return {"error": f"Error executing code: {str(e)}"}

    def _fetch_satellite_image(self, params):
        """Fetch a satellite image for a specific location
        
        Args:
            params: Dictionary with latitude, longitude, and optional zoom
            
        Returns:
            Path to the saved image
        """
        try:
            # Validate input
            if not isinstance(params, dict):
                return {"error": "Input must be a dictionary with 'latitude' and 'longitude'"}
            
            if "latitude" not in params or "longitude" not in params:
                return {"error": "Parameters 'latitude' and 'longitude' are required"}
            
            # Extract parameters
            latitude = params["latitude"]
            longitude = params["longitude"]
            zoom = params.get("zoom", 18)
            
            # Fetch image using multimodal database
            image_path = self.db.fetch_satellite_image(latitude, longitude, zoom)
            
            if not image_path:
                return {"error": "Failed to fetch satellite image"}
            
            return {"image_path": image_path}
        
        except Exception as e:
            return {"error": f"Error fetching satellite image: {str(e)}"}


# Example usage
if __name__ == "__main__":
    urban_fusion = UrbanFusionSystem()
    
    # Process a query
    result = urban_fusion.process_query(
        "Find restaurant locations in Dubai with a delivery radius of 3km of Dubai's city center, in residential areas, with less than 2 Italian cuisines in a 5km radius."
    )
    
    print("\nResults:")
    print(f"Found {len(result['locations'])} locations")
    print(f"Map: {result['map_path']}")
    print("\nExplanation:")
    print(result['explanation'])
