"""
UrbanFusion - Multimodal Database
This file implements a multimodal database for the UrbanFusion system
with support for text, geospatial data, and satellite imagery from APIs
"""

import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import List, Dict, Any, Optional, Union
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
from PIL import Image
import base64
import io
import requests
import time
import urllib.parse

# Vector database
try:
    import chromadb
    from chromadb.utils import embedding_functions
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    print("Warning: ChromaDB not available. Vector search functionality will be limited.")


class MultimodalDatabase:
    """Multimodal database for the UrbanFusion system"""

    def __init__(self, data_dir="data"):
        """Initialize the multimodal database
        
        Args:
            data_dir: Directory for data storage
        """
        self.data_dir = data_dir
        self.embeddings_dir = os.path.join(data_dir, "embeddings")
        self.images_dir = os.path.join(data_dir, "images")
        
        # Create directories if they don't exist
        os.makedirs(self.embeddings_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        
        # Initialize vector database if available
        self.vector_db = None
        self.embedding_function = None
        if CHROMA_AVAILABLE:
            # Use default embedding function (all-MiniLM-L6-v2)
            self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
            self.vector_db = chromadb.PersistentClient(path=self.embeddings_dir)
            
            # Create collections if they don't exist
            try:
                self.text_collection = self.vector_db.get_or_create_collection(
                    name="text_data",
                    embedding_function=self.embedding_function
                )
                self.geo_collection = self.vector_db.get_or_create_collection(
                    name="geo_data",
                    embedding_function=self.embedding_function
                )
                self.image_collection = self.vector_db.get_or_create_collection(
                    name="image_data",
                    embedding_function=self.embedding_function
                )
            except Exception as e:
                print(f"Error initializing vector database: {str(e)}")
                self.vector_db = None
        
        # Data caches
        self.text_data_cache = {}
        self.geo_data_cache = {}
        self.image_data_cache = {}
        
        # Load existing satellite images if available
        self._load_existing_satellite_images()

    def _load_existing_satellite_images(self):
        """Load existing satellite images from the images directory"""
        if not os.path.exists(self.images_dir):
            return
            
        # Check for metadata file
        metadata_path = os.path.join(self.images_dir, "metadata.json")
        metadata_dict = {}
        
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata_dict = json.load(f)
            except Exception as e:
                print(f"Error loading metadata file: {str(e)}")
        
        # Load images
        image_count = 0
        for filename in os.listdir(self.images_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                image_path = os.path.join(self.images_dir, filename)
                
                # Get metadata for this image if available
                image_id = os.path.splitext(filename)[0]
                metadata = metadata_dict.get(filename, metadata_dict.get(image_id, {}))
                
                if not metadata:
                    # If no specific metadata, use filename as description
                    metadata = {"description": f"Satellite image: {filename}"}
                
                # Add to cache
                self.image_data_cache[image_id] = {
                    "text": f"Satellite image: {metadata.get('description', filename)}",
                    "metadata": metadata,
                    "image_path": image_path
                }
                
                # Add to vector database if available
                if self.vector_db is not None:
                    try:
                        self.image_collection.add(
                            documents=[self.image_data_cache[image_id]["text"]],
                            metadatas=[metadata],
                            ids=[image_id]
                        )
                    except Exception as e:
                        # Skip if already exists
                        pass
                
                image_count += 1
        
        if image_count > 0:
            print(f"Loaded {image_count} existing satellite images")

    def add_text_data(self, texts: List[str], metadatas: List[Dict[str, Any]], ids: List[str]) -> bool:
        """Add text data to the database
        
        Args:
            texts: List of text strings
            metadatas: List of metadata dictionaries
            ids: List of unique IDs
            
        Returns:
            Success status
        """
        # Cache the data
        for i, id in enumerate(ids):
            self.text_data_cache[id] = {
                "text": texts[i],
                "metadata": metadatas[i]
            }
        
        # Add to vector database if available
        if self.vector_db is not None:
            try:
                self.text_collection.add(
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids
                )
                return True
            except Exception as e:
                print(f"Error adding text data to vector database: {str(e)}")
                return False
        
        return True

    def add_geospatial_data(self, gdf: gpd.GeoDataFrame, id_column: str = None) -> bool:
        """Add geospatial data to the database
        
        Args:
            gdf: GeoDataFrame with geospatial data
            id_column: Column to use as ID (if None, generates IDs)
            
        Returns:
            Success status
        """
        if gdf.empty:
            return False
        
        # Generate IDs if not provided
        if id_column is None or id_column not in gdf.columns:
            ids = [f"geo_{i}" for i in range(len(gdf))]
        else:
            ids = [f"geo_{str(id)}" for id in gdf[id_column]]
        
        # Convert to text representations
        texts = []
        metadatas = []
        
        for i, row in gdf.iterrows():
            # Create text representation
            properties = ". ".join([f"{k}: {v}" for k, v in row.items() if k != 'geometry'])
            text = f"Geometry type: {row.geometry.geom_type}. Coordinates: {row.geometry.centroid}. {properties}"
            texts.append(text)
            
            # Create metadata
            metadata = {
                "type": "geospatial",
                "geometry_type": row.geometry.geom_type,
                "centroid": str(row.geometry.centroid),
                **{k: str(v) for k, v in row.items() if k != 'geometry'}
            }
            metadatas.append(metadata)
            
            # Cache the data
            self.geo_data_cache[ids[i]] = {
                "text": text,
                "metadata": metadata,
                "geometry": row.geometry
            }
        
        # Add to vector database if available
        if self.vector_db is not None:
            try:
                self.geo_collection.add(
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids
                )
                return True
            except Exception as e:
                print(f"Error adding geospatial data to vector database: {str(e)}")
                return False
        
        return True

    def fetch_satellite_image(self, latitude: float, longitude: float, zoom: int = 18, 
                             size: str = "600x600", api_type: str = "google_maps") -> Optional[str]:
        """Fetch a satellite image from an API
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            zoom: Zoom level (higher = more detailed)
            size: Image size in pixels (width x height)
            api_type: Type of API to use ('google_maps', 'mapbox', or 'bing')
            
        Returns:
            Path to saved image if successful, None otherwise
        """
        image_id = f"satellite_{latitude:.6f}_{longitude:.6f}_z{zoom}"
        image_path = os.path.join(self.images_dir, f"{image_id}.png")
        
        # Check if image already exists
        if os.path.exists(image_path):
            print(f"Using cached satellite image: {image_path}")
            return image_path
        
        try:
            if api_type == "google_maps":
                # Note: This requires an API key and billing account
                # For educational purposes only - replace with your own API key
                api_key = os.getenv("GOOGLE_MAPS_API_KEY", "")
                if not api_key:
                    print("Warning: No Google Maps API key found. Using static map fallback.")
                    return self._create_static_map(latitude, longitude, image_path)
                
                url = f"https://maps.googleapis.com/maps/api/staticmap?center={latitude},{longitude}&zoom={zoom}&size={size}&maptype=satellite&key={api_key}"
                
            elif api_type == "mapbox":
                # Note: This requires an API key
                # For educational purposes only - replace with your own API key
                api_key = os.getenv("MAPBOX_API_KEY", "")
                if not api_key:
                    print("Warning: No Mapbox API key found. Using static map fallback.")
                    return self._create_static_map(latitude, longitude, image_path)
                
                width, height = size.split("x")
                url = f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/{longitude},{latitude},{zoom}/{width}x{height}?access_token={api_key}"
                
            elif api_type == "bing":
                # Note: This requires an API key
                # For educational purposes only - replace with your own API key
                api_key = os.getenv("BING_MAPS_API_KEY", "")
                if not api_key:
                    print("Warning: No Bing Maps API key found. Using static map fallback.")
                    return self._create_static_map(latitude, longitude, image_path)
                
                url = f"https://dev.virtualearth.net/REST/v1/Imagery/Map/Aerial/{latitude},{longitude}/{zoom}?mapSize={size}&key={api_key}"
                
            else:
                print(f"Unknown API type: {api_type}")
                return self._create_static_map(latitude, longitude, image_path)
            
            # Download the image
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(image_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        f.write(chunk)
                
                print(f"Downloaded satellite image: {image_path}")
                
                # Create metadata
                metadata = {
                    "latitude": latitude,
                    "longitude": longitude,
                    "zoom": zoom,
                    "api_type": api_type,
                    "description": f"Satellite image at {latitude:.6f}, {longitude:.6f}",
                    "capture_date": time.strftime("%Y-%m-%d")
                }
                
                # Add to cache
                self.image_data_cache[image_id] = {
                    "text": f"Satellite image at {latitude:.6f}, {longitude:.6f}",
                    "metadata": metadata,
                    "image_path": image_path
                }
                
                # Add to vector database if available
                if self.vector_db is not None:
                    try:
                        self.image_collection.add(
                            documents=[self.image_data_cache[image_id]["text"]],
                            metadatas=[metadata],
                            ids=[image_id]
                        )
                    except Exception as e:
                        print(f"Error adding image to vector database: {str(e)}")
                
                # Update metadata file
                self._update_metadata_file(image_id, metadata)
                
                return image_path
            else:
                print(f"Error downloading image: {response.status_code}")
                return self._create_static_map(latitude, longitude, image_path)
                
        except Exception as e:
            print(f"Error fetching satellite image: {str(e)}")
            return self._create_static_map(latitude, longitude, image_path)

    def _create_static_map(self, latitude: float, longitude: float, image_path: str) -> str:
        """Create a static map image as a fallback
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            image_path: Path to save the image
            
        Returns:
            Path to saved image
        """
        try:
            # Create a simple map using matplotlib
            plt.figure(figsize=(6, 6))
            
            # Create a simple representation
            plt.scatter([longitude], [latitude], c='red', s=100, marker='o')
            plt.text(longitude, latitude, f"({latitude:.6f}, {longitude:.6f})", 
                    fontsize=12, ha='center', va='bottom')
            
            # Add a grid
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Set limits around the point
            plt.xlim(longitude - 0.01, longitude + 0.01)
            plt.ylim(latitude - 0.01, latitude + 0.01)
            
            # Add title
            plt.title(f"Location: {latitude:.6f}, {longitude:.6f}")
            
            # Save the image
            plt.savefig(image_path)
            plt.close()
            
            print(f"Created static map image: {image_path}")
            
            # Create metadata
            metadata = {
                "latitude": latitude,
                "longitude": longitude,
                "description": f"Static map at {latitude:.6f}, {longitude:.6f}",
                "is_fallback": True,
                "capture_date": time.strftime("%Y-%m-%d")
            }
            
            # Add to cache
            image_id = os.path.splitext(os.path.basename(image_path))[0]
            self.image_data_cache[image_id] = {
                "text": f"Static map at {latitude:.6f}, {longitude:.6f}",
                "metadata": metadata,
                "image_path": image_path
            }
            
            # Add to vector database if available
            if self.vector_db is not None:
                try:
                    self.image_collection.add(
                        documents=[self.image_data_cache[image_id]["text"]],
                        metadatas=[metadata],
                        ids=[image_id]
                    )
                except Exception as e:
                    print(f"Error adding image to vector database: {str(e)}")
            
            # Update metadata file
            self._update_metadata_file(image_id, metadata)
            
            return image_path
            
        except Exception as e:
            print(f"Error creating static map: {str(e)}")
            
            # Create an even simpler fallback - just a blank image with text
            img = Image.new('RGB', (600, 600), color=(240, 240, 240))
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(img)
            draw.text((300, 300), f"Location: {latitude:.6f}, {longitude:.6f}", 
                     fill=(0, 0, 0), anchor="mm")
            img.save(image_path)
            
            return image_path

    def _update_metadata_file(self, image_id: str, metadata: Dict[str, Any]):
        """Update the metadata file with new image information
        
        Args:
            image_id: Image ID
            metadata: Metadata dictionary
        """
        metadata_path = os.path.join(self.images_dir, "metadata.json")
        metadata_dict = {}
        
        # Load existing metadata if available
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata_dict = json.load(f)
            except Exception as e:
                print(f"Error loading metadata file: {str(e)}")
        
        # Add or update metadata
        metadata_dict[image_id] = metadata
        
        # Save metadata
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata_dict, f, indent=2)
        except Exception as e:
            print(f"Error saving metadata file: {str(e)}")

    def fetch_satellite_images_for_locations(self, locations: List[Dict[str, Any]], 
                                           zoom: int = 18) -> List[str]:
        """Fetch satellite images for a list of locations
        
        Args:
            locations: List of location dictionaries with lat/lng or latitude/longitude
            zoom: Zoom level (higher = more detailed)
            
        Returns:
            List of paths to saved images
        """
        image_paths = []
        
        for location in locations:
            # Extract coordinates
            lat = location.get('lat', location.get('latitude'))
            lng = location.get('lng', location.get('longitude'))
            
            if lat is not None and lng is not None:
                # Fetch image
                image_path = self.fetch_satellite_image(lat, lng, zoom)
                if image_path:
                    image_paths.append(image_path)
                    
                # Add a small delay to avoid rate limiting
                time.sleep(0.5)
        
        return image_paths

    def query_text(self, query: str, n_results: int = 5, filter_dict: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Query the database for text data
        
        Args:
            query: Query string
            n_results: Number of results to return
            filter_dict: Dictionary of metadata filters
            
        Returns:
            List of matching results
        """
        results = []
        
        # Use vector database if available
        if self.vector_db is not None:
            try:
                query_results = self.text_collection.query(
                    query_texts=[query],
                    n_results=n_results,
                    where=filter_dict
                )
                
                for i, id in enumerate(query_results['ids'][0]):
                    results.append({
                        "id": id,
                        "text": query_results['documents'][0][i],
                        "metadata": query_results['metadatas'][0][i],
                        "distance": query_results['distances'][0][i] if 'distances' in query_results else None
                    })
                
                return results
            except Exception as e:
                print(f"Error querying vector database: {str(e)}")
        
        # Fallback to simple keyword matching
        query_terms = query.lower().split()
        for id, data in self.text_data_cache.items():
            if filter_dict:
                # Check if metadata matches filter
                metadata_match = True
                for key, value in filter_dict.items():
                    if key not in data["metadata"] or data["metadata"][key] != value:
                        metadata_match = False
                        break
                
                if not metadata_match:
                    continue
            
            # Simple keyword matching
            text = data["text"].lower()
            match_count = sum(1 for term in query_terms if term in text)
            
            if match_count > 0:
                results.append({
                    "id": id,
                    "text": data["text"],
                    "metadata": data["metadata"],
                    "match_score": match_count / len(query_terms)
                })
        
        # Sort by match score and limit results
        results.sort(key=lambda x: x.get("match_score", 0), reverse=True)
        return results[:n_results]

    def query_geospatial(self, query: str = None, location: Point = None, radius_km: float = None, 
                        n_results: int = 5, filter_dict: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Query the database for geospatial data
        
        Args:
            query: Text query (optional)
            location: Center point for spatial search (optional)
            radius_km: Search radius in kilometers (optional)
            n_results: Number of results to return
            filter_dict: Dictionary of metadata filters
            
        Returns:
            List of matching results
        """
        results = []
        
        # Use vector database for text query if available
        if query and self.vector_db is not None:
            try:
                query_results = self.geo_collection.query(
                    query_texts=[query],
                    n_results=n_results,
                    where=filter_dict
                )
                
                for i, id in enumerate(query_results['ids'][0]):
                    if id in self.geo_data_cache:
                        results.append({
                            "id": id,
                            "text": query_results['documents'][0][i],
                            "metadata": query_results['metadatas'][0][i],
                            "geometry": self.geo_data_cache[id]["geometry"],
                            "distance": query_results['distances'][0][i] if 'distances' in query_results else None
                        })
                
                # If we have a location and radius, filter by distance
                if location and radius_km and results:
                    filtered_results = []
                    for result in results:
                        if result["geometry"].centroid.distance(location) * 111 <= radius_km:  # Approx conversion to km
                            filtered_results.append(result)
                    results = filtered_results
                
                return results
            except Exception as e:
                print(f"Error querying vector database: {str(e)}")
        
        # Spatial search only
        if location and radius_km:
            for id, data in self.geo_data_cache.items():
                if filter_dict:
                    # Check if metadata matches filter
                    metadata_match = True
                    for key, value in filter_dict.items():
                        if key not in data["metadata"] or data["metadata"][key] != value:
                            metadata_match = False
                            break
                    
                    if not metadata_match:
                        continue
                
                # Check distance
                distance_km = data["geometry"].centroid.distance(location) * 111  # Approx conversion to km
                if distance_km <= radius_km:
                    results.append({
                        "id": id,
                        "text": data["text"],
                        "metadata": data["metadata"],
                        "geometry": data["geometry"],
                        "distance_km": distance_km
                    })
            
            # Sort by distance and limit results
            results.sort(key=lambda x: x.get("distance_km", float('inf')))
            return results[:n_results]
        
        # Text search fallback
        if query:
            query_terms = query.lower().split()
            for id, data in self.geo_data_cache.items():
                if filter_dict:
                    # Check if metadata matches filter
                    metadata_match = True
                    for key, value in filter_dict.items():
                        if key not in data["metadata"] or data["metadata"][key] != value:
                            metadata_match = False
                            break
                    
                    if not metadata_match:
                        continue
                
                # Simple keyword matching
                text = data["text"].lower()
                match_count = sum(1 for term in query_terms if term in text)
                
                if match_count > 0:
                    results.append({
                        "id": id,
                        "text": data["text"],
                        "metadata": data["metadata"],
                        "geometry": data["geometry"],
                        "match_score": match_count / len(query_terms)
                    })
            
            # Sort by match score and limit results
            results.sort(key=lambda x: x.get("match_score", 0), reverse=True)
            return results[:n_results]
        
        return []

    def query_satellite_images(self, query: str = None, location: Point = None, radius_km: float = None,
                              n_results: int = 5, filter_dict: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Query the database for satellite images
        
        Args:
            query: Text query (optional)
            location: Center point for spatial search (optional)
            radius_km: Search radius in kilometers (optional)
            n_results: Number of results to return
            filter_dict: Dictionary of metadata filters
            
        Returns:
            List of matching results
        """
        results = []
        
        # Use vector database for text query if available
        if query and self.vector_db is not None:
            try:
                query_results = self.image_collection.query(
                    query_texts=[query],
                    n_results=n_results,
                    where=filter_dict
                )
                
                for i, id in enumerate(query_results['ids'][0]):
                    if id in self.image_data_cache:
                        results.append({
                            "id": id,
                            "text": query_results['documents'][0][i],
                            "metadata": query_results['metadatas'][0][i],
                            "image_path": self.image_data_cache[id]["image_path"],
                            "distance": query_results['distances'][0][i] if 'distances' in query_results else None
                        })
                
                # If we have a location and radius, filter by distance
                if location and radius_km and results:
                    filtered_results = []
                    for result in results:
                        if "latitude" in result["metadata"] and "longitude" in result["metadata"]:
                            img_location = Point(float(result["metadata"]["longitude"]), float(result["metadata"]["latitude"]))
                            if img_location.distance(location) * 111 <= radius_km:  # Approx conversion to km
                                filtered_results.append(result)
                    results = filtered_results
                
                return results
            except Exception as e:
                print(f"Error querying vector database: {str(e)}")
        
        # Spatial search only
        if location and radius_km:
            for id, data in self.image_data_cache.items():
                if filter_dict:
                    # Check if metadata matches filter
                    metadata_match = True
                    for key, value in filter_dict.items():
                        if key not in data["metadata"] or data["metadata"][key] != value:
                            metadata_match = False
                            break
                    
                    if not metadata_match:
                        continue
                
                # Check if image has location data
                if "latitude" in data["metadata"] and "longitude" in data["metadata"]:
                    img_location = Point(float(data["metadata"]["longitude"]), float(data["metadata"]["latitude"]))
                    distance_km = img_location.distance(location) * 111  # Approx conversion to km
                    if distance_km <= radius_km:
                        results.append({
                            "id": id,
                            "text": data["text"],
                            "metadata": data["metadata"],
                            "image_path": data["image_path"],
                            "distance_km": distance_km
                        })
            
            # Sort by distance and limit results
            results.sort(key=lambda x: x.get("distance_km", float('inf')))
            return results[:n_results]
        
        # Text search fallback
        if query:
            query_terms = query.lower().split()
            for id, data in self.image_data_cache.items():
                if filter_dict:
                    # Check if metadata matches filter
                    metadata_match = True
                    for key, value in filter_dict.items():
                        if key not in data["metadata"] or data["metadata"][key] != value:
                            metadata_match = False
                            break
                    
                    if not metadata_match:
                        continue
                
                # Simple keyword matching
                text = data["text"].lower()
                match_count = sum(1 for term in query_terms if term in text)
                
                if match_count > 0:
                    results.append({
                        "id": id,
                        "text": data["text"],
                        "metadata": data["metadata"],
                        "image_path": data["image_path"],
                        "match_score": match_count / len(query_terms)
                    })
            
            # Sort by match score and limit results
            results.sort(key=lambda x: x.get("match_score", 0), reverse=True)
            return results[:n_results]
        
        return []

    def get_image(self, image_id: str) -> Optional[Dict[str, Any]]:
        """Get a satellite image by ID
        
        Args:
            image_id: Image ID
            
        Returns:
            Dictionary with image data and metadata
        """
        if image_id in self.image_data_cache:
            data = self.image_data_cache[image_id]
            
            # Check if image file exists
            if os.path.exists(data["image_path"]):
                return {
                    "id": image_id,
                    "metadata": data["metadata"],
                    "image_path": data["image_path"]
                }
        
        return None

    def visualize_satellite_image(self, image_id: str, output_path: str = None) -> Optional[str]:
        """Visualize a satellite image
        
        Args:
            image_id: Image ID
            output_path: Path to save visualization (optional)
            
        Returns:
            Path to saved visualization if successful, None otherwise
        """
        image_data = self.get_image(image_id)
        if not image_data:
            return None
        
        try:
            # Load the image
            img = Image.open(image_data["image_path"])
            
            # Create figure
            plt.figure(figsize=(10, 8))
            plt.imshow(img)
            
            # Add metadata as title
            title = f"Satellite Image: {image_id}"
            if "description" in image_data["metadata"]:
                title += f"\n{image_data['metadata']['description']}"
            plt.title(title)
            
            # Add coordinates if available
            if "latitude" in image_data["metadata"] and "longitude" in image_data["metadata"]:
                plt.xlabel(f"Location: {image_data['metadata']['latitude']}, {image_data['metadata']['longitude']}")
            
            # Save or show
            if output_path:
                plt.savefig(output_path)
                plt.close()
                return output_path
            else:
                output_path = os.path.join(self.data_dir, "output", f"{image_id}_visualization.png")
                plt.savefig(output_path)
                plt.close()
                return output_path
        
        except Exception as e:
            print(f"Error visualizing image: {str(e)}")
            return None

    def multimodal_query(self, query: str, location: Point = None, radius_km: float = None,
                        include_text: bool = True, include_geo: bool = True, include_images: bool = True,
                        n_results: int = 5, filter_dict: Dict[str, Any] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Perform a multimodal query across all data types
        
        Args:
            query: Query string
            location: Center point for spatial search (optional)
            radius_km: Search radius in kilometers (optional)
            include_text: Whether to include text results
            include_geo: Whether to include geospatial results
            include_images: Whether to include image results
            n_results: Number of results to return per type
            filter_dict: Dictionary of metadata filters
            
        Returns:
            Dictionary with results for each data type
        """
        results = {}
        
        if include_text:
            results["text"] = self.query_text(query, n_results, filter_dict)
        
        if include_geo:
            results["geospatial"] = self.query_geospatial(query, location, radius_km, n_results, filter_dict)
        
        if include_images:
            results["images"] = self.query_satellite_images(query, location, radius_km, n_results, filter_dict)
        
        return results

    def create_multimodal_visualization(self, query_results: Dict[str, List[Dict[str, Any]]], 
                                      output_path: str = None) -> Optional[str]:
        """Create a visualization of multimodal query results
        
        Args:
            query_results: Results from multimodal_query
            output_path: Path to save visualization (optional)
            
        Returns:
            Path to saved visualization if successful, None otherwise
        """
        # Check if we have any results
        if not query_results or all(len(results) == 0 for results in query_results.values()):
            return None
        
        try:
            # Determine number of subplots needed
            n_geo = len(query_results.get("geospatial", []))
            n_img = len(query_results.get("images", []))
            
            if n_geo == 0 and n_img == 0:
                return None
            
            # Create figure
            fig = plt.figure(figsize=(15, 10))
            
            # Add geospatial results
            if n_geo > 0:
                ax_geo = fig.add_subplot(1, 2, 1)
                
                # Create a GeoDataFrame from results
                geometries = [result["geometry"] for result in query_results["geospatial"]]
                geo_data = {
                    "id": [result["id"] for result in query_results["geospatial"]],
                    "geometry": geometries
                }
                gdf = gpd.GeoDataFrame(geo_data, geometry="geometry")
                
                # Plot the GeoDataFrame
                gdf.plot(ax=ax_geo)
                
                # Add labels
                for i, result in enumerate(query_results["geospatial"]):
                    centroid = result["geometry"].centroid
                    ax_geo.text(centroid.x, centroid.y, str(i+1), fontsize=12, ha='center', va='center')
                
                ax_geo.set_title("Geospatial Results")
            
            # Add image results
            if n_img > 0:
                ax_img = fig.add_subplot(1, 2, 2)
                
                # Get the first image
                image_path = query_results["images"][0]["image_path"]
                img = Image.open(image_path)
                
                # Display the image
                ax_img.imshow(img)
                ax_img.set_title(f"Satellite Image: {query_results['images'][0]['id']}")
                ax_img.axis('off')
            
            # Save or show
            if output_path:
                plt.savefig(output_path)
                plt.close()
                return output_path
            else:
                output_path = os.path.join(self.data_dir, "output", "multimodal_results.png")
                plt.savefig(output_path)
                plt.close()
                return output_path
        
        except Exception as e:
            print(f"Error creating multimodal visualization: {str(e)}")
            return None


# Example usage
if __name__ == "__main__":
    db = MultimodalDatabase()
    
    # Fetch a satellite image
    image_path = db.fetch_satellite_image(25.2048, 55.2708)  # Dubai center
    print(f"Satellite image saved to: {image_path}")
    
    # Query
    results = db.query_text("restaurant")
    print(results)
