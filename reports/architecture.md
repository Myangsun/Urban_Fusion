# UrbanFusion Framework Architecture

## System Components

### 1. Coordinator Agent
The central orchestration component responsible for:
- Managing agent registration and invocation
- Routing queries to appropriate specialized agents
- Coordinating collaboration between agents
- Synthesizing final responses

### 2. Specialized Agents

#### GeoAgent
- Handles geospatial queries and analysis
- Processes location data and spatial relationships
- Performs proximity calculations and spatial filtering
- Interfaces with GIS data sources

#### Constraints Agent
- Enforces and evaluates constraints
- Handles both hard constraints (must be satisfied) and soft preferences (can be optimized)
- Validates potential solutions against constraint criteria
- Ranks solutions based on constraint satisfaction

#### Evaluation Agent
- Validates results against established benchmarks
- Calculates metrics like accuracy and pass rate
- Performs intermediate testing on step results
- Provides feedback for system improvement

#### Explanation Agent
- Generates interpretable explanations of site recommendations
- Provides transparency into decision-making process
- Creates human-readable justifications for recommendations
- Highlights key factors influencing decisions

### 3. Functional Tools

#### MapTool
- Generates interactive maps
- Visualizes geospatial data
- Displays potential site locations
- Shows spatial relationships between entities

#### Data Analysis Tool
- Processes structured data
- Analyzes financial metrics
- Performs statistical calculations
- Extracts insights from CSV data

#### Visualization Tool
- Creates charts and graphs
- Supports decision-making with visual aids
- Visualizes comparison between options
- Presents metrics in an understandable format

#### Code Generation Tool
- Enables agents to write and execute Python code
- Performs complex geospatial analysis
- Implements custom algorithms
- Extends system capabilities beyond pre-built functions

### 4. Multimodal Embedding Database
- Integrates diverse data types (GIS, CSV, images)
- Uses vector embeddings for efficient similarity search
- Stores pre-trained multimodal embeddings
- Implemented with Chroma for vector database functionality

## Data Flow

1. User submits natural language query
2. Coordinator processes query and determines required agents/tools
3. Specialized agents are invoked with relevant context
4. Agents use tools to access and process data from the multimodal database
5. Results are validated by the Evaluation Agent
6. Explanation Agent generates interpretable explanations
7. Coordinator synthesizes final response and returns to user

## Implementation Approach

The system is implemented as a directed graph G = (V, E), where:
- Vertices V represent agents and tools
- Edges E represent information flow between components

This architecture enables flexible composition of agents and tools to address complex urban site selection problems with multiple constraints and data modalities.
