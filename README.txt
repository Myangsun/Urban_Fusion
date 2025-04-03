# UrbanFusion Prototype

## Overview

UrbanFusion is a multimodal geo-agent framework designed for urban site selection in Dubai. The system leverages large language models (LLMs) and vision-language models (VLMs) to process diverse data types and employ reasoning capabilities through tool usage and code generation.

This prototype implements the core components of the UrbanFusion framework using LangChain for the agent framework, providing a foundation for multimodal data integration and constraint-driven site selection.

## Components

The prototype consists of the following components:

1. **Agent Framework** (`urban_fusion.py`): Implements the LangChain-based agent framework with specialized agents:

   - Coordinator Agent: Manages agent registration, invocation, and collaboration
   - GeoAgent: Handles geospatial queries and analysis
   - Constraints Agent: Enforces and evaluates constraints
   - Evaluation Agent: Validates results against benchmarks
   - Explanation Agent: Generates interpretable explanations

2. **Multimodal Database** (`multimodal_database.py`): Implements a multimodal embedding database that integrates diverse data types:

   - GIS Data: Spatial information including boundaries, road networks, and existing facilities
   - CSV Data: Structured data such as financial metrics, demographic statistics, and order volumes
   - Image Data: Satellite imagery, street views, and visual representations of urban patterns

3. **Data Processor** (`data_processor.py`): Implements data processing functionality for analyzing and visualizing data:

   - Processing order data to extract insights
   - Creating heatmaps and visualizations
   - Analyzing constraints and calculating metrics

4. **Main Application** (`app.py`): Implements the main application that ties everything together:

   - Processing user queries
   - Analyzing data
   - Evaluating locations against constraints
   - Running demonstrations

5. **Environment Setup** (`setup.py`): Sets up the environment for the UrbanFusion system:

   - Creating necessary directories
   - Setting up environment variables

6. **Testing Module** (`test_urban_fusion.py`): Implements testing functionality for the UrbanFusion system:
   - Functional tests for components
   - Unit tests for specific functionality
   - Integration tests for query processing

## Installation

1. Clone the repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up the environment:
   ```
   python setup.py
   ```
4. Add your OpenAI API key to the `.env` file:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Future Work

Future enhancements to the prototype could include:

1. Implementing multimodal fusion approach
2. Adding support for more data types and sources
3. Enhancing the constraint resolution mechanism
4. Improving the explanation generation capabilities
5. Implementing cross-city transfer learning
6. Adding support for collaborative multi-stakeholder planning

## License

This project is licensed under the MIT License - see the LICENSE file for details.
