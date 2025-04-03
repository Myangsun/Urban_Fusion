"""
Urban Fusion Agent with LangChain and GPT-4 Integration

This module implements an agent framework for answering site selection benchmark questions
using LangChain and GPT-4. The agent uses specialized components to process different types
of constraints and generate answers.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set
import matplotlib.pyplot as plt

from langchain.chains import LLMChain  # <-- Add this import
# Updated LangChain imports with all fixes
from langchain_core.agents import AgentAction, AgentFinish
from langchain.agents.agent import AgentExecutor  # Fixed import for AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

from langchain_core.tools import Tool
from pydantic import BaseModel, Field  # Fixed import from pydantic directly
# Fixed import for initialize_agent
from langchain.agents import initialize_agent, AgentType
# from langchain_community.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_community.callbacks.manager import get_openai_callback

# Import custom tools and GIS functions
import langchain_tools as tools
from gis_analysis_functions import (
    load_geojson_data,
    load_csv_data,
    load_benchmark_data,
    process_question,
    process_all_questions,
    evaluate_results,
    create_map_visualization
)

# File paths
GEOJSON_PATH = 'upload/dubai_sector3_sites.geojson'
CSV_PATH = 'upload/talabat_sample.csv'
BENCHMARK_PATH = 'upload/site_selection_benchmark.json'
OUTPUT_DIR = 'output'

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)


class CoordinatorAgent:
    """
    Coordinator agent that orchestrates the workflow for processing site selection questions.
    """

    def __init__(self, api_key: Optional[str] = None, use_gpt4: bool = True):
        """
        Initialize the coordinator agent.

        Args:
            api_key: OpenAI API key
            use_gpt4: Whether to use GPT-4 for reasoning
        """
        # Set API key if provided
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

        # Initialize LLM
        self.llm = ChatOpenAI(
            model_name="gpt-4" if use_gpt4 else "gpt-3.5-turbo",
            temperature=0.2
        )

        # Initialize specialized agents
        self.geo_agent = GeoAgent(self.llm)
        self.constraints_agent = ConstraintsAgent(self.llm)
        self.explanation_agent = ExplanationAgent(self.llm)

        # Initialize agent with tools
        self.agent = initialize_agent(
            tools=[
                tools.MapTool(),
                tools.DataAnalysisTool(),
                tools.VisualizationTool()
            ],
            llm=self.llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )

        # Load data
        self.sites_gdf = load_geojson_data(GEOJSON_PATH)

        # Check and add/rename site_id
        if 'site_id' not in self.sites_gdf.columns:
            if 'id' in self.sites_gdf.columns:
                self.sites_gdf.rename(columns={'id': 'site_id'}, inplace=True)
            else:
                # create a default numeric ID
                self.sites_gdf['site_id'] = range(1, len(self.sites_gdf) + 1)

        self.orders_df = load_csv_data(CSV_PATH)
        self.benchmark_data = load_benchmark_data(BENCHMARK_PATH)

    def process_question(self, question_id: int) -> Dict[str, Any]:
        """
        Process a single question and find parcels that satisfy all constraints.

        Args:
            question_id: ID of the question to process

        Returns:
            Dictionary with question ID, selected parcels, and count
        """
        # Get question data
        question_data = None
        for question in self.benchmark_data.get('questions', []):
            if question.get('question_id') == question_id:
                question_data = question
                break

        if not question_data:
            return {
                'question_id': question_id,
                'parcels': [],
                'count': 0,
                'error': f"Question {question_id} not found"
            }

        print(f"\n=== Processing Question {question_id} ===")
        print(
            f"Processing question {question_id}: {question_data.get('question')}")
        print(f"Constraints: {question_data.get('constraints')}")

        # Use GIS analysis functions to process the question
        result = process_question(question_data)

        # Add explanation using the explanation agent
        explanation = self.explanation_agent.generate_explanation(
            question_data.get('question'),
            question_data.get('constraints'),
            result.get('parcels')
        )

        result['explanation'] = explanation

        return result

    def run_benchmark(self, question_ids: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Run the benchmark on specified questions or all questions.

        Args:
            question_ids: List of question IDs to process, or None for all questions

        Returns:
            Dictionary with results for all processed questions
        """
        results = {}
        evaluation = {}

        # Determine which questions to process
        if question_ids:
            questions_to_process = [q for q in self.benchmark_data.get('questions', [])
                                    if q.get('question_id') in question_ids]
        else:
            questions_to_process = self.benchmark_data.get('questions', [])

        print(f"Running Urban Fusion agent on benchmark questions...")
        print(
            f"Testing on questions: {[q.get('question_id') for q in questions_to_process]}")

        # Process each question
        for question in questions_to_process:
            question_id = question.get('question_id')

            # Process question
            result = self.process_question(question_id)

            # Evaluate result
            ground_truth = question.get('answer', {}).get('parcels', [])
            metrics = evaluate_results(result.get('parcels', []), ground_truth)

            # Store results and evaluation
            results[question_id] = result
            evaluation[question_id] = metrics

        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            OUTPUT_DIR, f"benchmark_results_{timestamp}.json")

        with open(output_path, 'w') as f:
            json.dump({
                'results': results,
                'evaluation': evaluation
            }, f, indent=2)

        print(f"\n=== Benchmark Results ===")
        print(f"Results saved to: {output_path}")

        return {
            'results': results,
            'evaluation': evaluation,
            'output_path': output_path
        }


class GeoAgent:
    """
    Specialized agent for handling geospatial constraints.
    """

    def __init__(self, llm):
        """
        Initialize the geo agent.

        Args:
            llm: Language model to use for reasoning
        """
        self.llm = llm

        # Initialize prompt template
        self.prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(
                    "You are a geospatial analysis expert."
                ),
                HumanMessagePromptTemplate.from_template(
                    """
            Analyze the following site selection question and constraints:
            
            Question: {question}
            
            Geospatial Constraints:
            {constraints}
            
            Identify the key geospatial parameters needed to solve this problem:
            1. What are the buffer distances?
            2. What are the proximity requirements?
            3. What are the target features?
            
            Provide your analysis in a structured format.
            """
                ),
            ]
        )

        # Initialize chain
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def analyze_constraints(self, question: str, constraints: List[str]) -> Dict[str, Any]:
        """
        Analyze geospatial constraints.

        Args:
            question: Question text
            constraints: List of constraint texts

        Returns:
            Dictionary with analyzed constraint parameters
        """
        # Filter for geospatial constraints
        geo_constraints = [c for c in constraints if "geospatial" in c.lower()]

        if not geo_constraints:
            return {}

        # Run chain
        result = self.chain.run(
            question=question, constraints="\n".join(geo_constraints))

        # Parse result (simplified for demonstration)
        parsed = {
            "buffer_distances": [0.5, 1.0],  # Default values
            "proximity_requirements": [],
            "target_features": []
        }

        return parsed


class ConstraintsAgent:
    """
    Specialized agent for handling various types of constraints.
    """

    def __init__(self, llm):
        """
        Initialize the constraints agent.

        Args:
            llm: Language model to use for reasoning
        """
        self.llm = llm

        # Initialize prompt template
        self.prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(
                    "You are a constraints analysis expert."
                ),
                HumanMessagePromptTemplate.from_template("""
            Analyze the following site selection question and constraints:

            Question: {question}

            Constraints:
            {constraints}

            Categorize each constraint by type (geospatial, temporal, economic, market) and extract the key parameters:
            - Comparison operators (>, <, >=, <=)
            - Threshold values
            - Target features
            - Time periods
            - Radius values

            Provide your analysis in a structured format.
        """)
            ]
        )
        # Initialize chain
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def analyze_constraints(self, question: str, constraints: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Analyze all constraints.

        Args:
            question: Question text
            constraints: List of constraint texts

        Returns:
            Dictionary with analyzed constraint parameters by type
        """
        # Run chain
        result = self.chain.run(
            question=question, constraints="\n".join(constraints))

        # Parse result (simplified for demonstration)
        parsed = {
            "geospatial": [],
            "temporal": [],
            "economic": [],
            "market": []
        }

        # Add some sample parsed constraints for demonstration
        for constraint in constraints:
            if "geospatial" in constraint.lower():
                parsed["geospatial"].append({
                    "type": "buffer",
                    "radius_km": 0.5
                })
            elif "temporal" in constraint.lower():
                parsed["temporal"].append({
                    "type": "time_of_day",
                    "period": "evening"
                })
            elif "economic" in constraint.lower():
                parsed["economic"].append({
                    "type": "threshold",
                    "metric": "average_order_value",
                    "operator": ">",
                    "value": 100
                })
            elif "market" in constraint.lower():
                parsed["market"].append({
                    "type": "count",
                    "target": "coffee_shops",
                    "operator": "<=",
                    "value": 4
                })

        return parsed


class ExplanationAgent:
    """
    Specialized agent for generating explanations.
    """

    def __init__(self, llm):
        """
        Initialize the explanation agent.

        Args:
            llm: Language model to use for reasoning
        """
        self.llm = llm

        # Initialize prompt template
        self.prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(
                    "You are an urban planning expert specializing in GIS analysis."
                ),
                HumanMessagePromptTemplate.from_template("""
                    Generate an explanation for the following site selection result:

                    Question: {question}

                    Constraints:
                    {constraints}

                    Selected Parcels: {parcels}

                    Provide a clear explanation detailing:
                    - Why these parcels were selected
                    - How they satisfy each specific constraint
                    - Why these parcels are optimal choices for the given requirements

                    Use urban planning terminology and GIS concepts in your explanation.
                """)
            ]
        )

        # Initialize chain
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def generate_explanation(self, question: str, constraints: List[str], parcels: List[int]) -> str:
        """
        Generate an explanation for the selected parcels.

        Args:
            question: Question text
            constraints: List of constraint texts
            parcels: List of selected parcel IDs

        Returns:
            Explanation text
        """
        # Run chain
        result = self.chain.invoke({
            "question": question,
            "constraints": "\n".join(constraints),
            "parcels": ", ".join(map(str, parcels[:10])) + ("..." if len(parcels) > 10 else "")
        })

        # Safely return the output regardless of structure
        if isinstance(result, dict):
            return result.get("output", str(result))
        return getattr(result, "content", str(result))


def main():
    """Main function to run the Urban Fusion agent."""
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run Urban Fusion agent on benchmark questions")
    parser.add_argument("--questions", type=str,
                        help="Comma-separated list of question IDs to test")
    parser.add_argument("--use_gpt4", action="store_true",
                        help="Use GPT-4 for reasoning")
    parser.add_argument("--api_key", type=str, help="OpenAI API key")

    args = parser.parse_args()

    # Parse question IDs
    question_ids = None
    if args.questions:
        question_ids = [int(q.strip()) for q in args.questions.split(",")]

    # Initialize coordinator agent
    coordinator = CoordinatorAgent(
        api_key=args.api_key, use_gpt4=args.use_gpt4)

    # Run benchmark
    results = coordinator.run_benchmark(question_ids)

    # Print summary
    print("\nBenchmark Summary:")
    for question_id, evaluation in results['evaluation'].items():
        print(f"Question {question_id}:")
        print(f"  Precision: {evaluation['precision']:.2f}")
        print(f"  Recall: {evaluation['recall']:.2f}")
        print(f"  F1 Score: {evaluation['f1_score']:.2f}")
        print(f"  Accuracy: {evaluation['accuracy']:.2f}")

    # Calculate overall metrics
    precisions = [e['precision'] for e in results['evaluation'].values()]
    recalls = [e['recall'] for e in results['evaluation'].values()]
    f1_scores = [e['f1_score'] for e in results['evaluation'].values()]
    accuracies = [e['accuracy'] for e in results['evaluation'].values()]

    print("\nOverall Metrics:")
    print(f"  Average Precision: {np.mean(precisions):.2f}")
    print(f"  Average Recall: {np.mean(recalls):.2f}")
    print(f"  Average F1 Score: {np.mean(f1_scores):.2f}")
    print(f"  Average Accuracy: {np.mean(accuracies):.2f}")


if __name__ == "__main__":
    main()
