"""
Process Questions Script for Urban Fusion Agent

This script processes site selection benchmark questions using the Urban Fusion agent
and evaluates the results against ground truth answers.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
from typing import Dict, List, Any, Optional, Tuple, Set

# Import Urban Fusion agent
from urban_fusion_agent import CoordinatorAgent

# Import GIS analysis functions
from gis_analysis_functions import (
    load_benchmark_data,
    evaluate_results,
    create_evaluation_visualization
)

# File paths
BENCHMARK_PATH = 'upload/site_selection_benchmark.json'
OUTPUT_DIR = 'output'

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)


def process_all_questions(api_key: Optional[str] = None, use_gpt4: bool = True) -> Dict[str, Any]:
    """
    Process all questions in the benchmark.

    Args:
        api_key: OpenAI API key
        use_gpt4: Whether to use GPT-4 for reasoning

    Returns:
        Dictionary with results for all questions
    """
    # Load benchmark data
    benchmark_data = load_benchmark_data(BENCHMARK_PATH)

    # Initialize coordinator agent
    coordinator = CoordinatorAgent(api_key=api_key, use_gpt4=use_gpt4)

    # Run benchmark on all questions
    results = coordinator.run_benchmark()

    # Create evaluation visualization
    evaluation_viz_path = create_evaluation_visualization(
        results['evaluation'])

    # Save detailed results to CSV
    save_results_to_csv(results)

    # Create HTML report
    create_html_report(results)

    return results


def process_question_range(start_id: int, end_id: int, api_key: Optional[str] = None, use_gpt4: bool = True) -> Dict[str, Any]:
    """
    Process a range of questions in the benchmark.

    Args:
        start_id: Starting question ID
        end_id: Ending question ID
        api_key: OpenAI API key
        use_gpt4: Whether to use GPT-4 for reasoning

    Returns:
        Dictionary with results for the specified range of questions
    """
    # Load benchmark data
    benchmark_data = load_benchmark_data(BENCHMARK_PATH)

    # Get question IDs in the specified range
    question_ids = list(range(start_id, end_id + 1))

    # Initialize coordinator agent
    coordinator = CoordinatorAgent(api_key=api_key, use_gpt4=use_gpt4)

    # Run benchmark on specified questions
    results = coordinator.run_benchmark(question_ids)

    # Create evaluation visualization
    evaluation_viz_path = create_evaluation_visualization(
        results['evaluation'])

    # Save detailed results to CSV
    save_results_to_csv(results)

    # Create HTML report
    create_html_report(results)

    return results


def process_single_question(question_id: int, api_key: Optional[str] = None, use_gpt4: bool = True) -> Dict[str, Any]:
    """
    Process a single question in the benchmark.

    Args:
        question_id: Question ID to process
        api_key: OpenAI API key
        use_gpt4: Whether to use GPT-4 for reasoning

    Returns:
        Dictionary with results for the specified question
    """
    # Initialize coordinator agent
    coordinator = CoordinatorAgent(api_key=api_key, use_gpt4=use_gpt4)

    # Process the question
    result = coordinator.process_question(question_id)

    # Get ground truth
    benchmark_data = load_benchmark_data(BENCHMARK_PATH)
    question_data = None
    for question in benchmark_data.get('questions', []):
        if question.get('question_id') == question_id:
            question_data = question
            break

    if question_data:
        ground_truth = question_data.get('answer', {}).get('parcels', [])
        metrics = evaluate_results(result.get('parcels', []), ground_truth)

        # Create a simple results structure
        results = {
            'results': {question_id: result},
            'evaluation': {question_id: metrics}
        }

        # Save detailed results to CSV
        save_results_to_csv(results)

        # Create HTML report
        create_html_report(results)

        return results
    else:
        return {
            'results': {question_id: result},
            'evaluation': {question_id: {}}
        }


def save_results_to_csv(results: Dict[str, Any]) -> str:
    """
    Save detailed results to CSV.

    Args:
        results: Dictionary with results and evaluation

    Returns:
        Path to the saved CSV file
    """
    # Create a list of records
    records = []
    for question_id, result in results['results'].items():
        evaluation = results['evaluation'].get(question_id, {})

        record = {
            'question_id': question_id,
            'predicted_count': result.get('count', 0),
            'ground_truth_count': len(evaluation.get('true_positives', [])) + len(evaluation.get('false_negatives', [])),
            'precision': evaluation.get('precision', 0),
            'recall': evaluation.get('recall', 0),
            'f1_score': evaluation.get('f1_score', 0),
            'accuracy': evaluation.get('accuracy', 0),
            'true_positives': evaluation.get('true_positives', 0),
            'false_positives': evaluation.get('false_positives', 0),
            'false_negatives': evaluation.get('false_negatives', 0)
        }

        records.append(record)

    # Create DataFrame
    df = pd.DataFrame(records)

    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(OUTPUT_DIR, f"detailed_results_{timestamp}.csv")
    df.to_csv(output_path, index=False)

    return output_path


def create_html_report(results: Dict[str, Any]) -> str:
    """
    Create an HTML report with results and visualizations.

    Args:
        results: Dictionary with results and evaluation

    Returns:
        Path to the saved HTML file
    """
    # Create HTML content
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Urban Fusion Agent Benchmark Results</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #2c3e50; }
            h2 { color: #3498db; }
            table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            .metric-good { color: green; }
            .metric-medium { color: orange; }
            .metric-poor { color: red; }
            .visualization { margin: 20px 0; }
        </style>
    </head>
    <body>
        <h1>Urban Fusion Agent Benchmark Results</h1>
        <h2>Summary</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
    """

    # Calculate overall metrics
    precisions = [e.get('precision', 0)
                  for e in results['evaluation'].values()]
    recalls = [e.get('recall', 0) for e in results['evaluation'].values()]
    f1_scores = [e.get('f1_score', 0) for e in results['evaluation'].values()]
    accuracies = [e.get('accuracy', 0) for e in results['evaluation'].values()]

    avg_precision = np.mean(precisions) if precisions else 0
    avg_recall = np.mean(recalls) if recalls else 0
    avg_f1 = np.mean(f1_scores) if f1_scores else 0
    avg_accuracy = np.mean(accuracies) if accuracies else 0

    # Add overall metrics to HTML
    html_content += f"""
            <tr>
                <td>Average Precision</td>
                <td class="{'metric-good' if avg_precision >= 0.8 else 'metric-medium' if avg_precision >= 0.5 else 'metric-poor'}">{avg_precision:.2f}</td>
            </tr>
            <tr>
                <td>Average Recall</td>
                <td class="{'metric-good' if avg_recall >= 0.8 else 'metric-medium' if avg_recall >= 0.5 else 'metric-poor'}">{avg_recall:.2f}</td>
            </tr>
            <tr>
                <td>Average F1 Score</td>
                <td class="{'metric-good' if avg_f1 >= 0.8 else 'metric-medium' if avg_f1 >= 0.5 else 'metric-poor'}">{avg_f1:.2f}</td>
            </tr>
            <tr>
                <td>Average Accuracy</td>
                <td class="{'metric-good' if avg_accuracy >= 0.8 else 'metric-medium' if avg_accuracy >= 0.5 else 'metric-poor'}">{avg_accuracy:.2f}</td>
            </tr>
            <tr>
                <td>Questions Processed</td>
                <td>{len(results['results'])}</td>
            </tr>
        </table>
        
        <h2>Detailed Results</h2>
        <table>
            <tr>
                <th>Question ID</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1 Score</th>
                <th>Accuracy</th>
                <th>Predicted Count</th>
                <th>Ground Truth Count</th>
            </tr>
    """

    # Add detailed results to HTML
    for question_id, result in results['results'].items():
        evaluation = results['evaluation'].get(question_id, {})
        precision = evaluation.get('precision', 0)
        recall = evaluation.get('recall', 0)
        f1_score = evaluation.get('f1_score', 0)
        accuracy = evaluation.get('accuracy', 0)
        predicted_count = result.get('count', 0)
        ground_truth_count = len(evaluation.get(
            'true_positives', [])) + len(evaluation.get('false_negatives', []))

        html_content += f"""
            <tr>
                <td>{question_id}</td>
                <td class="{'metric-good' if precision >= 0.8 else 'metric-medium' if precision >= 0.5 else 'metric-poor'}">{precision:.2f}</td>
                <td class="{'metric-good' if recall >= 0.8 else 'metric-medium' if recall >= 0.5 else 'metric-poor'}">{recall:.2f}</td>
                <td class="{'metric-good' if f1_score >= 0.8 else 'metric-medium' if f1_score >= 0.5 else 'metric-poor'}">{f1_score:.2f}</td>
                <td class="{'metric-good' if accuracy >= 0.8 else 'metric-medium' if accuracy >= 0.5 else 'metric-poor'}">{accuracy:.2f}</td>
                <td>{predicted_count}</td>
                <td>{ground_truth_count}</td>
            </tr>
        """

    html_content += """
        </table>
        
        <h2>Visualizations</h2>
        <div class="visualization">
            <img src="evaluation_metrics.jpg" alt="Evaluation Metrics" style="max-width: 100%;">
        </div>
        
        <h2>Optimization Suggestions</h2>
        <ul>
            <li>Improve geospatial calculations with more accurate coordinate system transformations</li>
            <li>Enhance constraint parsing with more robust regex patterns</li>
            <li>Implement more sophisticated customer deduplication logic</li>
            <li>Add support for more complex temporal constraints</li>
            <li>Optimize buffer operations for better performance</li>
        </ul>
    </body>
    </html>
    """

    # Save HTML to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(
        OUTPUT_DIR, f"benchmark_report_{timestamp}.html")

    with open(output_path, 'w') as f:
        f.write(html_content)

    return output_path


def main():
    """Main function to run the process_questions script."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Process site selection benchmark questions")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true",
                       help="Process all questions")
    group.add_argument("--range", type=int, nargs=2,
                       metavar=("START", "END"), help="Process a range of questions")
    group.add_argument("--question", type=int,
                       help="Process a single question")
    parser.add_argument("--api_key", type=str, help="OpenAI API key")
    parser.add_argument("--use_gpt4", action="store_true",
                        help="Use GPT-4 for reasoning")

    args = parser.parse_args()

    # Process questions based on arguments
    if args.all:
        results = process_all_questions(
            api_key=args.api_key, use_gpt4=args.use_gpt4)
    elif args.range:
        start_id, end_id = args.range
        results = process_question_range(
            start_id, end_id, api_key=args.api_key, use_gpt4=args.use_gpt4)
    elif args.question:
        results = process_single_question(
            args.question, api_key=args.api_key, use_gpt4=args.use_gpt4)

    # Print summary
    print("\nBenchmark Summary:")
    for question_id, evaluation in results['evaluation'].items():
        print(f"Question {question_id}:")
        print(f"  Precision: {evaluation.get('precision', 0):.2f}")
        print(f"  Recall: {evaluation.get('recall', 0):.2f}")
        print(f"  F1 Score: {evaluation.get('f1_score', 0):.2f}")
        print(f"  Accuracy: {evaluation.get('accuracy', 0):.2f}")

    # Calculate overall metrics
    precisions = [e.get('precision', 0)
                  for e in results['evaluation'].values()]
    recalls = [e.get('recall', 0) for e in results['evaluation'].values()]
    f1_scores = [e.get('f1_score', 0) for e in results['evaluation'].values()]
    accuracies = [e.get('accuracy', 0) for e in results['evaluation'].values()]

    print("\nOverall Metrics:")
    print(
        f"  Average Precision: {np.mean(precisions) if precisions else 0:.2f}")
    print(f"  Average Recall: {np.mean(recalls) if recalls else 0:.2f}")
    print(f"  Average F1 Score: {np.mean(f1_scores) if f1_scores else 0:.2f}")
    print(
        f"  Average Accuracy: {np.mean(accuracies) if accuracies else 0:.2f}")


if __name__ == "__main__":
    main()
