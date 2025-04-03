"""
Evaluation system for benchmark questions to assess the performance of the Urban Fusion agent.
This system calculates metrics like accuracy, precision, recall, F1 score, and constraint satisfaction.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional

# Load benchmark questions
BENCHMARK_PATH = '/home/ubuntu/upload/site_selection_benchmark.json'
OUTPUT_DIR = '/home/ubuntu/output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load benchmark questions
with open(BENCHMARK_PATH, 'r') as f:
    benchmark_questions = json.load(f)

class BenchmarkEvaluator:
    def __init__(self):
        """Initialize the benchmark evaluator."""
        self.benchmark_questions = benchmark_questions
        self.results = []
    
    def evaluate_question(self, question_id, predicted_parcels, constraint_satisfaction=None):
        """
        Evaluate the predicted answer against the ground truth for a specific question.
        
        Args:
            question_id: ID of the benchmark question
            predicted_parcels: List of parcel IDs predicted by the agent
            constraint_satisfaction: Dictionary mapping constraint types to satisfaction scores (0-1)
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Get the ground truth from the benchmark
        question_data = next((q for q in self.benchmark_questions if q["question_id"] == question_id), None)
        if not question_data:
            return {"error": f"Question {question_id} not found in benchmark"}
        
        ground_truth_parcels = question_data["answer"]["parcels"]
        
        # Calculate metrics
        metrics = self._calculate_metrics(predicted_parcels, ground_truth_parcels)
        
        # Add constraint satisfaction if provided
        if constraint_satisfaction:
            metrics["constraint_satisfaction"] = constraint_satisfaction
        
        # Store the evaluation result
        evaluation_result = {
            "question_id": question_id,
            "question": question_data["question"],
            "constraints": question_data["constraints"],
            "metrics": metrics,
            "ground_truth": {
                "parcels": ground_truth_parcels,
                "count": len(ground_truth_parcels)
            },
            "prediction": {
                "parcels": predicted_parcels,
                "count": len(predicted_parcels)
            }
        }
        
        self.results.append(evaluation_result)
        
        return evaluation_result
    
    def _calculate_metrics(self, predicted_parcels, ground_truth_parcels):
        """
        Calculate evaluation metrics.
        
        Args:
            predicted_parcels: List of parcel IDs predicted by the agent
            ground_truth_parcels: List of parcel IDs from the ground truth
            
        Returns:
            Dictionary containing metrics (accuracy, precision, recall, F1 score)
        """
        # Convert to sets for easier operations
        predicted_set = set(predicted_parcels)
        ground_truth_set = set(ground_truth_parcels)
        
        # Calculate metrics
        true_positives = len(predicted_set.intersection(ground_truth_set))
        false_positives = len(predicted_set - ground_truth_set)
        false_negatives = len(ground_truth_set - predicted_set)
        
        # Accuracy (exact match)
        accuracy = 1.0 if predicted_set == ground_truth_set else 0.0
        
        # Precision
        precision = true_positives / len(predicted_set) if len(predicted_set) > 0 else 0.0
        
        # Recall
        recall = true_positives / len(ground_truth_set) if len(ground_truth_set) > 0 else 0.0
        
        # F1 Score
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives
        }
    
    def calculate_overall_metrics(self):
        """
        Calculate overall metrics across all evaluated questions.
        
        Returns:
            Dictionary containing overall metrics
        """
        if not self.results:
            return {"error": "No questions have been evaluated yet"}
        
        # Calculate average metrics
        overall_metrics = {
            "accuracy": sum(r["metrics"]["accuracy"] for r in self.results) / len(self.results),
            "precision": sum(r["metrics"]["precision"] for r in self.results) / len(self.results),
            "recall": sum(r["metrics"]["recall"] for r in self.results) / len(self.results),
            "f1_score": sum(r["metrics"]["f1_score"] for r in self.results) / len(self.results)
        }
        
        # Calculate constraint satisfaction if available
        constraint_satisfaction = {}
        constraint_counts = {}
        
        for result in self.results:
            if "constraint_satisfaction" in result["metrics"]:
                for constraint_type, score in result["metrics"]["constraint_satisfaction"].items():
                    if constraint_type not in constraint_satisfaction:
                        constraint_satisfaction[constraint_type] = 0
                        constraint_counts[constraint_type] = 0
                    
                    constraint_satisfaction[constraint_type] += score
                    constraint_counts[constraint_type] += 1
        
        # Calculate average constraint satisfaction
        if constraint_satisfaction:
            overall_metrics["constraint_satisfaction"] = {
                constraint_type: score / constraint_counts[constraint_type]
                for constraint_type, score in constraint_satisfaction.items()
            }
        
        return overall_metrics
    
    def generate_report(self, output_path=None):
        """
        Generate a comprehensive evaluation report.
        
        Args:
            output_path: Path to save the report (default: OUTPUT_DIR/evaluation_report.json)
            
        Returns:
            Dictionary containing the report
        """
        if not self.results:
            return {"error": "No questions have been evaluated yet"}
        
        # Calculate overall metrics
        overall_metrics = self.calculate_overall_metrics()
        
        # Create the report
        report = {
            "overall_metrics": overall_metrics,
            "question_results": self.results
        }
        
        # Save the report if output_path is provided
        if output_path:
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2)
            print(f"Evaluation report saved to {output_path}")
        
        return report
    
    def visualize_metrics(self, output_dir=OUTPUT_DIR):
        """
        Create visualizations of the evaluation metrics.
        
        Args:
            output_dir: Directory to save the visualizations
            
        Returns:
            List of paths to the generated visualizations
        """
        if not self.results:
            return {"error": "No questions have been evaluated yet"}
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract metrics for visualization
        question_ids = [r["question_id"] for r in self.results]
        precision_values = [r["metrics"]["precision"] for r in self.results]
        recall_values = [r["metrics"]["recall"] for r in self.results]
        f1_values = [r["metrics"]["f1_score"] for r in self.results]
        
        # Create bar chart for precision, recall, and F1 score
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(question_ids))
        width = 0.25
        
        ax.bar(x - width, precision_values, width, label='Precision')
        ax.bar(x, recall_values, width, label='Recall')
        ax.bar(x + width, f1_values, width, label='F1 Score')
        
        ax.set_xlabel('Question ID')
        ax.set_ylabel('Score')
        ax.set_title('Precision, Recall, and F1 Score by Question')
        ax.set_xticks(x)
        ax.set_xticklabels(question_ids)
        ax.legend()
        
        # Save the chart
        metrics_chart_path = os.path.join(output_dir, "metrics_by_question.jpg")
        plt.savefig(metrics_chart_path)
        plt.close()
        
        # Create a chart for constraint satisfaction if available
        constraint_chart_path = None
        
        if any("constraint_satisfaction" in r["metrics"] for r in self.results):
            # Extract constraint satisfaction scores
            constraint_data = {}
            
            for result in self.results:
                if "constraint_satisfaction" in result["metrics"]:
                    for constraint_type, score in result["metrics"]["constraint_satisfaction"].items():
                        if constraint_type not in constraint_data:
                            constraint_data[constraint_type] = []
                        
                        # Extend the list if needed
                        while len(constraint_data[constraint_type]) < result["question_id"] - 1:
                            constraint_data[constraint_type].append(None)
                        
                        constraint_data[constraint_type].append(score)
            
            # Create line chart for constraint satisfaction
            fig, ax = plt.subplots(figsize=(12, 6))
            
            for constraint_type, scores in constraint_data.items():
                # Filter out None values
                valid_scores = [(i+1, score) for i, score in enumerate(scores) if score is not None]
                if valid_scores:
                    x_values, y_values = zip(*valid_scores)
                    ax.plot(x_values, y_values, marker='o', label=constraint_type)
            
            ax.set_xlabel('Question ID')
            ax.set_ylabel('Satisfaction Score (0-1)')
            ax.set_title('Constraint Satisfaction by Question and Constraint Type')
            ax.set_xticks(range(1, max(question_ids) + 1))
            ax.set_ylim(0, 1.1)
            ax.legend()
            
            # Save the chart
            constraint_chart_path = os.path.join(output_dir, "constraint_satisfaction.jpg")
            plt.savefig(constraint_chart_path)
            plt.close()
        
        # Return paths to the generated visualizations
        visualization_paths = [metrics_chart_path]
        if constraint_chart_path:
            visualization_paths.append(constraint_chart_path)
        
        return visualization_paths

# Example usage
if __name__ == "__main__":
    # Initialize the evaluator
    evaluator = BenchmarkEvaluator()
    
    # Example: Evaluate a question with perfect prediction
    question_id = 1
    ground_truth = next(q["answer"]["parcels"] for q in benchmark_questions if q["question_id"] == question_id)
    
    # Perfect prediction (same as ground truth)
    perfect_result = evaluator.evaluate_question(
        question_id=question_id,
        predicted_parcels=ground_truth,
        constraint_satisfaction={
            "Geospatial": 1.0,
            "Market": 1.0
        }
    )
    print(f"Perfect prediction metrics: {perfect_result['metrics']}")
    
    # Imperfect prediction (missing some parcels)
    imperfect_parcels = ground_truth[:-2]  # Remove the last 2 parcels
    imperfect_result = evaluator.evaluate_question(
        question_id=2,
        predicted_parcels=imperfect_parcels,
        constraint_satisfaction={
            "Geospatial": 0.8,
            "Temporal": 0.9,
            "Economic": 0.7,
            "Market": 0.85
        }
    )
    print(f"Imperfect prediction metrics: {imperfect_result['metrics']}")
    
    # Generate overall report
    report = evaluator.generate_report(os.path.join(OUTPUT_DIR, "evaluation_report.json"))
    print(f"Overall metrics: {report['overall_metrics']}")
    
    # Create visualizations
    visualization_paths = evaluator.visualize_metrics()
    print(f"Visualizations created: {visualization_paths}")
