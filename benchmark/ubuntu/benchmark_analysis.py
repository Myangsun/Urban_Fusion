"""
Analyze the site selection benchmark JSON to understand question patterns and constraint types.
This will help in designing the agent framework architecture.
"""

import json
import pandas as pd
from collections import Counter

# Load the benchmark JSON
with open('/home/ubuntu/upload/site_selection_benchmark.json', 'r') as f:
    benchmark_data = json.load(f)

# Analyze the structure of the benchmark questions
print(f"Total number of benchmark questions: {len(benchmark_data)}")

# Analyze constraint types
constraint_types = []
for question in benchmark_data:
    for constraint in question['constraints']:
        constraint_type = constraint.split(':')[0].strip()
        constraint_types.append(constraint_type)

constraint_counts = Counter(constraint_types)
print("\nConstraint type distribution:")
for constraint_type, count in constraint_counts.items():
    print(f"  {constraint_type}: {count}")

# Analyze geospatial constraints
geospatial_constraints = []
for question in benchmark_data:
    for constraint in question['constraints']:
        if constraint.startswith('Geospatial:'):
            geospatial_constraints.append(constraint)

print("\nGeospatial constraint patterns:")
for constraint in set(geospatial_constraints):
    count = geospatial_constraints.count(constraint)
    print(f"  {constraint} (appears {count} times)")

# Analyze temporal constraints
temporal_constraints = []
for question in benchmark_data:
    for constraint in question['constraints']:
        if constraint.startswith('Temporal:'):
            temporal_constraints.append(constraint)

print("\nTemporal constraint patterns:")
for constraint in set(temporal_constraints):
    count = temporal_constraints.count(constraint)
    print(f"  {constraint} (appears {count} times)")

# Analyze economic constraints
economic_constraints = []
for question in benchmark_data:
    for constraint in question['constraints']:
        if constraint.startswith('Economic:'):
            economic_constraints.append(constraint)

print("\nEconomic constraint patterns:")
for constraint in set(economic_constraints):
    count = economic_constraints.count(constraint)
    print(f"  {constraint} (appears {count} times)")

# Analyze market constraints
market_constraints = []
for question in benchmark_data:
    for constraint in question['constraints']:
        if constraint.startswith('Market:'):
            market_constraints.append(constraint)

print("\nMarket constraint patterns:")
for constraint in set(market_constraints):
    count = market_constraints.count(constraint)
    print(f"  {constraint} (appears {count} times)")

# Analyze answer sizes
answer_sizes = [len(question['answer']['parcels']) for question in benchmark_data]
print(f"\nAnswer size statistics:")
print(f"  Min: {min(answer_sizes)}")
print(f"  Max: {max(answer_sizes)}")
print(f"  Average: {sum(answer_sizes) / len(answer_sizes):.2f}")

# Analyze common patterns in questions
print("\nCommon patterns in questions:")
radius_patterns = []
for question in benchmark_data:
    question_text = question['question']
    if 'radius' in question_text:
        radius_values = []
        for constraint in question['constraints']:
            if 'radius' in constraint:
                parts = constraint.split('radius')
                for part in parts:
                    if 'km' in part:
                        radius = part.split('km')[0].strip()
                        if radius.startswith('buffer'):
                            radius = radius.split('buffer')[1].strip()
                        radius_values.append(radius)
        radius_patterns.append(f"Question {question['question_id']}: {', '.join(radius_values)} km")

print("  Radius values used in questions:")
for pattern in radius_patterns:
    print(f"    {pattern}")

# Summary of findings
print("\nSummary of findings:")
print(f"1. The benchmark contains {len(benchmark_data)} questions with varying constraints.")
print(f"2. Most common constraint type: {constraint_counts.most_common(1)[0][0]} ({constraint_counts.most_common(1)[0][1]} occurrences)")
print(f"3. Average number of parcels in answers: {sum(answer_sizes) / len(answer_sizes):.2f}")
print(f"4. All questions involve geospatial constraints with radius buffers.")
print(f"5. Questions combine multiple constraint types to create complex site selection scenarios.")
