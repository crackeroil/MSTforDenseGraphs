import csv
import os
from datetime import datetime

class BenchmarkLogger:
    def __init__(self, filename='benchmark_results.csv'):
        self.filename = filename
        self.initialize_file()

    def initialize_file(self):
        """Creates the file with headers if it doesn't exist."""
        if not os.path.exists(self.filename):
            with open(self.filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'Algorithm', 'Dataset', 'Epsilon', 'Phase', 'Round', 'Edges_Count', 'Duration_Seconds', 'Notes'])

    def log(self, algorithm, dataset, epsilon, phase, round_num, edges_count, duration, notes=''):
        """Logs a single entry to the CSV file."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, algorithm, dataset, epsilon, phase, round_num, edges_count, duration, notes])
        
        # Also print to console for immediate feedback
        print(f"[LOG] {algorithm} | {dataset} | eps={epsilon} | {phase} | Round {round_num} | Edges: {edges_count} | Time: {duration}s")
