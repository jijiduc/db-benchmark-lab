#!/usr/bin/env python3
"""
Module to run benchmarks comparing MongoDB, PostgreSQL, and Redis.
"""
import time
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import sys

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.mongodb.client import MongoDBClient
from src.postgresql.client import PostgreSQLClient
from src.redis.client import RedisClient
from src.common.generate_data import generate_dataset

class DatabaseBenchmark:
    """Class to run benchmarks comparing different databases."""
    
    def __init__(self, db_clients, data_sizes=None, batch_sizes=None, result_dir='results'):
        """Initialize the benchmark with the database clients."""
        self.db_clients = db_clients
        self.data_sizes = data_sizes or [100, 1000, 5000, 10000]
        self.batch_sizes = batch_sizes or [1, 10, 100, 1000]
        self.result_dir = result_dir
        os.makedirs(result_dir, exist_ok=True)
        
    def _generate_test_data(self, size):
        """Generate test data for the benchmark."""
        return generate_dataset(size)
        
    def run_insert_benchmark(self):
        """Run benchmark for insert operations."""
        results = []
        
        for db_name, client in self.db_clients.items():
            print(f"Running insert benchmark for {db_name}...")
            
            # Clear previous data
            if db_name == 'mongodb':
                client.clear_collection()
            elif db_name == 'postgresql':
                client.clear_table()
            elif db_name == 'redis':
                client.clear_data()
            
            # Test single inserts
            for data_size in self.data_sizes:
                events = self._generate_test_data(data_size)
                
                # Measure time for single inserts
                start_time = time.time()
                for event in events:
                    client.insert_event(event)
                end_time = time.time()
                
                single_time = end_time - start_time
                results.append({
                    'database': db_name,
                    'operation': 'insert_single',
                    'data_size': data_size,
                    'batch_size': 1,
                    'time': single_time,
                    'ops_per_second': data_size / single_time if single_time > 0 else 0
                })
                print(f"{db_name} - Single inserts - {data_size} events: {single_time:.4f}s")
                
                # Clear again for batch tests
                if db_name == 'mongodb':
                    client.clear_collection()
                elif db_name == 'postgresql':
                    client.clear_table()
                elif db_name == 'redis':
                    client.clear_data()
                
            # Test batch inserts
            for batch_size in self.batch_sizes:
                if batch_size <= max(self.data_sizes):
                    events = self._generate_test_data(batch_size)
                    
                    start_time = time.time()
                    client.insert_events(events)
                    end_time = time.time()
                    
                    batch_time = end_time - start_time
                    results.append({
                        'database': db_name,
                        'operation': 'insert_batch',
                        'data_size': batch_size,
                        'batch_size': batch_size,
                        'time': batch_time,
                        'ops_per_second': batch_size / batch_time if batch_time > 0 else 0
                    })
                    print(f"{db_name} - Batch insert - {batch_size} events: {batch_time:.4f}s")
        
        # Save results
        df = pd.DataFrame(results)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        df.to_csv(f"{self.result_dir}/insert_benchmark_{timestamp}.csv", index=False)
        
        return df
        
    def run_query_benchmark(self, setup_data_size=1000):
        """Run benchmark for query operations."""
        results = []
        
        # Generate and insert test data
        events = self._generate_test_data(setup_data_size)
        user_ids = list(set(event['user_id'] for event in events))[:5]  # Take 5 random users
        pages = list(set(event['page'] for event in events if 'page' in event))[:3]  # Take 3 random pages
        
        for db_name, client in self.db_clients.items():
            print(f"Running query benchmark for {db_name}...")
            
            # Clear previous data
            if db_name == 'mongodb':
                client.clear_collection()
            elif db_name == 'postgresql':
                client.clear_table()
            elif db_name == 'redis':
                client.clear_data()
                
            # Insert test data
            client.insert_events(events)
            
            # Benchmark find_events_by_user
            for user_id in user_ids:
                start_time = time.time()
                client.find_events_by_user(user_id)
                end_time = time.time()
                
                query_time = end_time - start_time
                results.append({
                    'database': db_name,
                    'operation': 'query_by_user',
                    'data_size': setup_data_size,
                    'query_param': user_id,
                    'time': query_time
                })
                print(f"{db_name} - Query by user {user_id}: {query_time:.4f}s")
                
            # Benchmark find_events_by_page
            for page in pages:
                start_time = time.time()
                client.find_events_by_page(page)
                end_time = time.time()
                
                query_time = end_time - start_time
                results.append({
                    'database': db_name,
                    'operation': 'query_by_page',
                    'data_size': setup_data_size,
                    'query_param': page,
                    'time': query_time
                })
                print(f"{db_name} - Query by page {page}: {query_time:.4f}s")
                
            # Benchmark count_events
            start_time = time.time()
            client.count_events()
            end_time = time.time()
            
            count_time = end_time - start_time
            results.append({
                'database': db_name,
                'operation': 'count_events',
                'data_size': setup_data_size,
                'query_param': 'N/A',
                'time': count_time
            })
            print(f"{db_name} - Count events: {count_time:.4f}s")
        
        # Save results
        df = pd.DataFrame(results)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        df.to_csv(f"{self.result_dir}/query_benchmark_{timestamp}.csv", index=False)
        
        return df
        
    def visualize_results(self, insert_df=None, query_df=None):
        """Visualize benchmark results."""
        plt.figure(figsize=(12, 8))
        
        # Visualize insert benchmark
        if insert_df is not None:
            plt.subplot(2, 1, 1)
            single_inserts = insert_df[insert_df['operation'] == 'insert_single']
            
            for db_name in single_inserts['database'].unique():
                db_data = single_inserts[single_inserts['database'] == db_name]
                plt.plot(db_data['data_size'], db_data['ops_per_second'], marker='o', label=db_name)
                
            plt.title('Single Insert Performance')
            plt.xlabel('Number of Events')
            plt.ylabel('Operations per Second')
            plt.legend()
            plt.grid(True)
            
        # Visualize query benchmark
        if query_df is not None:
            plt.subplot(2, 1, 2)
            query_by_user = query_df[query_df['operation'] == 'query_by_user']
            
            # Group by database and calculate mean time
            query_mean = query_by_user.groupby('database')['time'].mean().reset_index()
            
            plt.bar(query_mean['database'], query_mean['time'])
            plt.title('Query by User Performance (Average)')
            plt.xlabel('Database')
            plt.ylabel('Time (seconds)')
            plt.grid(True, axis='y')
            
        plt.tight_layout()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(f"{self.result_dir}/benchmark_visualization_{timestamp}.png")
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Run database benchmarks.')
    parser.add_argument('--data-sizes', type=int, nargs='+', default=[100, 1000, 5000, 10000],
                        help='Data sizes to test')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 10, 100, 1000],
                        help='Batch sizes to test')
    parser.add_argument('--setup-data-size', type=int, default=1000,
                        help='Data size for query benchmarks')
    parser.add_argument('--result-dir', type=str, default='results',
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    # Initialize database clients
    db_clients = {
        'mongodb': MongoDBClient(),
        'postgresql': PostgreSQLClient(connection_params="dbname=mousebenchmark user=jduc password=password"),
        'redis': RedisClient()
    }
    
    # Run benchmarks
    benchmark = DatabaseBenchmark(
        db_clients=db_clients,
        data_sizes=args.data_sizes,
        batch_sizes=args.batch_sizes,
        result_dir=args.result_dir
    )
    
    insert_df = benchmark.run_insert_benchmark()
    query_df = benchmark.run_query_benchmark(setup_data_size=args.setup_data_size)
    
    # Visualize results
    benchmark.visualize_results(insert_df, query_df)
    
    # Close connections
    for client in db_clients.values():
        client.close()

if __name__ == "__main__":
    main()