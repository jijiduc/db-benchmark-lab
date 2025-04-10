#!/usr/bin/env python3
"""
Module to compare performance of MongoDB, PostgreSQL and Redis
"""
import random
import time
import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from datetime import datetime, timedelta
import argparse
import sys
import uuid

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.mongodb.client import MongoDBClient
from src.postgresql.client import PostgreSQLClient
from src.redis.client import RedisClient
from src.common.generate_data import generate_dataset, generate_advanced_dataset

def set_uniform_y_limits(axes_list, padding=0.1):
    """
    Set uniform Y limits across multiple axes.
    
    Args:
        axes_list (list): List of matplotlib axes to equalize
        padding (float): Padding percentage above max value
    """
    y_min = float('inf')
    y_max = float('-inf')
    
    # Find global min and max
    for ax in axes_list:
        bottom, top = ax.get_ylim()
        y_min = min(y_min, bottom)
        y_max = max(y_max, top)
    
    # Add padding to top
    y_max += (y_max - y_min) * padding
    
    # Set uniform limits
    for ax in axes_list:
        ax.set_ylim(y_min, y_max)

def annotate_key_points(ax, x_data, y_data, labels=None, threshold=0.9):
    """
    Annotate important points in the data (maximum, minimum, crossovers).
    
    Args:
        ax (Axes): Matplotlib axes to annotate
        x_data (list): X coordinates
        y_data (list): Y coordinates
        labels (list): Optional labels for the data series
        threshold (float): Threshold ratio to max value for annotation
    """
    max_idx = np.argmax(y_data)
    min_idx = np.argmin(y_data)
    
    # Maximum point
    ax.annotate(f"Max: {y_data[max_idx]:.1f}",
                xy=(x_data[max_idx], y_data[max_idx]),
                xytext=(0, 15), textcoords='offset points',
                ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Minimum point if significantly different from maximum
    if y_data[min_idx] < y_data[max_idx] * threshold:
        ax.annotate(f"Min: {y_data[min_idx]:.1f}",
                    xy=(x_data[min_idx], y_data[min_idx]),
                    xytext=(0, -15), textcoords='offset points',
                    ha='center', va='top',
                    bbox=dict(boxstyle='round,pad=0.3', fc='lightblue', alpha=0.5),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

def apply_hatches_to_bars(ax):
    """Apply hatches to bars to distinguish them in addition to colors."""
    hatches = ['/', '\\', 'x', '.', 'o', '*']
    
    for i, patch in enumerate(ax.patches):
        # Apply hatch pattern based on bar index
        hatch_idx = i % len(hatches)
        patch.set_hatch(hatches[hatch_idx])
        # Make hatch visible but not overwhelming
        patch.set_edgecolor('black')
        patch.set_linewidth(0.5)

class SimplifiedDatabaseBenchmark:
    """Class for running benchmarks comparing different databases."""
    
    def __init__(self, db_clients, data_sizes=None, batch_sizes=None, iterations=3, result_dir='results'):
        """Initialize the benchmark with database clients."""
        self.db_clients = db_clients
        self.data_sizes = data_sizes or [100, 1000, 5000, 10000]
        self.batch_sizes = batch_sizes or [1, 10, 50, 100, 500, 1000]
        self.iterations = iterations
        self.result_dir = result_dir
        
        # Create results directory if it doesn't exist
        os.makedirs(result_dir, exist_ok=True)
        
        # Timestamp for results files
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
    def _generate_test_data(self, size):
        """Generate test data for the benchmark."""
        events = generate_dataset(size)
        
        # Ensure each event has an event_id
        for event in events:
            if 'event_id' not in event:
                event['event_id'] = str(uuid.uuid4())
                
        return events
        
    def run_insert_benchmark(self):
        """Run the benchmark for insert operations with multiple iterations."""
        all_results = []
        
        for iteration in range(1, self.iterations + 1):
            print(f"\nIteration {iteration}/{self.iterations} of insert tests")
            
            for db_name, client in self.db_clients.items():
                print(f"Running insert benchmark for {db_name}...")
                
                # Test individual inserts
                for data_size in self.data_sizes:
                    events = self._generate_test_data(data_size)
                    
                    # Clean previous data
                    if db_name == 'mongodb':
                        client.clear_collection()
                    elif db_name == 'postgresql':
                        client.clear_table()
                    elif db_name == 'redis':
                        client.clear_data()
                    
                    # Measure time for individual inserts
                    start_time = time.time()
                    for event in events:
                        client.insert_event(event)
                    end_time = time.time()
                    
                    single_time = end_time - start_time
                    all_results.append({
                        'database': db_name,
                        'operation': 'insert_single',
                        'data_size': data_size,
                        'batch_size': 1,
                        'time': single_time,
                        'ops_per_second': data_size / single_time if single_time > 0 else 0,
                        'iteration': iteration
                    })
                    print(f"{db_name} - Individual inserts - {data_size} events: {single_time:.4f}s")
                    
                    # Clean again for batch tests
                    if db_name == 'mongodb':
                        client.clear_collection()
                    elif db_name == 'postgresql':
                        client.clear_table()
                    elif db_name == 'redis':
                        client.clear_data()
                
                # Test batch inserts
                for batch_size in self.batch_sizes:
                    # Check that batch size is valid
                    if batch_size <= max(self.data_sizes):
                        events = self._generate_test_data(batch_size)
                        
                        # Clean previous data
                        if db_name == 'mongodb':
                            client.clear_collection()
                        elif db_name == 'postgresql':
                            client.clear_table()
                        elif db_name == 'redis':
                            client.clear_data()
                        
                        start_time = time.time()
                        client.insert_events(events)
                        end_time = time.time()
                        
                        batch_time = end_time - start_time
                        all_results.append({
                            'database': db_name,
                            'operation': 'insert_batch',
                            'data_size': batch_size,
                            'batch_size': batch_size,
                            'time': batch_time,
                            'ops_per_second': batch_size / batch_time if batch_time > 0 else 0,
                            'iteration': iteration
                        })
                        print(f"{db_name} - Batch insert - {batch_size} events: {batch_time:.4f}s")
        
        # Convert results to DataFrame and save
        df = pd.DataFrame(all_results)
        df.to_csv(f"{self.result_dir}/insert_benchmark_{self.timestamp}.csv", index=False)
        
        return df
        
    def run_query_benchmark(self, setup_data_size=5000):
        """Run the benchmark for simple query operations."""
        all_results = []
        
        # Generate and insert test data
        events = self._generate_test_data(setup_data_size)
        
        # Extract unique values for query parameters
        user_ids = list(set(event['user_id'] for event in events))[:5]
        pages = list(set(event['page'] for event in events))[:3]
        
        for iteration in range(1, self.iterations + 1):
            print(f"\nIteration {iteration}/{self.iterations} of query tests")
            
            for db_name, client in self.db_clients.items():
                print(f"Running query benchmark for {db_name}...")
                
                # Clean previous data
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
                    results = client.find_events_by_user(user_id)
                    end_time = time.time()
                    
                    query_time = end_time - start_time
                    all_results.append({
                        'database': db_name,
                        'operation': 'query_by_user',
                        'data_size': setup_data_size,
                        'query_param': user_id,
                        'result_size': len(results) if results else 0,
                        'time': query_time,
                        'iteration': iteration
                    })
                    print(f"{db_name} - Query by user {user_id}: {query_time:.4f}s")
                    
                # Benchmark find_events_by_page
                for page in pages:
                    start_time = time.time()
                    results = client.find_events_by_page(page)
                    end_time = time.time()
                    
                    query_time = end_time - start_time
                    all_results.append({
                        'database': db_name,
                        'operation': 'query_by_page',
                        'data_size': setup_data_size,
                        'query_param': page,
                        'result_size': len(results) if results else 0,
                        'time': query_time,
                        'iteration': iteration
                    })
                    print(f"{db_name} - Query by page {page}: {query_time:.4f}s")
                    
                # Benchmark count_events
                try:
                    start_time = time.time()
                    count = client.count_events()
                    end_time = time.time()
                    
                    count_time = end_time - start_time
                    all_results.append({
                        'database': db_name,
                        'operation': 'count_events',
                        'data_size': setup_data_size,
                        'query_param': 'N/A',
                        'result_size': count if count else 0,
                        'time': count_time,
                        'iteration': iteration
                    })
                    print(f"{db_name} - Count events: {count_time:.4f}s")
                except Exception as e:
                    print(f"Error counting events for {db_name}: {e}")
        
        # Convert results to DataFrame and save
        df = pd.DataFrame(all_results)
        df.to_csv(f"{self.result_dir}/query_benchmark_{self.timestamp}.csv", index=False)
        
        return df
    
    def run_advanced_query_benchmark(self, setup_data_size=5000, iterations=3):
        """Run benchmark for database-specific advanced query operations."""
        all_results = []
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Generate and insert advanced test data
        events = generate_advanced_dataset(setup_data_size)
        
        # Define time range for queries
        now = datetime.now()
        start_date = now - timedelta(days=7)
        end_date = now
        
        # Define screen zone for queries
        x_min, x_max = 100, 800
        y_min, y_max = 100, 600
        
        for iteration in range(1, iterations + 1):
            print(f"\nIteration {iteration}/{iterations} of advanced query tests")
            
            for db_name, client in self.db_clients.items():
                print(f"Running advanced query benchmark for {db_name}...")
                
                # Clean previous data
                if db_name == 'mongodb':
                    client.clear_collection()
                elif db_name == 'postgresql':
                    client.clear_table()
                elif db_name == 'redis':
                    client.clear_data()
                    
                # Insert test data
                client.insert_events(events)
                
                # Test time range query
                try:
                    start_time = time.time()
                    time_range_results = client.find_events_by_time_range(start_date, end_date)
                    end_time = time.time()
                    
                    time_range_time = end_time - start_time
                    all_results.append({
                        'database': db_name,
                        'operation': 'query_time_range',
                        'data_size': setup_data_size,
                        'result_size': len(time_range_results) if time_range_results else 0,
                        'time': time_range_time,
                        'iteration': iteration
                    })
                    print(f"{db_name} - Time range query: {time_range_time:.4f}s")
                except Exception as e:
                    print(f"Error in time range query for {db_name}: {e}")
                
                # Test screen zone query
                try:
                    start_time = time.time()
                    zone_results = client.find_events_in_screen_zone(x_min, x_max, y_min, y_max)
                    end_time = time.time()
                    
                    zone_time = end_time - start_time
                    all_results.append({
                        'database': db_name,
                        'operation': 'query_screen_zone',
                        'data_size': setup_data_size,
                        'result_size': len(zone_results) if zone_results else 0,
                        'time': zone_time,
                        'iteration': iteration
                    })
                    print(f"{db_name} - Screen zone query: {zone_time:.4f}s")
                except Exception as e:
                    print(f"Error in screen zone query for {db_name}: {e}")
                
                # Test aggregation by user
                try:
                    start_time = time.time()
                    user_agg_results = client.aggregate_events_by_user()
                    end_time = time.time()
                    
                    user_agg_time = end_time - start_time
                    all_results.append({
                        'database': db_name,
                        'operation': 'aggregate_by_user',
                        'data_size': setup_data_size,
                        'result_size': len(user_agg_results) if user_agg_results else 0,
                        'time': user_agg_time,
                        'iteration': iteration
                    })
                    print(f"{db_name} - Aggregate by user: {user_agg_time:.4f}s")
                except Exception as e:
                    print(f"Error in user aggregation for {db_name}: {e}")
                
                # Test aggregation by page
                try:
                    start_time = time.time()
                    page_agg_results = client.aggregate_events_by_page()
                    end_time = time.time()
                    
                    page_agg_time = end_time - start_time
                    all_results.append({
                        'database': db_name,
                        'operation': 'aggregate_by_page',
                        'data_size': setup_data_size,
                        'result_size': len(page_agg_results) if page_agg_results else 0,
                        'time': page_agg_time,
                        'iteration': iteration
                    })
                    print(f"{db_name} - Aggregate by page: {page_agg_time:.4f}s")
                except Exception as e:
                    print(f"Error in page aggregation for {db_name}: {e}")
                
                # Test aggregation by device
                try:
                    start_time = time.time()
                    device_agg_results = client.aggregate_events_by_device()
                    end_time = time.time()
                    
                    device_agg_time = end_time - start_time
                    all_results.append({
                        'database': db_name,
                        'operation': 'aggregate_by_device',
                        'data_size': setup_data_size,
                        'result_size': len(device_agg_results) if device_agg_results else 0,
                        'time': device_agg_time,
                        'iteration': iteration
                    })
                    print(f"{db_name} - Aggregate by device: {device_agg_time:.4f}s")
                except Exception as e:
                    print(f"Error in device aggregation for {db_name}: {e}")
        
        # Convert results to DataFrame and save
        df = pd.DataFrame(all_results)
        df.to_csv(f"{self.result_dir}/advanced_query_benchmark_{timestamp}.csv", index=False)
        
        return df
    
    def run_update_benchmark(self, setup_data_size=5000, update_percentage=0.1, iterations=3):
        """Run benchmark for update operations."""
        all_results = []
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Generate and insert test data
        events = generate_dataset(setup_data_size)
        
        for iteration in range(1, iterations + 1):
            print(f"\nIteration {iteration}/{iterations} of update tests")
            
            for db_name, client in self.db_clients.items():
                print(f"Running update benchmark for {db_name}...")
                
                # Clean previous data
                if db_name == 'mongodb':
                    client.clear_collection()
                elif db_name == 'postgresql':
                    client.clear_table()
                elif db_name == 'redis':
                    client.clear_data()
                    
                # Insert test data
                inserted_events = client.insert_events(events)
                
                # Calculate number of events to update
                update_count = int(setup_data_size * update_percentage)
                
                # Single update test
                event_to_update = events[0]
                update_data = {'updated': True, 'update_timestamp': datetime.now().isoformat()}
                
                start_time = time.time()
                client.update_event(event_to_update['event_id'], update_data)
                end_time = time.time()
                
                single_update_time = end_time - start_time
                all_results.append({
                    'database': db_name,
                    'operation': 'update_single',
                    'data_size': setup_data_size,
                    'updated_count': 1,
                    'time': single_update_time,
                    'ops_per_second': 1 / single_update_time if single_update_time > 0 else 0,
                    'iteration': iteration
                })
                print(f"{db_name} - Single update: {single_update_time:.4f}s")
                
                # Batch update test
                events_to_update = random.sample(events, update_count)
                event_ids = [e['event_id'] for e in events_to_update]
                batch_update_data = {'batch_updated': True, 'update_timestamp': datetime.now().isoformat()}
                
                start_time = time.time()
                updated = client.update_events_batch(event_ids, batch_update_data)
                end_time = time.time()
                
                batch_update_time = end_time - start_time
                all_results.append({
                    'database': db_name,
                    'operation': 'update_batch',
                    'data_size': setup_data_size,
                    'updated_count': update_count,
                    'time': batch_update_time,
                    'ops_per_second': update_count / batch_update_time if batch_update_time > 0 else 0,
                    'iteration': iteration
                })
                print(f"{db_name} - Batch update ({update_count} events): {batch_update_time:.4f}s")
                
                # Conditional update test
                condition = {'page': '/page_1'}
                conditional_update_data = {'page_updated': True, 'update_timestamp': datetime.now().isoformat()}
                
                start_time = time.time()
                updated = client.update_events_conditional(condition, conditional_update_data)
                end_time = time.time()
                
                condition_update_time = end_time - start_time
                all_results.append({
                    'database': db_name,
                    'operation': 'update_conditional',
                    'data_size': setup_data_size,
                    'updated_count': updated if isinstance(updated, int) else 0,
                    'time': condition_update_time,
                    'ops_per_second': (updated if isinstance(updated, int) else 0) / condition_update_time if condition_update_time > 0 else 0,
                    'iteration': iteration
                })
                print(f"{db_name} - Conditional update: {condition_update_time:.4f}s")
        
        # Convert results to DataFrame and save
        df = pd.DataFrame(all_results)
        df.to_csv(f"{self.result_dir}/update_benchmark_{timestamp}.csv", index=False)
        
        return df

    def run_delete_benchmark(self, setup_data_size=5000, delete_percentage=0.1, iterations=3):
        """Run benchmark for delete operations."""
        all_results = []
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Generate test data
        events = generate_dataset(setup_data_size)
        
        for iteration in range(1, iterations + 1):
            print(f"\nIteration {iteration}/{iterations} of delete tests")
            
            for db_name, client in self.db_clients.items():
                print(f"Running delete benchmark for {db_name}...")
                
                # Clean previous data
                if db_name == 'mongodb':
                    client.clear_collection()
                elif db_name == 'postgresql':
                    client.clear_table()
                elif db_name == 'redis':
                    client.clear_data()
                    
                # Insert test data
                client.insert_events(events)
                
                # Calculate number of events to delete
                delete_count = int(setup_data_size * delete_percentage)
                
                # Single delete test
                event_to_delete = events[0]
                
                start_time = time.time()
                client.delete_event(event_to_delete['event_id'])
                end_time = time.time()
                
                single_delete_time = end_time - start_time
                all_results.append({
                    'database': db_name,
                    'operation': 'delete_single',
                    'data_size': setup_data_size,
                    'deleted_count': 1,
                    'time': single_delete_time,
                    'ops_per_second': 1 / single_delete_time if single_delete_time > 0 else 0,
                    'iteration': iteration
                })
                print(f"{db_name} - Single delete: {single_delete_time:.4f}s")
                
                # Batch delete test
                events_to_delete = random.sample(events[1:], delete_count)  # Skip the first event as it was already deleted
                event_ids = [e['event_id'] for e in events_to_delete]
                
                start_time = time.time()
                deleted = client.delete_events_batch(event_ids)
                end_time = time.time()
                
                batch_delete_time = end_time - start_time
                all_results.append({
                    'database': db_name,
                    'operation': 'delete_batch',
                    'data_size': setup_data_size,
                    'deleted_count': delete_count,
                    'time': batch_delete_time,
                    'ops_per_second': delete_count / batch_delete_time if batch_delete_time > 0 else 0,
                    'iteration': iteration
                })
                print(f"{db_name} - Batch delete ({delete_count} events): {batch_delete_time:.4f}s")
                
                # Conditional delete test
                condition = {'page': '/page_2'}
                
                start_time = time.time()
                deleted = client.delete_events_conditional(condition)
                end_time = time.time()
                
                condition_delete_time = end_time - start_time
                all_results.append({
                    'database': db_name,
                    'operation': 'delete_conditional',
                    'data_size': setup_data_size,
                    'deleted_count': deleted if isinstance(deleted, int) else 0,
                    'time': condition_delete_time,
                    'ops_per_second': (deleted if isinstance(deleted, int) else 0) / condition_delete_time if condition_delete_time > 0 else 0,
                    'iteration': iteration
                })
                print(f"{db_name} - Conditional delete: {condition_delete_time:.4f}s")
        
        # Convert results to DataFrame and save
        df = pd.DataFrame(all_results)
        df.to_csv(f"{self.result_dir}/delete_benchmark_{timestamp}.csv", index=False)
        
        return df
    
    def _visualize_insert_performance(self, df, viz_dir):
        """Generate visualizations for insert performance."""
        # 1.1 Individual insert performance
        plt.figure(figsize=(12, 8))
        
        # Aggregate data by database and data size
        single_inserts = df[df['operation'] == 'insert_single']
        grouped = single_inserts.groupby(['database', 'data_size'])['ops_per_second'].agg(['mean', 'std']).reset_index()
        
        # Create a chart for each database
        for db_name in grouped['database'].unique():
            db_data = grouped[grouped['database'] == db_name]
            line, = plt.plot(db_data['data_size'], db_data['mean'], marker='o', linewidth=2, label=db_name)
            plt.fill_between(
                db_data['data_size'],
                db_data['mean'] - db_data['std'],
                db_data['mean'] + db_data['std'],
                alpha=0.2
            )
            
            # Annotate maximum point
            max_idx = db_data['mean'].idxmax()
            if not pd.isna(max_idx):
                max_val = db_data.iloc[max_idx]['mean']
                max_x = db_data.iloc[max_idx]['data_size']
                plt.annotate(f"{db_name} max: {max_val:.0f} ops/s",
                         xy=(max_x, max_val),
                         xytext=(0, 10),  # vertical offset
                         textcoords='offset points',
                         ha='center',
                         va='bottom',
                         bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.5),
                         color=line.get_color())
        
        # Add reference line for acceptable throughput
        target_throughput = 3000  # Target operations per second
        plt.axhline(y=target_throughput, color='g', linestyle='--', alpha=0.7)
        plt.text(max(db_data['data_size'])*0.5, target_throughput*1.05, 
                 'Target Throughput (3000 ops/s)', ha='center', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
        
        plt.title('Individual Insert Performance', fontsize=16)
        plt.xlabel('Number of Events', fontsize=14)
        plt.ylabel('Operations per Second', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{viz_dir}/insert_single_performance.png", dpi=300)
        plt.close()
        
        # 1.2 Batch insert performance
        plt.figure(figsize=(12, 8))
        
        batch_inserts = df[df['operation'] == 'insert_batch']
        grouped = batch_inserts.groupby(['database', 'batch_size'])['ops_per_second'].agg(['mean', 'std']).reset_index()
        
        for db_name in grouped['database'].unique():
            db_data = grouped[grouped['database'] == db_name]
            line, = plt.plot(db_data['batch_size'], db_data['mean'], marker='s', linewidth=2, label=db_name)
            plt.fill_between(
                db_data['batch_size'],
                db_data['mean'] - db_data['std'],
                db_data['mean'] + db_data['std'],
                alpha=0.2
            )
            
            # Annotate maximum point
            max_idx = db_data['mean'].idxmax()
            if not pd.isna(max_idx):
                max_val = db_data.iloc[max_idx]['mean']
                max_x = db_data.iloc[max_idx]['batch_size']
                plt.annotate(f"{db_name} max: {max_val:.0f} ops/s",
                         xy=(max_x, max_val),
                         xytext=(0, 10),  # vertical offset
                         textcoords='offset points',
                         ha='center',
                         va='bottom',
                         bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.5),
                         color=line.get_color())
        
        plt.title('Batch Insert Performance', fontsize=16)
        plt.xlabel('Batch Size', fontsize=14)
        plt.ylabel('Operations per Second', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{viz_dir}/insert_batch_performance.png", dpi=300)
        plt.close()
        
        # 1.3 Batch performance gain
        plt.figure(figsize=(14, 10))
        
        # For each database, compare batch efficiency vs individual inserts
        for db_name in df['database'].unique():
            db_single = df[(df['database'] == db_name) & (df['operation'] == 'insert_single')]
            db_batch = df[(df['database'] == db_name) & (df['operation'] == 'insert_batch')]
            
            # Calculate average performance gain for each batch size
            performance_gains = []
            batch_sizes = []
            
            for batch_size in sorted(db_batch['batch_size'].unique()):
                # Find the individual insert equivalent
                single_equiv = db_single[db_single['data_size'] == batch_size]
                batch_equiv = db_batch[db_batch['batch_size'] == batch_size]
                
                if not single_equiv.empty and not batch_equiv.empty:
                    single_ops = single_equiv['ops_per_second'].mean()
                    batch_ops = batch_equiv['ops_per_second'].mean()
                    
                    # Calculate the ratio (how many times faster)
                    gain = batch_ops / single_ops if single_ops > 0 else 0
                    performance_gains.append(gain)
                    batch_sizes.append(batch_size)
            
            if performance_gains:  # Only plot if we have data
                line, = plt.plot(batch_sizes, performance_gains, marker='D', linewidth=2, label=db_name)
                
                # Annotate maximum gain
                if performance_gains:
                    max_idx = np.argmax(performance_gains)
                    max_gain = performance_gains[max_idx]
                    max_batch = batch_sizes[max_idx]
                    plt.annotate(f"{db_name} max gain: {max_gain:.1f}x",
                            xy=(max_batch, max_gain),
                            xytext=(0, 10),
                            textcoords='offset points',
                            ha='center',
                            va='bottom',
                            bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.5),
                            color=line.get_color())
        
        plt.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='Equivalence (no gain)')
        plt.title('Performance Gain of Batch Inserts vs. Individual Inserts', fontsize=16)
        plt.xlabel('Batch Size', fontsize=14)
        plt.ylabel('Speedup Factor', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{viz_dir}/insert_batch_efficiency.png", dpi=300)
        plt.close()
    
    def _visualize_query_performance(self, df, viz_dir):
        """Generate visualizations for query performance."""
        # 2.1 Average execution time by query type
        plt.figure(figsize=(14, 10))
        
        # Aggregate data by database and operation type
        grouped = df.groupby(['database', 'operation'])['time'].agg(['mean', 'std']).reset_index()
        
        # Create grouped bar chart
        operations = sorted(grouped['operation'].unique())
        num_operations = len(operations)
        bar_width = 0.8 / len(grouped['database'].unique())
        index = np.arange(num_operations)
        
        for i, db_name in enumerate(sorted(grouped['database'].unique())):
            db_data = grouped[grouped['database'] == db_name]
            means = []
            stds = []
            
            for op in operations:
                op_data = db_data[db_data['operation'] == op]
                if not op_data.empty:
                    means.append(op_data['mean'].values[0])
                    stds.append(op_data['std'].values[0])
                else:
                    means.append(0)
                    stds.append(0)
            
            position = index + i * bar_width
            plt.bar(position, means, bar_width, yerr=stds, capsize=5, label=db_name)
            
            # Annotate bars with values
            for j, v in enumerate(means):
                if v > 0:
                    plt.text(position[j], v + stds[j] + 0.001, 
                             f"{v:.4f}s", 
                             ha='center', va='bottom', 
                             fontsize=8, rotation=45)
        
        # Apply hatches to bars
        apply_hatches_to_bars(plt.gca())
        
        # Add reference line for acceptable query time
        acceptable_query_time = 0.01  # 10ms
        plt.axhline(y=acceptable_query_time, color='r', linestyle='--', alpha=0.7)
        plt.text(index[-1], acceptable_query_time*1.1, 
                 'Acceptable Query Time (10ms)', ha='right', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
        
        plt.xticks(index + bar_width * (len(grouped['database'].unique()) - 1) / 2, 
                   [op.replace('_', ' ').title() for op in operations], 
                   rotation=45, ha='right')
        plt.title('Average Execution Time by Query Type', fontsize=16)
        plt.xlabel('Query Type', fontsize=14)
        plt.ylabel('Time (seconds)', fontsize=14)
        plt.legend(title='Database', fontsize=12)
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{viz_dir}/query_performance_by_type.png", dpi=300)
        plt.close()
        
        # 2.2 Boxplot of query times
        plt.figure(figsize=(14, 10))
        
        # Create a boxplot for each query type
        sns.boxplot(x='operation', y='time', hue='database', data=df)
        
        # Add reference line for acceptable query time
        plt.axhline(y=acceptable_query_time, color='r', linestyle='--', alpha=0.7)
        plt.text(len(operations)-1, acceptable_query_time*1.1, 
                 'Acceptable Query Time (10ms)', ha='right', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
        
        plt.title('Distribution of Execution Times by Query Type', fontsize=16)
        plt.xlabel('Query Type', fontsize=14)
        plt.ylabel('Time (seconds)', fontsize=14)
        plt.xticks(rotation=45)
        plt.legend(title='Database', fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{viz_dir}/query_time_distribution.png", dpi=300)
        plt.close()
    
    def _create_latency_throughput_graph(self, insert_df, query_df, viz_dir):
        """Create a combined graph showing the inverse relationship between latency and throughput."""
        plt.figure(figsize=(14, 10))
        
        # Create dual axis plot
        fig, ax1 = plt.subplots(figsize=(14, 8))
        ax2 = ax1.twinx()
        
        # Extract data
        for db_name in sorted(insert_df['database'].unique()):
            # Get throughput data
            throughput_data = insert_df[(insert_df['database'] == db_name) & 
                                    (insert_df['operation'] == 'insert_batch')]
            throughput_grouped = throughput_data.groupby('data_size')['ops_per_second'].mean().reset_index()
            
            # Get latency data 
            latency_data = query_df[query_df['database'] == db_name]
            latency_grouped = latency_data.groupby('operation')['time'].mean().reset_index()
            
            # Plot throughput on primary axis
            line1, = ax1.plot(throughput_grouped['data_size'], throughput_grouped['ops_per_second'], 
                     marker='o', linestyle='-', linewidth=2, 
                     label=f"{db_name} (Throughput)")
            
            # Plot latency on secondary axis if available
            if not latency_grouped.empty:
                avg_latency = latency_grouped['time'].mean()
                latency_values = [avg_latency] * len(throughput_grouped)
                line2, = ax2.plot(throughput_grouped['data_size'], 
                         latency_values, 
                         marker='s', linestyle='--', linewidth=2,
                         color=line1.get_color(), alpha=0.5,
                         label=f"{db_name} (Latency)")
        
        # Set labels and title
        ax1.set_xlabel('Batch Size / Number of Events', fontsize=14)
        ax1.set_ylabel('Throughput (Operations per Second)', fontsize=14, color='blue')
        ax2.set_ylabel('Average Latency (seconds)', fontsize=14, color='red')
        
        # Customize ticks
        ax1.tick_params(axis='y', labelcolor='blue')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Add grid and legend
        ax1.grid(True, alpha=0.3)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.1),
                   ncol=3, fontsize=12)
        
        # Add an annotation explaining the relationship
        plt.figtext(0.5, 0.01, 
                   "Inverse relationship: As throughput increases, systems typically experience higher latency.",
                   ha="center", fontsize=10, 
                   bbox={"facecolor":"orange", "alpha":0.1, "pad":5})
        
        plt.title('Relationship Between Throughput and Latency', fontsize=16)
        plt.tight_layout()
        plt.savefig(f"{viz_dir}/latency_throughput_relationship.png", dpi=300)
        plt.close()
    
    def _create_radar_chart(self, insert_df, query_df, advanced_query_df, viz_dir):
        """Create a radar chart comparing overall performance across all dimensions."""
        plt.figure(figsize=(12, 12))
        
        # Prepare data for radar chart
        categories = ['Individual Insert', 'Batch Insert', 'User Query', 
                      'Page Query', 'Advanced Queries', 'Concurrency Scalability']
        
        # Calculate normalized scores (higher is better)
        scores = {}
        
        for db_name in insert_df['database'].unique():
            db_scores = []
            
            # Single insert performance
            single_inserts = insert_df[(insert_df['database'] == db_name) & 
                                  (insert_df['operation'] == 'insert_single')]
            if not single_inserts.empty:
                avg_ops = single_inserts['ops_per_second'].mean()
                db_scores.append(avg_ops)
            else:
                db_scores.append(0)
                
            # Batch insert performance
            batch_inserts = insert_df[(insert_df['database'] == db_name) & 
                                 (insert_df['operation'] == 'insert_batch')]
            if not batch_inserts.empty:
                avg_ops = batch_inserts['ops_per_second'].mean()
                db_scores.append(avg_ops)
            else:
                db_scores.append(0)
                
            # Query by user performance (lower time is better, so invert)
            user_queries = query_df[(query_df['database'] == db_name) & 
                              (query_df['operation'] == 'query_by_user')]
            if not user_queries.empty:
                avg_time = user_queries['time'].mean()
                # Invert so higher is better
                db_scores.append(1.0 / (avg_time + 0.0001))  # Add small constant to avoid division by zero
            else:
                db_scores.append(0)
                
            # Query by page performance (lower time is better, so invert)
            page_queries = query_df[(query_df['database'] == db_name) & 
                              (query_df['operation'] == 'query_by_page')]
            if not page_queries.empty:
                avg_time = page_queries['time'].mean()
                # Invert so higher is better
                db_scores.append(1.0 / (avg_time + 0.0001))
            else:
                db_scores.append(0)
                
            # Advanced queries performance if available
            if advanced_query_df is not None:
                adv_queries = advanced_query_df[advanced_query_df['database'] == db_name]
                if not adv_queries.empty:
                    avg_time = adv_queries['time'].mean()
                    # Invert so higher is better
                    db_scores.append(1.0 / (avg_time + 0.0001))
                else:
                    db_scores.append(0)
            else:
                db_scores.append(0)
                
            # Concurrency scalability (estimate from batch performance trend)
            if not batch_inserts.empty:
                batch_sizes = batch_inserts['batch_size'].unique()
                if len(batch_sizes) >= 2:
                    small_batch = batch_inserts[batch_inserts['batch_size'] == min(batch_sizes)]['ops_per_second'].mean()
                    large_batch = batch_inserts[batch_inserts['batch_size'] == max(batch_sizes)]['ops_per_second'].mean()
                    # If scales well, large_batch should be higher than small_batch
                    scalability = large_batch / (small_batch + 0.0001)
                    db_scores.append(scalability)
                else:
                    db_scores.append(1)  # Neutral score
            else:
                db_scores.append(0)
                
            scores[db_name] = db_scores
        
        # Normalize scores across databases
        max_scores = [max([scores[db][i] for db in scores.keys()]) for i in range(len(categories))]
        normalized_scores = {}
        
        for db_name, db_scores in scores.items():
            normalized_scores[db_name] = [db_scores[i] / max_scores[i] if max_scores[i] > 0 else 0 
                                         for i in range(len(categories))]
        
        # Create radar chart
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        ax = plt.subplot(111, polar=True)
        
        # Plot each database
        for db_name, db_scores in normalized_scores.items():
            db_scores += db_scores[:1]  # Close the loop
            ax.plot(angles, db_scores, linewidth=2, label=db_name)
            ax.fill(angles, db_scores, alpha=0.1)
        
        # Set category labels
        plt.xticks(angles[:-1], categories, fontsize=12)
        
        # Set y ticks and hide them
        plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
        plt.ylim(0, 1)
        
        # Add legend and title
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=12)
        plt.title('Overall Database Performance', fontsize=16, y=1.08)
        
        # Add annotations explaining the chart
        plt.figtext(0.5, 0.01, 
                   "Radar chart shows relative performance across dimensions (higher is better).\nValues are normalized with the best performer in each category set to 1.0.",
                   ha="center", fontsize=10, 
                   bbox={"facecolor":"lightblue", "alpha":0.2, "pad":5})
        
        plt.tight_layout()
        plt.savefig(f"{viz_dir}/performance_radar_chart.png", dpi=300)
        plt.close()
    
    def visualize_results(self, insert_df=None, query_df=None):
        """Visualize benchmark results."""
        # Create specific directory for visualizations
        viz_dir = f"{self.result_dir}/visualizations_{self.timestamp}"
        os.makedirs(viz_dir, exist_ok=True)
        
        # Configure global style for charts
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("Set2")
        
        # 1. Visualize insert performance
        if insert_df is not None:
            self._visualize_insert_performance(insert_df, viz_dir)
        
        # 2. Visualize query performance
        if query_df is not None:
            self._visualize_query_performance(query_df, viz_dir)
        
        # 3. Create a latency/throughput combined graph
        if insert_df is not None and query_df is not None:
            self._create_latency_throughput_graph(insert_df, query_df, viz_dir)
        
        # 4. Create a radar chart for global comparison
        if insert_df is not None and query_df is not None:
            self._create_radar_chart(insert_df, query_df, None, viz_dir)
        
        # 5. Create a summary HTML dashboard
        self._create_performance_dashboard(viz_dir)
        
        print(f"Visualizations generated in directory: {viz_dir}")
    
    def _create_performance_dashboard(self, viz_dir):
        """Create an HTML dashboard with visualizations."""
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Database Benchmark Results</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    color: #333;
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                }}
                .dashboard-section {{
                    margin-bottom: 40px;
                    padding: 20px;
                    background-color: #f9f9f9;
                    border-radius: 8px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                .visualization {{
                    margin: 20px 0;
                    text-align: center;
                }}
                .visualization img {{
                    max-width: 100%;
                    height: auto;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                }}
                .summary {{
                    background-color: #e9f7ef;
                    padding: 15px;
                    border-left: 5px solid #27ae60;
                    margin-bottom: 20px;
                }}
                footer {{
                    margin-top: 40px;
                    padding-top: 20px;
                    border-top: 1px solid #eee;
                    text-align: center;
                    font-size: 0.9em;
                    color: #777;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Database Benchmark Results for Mouse Tracking</h1>
                <p>Execution Date: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>
                
                <div class="dashboard-section">
                    <h2>Performance Summary</h2>
                    <div class="summary">
                        <p>This dashboard presents the results of comparing MongoDB, PostgreSQL and Redis 
                        for a mouse movement tracking application. Tests were performed on insert and query operations 
                        with different data sizes.</p>
                    </div>
                </div>
                
                <div class="dashboard-section">
                    <h2>Insert Performance</h2>
                    
                    <div class="visualization">
                        <h3>Individual Inserts</h3>
                        <img src="insert_single_performance.png" alt="Individual Insert Performance">
                        <p>This chart shows the number of operations per second for individual inserts based on data size.</p>
                    </div>
                    
                    <div class="visualization">
                        <h3>Batch Inserts</h3>
                        <img src="insert_batch_performance.png" alt="Batch Insert Performance">
                        <p>This chart shows the number of operations per second for batch inserts based on batch size.</p>
                    </div>
                    
                    <div class="visualization">
                        <h3>Batch Performance Gain</h3>
                        <img src="insert_batch_efficiency.png" alt="Batch Insert Efficiency">
                        <p>This chart shows the speedup factor of batch inserts compared to individual inserts.</p>
                    </div>
                </div>
                
                <div class="dashboard-section">
                    <h2>Query Performance</h2>
                    
                    <div class="visualization">
                        <h3>Execution Time by Query Type</h3>
                        <img src="query_performance_by_type.png" alt="Performance by Query Type">
                        <p>This chart shows the average execution time for each query type.</p>
                    </div>
                    
                    <div class="visualization">
                        <h3>Query Time Distribution</h3>
                        <img src="query_time_distribution.png" alt="Query Time Distribution">
                        <p>This chart shows the distribution of execution times for each query type.</p>
                    </div>
                </div>
                
                <div class="dashboard-section">
                    <h2>Advanced Analysis</h2>
                    
                    <div class="visualization">
                        <h3>Relationship Between Throughput and Latency</h3>
                        <img src="latency_throughput_relationship.png" alt="Throughput/Latency Relationship">
                        <p>This chart shows the inverse relationship between throughput and latency for each database.</p>
                    </div>
                    
                    <div class="visualization">
                        <h3>Overall Performance Comparison</h3>
                        <img src="performance_radar_chart.png" alt="Performance Radar Chart">
                        <p>This radar chart provides an overview of relative performance across all dimensions.</p>
                    </div>
                </div>
                
                <footer>
                    <p>Benchmark conducted as part of the "Beyond Relational Databases" (205.2) course at HES-SO Valais.</p>
                    <p>&copy; {datetime.now().year} - Database Comparison Laboratory</p>
                </footer>
            </div>
        </body>
        </html>
        """
        
        # Save the HTML dashboard
        with open(f"{viz_dir}/dashboard.html", 'w') as f:
            f.write(html_content)


def main():
    """Main function to run the benchmarks."""
    parser = argparse.ArgumentParser(description='Run database benchmarks.')
    parser.add_argument('--data-sizes', type=int, nargs='+', default=[100, 1000, 5000, 10000],
                        help='Data sizes to test')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 10, 50, 100, 500, 1000],
                        help='Batch sizes to test')
    parser.add_argument('--iterations', type=int, default=3,
                        help='Number of iterations for each test')
    parser.add_argument('--setup-data-size', type=int, default=5000,
                        help='Data size for query benchmarks')
    parser.add_argument('--result-dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--tests', type=str, nargs='+', 
                        default=['insert', 'query', 'advanced_query', 'update', 'delete', 'concurrent'],
                        help='Tests to run (insert, query, advanced_query, update, delete, concurrent)')
    parser.add_argument('--monitor', action='store_true',
                        help='Enable system resource monitoring')
    parser.add_argument('--concurrency-levels', type=int, nargs='+', default=[1, 2, 5, 10, 20],
                        help='Concurrency levels to test')
    
    args = parser.parse_args()
    
    # Initialize database clients
    db_clients = {
        'mongodb': MongoDBClient(),
        'postgresql': PostgreSQLClient(connection_params="dbname=mousebenchmark user=jduc password=password"),
        'redis': RedisClient()
    }
    
    # Create a benchmark instance
    benchmark = SimplifiedDatabaseBenchmark(
        db_clients=db_clients,
        data_sizes=args.data_sizes,
        batch_sizes=args.batch_sizes,
        iterations=args.iterations,
        result_dir=args.result_dir
    )
    
    # Initialize a system monitor if requested
    system_monitor = None
    if args.monitor:
        from src.common.system_monitor import SystemMonitor
        system_monitor = SystemMonitor(interval=1.0, result_dir=args.result_dir)
        
    # Variables to store results
    insert_df = None
    query_df = None
    advanced_query_df = None
    update_df = None
    delete_df = None
    concurrent_insert_df = None
    concurrent_query_df = None
    
    # Run selected benchmarks
    if 'insert' in args.tests:
        print("\n=== Running Insert Benchmarks ===")
        insert_df = benchmark.run_insert_benchmark()
    
    if 'query' in args.tests:
        print("\n=== Running Query Benchmarks ===")
        query_df = benchmark.run_query_benchmark(setup_data_size=args.setup_data_size)
        
    if 'advanced_query' in args.tests:
        print("\n=== Running Advanced Query Benchmarks ===")
        advanced_query_df = benchmark.run_advanced_query_benchmark(setup_data_size=args.setup_data_size)
        
    if 'update' in args.tests:
        print("\n=== Running Update Benchmarks ===")
        update_df = benchmark.run_update_benchmark(setup_data_size=args.setup_data_size)
        
    if 'delete' in args.tests:
        print("\n=== Running Delete Benchmarks ===")
        delete_df = benchmark.run_delete_benchmark(setup_data_size=args.setup_data_size)
        
    if 'concurrent' in args.tests:
        print("\n=== Running Concurrent Benchmarks ===")
        
        # Initialize concurrent benchmark
        from src.common.concurrent_benchmark import ConcurrentBenchmark
        concurrent_bench = ConcurrentBenchmark(db_clients=db_clients, result_dir=args.result_dir)
        
        # Run concurrent insert benchmark
        print("\n--- Running Concurrent Insert Benchmark ---")
        concurrent_insert_df = concurrent_bench.run_concurrent_inserts(
            concurrency_levels=args.concurrency_levels,
            events_per_thread=100,
            iterations=args.iterations
        )
        
        # Run concurrent query benchmark
        print("\n--- Running Concurrent Query Benchmark ---")
        concurrent_query_df = concurrent_bench.run_concurrent_queries(
            concurrency_levels=args.concurrency_levels,
            queries_per_thread=100,
            setup_data_size=args.setup_data_size,
            iterations=args.iterations
        )
        
        # Visualize concurrent results
        print("\n--- Generating Concurrent Benchmark Visualizations ---")
        concurrent_bench.visualize_concurrent_results(
            insert_df=concurrent_insert_df,
            query_df=concurrent_query_df
        )
    
    # Visualize basic results
    print("\n=== Generating Visualizations ===")
    benchmark.visualize_results(insert_df, query_df)
    
    # Perform statistical analysis
    print("\n=== Performing Statistical Analysis ===")
    from src.common.statistics import BenchmarkAnalyzer
    analyzer = BenchmarkAnalyzer(result_dir=args.result_dir)
    
    # Analyze insert results
    if insert_df is not None:
        # Calculate statistics
        insert_stats = analyzer.calculate_statistics(insert_df)
        
        # Calculate confidence intervals
        insert_ci = analyzer.calculate_confidence_intervals(insert_df)
        
        # Perform t-tests
        insert_t_tests = analyzer.perform_t_tests(insert_df, 'time', 'operation', 'database')
        
        # Detect outliers
        insert_outliers = analyzer.detect_outliers(insert_df, ['time', 'ops_per_second'])
        
        # Visualize confidence intervals
        analyzer.visualize_confidence_intervals(insert_ci, 'time')
        analyzer.visualize_confidence_intervals(insert_ci, 'ops_per_second')
        
        # Visualize t-test results
        analyzer.visualize_t_test_matrix(insert_df, 'time')
        
        # Visualize outliers
        analyzer.visualize_outliers(insert_outliers, 'time')
        
        # Generate summary report
        analyzer.generate_summary_report(
            insert_stats, insert_ci, insert_t_tests,
            metric_cols=['time', 'ops_per_second']
        )
    
    # Close connections
    for client in db_clients.values():
        client.close()
    
    print("\nBenchmarks completed. Results have been saved to directory:", args.result_dir)

if __name__ == "__main__":
    main()