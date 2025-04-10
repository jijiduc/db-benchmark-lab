"""
Module for concurrent database operations benchmarking.
"""
import time
import random
import threading
import pandas as pd
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor
from src.common.generate_data import generate_dataset
from src.mongodb.client import MongoDBClient
from src.postgresql.client import PostgreSQLClient
from src.redis.client import RedisClient

class ConcurrentBenchmark:
    """Run concurrent database operations to simulate real-world load."""
    
    def __init__(self, db_clients, result_dir='results'):
        """
        Initialize the concurrent benchmark.
        
        Args:
            db_clients (dict): Dictionary of database clients
            result_dir (str): Directory to save results
        """
        self.db_clients = db_clients
        self.result_dir = result_dir
        
        # Create results directory if needed
        os.makedirs(result_dir, exist_ok=True)
        
    def run_concurrent_inserts(self, concurrency_levels=None, events_per_thread=100, iterations=3):
        """
        Run concurrent insert operations.
        
        Args:
            concurrency_levels (list): List of concurrency levels to test
            events_per_thread (int): Number of events for each thread to insert
            iterations (int): Number of iterations for each test
            
        Returns:
            DataFrame: Results of the concurrent insert benchmark
        """
        if concurrency_levels is None:
            concurrency_levels = [1, 2, 5, 10, 20, 50]
            
        all_results = []
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for db_name, client in self.db_clients.items():
            print(f"\nRunning concurrent insert benchmark for {db_name}...")
            
            for concurrency in concurrency_levels:
                print(f"  Testing with {concurrency} concurrent threads...")
                
                for iteration in range(1, iterations + 1):
                    # Clean previous data
                    if db_name == 'mongodb':
                        client.clear_collection()
                    elif db_name == 'postgresql':
                        client.clear_table()
                    elif db_name == 'redis':
                        client.clear_data()
                        
                    # Generate data for each thread
                    thread_data = [generate_dataset(events_per_thread) for _ in range(concurrency)]
                    
                    # Create connection parameters for PostgreSQL
                    postgresql_connection_params = "dbname=mousebenchmark user=jduc password=password"
                    
                    # Prepare thread functions
                    def insert_thread_events(thread_id):
                        thread_start_time = time.time()
                        latencies = []
                        
                        # Create a thread-specific client for PostgreSQL to avoid connection sharing issues
                        thread_client = client
                        if db_name == 'postgresql':
                            thread_client = PostgreSQLClient(connection_params=postgresql_connection_params)
                        
                        try:
                            for event in thread_data[thread_id]:
                                event_start = time.time()
                                # Insert single event
                                thread_client.insert_event(event)
                                latencies.append((time.time() - event_start) * 1000)  # Convert to ms
                        finally:
                            # Close the thread's connection if it's PostgreSQL
                            if db_name == 'postgresql' and thread_client is not client:
                                thread_client.close()
                            
                        thread_end_time = time.time()
                        thread_duration = thread_end_time - thread_start_time
                        
                        # Calculate percentiles
                        latency_p50 = np.percentile(latencies, 50) if latencies else 0
                        latency_p95 = np.percentile(latencies, 95) if latencies else 0
                        latency_p99 = np.percentile(latencies, 99) if latencies else 0
                        
                        return {
                            'thread_id': thread_id,
                            'events_processed': events_per_thread,
                            'thread_duration': thread_duration,
                            'events_per_second': events_per_thread / thread_duration if thread_duration > 0 else 0,
                            'avg_latency_ms': np.mean(latencies) if latencies else 0,
                            'min_latency_ms': np.min(latencies) if latencies else 0,
                            'max_latency_ms': np.max(latencies) if latencies else 0,
                            'p50_latency_ms': latency_p50,
                            'p95_latency_ms': latency_p95,
                            'p99_latency_ms': latency_p99
                        }
                    
                    # Run the threads
                    start_time = time.time()
                    with ThreadPoolExecutor(max_workers=concurrency) as executor:
                        thread_results = list(executor.map(insert_thread_events, range(concurrency)))
                    end_time = time.time()
                    
                    # Calculate overall statistics
                    total_duration = end_time - start_time
                    total_events = concurrency * events_per_thread
                    overall_events_per_second = total_events / total_duration if total_duration > 0 else 0
                    
                    # Aggregate thread results
                    thread_durations = [r['thread_duration'] for r in thread_results]
                    avg_thread_duration = np.mean(thread_durations) if thread_durations else 0
                    max_thread_duration = np.max(thread_durations) if thread_durations else 0
                    
                    # Aggregate latency results
                    avg_latencies = [r['avg_latency_ms'] for r in thread_results]
                    overall_avg_latency = np.mean(avg_latencies) if avg_latencies else 0
                    overall_p50_latency = np.mean([r['p50_latency_ms'] for r in thread_results]) if thread_results else 0
                    overall_p95_latency = np.mean([r['p95_latency_ms'] for r in thread_results]) if thread_results else 0
                    overall_p99_latency = np.mean([r['p99_latency_ms'] for r in thread_results]) if thread_results else 0
                    
                    # Store the result
                    result = {
                        'database': db_name,
                        'operation': 'concurrent_insert',
                        'concurrency': concurrency,
                        'events_per_thread': events_per_thread,
                        'total_events': total_events,
                        'total_duration': total_duration,
                        'overall_events_per_second': overall_events_per_second,
                        'avg_thread_duration': avg_thread_duration,
                        'max_thread_duration': max_thread_duration,
                        'overall_avg_latency_ms': overall_avg_latency,
                        'overall_p50_latency_ms': overall_p50_latency,
                        'overall_p95_latency_ms': overall_p95_latency,
                        'overall_p99_latency_ms': overall_p99_latency,
                        'iteration': iteration
                    }
                    
                    all_results.append(result)
                    print(f"    Iteration {iteration}: {overall_events_per_second:.2f} events/sec, Avg latency: {overall_avg_latency:.2f} ms")
        
        # Convert results to DataFrame and save
        df = pd.DataFrame(all_results)
        df.to_csv(f"{self.result_dir}/concurrent_insert_benchmark_{timestamp}.csv", index=False)
        
        return df
    
    def run_concurrent_queries(self, concurrency_levels=None, queries_per_thread=100, setup_data_size=5000, iterations=3):
        """
        Run concurrent query operations.
        
        Args:
            concurrency_levels (list): List of concurrency levels to test
            queries_per_thread (int): Number of queries for each thread to execute
            setup_data_size (int): Size of data to set up before querying
            iterations (int): Number of iterations for each test
            
        Returns:
            DataFrame: Results of the concurrent query benchmark
        """
        if concurrency_levels is None:
            concurrency_levels = [1, 2, 5, 10, 20, 50]
            
        all_results = []
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Generate a dataset for setup
        setup_events = generate_dataset(setup_data_size)
        
        # Extract unique values for query parameters
        all_user_ids = list(set(event['user_id'] for event in setup_events))
        all_pages = list(set(event['page'] for event in setup_events))
        
        # Create connection parameters for PostgreSQL
        postgresql_connection_params = "dbname=mousebenchmark user=jduc password=password"
        
        for db_name, client in self.db_clients.items():
            print(f"\nRunning concurrent query benchmark for {db_name}...")
            
            for concurrency in concurrency_levels:
                print(f"  Testing with {concurrency} concurrent threads...")
                
                for iteration in range(1, iterations + 1):
                    # Clean and set up data
                    if db_name == 'mongodb':
                        client.clear_collection()
                    elif db_name == 'postgresql':
                        client.clear_table()
                    elif db_name == 'redis':
                        client.clear_data()
                        
                    # Insert setup data
                    client.insert_events(setup_events)
                    
                    # Prepare query parameters for each thread
                    thread_query_params = []
                    for _ in range(concurrency):
                        # Create a mix of query types
                        query_types = []
                        for _ in range(queries_per_thread):
                            query_type = random.choice(['user', 'page', 'count'])
                            if query_type == 'user':
                                param = random.choice(all_user_ids)
                            elif query_type == 'page':
                                param = random.choice(all_pages)
                            else:  # count
                                param = None
                            query_types.append((query_type, param))
                        thread_query_params.append(query_types)
                    
                    # Prepare thread functions
                    def query_thread(thread_id):
                        thread_start_time = time.time()
                        latencies = []
                        
                        # Create a thread-specific client for PostgreSQL to avoid connection sharing issues
                        thread_client = client
                        if db_name == 'postgresql':
                            thread_client = PostgreSQLClient(connection_params=postgresql_connection_params)
                        
                        try:
                            for query_type, param in thread_query_params[thread_id]:
                                query_start = time.time()
                                
                                if query_type == 'user':
                                    thread_client.find_events_by_user(param)
                                elif query_type == 'page':
                                    thread_client.find_events_by_page(param)
                                else:  # count
                                    thread_client.count_events()
                                    
                                latencies.append((time.time() - query_start) * 1000)  # Convert to ms
                        finally:
                            # Close the thread's connection if it's PostgreSQL
                            if db_name == 'postgresql' and thread_client is not client:
                                thread_client.close()
                            
                        thread_end_time = time.time()
                        thread_duration = thread_end_time - thread_start_time
                        
                        # Calculate percentiles
                        latency_p50 = np.percentile(latencies, 50) if latencies else 0
                        latency_p95 = np.percentile(latencies, 95) if latencies else 0
                        latency_p99 = np.percentile(latencies, 99) if latencies else 0
                        
                        return {
                            'thread_id': thread_id,
                            'queries_processed': queries_per_thread,
                            'thread_duration': thread_duration,
                            'queries_per_second': queries_per_thread / thread_duration if thread_duration > 0 else 0,
                            'avg_latency_ms': np.mean(latencies) if latencies else 0,
                            'min_latency_ms': np.min(latencies) if latencies else 0,
                            'max_latency_ms': np.max(latencies) if latencies else 0,
                            'p50_latency_ms': latency_p50,
                            'p95_latency_ms': latency_p95,
                            'p99_latency_ms': latency_p99
                        }
                    
                    # Run the threads
                    start_time = time.time()
                    with ThreadPoolExecutor(max_workers=concurrency) as executor:
                        thread_results = list(executor.map(query_thread, range(concurrency)))
                    end_time = time.time()
                    
                    # Calculate overall statistics
                    total_duration = end_time - start_time
                    total_queries = concurrency * queries_per_thread
                    overall_queries_per_second = total_queries / total_duration if total_duration > 0 else 0
                    
                    # Aggregate thread results
                    thread_durations = [r['thread_duration'] for r in thread_results]
                    avg_thread_duration = np.mean(thread_durations) if thread_durations else 0
                    max_thread_duration = np.max(thread_durations) if thread_durations else 0
                    
                    # Aggregate latency results
                    avg_latencies = [r['avg_latency_ms'] for r in thread_results]
                    overall_avg_latency = np.mean(avg_latencies) if avg_latencies else 0
                    overall_p50_latency = np.mean([r['p50_latency_ms'] for r in thread_results]) if thread_results else 0
                    overall_p95_latency = np.mean([r['p95_latency_ms'] for r in thread_results]) if thread_results else 0
                    overall_p99_latency = np.mean([r['p99_latency_ms'] for r in thread_results]) if thread_results else 0
                    
                    # Store the result
                    result = {
                        'database': db_name,
                        'operation': 'concurrent_query',
                        'concurrency': concurrency,
                        'queries_per_thread': queries_per_thread,
                        'total_queries': total_queries,
                        'data_size': setup_data_size,
                        'total_duration': total_duration,
                        'overall_queries_per_second': overall_queries_per_second,
                        'avg_thread_duration': avg_thread_duration,
                        'max_thread_duration': max_thread_duration,
                        'overall_avg_latency_ms': overall_avg_latency,
                        'overall_p50_latency_ms': overall_p50_latency,
                        'overall_p95_latency_ms': overall_p95_latency,
                        'overall_p99_latency_ms': overall_p99_latency,
                        'iteration': iteration
                    }
                    
                    all_results.append(result)
                    print(f"    Iteration {iteration}: {overall_queries_per_second:.2f} queries/sec, Avg latency: {overall_avg_latency:.2f} ms")
        
        # Convert results to DataFrame and save
        df = pd.DataFrame(all_results)
        df.to_csv(f"{self.result_dir}/concurrent_query_benchmark_{timestamp}.csv", index=False)
        
        return df
    
    def visualize_concurrent_results(self, insert_df=None, query_df=None):
        """
        Visualize concurrent benchmark results.
        
        Args:
            insert_df (DataFrame): Results of concurrent insert benchmark
            query_df (DataFrame): Results of concurrent query benchmark
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        viz_dir = f"{self.result_dir}/concurrent_viz_{timestamp}"
        os.makedirs(viz_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("deep")
        
        # Visualize insert results
        if insert_df is not None:
            # Aggregate results by database and concurrency level
            grouped = insert_df.groupby(['database', 'concurrency'])[
                ['overall_events_per_second', 'overall_avg_latency_ms', 
                 'overall_p95_latency_ms', 'overall_p99_latency_ms']
            ].agg(['mean', 'std']).reset_index()
            
            # Plot throughput vs concurrency
            plt.figure(figsize=(12, 8))
            for db_name in grouped['database'].unique():
                db_data = grouped[grouped['database'] == db_name]
                mean_col = ('overall_events_per_second', 'mean')
                std_col = ('overall_events_per_second', 'std')
                
                line, = plt.plot(db_data['concurrency'], db_data[mean_col], marker='o', linewidth=2, label=db_name)
                plt.fill_between(
                    db_data['concurrency'],
                    db_data[mean_col] - db_data[std_col],
                    db_data[mean_col] + db_data[std_col],
                    alpha=0.2
                )
                
                # Annotate maximum point
                max_idx = db_data[mean_col].idxmax()
                if not pd.isna(max_idx):
                    max_val = db_data.iloc[max_idx][mean_col]
                    max_conc = db_data.iloc[max_idx]['concurrency']
                    plt.annotate(f"Max: {max_val:.0f} evt/s",
                             xy=(max_conc, max_val),
                             xytext=(0, 10),
                             textcoords='offset points',
                             ha='center',
                             va='bottom',
                             bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.5),
                             color=line.get_color())
                
                # Find crossover points with other databases
                for other_db in grouped['database'].unique():
                    if other_db != db_name:
                        other_data = grouped[grouped['database'] == other_db]
                        for i in range(len(db_data) - 1):
                            db_x1 = db_data.iloc[i]['concurrency']
                            db_y1 = db_data.iloc[i][mean_col]
                            db_x2 = db_data.iloc[i+1]['concurrency']
                            db_y2 = db_data.iloc[i+1][mean_col]
                            
                            for j in range(len(other_data) - 1):
                                other_x1 = other_data.iloc[j]['concurrency']
                                other_y1 = other_data.iloc[j][mean_col]
                                other_x2 = other_data.iloc[j+1]['concurrency']
                                other_y2 = other_data.iloc[j+1][mean_col]
                                
                                # Check if segments intersect
                                if (db_x1 <= other_x2 and db_x2 >= other_x1) and \
                                   ((db_y1 <= other_y1 and db_y2 >= other_y1) or \
                                    (db_y1 >= other_y1 and db_y2 <= other_y2)):
                                    # Compute approximate intersection point
                                    # Simple linear interpolation for demonstration
                                    t = (other_y1 - db_y1) / ((db_y2 - db_y1) if db_y2 != db_y1 else 1)
                                    cross_x = db_x1 + t * (db_x2 - db_x1)
                                    cross_y = db_y1 + t * (db_y2 - db_y1)
                                    
                                    if db_x1 <= cross_x <= db_x2 and other_x1 <= cross_x <= other_x2:
                                        plt.scatter([cross_x], [cross_y], marker='x', s=80, color='red', zorder=10)
                                        plt.annotate(f"Crossover: {cross_y:.0f}",
                                                 xy=(cross_x, cross_y),
                                                 xytext=(0, -15),
                                                 textcoords='offset points',
                                                 ha='center',
                                                 va='top',
                                                 bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7),
                                                 color='red')
            
            # Add performance threshold reference
            acceptable_throughput = 2000  # Example threshold
            plt.axhline(y=acceptable_throughput, color='r', linestyle='--', alpha=0.7)
            plt.text(max(db_data['concurrency'])*0.5, acceptable_throughput*1.05, 
                     'Acceptable Throughput Threshold', ha='center', va='bottom',
                     bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
            
            plt.title('Throughput vs Concurrency Level - Insert Operations', fontsize=16)
            plt.xlabel('Concurrency Level (Number of Threads)', fontsize=14)
            plt.ylabel('Throughput (Events per Second)', fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{viz_dir}/insert_throughput_vs_concurrency.png", dpi=300)
            plt.close()
            
            # Plot latency percentiles vs concurrency
            plt.figure(figsize=(14, 10))
            
            # Create subplots for different latency metrics
            fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
            
            # Plot average latency
            for db_name in grouped['database'].unique():
                db_data = grouped[grouped['database'] == db_name]
                mean_col = ('overall_avg_latency_ms', 'mean')
                std_col = ('overall_avg_latency_ms', 'std')
                
                axs[0].plot(db_data['concurrency'], db_data[mean_col], marker='o', linewidth=2, label=db_name)
                axs[0].fill_between(
                    db_data['concurrency'],
                    db_data[mean_col] - db_data[std_col],
                    db_data[mean_col] + db_data[std_col],
                    alpha=0.2
                )
                
                # Annotate maximum latency
                max_idx = db_data[mean_col].idxmax()
                if not pd.isna(max_idx):
                    max_val = db_data.iloc[max_idx][mean_col]
                    max_conc = db_data.iloc[max_idx]['concurrency']
                    if max_val > 5:  # Only annotate if latency is significant
                        axs[0].annotate(f"{max_val:.1f} ms",
                                     xy=(max_conc, max_val),
                                     xytext=(0, 10),
                                     textcoords='offset points',
                                     ha='center',
                                     va='bottom',
                                     bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.5))
            
            # Add latency threshold
            acceptable_latency = 10  # 10ms threshold
            axs[0].axhline(y=acceptable_latency, color='r', linestyle='--', alpha=0.7)
            axs[0].text(max(db_data['concurrency'])*0.5, acceptable_latency*1.1, 
                     'Acceptable', ha='center', va='bottom',
                     bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
            
            axs[0].set_title('Average Latency', fontsize=14)
            axs[0].set_xlabel('Concurrency Level', fontsize=12)
            axs[0].set_ylabel('Latency (ms)', fontsize=12)
            axs[0].legend(fontsize=10)
            axs[0].grid(True, alpha=0.3)
            
            # Plot p95 latency
            for db_name in grouped['database'].unique():
                db_data = grouped[grouped['database'] == db_name]
                mean_col = ('overall_p95_latency_ms', 'mean')
                std_col = ('overall_p95_latency_ms', 'std')
                
                axs[1].plot(db_data['concurrency'], db_data[mean_col], marker='s', linewidth=2, label=db_name)
                axs[1].fill_between(
                    db_data['concurrency'],
                    db_data[mean_col] - db_data[std_col],
                    db_data[mean_col] + db_data[std_col],
                    alpha=0.2
                )
                
                # Annotate maximum latency
                max_idx = db_data[mean_col].idxmax()
                if not pd.isna(max_idx):
                    max_val = db_data.iloc[max_idx][mean_col]
                    max_conc = db_data.iloc[max_idx]['concurrency']
                    if max_val > 5:  # Only annotate if latency is significant
                        axs[1].annotate(f"{max_val:.1f} ms",
                                     xy=(max_conc, max_val),
                                     xytext=(0, 10),
                                     textcoords='offset points',
                                     ha='center',
                                     va='bottom',
                                     bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.5))
            
            # Add latency threshold
            p95_threshold = 50  # 50ms threshold for P95
            axs[1].axhline(y=p95_threshold, color='r', linestyle='--', alpha=0.7)
            axs[1].text(max(db_data['concurrency'])*0.5, p95_threshold*1.1, 
                     'P95 Threshold', ha='center', va='bottom',
                     bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
            
            axs[1].set_title('P95 Latency', fontsize=14)
            axs[1].set_xlabel('Concurrency Level', fontsize=12)
            axs[1].set_ylabel('Latency (ms)', fontsize=12)
            axs[1].legend(fontsize=10)
            axs[1].grid(True, alpha=0.3)
            
            # Plot p99 latency
            for db_name in grouped['database'].unique():
                db_data = grouped[grouped['database'] == db_name]
                mean_col = ('overall_p99_latency_ms', 'mean')
                std_col = ('overall_p99_latency_ms', 'std')
                
                axs[2].plot(db_data['concurrency'], db_data[mean_col], marker='D', linewidth=2, label=db_name)
                axs[2].fill_between(
                    db_data['concurrency'],
                    db_data[mean_col] - db_data[std_col],
                    db_data[mean_col] + db_data[std_col],
                    alpha=0.2
                )
                
                # Annotate maximum latency
                max_idx = db_data[mean_col].idxmax()
                if not pd.isna(max_idx):
                    max_val = db_data.iloc[max_idx][mean_col]
                    max_conc = db_data.iloc[max_idx]['concurrency']
                    if max_val > 5:  # Only annotate if latency is significant
                        axs[2].annotate(f"{max_val:.1f} ms",
                                     xy=(max_conc, max_val),
                                     xytext=(0, 10),
                                     textcoords='offset points',
                                     ha='center',
                                     va='bottom',
                                     bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.5))
            
            # Add latency threshold
            p99_threshold = 100  # 100ms threshold for P99
            axs[2].axhline(y=p99_threshold, color='r', linestyle='--', alpha=0.7)
            axs[2].text(max(db_data['concurrency'])*0.5, p99_threshold*1.1, 
                     'P99 Threshold', ha='center', va='bottom',
                     bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
            
            axs[2].set_title('P99 Latency', fontsize=14)
            axs[2].set_xlabel('Concurrency Level', fontsize=12)
            axs[2].set_ylabel('Latency (ms)', fontsize=12)
            axs[2].legend(fontsize=10)
            axs[2].grid(True, alpha=0.3)
            
            plt.suptitle('Latency Metrics vs Concurrency Level - Insert Operations', fontsize=16)
            plt.tight_layout()
            plt.savefig(f"{viz_dir}/insert_latency_vs_concurrency.png", dpi=300)
            plt.close()
        
        # Visualize query results
        if query_df is not None:
            # Aggregate results by database and concurrency level
            grouped = query_df.groupby(['database', 'concurrency'])[
                ['overall_queries_per_second', 'overall_avg_latency_ms', 
                 'overall_p95_latency_ms', 'overall_p99_latency_ms']
            ].agg(['mean', 'std']).reset_index()
            
            # Plot throughput vs concurrency
            plt.figure(figsize=(12, 8))
            for db_name in grouped['database'].unique():
                db_data = grouped[grouped['database'] == db_name]
                mean_col = ('overall_queries_per_second', 'mean')
                std_col = ('overall_queries_per_second', 'std')
                
                line, = plt.plot(db_data['concurrency'], db_data[mean_col], marker='o', linewidth=2, label=db_name)
                plt.fill_between(
                    db_data['concurrency'],
                    db_data[mean_col] - db_data[std_col],
                    db_data[mean_col] + db_data[std_col],
                    alpha=0.2
                )
                
                # Annotate maximum point
                max_idx = db_data[mean_col].idxmax()
                if not pd.isna(max_idx):
                    max_val = db_data.iloc[max_idx][mean_col]
                    max_conc = db_data.iloc[max_idx]['concurrency']
                    plt.annotate(f"Max: {max_val:.0f} q/s",
                             xy=(max_conc, max_val),
                             xytext=(0, 10),
                             textcoords='offset points',
                             ha='center',
                             va='bottom',
                             bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.5),
                             color=line.get_color())
            
            # Add performance threshold reference
            acceptable_throughput = 500  # Example threshold for queries
            plt.axhline(y=acceptable_throughput, color='r', linestyle='--', alpha=0.7)
            plt.text(max(db_data['concurrency'])*0.5, acceptable_throughput*1.05, 
                     'Acceptable Query Throughput', ha='center', va='bottom',
                     bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
            
            plt.title('Throughput vs Concurrency Level - Query Operations', fontsize=16)
            plt.xlabel('Concurrency Level (Number of Threads)', fontsize=14)
            plt.ylabel('Throughput (Queries per Second)', fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{viz_dir}/query_throughput_vs_concurrency.png", dpi=300)
            plt.close()
            
            # Plot latency percentiles vs concurrency
            # Create subplots for different latency metrics
            fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
            
            # Plot average latency
            for db_name in grouped['database'].unique():
                db_data = grouped[grouped['database'] == db_name]
                mean_col = ('overall_avg_latency_ms', 'mean')
                std_col = ('overall_avg_latency_ms', 'std')
                
                axs[0].plot(db_data['concurrency'], db_data[mean_col], marker='o', linewidth=2, label=db_name)
                axs[0].fill_between(
                    db_data['concurrency'],
                    db_data[mean_col] - db_data[std_col],
                    db_data[mean_col] + db_data[std_col],
                    alpha=0.2
                )
                
                # Annotate maximum latency
                max_idx = db_data[mean_col].idxmax()
                if not pd.isna(max_idx):
                    max_val = db_data.iloc[max_idx][mean_col]
                    max_conc = db_data.iloc[max_idx]['concurrency']
                    if max_val > 50:  # Only annotate if latency is significant
                        axs[0].annotate(f"{max_val:.1f} ms",
                                     xy=(max_conc, max_val),
                                     xytext=(0, 10),
                                     textcoords='offset points',
                                     ha='center',
                                     va='bottom',
                                     bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.5))
            
            # Add latency threshold
            acceptable_query_latency = 100  # 100ms threshold for queries
            axs[0].axhline(y=acceptable_query_latency, color='r', linestyle='--', alpha=0.7)
            axs[0].text(max(db_data['concurrency'])*0.5, acceptable_query_latency*1.1, 
                     'Acceptable', ha='center', va='bottom',
                     bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
            
            axs[0].set_title('Average Latency', fontsize=14)
            axs[0].set_xlabel('Concurrency Level', fontsize=12)
            axs[0].set_ylabel('Latency (ms)', fontsize=12)
            axs[0].legend(fontsize=10)
            axs[0].grid(True, alpha=0.3)
            
            # Plot p95 latency
            for db_name in grouped['database'].unique():
                db_data = grouped[grouped['database'] == db_name]
                mean_col = ('overall_p95_latency_ms', 'mean')
                std_col = ('overall_p95_latency_ms', 'std')
                
                axs[1].plot(db_data['concurrency'], db_data[mean_col], marker='s', linewidth=2, label=db_name)
                axs[1].fill_between(
                    db_data['concurrency'],
                    db_data[mean_col] - db_data[std_col],
                    db_data[mean_col] + db_data[std_col],
                    alpha=0.2
                )
                
                # Annotate maximum latency
                max_idx = db_data[mean_col].idxmax()
                if not pd.isna(max_idx):
                    max_val = db_data.iloc[max_idx][mean_col]
                    max_conc = db_data.iloc[max_idx]['concurrency']
                    if max_val > 100:  # Only annotate if latency is significant
                        axs[1].annotate(f"{max_val:.1f} ms",
                                     xy=(max_conc, max_val),
                                     xytext=(0, 10),
                                     textcoords='offset points',
                                     ha='center',
                                     va='bottom',
                                     bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.5))
            
            # Add latency threshold
            query_p95_threshold = 500  # 500ms threshold for P95 query latency
            axs[1].axhline(y=query_p95_threshold, color='r', linestyle='--', alpha=0.7)
            axs[1].text(max(db_data['concurrency'])*0.5, query_p95_threshold*1.1, 
                     'P95 Threshold', ha='center', va='bottom',
                     bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
            
            axs[1].set_title('P95 Latency', fontsize=14)
            axs[1].set_xlabel('Concurrency Level', fontsize=12)
            axs[1].set_ylabel('Latency (ms)', fontsize=12)
            axs[1].legend(fontsize=10)
            axs[1].grid(True, alpha=0.3)
            
            # Plot p99 latency
            for db_name in grouped['database'].unique():
                db_data = grouped[grouped['database'] == db_name]
                mean_col = ('overall_p99_latency_ms', 'mean')
                std_col = ('overall_p99_latency_ms', 'std')
                
                axs[2].plot(db_data['concurrency'], db_data[mean_col], marker='D', linewidth=2, label=db_name)
                axs[2].fill_between(
                    db_data['concurrency'],
                    db_data[mean_col] - db_data[std_col],
                    db_data[mean_col] + db_data[std_col],
                    alpha=0.2
                )
                
                # Annotate maximum latency
                max_idx = db_data[mean_col].idxmax()
                if not pd.isna(max_idx):
                    max_val = db_data.iloc[max_idx][mean_col]
                    max_conc = db_data.iloc[max_idx]['concurrency']
                    if max_val > 100:  # Only annotate if latency is significant
                        axs[2].annotate(f"{max_val:.1f} ms",
                                     xy=(max_conc, max_val),
                                     xytext=(0, 10),
                                     textcoords='offset points',
                                     ha='center',
                                     va='bottom',
                                     bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.5))
            
            # Add latency threshold
            query_p99_threshold = 1000  # 1000ms threshold for P99 query latency
            axs[2].axhline(y=query_p99_threshold, color='r', linestyle='--', alpha=0.7)
            axs[2].text(max(db_data['concurrency'])*0.5, query_p99_threshold*1.1, 
                     'P99 Threshold', ha='center', va='bottom',
                     bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
            
            axs[2].set_title('P99 Latency', fontsize=14)
            axs[2].set_xlabel('Concurrency Level', fontsize=12)
            axs[2].set_ylabel('Latency (ms)', fontsize=12)
            axs[2].legend(fontsize=10)
            axs[2].grid(True, alpha=0.3)
            
            plt.suptitle('Latency Metrics vs Concurrency Level - Query Operations', fontsize=16)
            plt.tight_layout()
            plt.savefig(f"{viz_dir}/query_latency_vs_concurrency.png", dpi=300)
            plt.close()
        
        return viz_dir