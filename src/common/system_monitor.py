"""
Module for monitoring system resources during benchmarks.
"""
import psutil
import threading
import time
import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class SystemMonitor:
    """Monitor system resources (CPU, memory, disk I/O) during benchmark runs."""
    
    def __init__(self, interval=1.0, result_dir='results'):
        """
        Initialize the system monitor.
        
        Args:
            interval (float): Sampling interval in seconds
            result_dir (str): Directory to save results
        """
        self.interval = interval
        self.result_dir = result_dir
        self.running = False
        self.thread = None
        self.measurements = []
        self.database = None
        self.operation = None
        self.data_size = None
        self.start_time = None
        
        # Create results directory if needed
        os.makedirs(result_dir, exist_ok=True)
        
    def start(self, database, operation, data_size):
        """
        Start monitoring system resources.
        
        Args:
            database (str): Name of the database being tested
            operation (str): Type of operation being performed
            data_size (int): Size of data being processed
        """
        if self.running:
            return
            
        self.database = database
        self.operation = operation
        self.data_size = data_size
        self.start_time = time.time()
        self.measurements = []
        self.running = True
        
        # Start monitoring in a separate thread
        self.thread = threading.Thread(target=self._monitor)
        self.thread.daemon = True
        self.thread.start()
        
    def stop(self):
        """Stop monitoring and save results."""
        if not self.running:
            return
            
        self.running = False
        if self.thread:
            self.thread.join(timeout=5.0)
            
        # Save results
        if self.measurements:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            df = pd.DataFrame(self.measurements)
            filename = f"{self.result_dir}/resources_{self.database}_{self.operation}_{self.data_size}_{timestamp}.csv"
            df.to_csv(filename, index=False)
            return df
        return None
        
    def _monitor(self):
        """Monitor system resources at regular intervals."""
        while self.running:
            # Get current timestamp relative to start
            current_time = time.time() - self.start_time
            
            # Get CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            
            # Get memory usage
            memory = psutil.virtual_memory()
            memory_used_percent = memory.percent
            memory_used_mb = memory.used / (1024 * 1024)  # Convert to MB
            
            # Get disk I/O since last call
            disk_io = psutil.disk_io_counters()
            disk_read_mb = disk_io.read_bytes / (1024 * 1024)
            disk_write_mb = disk_io.write_bytes / (1024 * 1024)
            
            # Get network I/O
            net_io = psutil.net_io_counters()
            net_sent_mb = net_io.bytes_sent / (1024 * 1024)
            net_recv_mb = net_io.bytes_recv / (1024 * 1024)
            
            # Store measurement
            self.measurements.append({
                'time': current_time,
                'database': self.database,
                'operation': self.operation,
                'data_size': self.data_size,
                'cpu_percent': cpu_percent,
                'memory_percent': memory_used_percent,
                'memory_used_mb': memory_used_mb,
                'disk_read_mb': disk_read_mb,
                'disk_write_mb': disk_write_mb,
                'net_sent_mb': net_sent_mb,
                'net_recv_mb': net_recv_mb
            })
            
            # Sleep for the specified interval
            time.sleep(self.interval)
    
    def visualize_resource_usage(self, result_df=None, viz_dir=None):
        """
        Generate visualizations of resource usage.
        
        Args:
            result_df (DataFrame): DataFrame containing resource measurements
            viz_dir (str): Directory to save visualizations
        """
        if result_df is None and not self.measurements:
            return
            
        if result_df is None:
            result_df = pd.DataFrame(self.measurements)
            
        if viz_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            viz_dir = f"{self.result_dir}/resource_viz_{timestamp}"
            
        # Create visualization directory
        os.makedirs(viz_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("deep")
        
        # Plot CPU usage over time
        plt.figure(figsize=(12, 6))
        for db in result_df['database'].unique():
            db_data = result_df[result_df['database'] == db]
            plt.plot(db_data['time'], db_data['cpu_percent'], label=db, linewidth=2)
            
        plt.title(f'CPU Usage Over Time - {self.operation} Operation', fontsize=16)
        plt.xlabel('Time (seconds)', fontsize=14)
        plt.ylabel('CPU Usage (%)', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{viz_dir}/cpu_usage_{self.operation}.png", dpi=300)
        plt.close()
        
        # Plot memory usage over time
        plt.figure(figsize=(12, 6))
        for db in result_df['database'].unique():
            db_data = result_df[result_df['database'] == db]
            plt.plot(db_data['time'], db_data['memory_used_mb'], label=db, linewidth=2)
            
        plt.title(f'Memory Usage Over Time - {self.operation} Operation', fontsize=16)
        plt.xlabel('Time (seconds)', fontsize=14)
        plt.ylabel('Memory Used (MB)', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{viz_dir}/memory_usage_{self.operation}.png", dpi=300)
        plt.close()
        
        # Plot disk I/O over time
        plt.figure(figsize=(12, 6))
        for db in result_df['database'].unique():
            db_data = result_df[result_df['database'] == db]
            plt.plot(db_data['time'], db_data['disk_write_mb'], label=f"{db} (Write)", linewidth=2)
            plt.plot(db_data['time'], db_data['disk_read_mb'], label=f"{db} (Read)", linewidth=2, linestyle='--')
            
        plt.title(f'Disk I/O Over Time - {self.operation} Operation', fontsize=16)
        plt.xlabel('Time (seconds)', fontsize=14)
        plt.ylabel('Disk I/O (MB)', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{viz_dir}/disk_io_{self.operation}.png", dpi=300)
        plt.close()
        
        # Plot network I/O over time
        plt.figure(figsize=(12, 6))
        for db in result_df['database'].unique():
            db_data = result_df[result_df['database'] == db]
            plt.plot(db_data['time'], db_data['net_sent_mb'], label=f"{db} (Sent)", linewidth=2)
            plt.plot(db_data['time'], db_data['net_recv_mb'], label=f"{db} (Received)", linewidth=2, linestyle='--')
            
        plt.title(f'Network I/O Over Time - {self.operation} Operation', fontsize=16)
        plt.xlabel('Time (seconds)', fontsize=14)
        plt.ylabel('Network I/O (MB)', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{viz_dir}/network_io_{self.operation}.png", dpi=300)
        plt.close()
        
        # Create a heatmap of resource usage by database
        resource_cols = ['cpu_percent', 'memory_percent', 'disk_read_mb', 'disk_write_mb']
        pivot_df = result_df.groupby(['database', 'operation'])[resource_cols].mean().reset_index()
        pivot_table = pivot_df.pivot(index='database', columns='operation', values=resource_cols)
        
        plt.figure(figsize=(14, 10))
        sns.heatmap(pivot_table, annot=True, cmap='YlGnBu', linewidths=.5, fmt='.2f')
        plt.title('Average Resource Usage by Database and Operation', fontsize=16)
        plt.tight_layout()
        plt.savefig(f"{viz_dir}/resource_heatmap.png", dpi=300)
        plt.close()
        
        return viz_dir