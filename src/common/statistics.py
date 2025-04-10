"""
Module for advanced statistical analysis of benchmark results.
"""
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

class BenchmarkAnalyzer:
    """Analyze benchmark results with advanced statistical methods."""
    
    def __init__(self, result_dir='results'):
        """
        Initialize the benchmark analyzer.
        
        Args:
            result_dir (str): Directory containing benchmark results
        """
        self.result_dir = result_dir
        self.viz_dir = f"{result_dir}/stats_viz_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.viz_dir, exist_ok=True)
        
    def load_results(self, filename):
        """
        Load benchmark results from a CSV file.
        
        Args:
            filename (str): Path to the CSV file
            
        Returns:
            DataFrame: Loaded benchmark results
        """
        return pd.read_csv(filename)
    
    def calculate_statistics(self, df, group_by=None, metric_cols=None):
        """
        Calculate advanced statistics for benchmark results.
        
        Args:
            df (DataFrame): Benchmark results
            group_by (list): Columns to group by
            metric_cols (list): Columns to calculate statistics for
            
        Returns:
            DataFrame: Statistical results
        """
        if group_by is None:
            group_by = ['database', 'operation']
            
        if metric_cols is None:
            # Find numeric columns that aren't in group_by
            metric_cols = [col for col in df.select_dtypes(include=[np.number]).columns
                          if col not in group_by and col != 'iteration']
            
        # Calculate various statistics
        stats_df = df.groupby(group_by)[metric_cols].agg([
            'count',          # Number of samples
            'mean',           # Mean
            'std',            # Standard deviation
            'min',            # Minimum
            'max',            # Maximum
            lambda x: x.quantile(0.25),  # 25th percentile
            lambda x: x.quantile(0.5),   # Median
            lambda x: x.quantile(0.75),  # 75th percentile
            lambda x: x.quantile(0.95),  # 95th percentile
            lambda x: x.quantile(0.99),  # 99th percentile
            lambda x: stats.sem(x)       # Standard error of the mean
        ]).reset_index()
        
        # Rename the custom quantile and sem functions
        stats_df = stats_df.rename(columns={
            '<lambda_0>': 'q25',
            '<lambda_1>': 'q50',
            '<lambda_2>': 'q75',
            '<lambda_3>': 'p95',
            '<lambda_4>': 'p99',
            '<lambda_5>': 'sem'
        })
        
        return stats_df
    
    def calculate_confidence_intervals(self, df, group_by=None, metric_cols=None, confidence=0.95):
        """
        Calculate confidence intervals for benchmark metrics.
        
        Args:
            df (DataFrame): Benchmark results
            group_by (list): Columns to group by
            metric_cols (list): Columns to calculate confidence intervals for
            confidence (float): Confidence level (default: 0.95 for 95% CI)
            
        Returns:
            DataFrame: Confidence interval results
        """
        if group_by is None:
            group_by = ['database', 'operation']
            
        if metric_cols is None:
            # Find numeric columns that aren't in group_by
            metric_cols = [col for col in df.select_dtypes(include=[np.number]).columns
                          if col not in group_by and col != 'iteration']
            
        # Calculate confidence intervals
        result_rows = []
        
        for name, group in df.groupby(group_by):
            row = {}
            
            # Add grouping columns to row
            if isinstance(name, tuple):
                for i, col in enumerate(group_by):
                    row[col] = name[i]
            else:
                row[group_by[0]] = name
                
            # Calculate confidence intervals for each metric
            for col in metric_cols:
                values = group[col].dropna()
                
                if len(values) >= 2:  # Need at least 2 samples for t-test
                    mean = values.mean()
                    sem = stats.sem(values)
                    interval = stats.t.interval(confidence, len(values)-1, loc=mean, scale=sem)
                    
                    row[f"{col}_mean"] = mean
                    row[f"{col}_ci_lower"] = interval[0]
                    row[f"{col}_ci_upper"] = interval[1]
                    row[f"{col}_ci_width"] = interval[1] - interval[0]
                    # Relative width as percentage of mean
                    row[f"{col}_ci_rel_width"] = ((interval[1] - interval[0]) / mean) * 100 if mean != 0 else float('inf')
                else:
                    row[f"{col}_mean"] = values.mean() if len(values) > 0 else np.nan
                    row[f"{col}_ci_lower"] = np.nan
                    row[f"{col}_ci_upper"] = np.nan
                    row[f"{col}_ci_width"] = np.nan
                    row[f"{col}_ci_rel_width"] = np.nan
                    
            result_rows.append(row)
            
        return pd.DataFrame(result_rows)
    
    def perform_t_tests(self, df, metric_col, group_by='operation', compare_col='database'):
        """
        Perform t-tests to compare databases for each operation.
        
        Args:
            df (DataFrame): Benchmark results
            metric_col (str): Column containing the metric to compare
            group_by (str): Column to group by (e.g., 'operation')
            compare_col (str): Column containing values to compare (e.g., 'database')
            
        Returns:
            DataFrame: T-test results
        """
        t_test_results = []
        
        # Get unique values for the grouping column
        operations = df[group_by].unique()
        
        # Get unique values for the comparison column
        databases = df[compare_col].unique()
        
        # Perform t-tests for each operation and each pair of databases
        for operation in operations:
            op_data = df[df[group_by] == operation]
            
            for i in range(len(databases)):
                for j in range(i+1, len(databases)):
                    db1, db2 = databases[i], databases[j]
                    
                    db1_values = op_data[op_data[compare_col] == db1][metric_col].dropna()
                    db2_values = op_data[op_data[compare_col] == db2][metric_col].dropna()
                    
                    if len(db1_values) >= 2 and len(db2_values) >= 2:
                        # Perform t-test
                        t_stat, p_value = stats.ttest_ind(db1_values, db2_values, equal_var=False)
                        
                        # Determine which is better (lower is better for time metrics)
                        better = db1 if db1_values.mean() < db2_values.mean() else db2
                        
                        # Calculate percent difference
                        pct_diff = abs(db1_values.mean() - db2_values.mean()) / min(db1_values.mean(), db2_values.mean()) * 100
                        
                        t_test_results.append({
                            group_by: operation,
                            'db1': db1,
                            'db2': db2,
                            'db1_mean': db1_values.mean(),
                            'db2_mean': db2_values.mean(),
                            'difference': db1_values.mean() - db2_values.mean(),
                            'percent_difference': pct_diff,
                            't_statistic': t_stat,
                            'p_value': p_value,
                            'significant': p_value < 0.05,
                            'better_db': better
                        })
        
        return pd.DataFrame(t_test_results)
    
    def detect_outliers(self, df, metric_cols=None, method='iqr', group_by=None):
        """
        Detect outliers in benchmark results.
        
        Args:
            df (DataFrame): Benchmark results
            metric_cols (list): Columns to check for outliers
            method (str): Outlier detection method ('iqr' or 'zscore')
            group_by (list): Columns to group by
            
        Returns:
            DataFrame: Results with outlier flags
        """
        if group_by is None:
            group_by = ['database', 'operation']
            
        if metric_cols is None:
            # Find numeric columns that aren't in group_by
            metric_cols = [col for col in df.select_dtypes(include=[np.number]).columns
                          if col not in group_by and col != 'iteration']
            
        # Create a copy of the dataframe
        result_df = df.copy()
        
        # Add outlier columns
        for col in metric_cols:
            result_df[f"{col}_is_outlier"] = False
            
        # Detect outliers within each group
        for name, group in df.groupby(group_by):
            for col in metric_cols:
                values = group[col].dropna()
                
                if len(values) >= 4:  # Need a reasonable sample size
                    # Get indices for this group
                    idx = group.index
                    
                    if method == 'iqr':
                        # IQR method
                        q1, q3 = np.percentile(values, [25, 75])
                        iqr = q3 - q1
                        lower_bound = q1 - (1.5 * iqr)
                        upper_bound = q3 + (1.5 * iqr)
                        
                        # Flag outliers
                        for i, val in zip(idx, values):
                            if val < lower_bound or val > upper_bound:
                                result_df.at[i, f"{col}_is_outlier"] = True
                                
                    elif method == 'zscore':
                        # Z-score method
                        mean = values.mean()
                        std = values.std()
                        
                        # Flag outliers
                        for i, val in zip(idx, values):
                            z = (val - mean) / std if std > 0 else 0
                            if abs(z) > 3:  # 3 standard deviations from the mean
                                result_df.at[i, f"{col}_is_outlier"] = True
        
        return result_df
    
    def visualize_confidence_intervals(self, ci_df, metric_col, group_by='operation'):
        """
        Visualize confidence intervals.
        
        Args:
            ci_df (DataFrame): Confidence interval results
            metric_col (str): Base metric column name
            group_by (str): Column to group by
            
        Returns:
            str: Path to saved visualization
        """
        plt.figure(figsize=(14, 8))
        
        # Get unique values for the grouping column
        operations = ci_df[group_by].unique()
        num_operations = len(operations)
        
        # Set up the plot
        width = 0.8 / len(ci_df['database'].unique())
        x = np.arange(num_operations)
        
        # Plot each database's confidence intervals
        for i, db in enumerate(sorted(ci_df['database'].unique())):
            db_data = ci_df[ci_df['database'] == db]
            db_data = db_data.set_index(group_by)
            
            means = []
            errors = []
            
            for op in operations:
                if op in db_data.index:
                    mean_col = f"{metric_col}_mean"
                    lower_col = f"{metric_col}_ci_lower"
                    upper_col = f"{metric_col}_ci_upper"
                    
                    mean = db_data.loc[op, mean_col]
                    lower = db_data.loc[op, lower_col]
                    upper = db_data.loc[op, upper_col]
                    
                    means.append(mean)
                    errors.append([mean - lower, upper - mean])
                else:
                    means.append(0)
                    errors.append([0, 0])
            
            errors = np.array(errors).T
            
            plt.bar(x + i * width, means, width, label=db, yerr=errors, capsize=5)
        
        # Set labels and title
        plt.xlabel(group_by.capitalize(), fontsize=14)
        plt.ylabel(metric_col.replace('_', ' ').title(), fontsize=14)
        plt.title(f"95% Confidence Intervals for {metric_col.replace('_', ' ').title()}", fontsize=16)
        plt.xticks(x + width * (len(ci_df['database'].unique()) - 1) / 2, operations)
        plt.legend(title='Database')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Save the plot
        filename = f"{self.viz_dir}/ci_{metric_col}_{group_by}.png"
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
        
        return filename
    
    def visualize_t_test_matrix(self, df, metric_col):
        """
        Visualize t-test results as a matrix.
        
        Args:
            df (DataFrame): Benchmark results
            metric_col (str): Column containing the metric to compare
            
        Returns:
            str: Path to saved visualization
        """
        operations = df['operation'].unique()
        databases = df['database'].unique()
        
        for operation in operations:
            op_data = df[df['operation'] == operation]
            
            # Create a matrix to store p-values
            p_matrix = pd.DataFrame(index=databases, columns=databases)
            
            # Fill the matrix
            for db1 in databases:
                for db2 in databases:
                    if db1 == db2:
                        p_matrix.loc[db1, db2] = 1.0  # Same database, p-value = 1
                    else:
                        db1_values = op_data[op_data['database'] == db1][metric_col].dropna()
                        db2_values = op_data[op_data['database'] == db2][metric_col].dropna()
                        
                        if len(db1_values) >= 2 and len(db2_values) >= 2:
                            _, p_value = stats.ttest_ind(db1_values, db2_values, equal_var=False)
                            p_matrix.loc[db1, db2] = p_value
                        else:
                            p_matrix.loc[db1, db2] = np.nan
            
            # Convert all values to float using the recommended method
            p_matrix = p_matrix.fillna(1.0)
            p_matrix = p_matrix.infer_objects(copy=False).astype(float)
            
            # Visualize the matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(p_matrix, annot=True, cmap='YlGnBu', vmin=0, vmax=0.1,
                    cbar_kws={'label': 'p-value'})
            
            plt.title(f"T-test p-values for {operation} - {metric_col}", fontsize=16)
            plt.tight_layout()
            
            # Save the plot
            filename = f"{self.viz_dir}/ttest_{operation}_{metric_col}.png"
            plt.savefig(filename, dpi=300)
            plt.close()
        
        return self.viz_dir
    
    def visualize_outliers(self, outlier_df, metric_col, group_by=None):
        """
        Visualize outliers in benchmark results.
        
        Args:
            outlier_df (DataFrame): Results with outlier flags
            metric_col (str): Column to visualize
            group_by (list): Columns to group by
            
        Returns:
            str: Path to saved visualization
        """
        if group_by is None:
            group_by = ['database', 'operation']
            
        # Create outlier flag column name
        outlier_col = f"{metric_col}_is_outlier"
        
        # Create visualization
        plt.figure(figsize=(14, 8))
        
        # Plot each group
        for name, group in outlier_df.groupby(group_by):
            # Extract values and outlier flags
            values = group[metric_col].values
            is_outlier = group[outlier_col].values
            
            # Create a label from the group name
            if isinstance(name, tuple):
                label = ' - '.join(str(x) for x in name)
            else:
                label = str(name)
                
            # Plot normal points
            normal_x = np.arange(len(values))[~is_outlier]
            normal_y = values[~is_outlier]
            plt.scatter(normal_x, normal_y, label=f"{label} (Normal)", alpha=0.7)
            
            # Plot outliers
            outlier_x = np.arange(len(values))[is_outlier]
            outlier_y = values[is_outlier]
            plt.scatter(outlier_x, outlier_y, label=f"{label} (Outlier)", marker='x', s=100, alpha=0.7)
        
        plt.title(f"Outlier Detection for {metric_col.replace('_', ' ').title()}", fontsize=16)
        plt.xlabel("Sample Index", fontsize=14)
        plt.ylabel(metric_col.replace('_', ' ').title(), fontsize=14)
        plt.legend(title='Group')
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        filename = f"{self.viz_dir}/outliers_{metric_col}.png"
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
        
        return filename
    
    def generate_summary_report(self, stats_df, ci_df, t_test_results, metric_cols=None):
        """
        Generate a summary report of the statistical analysis.
        
        Args:
            stats_df (DataFrame): Statistical results
            ci_df (DataFrame): Confidence interval results
            t_test_results (DataFrame): T-test results
            metric_cols (list): Columns to include in the report
            
        Returns:
            str: Path to saved report
        """
        # Create an HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Database Benchmark Statistical Analysis</title>
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
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin-bottom: 20px;
                }}
                th, td {{
                    text-align: left;
                    padding: 8px;
                    border: 1px solid #ddd;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
                .significant {{
                    font-weight: bold;
                    color: #2980b9;
                }}
                .warning {{
                    color: #e74c3c;
                }}
                .summary {{
                    background-color: #e9f7ef;
                    padding: 15px;
                    border-left: 5px solid #27ae60;
                    margin-bottom: 20px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Database Benchmark Statistical Analysis</h1>
                <p>Generated on: {datetime.now().strftime('%d %B %Y %H:%M:%S')}</p>
                
                <div class="summary">
                    <h2>Summary of Findings</h2>
                    <p>This report presents a detailed statistical analysis of the database benchmarks,
                    including descriptive statistics, confidence intervals, and significance tests.</p>
                </div>
                
                <h2>Descriptive Statistics</h2>
                <p>The following tables show the key statistical measures for each database and operation.</p>
        """
        
        # Add descriptive statistics tables
        for operation in stats_df['operation'].unique():
            op_stats = stats_df[stats_df['operation'] == operation]
            
            html_content += f"""
                <h3>Operation: {operation}</h3>
                <table>
                    <tr>
                        <th>Database</th>
                        <th>Metric</th>
                        <th>Count</th>
                        <th>Mean</th>
                        <th>Std Dev</th>
                        <th>Min</th>
                        <th>Median</th>
                        <th>Max</th>
                        <th>95th Percentile</th>
                    </tr>
            """
            
            for _, row in op_stats.iterrows():
                db = row['database']
                for metric in metric_cols or ['time', 'ops_per_second']:
                    if (metric, 'mean') in row:
                        html_content += f"""
                            <tr>
                                <td>{db}</td>
                                <td>{metric.replace('_', ' ').title()}</td>
                                <td>{row[(metric, 'count')]:.0f}</td>
                                <td>{row[(metric, 'mean')]:.4f}</td>
                                <td>{row[(metric, 'std')]:.4f}</td>
                                <td>{row[(metric, 'min')]:.4f}</td>
                                <td>{row[(metric, 'q50')]:.4f}</td>
                                <td>{row[(metric, 'max')]:.4f}</td>
                                <td>{row[(metric, 'p95')]:.4f}</td>
                            </tr>
                        """
            
            html_content += "</table>"
        
        # Add confidence interval section
        html_content += """
                <h2>Confidence Intervals (95%)</h2>
                <p>The following tables show 95% confidence intervals for key metrics.</p>
        """
        
        for operation in ci_df['operation'].unique():
            op_ci = ci_df[ci_df['operation'] == operation]
            
            html_content += f"""
                <h3>Operation: {operation}</h3>
                <table>
                    <tr>
                        <th>Database</th>
                        <th>Metric</th>
                        <th>Mean</th>
                        <th>Lower CI</th>
                        <th>Upper CI</th>
                        <th>CI Width</th>
                        <th>Relative Width (%)</th>
                    </tr>
            """
            
            for _, row in op_ci.iterrows():
                db = row['database']
                for metric in metric_cols or ['time', 'ops_per_second']:
                    mean_col = f"{metric}_mean"
                    lower_col = f"{metric}_ci_lower"
                    upper_col = f"{metric}_ci_upper"
                    width_col = f"{metric}_ci_width"
                    rel_width_col = f"{metric}_ci_rel_width"
                    
                    if mean_col in row and not pd.isna(row[mean_col]):
                        rel_width = row[rel_width_col]
                        width_class = "warning" if rel_width > 10 else ""
                        
                        html_content += f"""
                            <tr>
                                <td>{db}</td>
                                <td>{metric.replace('_', ' ').title()}</td>
                                <td>{row[mean_col]:.4f}</td>
                                <td>{row[lower_col]:.4f}</td>
                                <td>{row[upper_col]:.4f}</td>
                                <td>{row[width_col]:.4f}</td>
                                <td class="{width_class}">{rel_width:.2f}%</td>
                            </tr>
                        """
            
            html_content += "</table>"
        
        # Add t-test results section
        html_content += """
                <h2>Statistical Significance Tests</h2>
                <p>The following tables show the results of t-tests comparing different databases.</p>
        """
        
        for operation in t_test_results['operation'].unique():
            op_tests = t_test_results[t_test_results['operation'] == operation]
            
            html_content += f"""
                <h3>Operation: {operation}</h3>
                <table>
                    <tr>
                        <th>Database 1</th>
                        <th>Database 2</th>
                        <th>Difference</th>
                        <th>% Difference</th>
                        <th>p-value</th>
                        <th>Significant?</th>
                        <th>Better Database</th>
                    </tr>
            """
            
            for _, row in op_tests.iterrows():
                sig_class = "significant" if row['significant'] else ""
                
                html_content += f"""
                    <tr>
                        <td>{row['db1']}</td>
                        <td>{row['db2']}</td>
                        <td>{row['difference']:.4f}</td>
                        <td>{row['percent_difference']:.2f}%</td>
                        <td>{row['p_value']:.4f}</td>
                        <td class="{sig_class}">{"Yes" if row['significant'] else "No"}</td>
                        <td class="{sig_class if row['significant'] else ""}">{row['better_db']}</td>
                    </tr>
                """
            
            html_content += "</table>"
        
        # Close the HTML document
        html_content += """
            </div>
        </body>
        </html>
        """
        
        # Save the report
        report_path = f"{self.viz_dir}/statistical_report.html"
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        return report_path