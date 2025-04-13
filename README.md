# Database Comparison Laboratory for Mouse Tracking Analytics

This project is a laboratory component of the "205.2 Beyond Relational Databases" course taught at the School of Engineering (HEI), part of the University of Applied Sciences Western Switzerland (HES-SO) in Sion.

## Overview

This laboratory project conducts a comprehensive comparative analysis between MongoDB, PostgreSQL, and Redis database systems for storing and analyzing mouse tracking data. Mouse tracking represents an ideal use case for evaluating different database paradigms due to its high-frequency data generation, semi-structured nature, and diverse query requirements.

## Key Features

- **Comprehensive Testing Framework**: Automated testing suite with configurable parameters (data sizes, batch sizes, iterations)
- **Performance Metrics**: Detailed analysis of CRUD operations with particular focus on:
  - High-velocity insertions (individual and batch)
  - Simple and complex query operations
  - Update and delete operation efficiency
- **Concurrency Testing**: Evaluates database performance under different levels of concurrent access
- **Statistical Analysis**: Applies confidence intervals, significance testing, and outlier detection for reliable conclusions
- **Resource Monitoring**: Tracks CPU, memory, disk, and network utilization during database operations
- **Advanced Visualization**: Generates detailed charts, graphs, and HTML dashboards to visualize performance differences

## Project Structure

```
db-comparison-lab/
├── src/
│   ├── mongodb/          # MongoDB implementation
│   │   └── client.py     # MongoDB client class
│   ├── postgresql/       # PostgreSQL implementation
│   │   └── client.py     # PostgreSQL client class
│   ├── redis/            # Redis implementation
│   │   └── client.py     # Redis client class
│   └── common/           # Shared utilities
│       ├── generate_data.py     # Data generation utilities
│       ├── system_monitor.py    # System resource monitoring
│       ├── concurrent_benchmark.py  # Concurrent testing framework
│       └── statistics.py        # Statistical analysis utilities
├── tests/                # Test framework
│   ├── run_benchmarks.py  # Main benchmark runner
│   └── visualization_text.py  # Visualization testing script
└── results/              # Results directory
    ├── visualizations_*/ # Generated charts and graphs
    ├── concurrent_viz_*/ # Concurrency test visualizations
    ├── stats_viz_*/      # Statistical analysis visualizations
    └── resource_viz_*/   # System resource usage visualizations
```

## Installation and Setup

### Prerequisites
- Python 3.8+
- MongoDB 5.0+
- PostgreSQL 13.0+
- Redis 6.0+
- Required Python packages: psutil, pandas, matplotlib, seaborn, pymongo, psycopg2, redis, numpy, scipy

### Installation Steps
1. Clone the repository
2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   ```
3. Activate the environment:
   - Linux/Mac: `source venv/bin/activate`
   - Windows: `venv\Scripts\activate`
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Ensure database services are running:
   ```bash
   sudo systemctl start mongodb
   sudo systemctl start postgresql
   sudo systemctl start redis
   ```

## Running the Benchmarks

### Basic Benchmarks
```bash
# Run all benchmarks with default settings
python tests/run_benchmarks.py

# Run only specific tests
python tests/run_benchmarks.py --tests insert query update delete

# Run with custom data sizes
python tests/run_benchmarks.py --data-sizes 100 1000 5000 10000

# Run with custom batch sizes
python tests/run_benchmarks.py --batch-sizes 1 10 50 100 500 1000

# Run with more iterations for better statistical validity
python tests/run_benchmarks.py --iterations 5
```

### Advanced Benchmarks
```bash
# Run advanced query benchmarks
python tests/run_benchmarks.py --tests advanced_query

# Run with system resource monitoring
python tests/run_benchmarks.py --monitor

# Run concurrent benchmarks
python tests/run_benchmarks.py --tests concurrent

# Specify concurrency levels
python tests/run_benchmarks.py --tests concurrent --concurrency-levels 1 2 5 10 20 50 --iterations 3

# Comprehensive benchmark with all features
python tests/run_benchmarks.py --tests insert query advanced_query update delete concurrent --data-sizes 100 1000 5000 10000 --batch-sizes 1 10 50 100 500 1000 --monitor --concurrency-levels 1 5 10 20 --iterations 5
```

## Generated Visualizations

The benchmark runner automatically generates several types of visualizations:

- **Performance Charts**: Throughput and latency charts for each operation type
- **Comparison Graphs**: Side-by-side comparisons of database performance across different dimensions
- **Statistical Visualizations**: Confidence intervals, outlier detection, and significance testing
- **Resource Usage Graphs**: CPU, memory, disk, and network utilization during database operations
- **Interactive Dashboard**: HTML dashboard summarizing all key findings

### Viewing Results
```bash
# List generated result directories
ls -la results/

# View statistical report
firefox results/stats_viz_*/statistical_report.html

# View performance dashboard
firefox results/visualizations_*/dashboard.html
```

## Evaluation Methodology

The benchmarks evaluate databases across multiple dimensions:

- **Performance**: Execution time and throughput of operations with statistical confidence intervals
- **Concurrency**: Performance degradation under different levels of concurrent access
- **Resource Usage**: System impact in terms of CPU, memory, disk I/O, and network utilization
- **Development Efficiency**: Code complexity and implementation considerations
- **Query Capabilities**: Performance of simple lookups vs. complex analytical queries
- **Scalability**: Performance trends as data volume increases

## Customizing Tests

The test framework is highly configurable. Key parameters include:

- `--data-sizes`: Sizes of datasets to test (number of records)
- `--batch-sizes`: Batch sizes for bulk operations
- `--iterations`: Number of test repetitions for statistical validity
- `--concurrency-levels`: Number of concurrent threads to simulate
- `--monitor`: Enable system resource monitoring
- `--tests`: Specific test categories to run
- `--setup-data-size`: Size of dataset for query tests
- `--result-dir`: Custom directory for results

## Authors

[Jeremy Duc](https://github.com/jijiduc)

## License

This project is an academic exercise created for the course "Beyond Relational Databases" (205.2) under the MIT License.

## Acknowledgments

- Prof. Dr. Pamela Delgado
- Prof. Dr. René Schumann

---

*This project was developed with the assistance of Claude AI from Anthropic.*