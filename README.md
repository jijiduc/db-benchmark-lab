# Database Comparison Laboratory for Mouse Tracking Analytics

This project is a laboratory component of the "205.2 Beyond Relational Databases" course taught at the School of Engineering (HEI), part of the University of Applied Sciences Western Switzerland (HES-SO) in Sion.

## Description

This laboratory project conducts a comprehensive comparison between three distinct database systems—MongoDB, PostgreSQL, and Redis—in the context of a mouse-tracking application. Mouse tracking data represents a particularly interesting use case due to its high-velocity data generation, semi-structured nature, and complex querying requirements for subsequent analytics.

The study evaluates each database system across multiple dimensions:

- **Performance metrics**: We measure and compare execution times for CRUD operations, with particular focus on high-velocity insertions (typical of real-time tracking) and complex aggregation queries (essential for analytics).
  
- **Development efficiency**: We analyze implementation complexity, code maintainability, and the learning curve associated with each database technology.

- **Use case suitability**: We assess how each database's specific features align with the requirements of mouse tracking applications, including schema flexibility, indexing capabilities, and query expressiveness.

- **Scalability potential**: We evaluate how each solution might perform under increased data volume and user concurrency through dedicated concurrency testing.

- **Statistical rigor**: We apply advanced statistical analysis, including confidence intervals and significance testing, to ensure reliable conclusions.

- **System resource utilization**: We monitor CPU, memory, disk, and network utilization to understand the total system impact of each database solution.

The results provide evidence-based insights to guide technology selection for similar real-world applications and contribute to a deeper understanding of the strengths and limitations of different database paradigms.

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
│   └── common/           # Shared code
│       ├── generate_data.py     # Data generation utilities
│       ├── system_monitor.py    # System resource monitoring
│       ├── concurrent_benchmark.py  # Concurrent testing framework
│       └── statistics.py        # Statistical analysis utilities
├── tests/                # Unit and performance tests
│   └── run_benchmarks.py  # Main benchmark runner
├── results/              # Benchmark results and visualizations
│   ├── visualizations_/ # Generated charts and graphs
│   ├── concurrent_viz_/ # Concurrency test visualizations
│   ├── stats_viz_/      # Statistical analysis visualizations
└──   └── resource_viz_/   # System resource usage visualizations
```

## Installation and Setup

### Prerequisites
- Python 3.8+
- MongoDB 5.0+
- PostgreSQL 13.0+
- Redis 6.0+
- Required Python packages: psutil, pandas, matplotlib, seaborn, pymongo, psycopg2, redis, numpy, scipy

### Installation
1. Clone the repository
2. Create a virtual environment: `python3 -m venv venv`
3. Activate the environment: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`
5. Ensure database services are running:
   ```bash
   sudo systemctl start mongodb
   sudo systemctl start postgresql
   sudo systemctl start redis
   ```

## Running the Tests
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

# For CRUD operations testing
python tests/run_benchmarks.py --tests insert query update delete --data-sizes 1000 5000 --iterations 3

# For testing performance under concurrent load
python tests/run_benchmarks.py --tests concurrent --concurrency-levels 1 5 10 20 --iterations 3

# For analyzing resource utilization:
python tests/run_benchmarks.py --tests insert query --data-sizes 5000 10000 --monitor

# For testing complex query operations
python tests/run_benchmarks.py --tests advanced_query --setup-data-size 5000 --iterations 3

# Full comprehensive benchmark with all features
python tests/run_benchmarks.py --tests insert query advanced_query update delete concurrent --data-sizes 100 1000 5000 10000 --batch-sizes 1 10 50 100 500 1000 --monitor --concurrency-levels 1 5 10 20 --iterations 5
```
## Analyzing Results
```bash
# The benchmark runner automatically generates visualizations and reports
# Check the results directory:
ls -la results/

# View statistical report
firefox results/stats_viz_*/statistical_report.html

# View performance dashboard
firefox results/visualizations_*/dashboard.html
```
## Evaluation Methodology
- **Performance**: Execution time of CRUD operations, throughput under load, and response time variability with statistical confidence intervals
- **Concurrency**: Performance degradation under different levels of concurrent access
- **Resource Usage**: CPU, memory, disk I/O, and network utilization during database operations
- **Development**: Code complexity, implementation time, and maintenance requirements
- **Suitability**: Specific features for effective mouse tracking and subsequent analytics
- **Statistical Significance**: P-values and confidence intervals to ensure reliable comparisons

## Author

[Jeremy Duc](https://github.com/jijiduc)

## License

This project is an academic exercise created for the course "Beyond Relational Databases" (205.2) under the MIT License.

## Acknowledgments

- Prof. Dr. Pamela Delgado
- Prof. Dr. René Schumann
