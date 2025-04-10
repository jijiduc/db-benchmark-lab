# Database Comparison Laboratory for Mouse Tracking Analytics

This project is a laboratory component of the "205.2 Beyond Relational Databases" course taught at the School of Engineering (HEI), part of the University of Applied Sciences Western Switzerland (HES-SO) in Sion.

## Description

This laboratory project conducts a comprehensive comparison between three distinct database systems—MongoDB, PostgreSQL, and Redis—in the context of a mouse-tracking application. Mouse tracking data represents a particularly interesting use case due to its high-velocity data generation, semi-structured nature, and complex querying requirements for subsequent analytics.

The study evaluates each database system across multiple dimensions:

- **Performance metrics**: We measure and compare execution times for CRUD operations, with particular focus on high-velocity insertions (typical of real-time tracking) and complex aggregation queries (essential for analytics).
  
- **Development efficiency**: We analyze implementation complexity, code maintainability, and the learning curve associated with each database technology.

- **Use case suitability**: We assess how each database's specific features align with the requirements of mouse tracking applications, including schema flexibility, indexing capabilities, and query expressiveness.

- **Scalability potential**: We evaluate how each solution might perform under increased data volume and user concurrency.

The results provide evidence-based insights to guide technology selection for similar real-world applications and contribute to a deeper understanding of the strengths and limitations of different database paradigms.

## Project Structure
```
db-comparison-lab/
├── src/
│   ├── mongodb/    # MongoDB implementation
│   ├── postgresql/ # PostgreSQL implementation
│   ├── redis/      # Redis implementation
│   └── common/     # Shared code
├── tests/          # Unit and performance tests
├── data/           # Synthetic test data
├── results/        # Benchmark results
└── docs/           # Documentation
```

## Installation and Setup

### Prerequisites
- Python 3.8+
- MongoDB
- PostgreSQL
- Redis

### Installation
1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the environment: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`

## Running the Tests
```bash
# Generate test data
python src/common/generate_data.py

# Run all benchmarks
python tests/run_benchmarks.py

# Visualize results
python src/common/visualize_results.py
```

## Evaluation Methodology
- **Performance**: Execution time of CRUD operations, throughput under load, and response time variability
- **Development**: Code complexity, implementation time, and maintenance requirements
- **Suitability**: Specific features for effective mouse tracking and subsequent analytics

## Author

[Jeremy Duc](https://github.com/jijiduc)

## License

This project is an academic exercise created for the course "Beyond Relational Databases" (205.2) under the MIT License.

## Acknowledgments

- Prof. Dr. Pamela Delgado
- Prof. Dr. René Schumann
