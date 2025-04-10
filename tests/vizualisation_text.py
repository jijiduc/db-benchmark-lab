#!/usr/bin/env python3
"""
Script pour tester les fonctions de visualisation de run_benchmarks.py
sans exécuter les benchmarks complets.
"""
import os
import sys
import glob
import pandas as pd
from datetime import datetime

# Ajout du répertoire parent au chemin Python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import de la classe de benchmark
from tests.run_benchmarks import SimplifiedDatabaseBenchmark

def test_visualizations():
    """
    Teste les fonctions de visualisation en chargeant les résultats existants.
    """
    result_dir = 'results'
    if not os.path.exists(result_dir):
        print(f"Le répertoire {result_dir} n'existe pas.")
        return False
    
    # Trouver les fichiers CSV les plus récents
    insert_files = glob.glob(f"{result_dir}/insert_benchmark_*.csv")
    query_files = glob.glob(f"{result_dir}/query_benchmark_*.csv")
    
    if not insert_files or not query_files:
        print("Aucun fichier de résultats trouvé.")
        return False
    
    # Trier par date pour prendre les plus récents
    insert_csv = sorted(insert_files)[-1]
    query_csv = sorted(query_files)[-1]
    
    print(f"Chargement des données depuis :")
    print(f"  - {insert_csv}")
    print(f"  - {query_csv}")
    
    # Charger les données
    insert_df = pd.read_csv(insert_csv)
    query_df = pd.read_csv(query_csv)
    
    # Créer une instance de benchmark sans clients de BD
    benchmark = SimplifiedDatabaseBenchmark(
        db_clients={},  # Pas besoin de clients pour les visualisations
        data_sizes=[100, 1000, 5000],
        batch_sizes=[1, 10, 50],
        iterations=1,
        result_dir=result_dir
    )
    
    # Exécuter les visualisations
    try:
        print("Génération des visualisations...")
        benchmark.visualize_results(insert_df, query_df)
        print(f"Visualisations générées avec succès dans {result_dir}/visualizations_*")
        return True
    except Exception as e:
        import traceback
        print(f"Erreur lors de la génération des visualisations : {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_visualizations()
    sys.exit(0 if success else 1)