#!/usr/bin/env python3
"""
Module pour comparer les performances de MongoDB, PostgreSQL et Redis
"""
import time
import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse
import sys
import uuid

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.mongodb.client import MongoDBClient
from src.postgresql.client import PostgreSQLClient
from src.redis.client import RedisClient
from src.common.generate_data import generate_dataset

class SimplifiedDatabaseBenchmark:
    """Classe pour exécuter des benchmarks comparant différentes bases de données."""
    
    def __init__(self, db_clients, data_sizes=None, batch_sizes=None, iterations=3, result_dir='results'):
        """Initialise le benchmark avec les clients de base de données."""
        self.db_clients = db_clients
        self.data_sizes = data_sizes or [100, 1000, 5000, 10000]
        self.batch_sizes = batch_sizes or [1, 10, 50, 100, 500, 1000]
        self.iterations = iterations
        self.result_dir = result_dir
        
        # Créer le répertoire de résultats s'il n'existe pas
        os.makedirs(result_dir, exist_ok=True)
        
        # Timestamp pour les fichiers de résultats
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
    def _generate_test_data(self, size):
        """Génère des données de test pour le benchmark."""
        events = generate_dataset(size)
        
        # S'assurer que chaque événement a un event_id
        for event in events:
            if 'event_id' not in event:
                event['event_id'] = str(uuid.uuid4())
                
        return events
        
    def run_insert_benchmark(self):
        """Exécute le benchmark pour les opérations d'insertion avec plusieurs itérations."""
        all_results = []
        
        for iteration in range(1, self.iterations + 1):
            print(f"\nItération {iteration}/{self.iterations} des tests d'insertion")
            
            for db_name, client in self.db_clients.items():
                print(f"Exécution du benchmark d'insertion pour {db_name}...")
                
                # Test des insertions individuelles
                for data_size in self.data_sizes:
                    events = self._generate_test_data(data_size)
                    
                    # Nettoyer les données précédentes
                    if db_name == 'mongodb':
                        client.clear_collection()
                    elif db_name == 'postgresql':
                        client.clear_table()
                    elif db_name == 'redis':
                        client.clear_data()
                    
                    # Mesurer le temps pour les insertions individuelles
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
                    print(f"{db_name} - Insertions individuelles - {data_size} événements: {single_time:.4f}s")
                    
                    # Nettoyer à nouveau pour les tests par lot
                    if db_name == 'mongodb':
                        client.clear_collection()
                    elif db_name == 'postgresql':
                        client.clear_table()
                    elif db_name == 'redis':
                        client.clear_data()
                
                # Test des insertions par lot
                for batch_size in self.batch_sizes:
                    # Vérifier que la taille du lot est valide
                    if batch_size <= max(self.data_sizes):
                        events = self._generate_test_data(batch_size)
                        
                        # Nettoyer les données précédentes
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
                        print(f"{db_name} - Insertion par lot - {batch_size} événements: {batch_time:.4f}s")
        
        # Convertir les résultats en DataFrame et sauvegarder
        df = pd.DataFrame(all_results)
        df.to_csv(f"{self.result_dir}/insert_benchmark_{self.timestamp}.csv", index=False)
        
        return df
        
    def run_query_benchmark(self, setup_data_size=5000):
        """Exécute le benchmark pour les opérations de requête simples."""
        all_results = []
        
        # Générer et insérer les données de test
        events = self._generate_test_data(setup_data_size)
        
        # Extraire des valeurs uniques pour les paramètres de requête
        user_ids = list(set(event['user_id'] for event in events))[:5]
        pages = list(set(event['page'] for event in events))[:3]
        
        for iteration in range(1, self.iterations + 1):
            print(f"\nItération {iteration}/{self.iterations} des tests de requête")
            
            for db_name, client in self.db_clients.items():
                print(f"Exécution du benchmark de requête pour {db_name}...")
                
                # Nettoyer les données précédentes
                if db_name == 'mongodb':
                    client.clear_collection()
                elif db_name == 'postgresql':
                    client.clear_table()
                elif db_name == 'redis':
                    client.clear_data()
                    
                # Insérer les données de test
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
                    print(f"{db_name} - Requête par utilisateur {user_id}: {query_time:.4f}s")
                    
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
                    print(f"{db_name} - Requête par page {page}: {query_time:.4f}s")
                    
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
                    print(f"{db_name} - Comptage des événements: {count_time:.4f}s")
                except Exception as e:
                    print(f"Erreur lors du comptage des événements pour {db_name}: {e}")
        
        # Convertir les résultats en DataFrame et sauvegarder
        df = pd.DataFrame(all_results)
        df.to_csv(f"{self.result_dir}/query_benchmark_{self.timestamp}.csv", index=False)
        
        return df
    
    def visualize_results(self, insert_df=None, query_df=None):
        """Visualise les résultats du benchmark."""
        # Créer un répertoire spécifique pour les visualisations
        viz_dir = f"{self.result_dir}/visualizations_{self.timestamp}"
        os.makedirs(viz_dir, exist_ok=True)
        
        # Configurer le style global pour les graphiques
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("Set2")
        
        # 1. Visualisation des performances d'insertion
        if insert_df is not None:
            self._visualize_insert_performance(insert_df, viz_dir)
        
        # 2. Visualisation des performances de requête
        if query_df is not None:
            self._visualize_query_performance(query_df, viz_dir)
        
        # 3. Créer un dashboard récapitulatif HTML
        self._create_performance_dashboard(viz_dir)
        
        print(f"Visualisations générées dans le répertoire: {viz_dir}")
    
    def _visualize_insert_performance(self, df, viz_dir):
        """Génère des visualisations pour les performances d'insertion."""
        # 1.1 Performances des insertions individuelles
        plt.figure(figsize=(12, 8))
        
        # Agréger les données par base de données et taille de données
        single_inserts = df[df['operation'] == 'insert_single']
        grouped = single_inserts.groupby(['database', 'data_size'])['ops_per_second'].agg(['mean', 'std']).reset_index()
        
        # Créer un graphique pour chaque base de données
        for db_name in grouped['database'].unique():
            db_data = grouped[grouped['database'] == db_name]
            plt.plot(db_data['data_size'], db_data['mean'], marker='o', linewidth=2, label=db_name)
            plt.fill_between(
                db_data['data_size'],
                db_data['mean'] - db_data['std'],
                db_data['mean'] + db_data['std'],
                alpha=0.2
            )
        
        plt.title('Performance d\'insertion individuelle', fontsize=16)
        plt.xlabel('Nombre d\'événements', fontsize=14)
        plt.ylabel('Opérations par seconde', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{viz_dir}/insert_single_performance.png", dpi=300)
        plt.close()
        
        # 1.2 Performances des insertions par lot
        plt.figure(figsize=(12, 8))
        
        batch_inserts = df[df['operation'] == 'insert_batch']
        grouped = batch_inserts.groupby(['database', 'batch_size'])['ops_per_second'].agg(['mean', 'std']).reset_index()
        
        for db_name in grouped['database'].unique():
            db_data = grouped[grouped['database'] == db_name]
            plt.plot(db_data['batch_size'], db_data['mean'], marker='s', linewidth=2, label=db_name)
            plt.fill_between(
                db_data['batch_size'],
                db_data['mean'] - db_data['std'],
                db_data['mean'] + db_data['std'],
                alpha=0.2
            )
        
        plt.title('Performance d\'insertion par lot', fontsize=16)
        plt.xlabel('Taille du lot', fontsize=14)
        plt.ylabel('Opérations par seconde', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{viz_dir}/insert_batch_performance.png", dpi=300)
        plt.close()
        
        # 1.3 Gain de performance des lots
        plt.figure(figsize=(14, 10))
        
        # Pour chaque base de données, comparer l'efficacité des lots par rapport aux insertions individuelles
        for db_name in df['database'].unique():
            db_single = df[(df['database'] == db_name) & (df['operation'] == 'insert_single')]
            db_batch = df[(df['database'] == db_name) & (df['operation'] == 'insert_batch')]
            
            # Calculer le gain de performance moyen pour chaque taille de lot
            performance_gains = []
            batch_sizes = []
            
            for batch_size in sorted(db_batch['batch_size'].unique()):
                # Trouver l'équivalent en insertion individuelle
                single_equiv = db_single[db_single['data_size'] == batch_size]
                batch_equiv = db_batch[db_batch['batch_size'] == batch_size]
                
                if not single_equiv.empty and not batch_equiv.empty:
                    single_ops = single_equiv['ops_per_second'].mean()
                    batch_ops = batch_equiv['ops_per_second'].mean()
                    
                    # Calculer le ratio (combien de fois plus rapide)
                    gain = batch_ops / single_ops if single_ops > 0 else 0
                    performance_gains.append(gain)
                    batch_sizes.append(batch_size)
            
            if performance_gains:  # Ne tracer que s'il y a des données
                plt.plot(batch_sizes, performance_gains, marker='D', linewidth=2, label=db_name)
        
        plt.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='Équivalence (pas de gain)')
        plt.title('Gain de performance des insertions par lot vs. insertions individuelles', fontsize=16)
        plt.xlabel('Taille du lot', fontsize=14)
        plt.ylabel('Facteur d\'accélération', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{viz_dir}/insert_batch_efficiency.png", dpi=300)
        plt.close()
    
    def _visualize_query_performance(self, df, viz_dir):
        """Génère des visualisations pour les performances de requête."""
        # 2.1 Temps d'exécution moyen par type de requête
        plt.figure(figsize=(14, 10))
        
        # Agréger les données par base de données et type d'opération
        grouped = df.groupby(['database', 'operation'])['time'].agg(['mean', 'std']).reset_index()
        
        # Créer un graphique à barres groupées
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
        
        plt.xticks(index + bar_width * (len(grouped['database'].unique()) - 1) / 2, [op.replace('_', ' ').title() for op in operations], rotation=45, ha='right')
        plt.title('Temps d\'exécution moyen par type de requête', fontsize=16)
        plt.xlabel('Type de requête', fontsize=14)
        plt.ylabel('Temps (secondes)', fontsize=14)
        plt.legend(title='Base de données', fontsize=12)
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{viz_dir}/query_performance_by_type.png", dpi=300)
        plt.close()
        
        # 2.2 Boxplot des temps de requête
        plt.figure(figsize=(14, 10))
        
        # Créer un boxplot pour chaque type de requête
        sns.boxplot(x='operation', y='time', hue='database', data=df)
        
        plt.title('Distribution des temps d\'exécution par type de requête', fontsize=16)
        plt.xlabel('Type de requête', fontsize=14)
        plt.ylabel('Temps (secondes)', fontsize=14)
        plt.xticks(rotation=45)
        plt.legend(title='Base de données', fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{viz_dir}/query_time_distribution.png", dpi=300)
        plt.close()
    
    def _create_performance_dashboard(self, viz_dir):
        """Crée un tableau de bord HTML avec les visualisations."""
        html_content = f"""
        <!DOCTYPE html>
        <html lang="fr">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Benchmark de Bases de Données - Résultats</title>
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
                <h1>Résultats du Benchmark de Bases de Données pour le Suivi de Souris</h1>
                <p>Date d'exécution: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>
                
                <div class="dashboard-section">
                    <h2>Résumé des Performances</h2>
                    <div class="summary">
                        <p>Ce dashboard présente les résultats de la comparaison entre MongoDB, PostgreSQL et Redis 
                        pour une application de suivi de mouvement de souris. Les tests ont été effectués sur des opérations 
                        d'insertion et de requête avec différentes tailles de données.</p>
                    </div>
                </div>
                
                <div class="dashboard-section">
                    <h2>Performance d'Insertion</h2>
                    
                    <div class="visualization">
                        <h3>Insertions individuelles</h3>
                        <img src="insert_single_performance.png" alt="Performance d'insertion individuelle">
                        <p>Ce graphique montre le nombre d'opérations par seconde pour les insertions individuelles en fonction de la taille des données.</p>
                    </div>
                    
                    <div class="visualization">
                        <h3>Insertions par lot</h3>
                        <img src="insert_batch_performance.png" alt="Performance d'insertion par lot">
                        <p>Ce graphique montre le nombre d'opérations par seconde pour les insertions par lot en fonction de la taille du lot.</p>
                    </div>
                    
                    <div class="visualization">
                        <h3>Gain de performance des lots</h3>
                        <img src="insert_batch_efficiency.png" alt="Efficacité des insertions par lot">
                        <p>Ce graphique montre le facteur d'accélération des insertions par lot par rapport aux insertions individuelles.</p>
                    </div>
                </div>
                
                <div class="dashboard-section">
                    <h2>Performance de Requête</h2>
                    
                    <div class="visualization">
                        <h3>Temps d'exécution par type de requête</h3>
                        <img src="query_performance_by_type.png" alt="Performance par type de requête">
                        <p>Ce graphique montre le temps moyen d'exécution pour chaque type de requête.</p>
                    </div>
                    
                    <div class="visualization">
                        <h3>Distribution des temps de requête</h3>
                        <img src="query_time_distribution.png" alt="Distribution des temps de requête">
                        <p>Ce graphique montre la distribution des temps d'exécution pour chaque type de requête.</p>
                    </div>
                </div>
                
                <footer>
                    <p>Benchmark réalisé dans le cadre du cours "Beyond Relational Databases" (205.2) à la HES-SO Valais.</p>
                    <p>&copy; {datetime.now().year} - Laboratoire de comparaison de bases de données</p>
                </footer>
            </div>
        </body>
        </html>
        """
        
        # Enregistrer le tableau de bord HTML
        with open(f"{viz_dir}/dashboard.html", 'w') as f:
            f.write(html_content)

def main():
    """Fonction principale pour exécuter les benchmarks."""
    parser = argparse.ArgumentParser(description='Exécuter des benchmarks de bases de données.')
    parser.add_argument('--data-sizes', type=int, nargs='+', default=[100, 1000, 5000, 10000],
                        help='Tailles de données à tester')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 10, 50, 100, 500, 1000],
                        help='Tailles de lots à tester')
    parser.add_argument('--iterations', type=int, default=3,
                        help='Nombre d\'itérations pour chaque test')
    parser.add_argument('--setup-data-size', type=int, default=5000,
                        help='Taille de données pour les benchmarks de requête')
    parser.add_argument('--result-dir', type=str, default='results',
                        help='Répertoire pour enregistrer les résultats')
    parser.add_argument('--tests', type=str, nargs='+', default=['insert', 'query'],
                        help='Tests à exécuter (insert, query)')
    
    args = parser.parse_args()
    
    # Initialiser les clients de base de données
    db_clients = {
        'mongodb': MongoDBClient(),
        'postgresql': PostgreSQLClient(connection_params="dbname=mousebenchmark user=jduc password=password"),
        'redis': RedisClient()
    }
    
    # Créer une instance du benchmark
    benchmark = SimplifiedDatabaseBenchmark(
        db_clients=db_clients,
        data_sizes=args.data_sizes,
        batch_sizes=args.batch_sizes,
        iterations=args.iterations,
        result_dir=args.result_dir
    )
    
    # Variables pour stocker les résultats
    insert_df = None
    query_df = None
    
    # Exécuter les benchmarks sélectionnés
    if 'insert' in args.tests:
        print("\n=== Exécution des benchmarks d'insertion ===")
        insert_df = benchmark.run_insert_benchmark()
    
    if 'query' in args.tests:
        print("\n=== Exécution des benchmarks de requête ===")
        query_df = benchmark.run_query_benchmark(setup_data_size=args.setup_data_size)
    
    # Visualiser les résultats
    print("\n=== Génération des visualisations ===")
    benchmark.visualize_results(insert_df, query_df)
    
    # Fermer les connexions
    for client in db_clients.values():
        client.close()
    
    print("\nBenchmarks terminés. Les résultats ont été enregistrés dans le répertoire:", args.result_dir)

if __name__ == "__main__":
    main()