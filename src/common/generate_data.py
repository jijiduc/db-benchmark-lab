#!/usr/bin/env python3
"""
Module pour générer des données synthétiques de mouvements de souris
pour tester les différentes bases de données.
"""
import json
import random
import time
from datetime import datetime
import os
import argparse

def generate_mouse_event(user_id=None):
    """Génère un événement de souris aléatoire."""
    if user_id is None:
        user_id = random.randint(1, 100)
    
    return {
        "user_id": user_id,
        "x_pos": random.randint(0, 1920),
        "y_pos": random.randint(0, 1080),
        "event_type": random.choice(["move", "click", "scroll"]),
        "timestamp": datetime.now().isoformat(),
        "page": f"/page_{random.randint(1, 10)}",
        "session_id": f"session_{random.randint(1000, 9999)}"
    }

def generate_dataset(num_events=1000, num_users=10):
    """Génère un ensemble de données de mouvements de souris."""
    events = []
    for _ in range(num_events):
        user_id = random.randint(1, num_users)
        events.append(generate_mouse_event(user_id))
    return events

def save_to_file(data, filename="data/mouse_events.json"):
    """Sauvegarde les données générées dans un fichier JSON."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Données générées et sauvegardées dans {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Génère des données synthétiques de mouvements de souris.')
    parser.add_argument('--events', type=int, default=1000, help='Nombre d\'événements à générer')
    parser.add_argument('--users', type=int, default=10, help='Nombre d\'utilisateurs différents')
    parser.add_argument('--output', type=str, default="data/mouse_events.json", help='Fichier de sortie')
    
    args = parser.parse_args()
    
    dataset = generate_dataset(args.events, args.users)
    save_to_file(dataset, args.output)
