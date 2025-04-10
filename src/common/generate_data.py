"""
Module pour générer des données de test pour le benchmark des bases de données.
"""
import random
import uuid
import time
from datetime import datetime

def generate_dataset(size):
    """
    Génère un ensemble de données d'événements de souris pour les tests.
    
    Args:
        size (int): Nombre d'événements à générer
        
    Returns:
        list: Liste d'événements de souris
    """
    events = []
    
    # Pages possibles
    pages = [f"/page_{i}" for i in range(1, 10)]
    
    # Types d'événements possibles
    event_types = ["click", "mousemove", "mouseover", "mouseout", "mousedown", "mouseup"]
    
    # Nombre possible d'utilisateurs
    user_count = max(min(size // 10, 100), 5)  # Min 5, max 100 utilisateurs
    
    for i in range(size):
        # Générer des données aléatoires pour l'événement
        user_id = random.randint(1, user_count)
        page = random.choice(pages)
        event_type = random.choice(event_types)
        x_pos = random.randint(0, 1000)
        y_pos = random.randint(0, 800)
        
        # Générer un identifiant unique pour l'événement - IMPORTANT pour les opérations de mise à jour
        event_id = str(uuid.uuid4())
        
        # Créer l'événement
        event = {
            "event_id": event_id,
            "user_id": user_id,
            "page": page,
            "event_type": event_type,
            "x_pos": x_pos,
            "y_pos": y_pos,
            "timestamp": datetime.now().isoformat()
        }
        
        events.append(event)
    
    return events

def generate_advanced_dataset(size):
    """
    Génère un ensemble de données enrichies pour les tests avancés.
    
    Args:
        size (int): Nombre d'événements à générer
        
    Returns:
        list: Liste d'événements enrichis
    """
    # Obtenir les données de base
    basic_events = generate_dataset(size)
    
    # Enrichir avec des données supplémentaires
    for event in basic_events:
        # Ajouter des informations de durée
        event['duration'] = round(random.uniform(0.1, 5.0), 2)
        
        # Ajouter des informations de session
        event['session_id'] = f"session_{random.randint(1, 20)}"
        
        # Ajouter des informations de dispositif
        devices = ['desktop', 'mobile', 'tablet']
        event['device'] = random.choice(devices)
        
        # Ajouter des informations de viewport
        event['viewport_width'] = random.randint(320, 1920)
        event['viewport_height'] = random.randint(480, 1080)
    
    return basic_events