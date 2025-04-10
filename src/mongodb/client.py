#!/usr/bin/env python3
"""
Module pour interagir avec MongoDB dans le cadre du benchmark
de suivi des mouvements de souris.
"""
import pymongo
import json
import time
from datetime import datetime

class MongoDBClient:
    """Client pour interagir avec MongoDB."""
    
    def __init__(self, uri="mongodb://localhost:27017/", db_name="mouse_tracking"):
        """Initialise la connexion à MongoDB."""
        self.client = pymongo.MongoClient(uri)
        self.db = self.client[db_name]
        self.events = self.db.events
        
    def insert_event(self, event):
        """Insère un événement de souris dans la collection."""
        return self.events.insert_one(event)
        
    def insert_events(self, events):
        """Insère plusieurs événements de souris dans la collection."""
        return self.events.insert_many(events)
        
    def find_events_by_user(self, user_id):
        """Récupère tous les événements d'un utilisateur."""
        return list(self.events.find({"user_id": user_id}))
        
    def find_events_by_page(self, page):
        """Récupère tous les événements d'une page."""
        return list(self.events.find({"page": page}))
        
    def count_events(self):
        """Compte le nombre total d'événements."""
        return self.events.count_documents({})
        
    def clear_collection(self):
        """Vide la collection d'événements."""
        return self.events.delete_many({})
        
    def close(self):
        """Ferme la connexion à MongoDB."""
        self.client.close()
