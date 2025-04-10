#!/usr/bin/env python3
"""
Module to interact with MongoDB for the mouse tracking benchmark.
"""
import pymongo
import json
from bson import ObjectId
import time
from datetime import datetime

class MongoDBClient:
    """Client for interacting with MongoDB."""
    
    def __init__(self, uri="mongodb://localhost:27017/", db_name="mouse_tracking"):
        """Initialize the connection to MongoDB."""
        self.client = pymongo.MongoClient(uri)
        self.db = self.client[db_name]
        self.events = self.db.events
        
        # Create indexes for better query performance
        self.events.create_index("user_id")
        self.events.create_index("page")
        
    def insert_event(self, event):
        """Insert a mouse event in the collection."""
        # Make a copy to avoid modifying the original object
        clean_event = event.copy()
        # Ensure we don't have duplicate _id fields if reusing events
        if '_id' in clean_event:
            del clean_event['_id']
            
        result = self.events.insert_one(clean_event)
        return result.inserted_id
        
    def insert_events(self, events):
        """Insert multiple mouse events in the collection."""
        # Make copies to avoid modifying the original objects
        clean_events = []
        for event in events:
            clean_event = event.copy()
            # Ensure we don't have duplicate _id fields if reusing events
            if '_id' in clean_event:
                del clean_event['_id']
            clean_events.append(clean_event)
            
        result = self.events.insert_many(clean_events)
        return result.inserted_ids
        
    def find_events_by_user(self, user_id):
        """Retrieve all events from a user."""
        return list(self.events.find({"user_id": user_id}))
        
    def find_events_by_page(self, page):
        """Retrieve all events from a page."""
        return list(self.events.find({"page": page}))
        
    def count_events(self):
        """Count the total number of events."""
        return self.events.count_documents({})
        
    def clear_collection(self):
        """Clear the events collection."""
        return self.events.delete_many({})
        
    def to_json_serializable(self, mongo_obj):
        """Convert MongoDB objects to JSON serializable format."""
        if isinstance(mongo_obj, list):
            return [self.to_json_serializable(item) for item in mongo_obj]
        
        if isinstance(mongo_obj, dict):
            result = {}
            for key, value in mongo_obj.items():
                if key == '_id' and isinstance(value, ObjectId):
                    result[key] = str(value)
                else:
                    result[key] = self.to_json_serializable(value)
            return result
            
        if isinstance(mongo_obj, ObjectId):
            return str(mongo_obj)
            
        return mongo_obj
        
    def export_events_as_json(self, query=None):
        """Export events that match the query as JSON-serializable objects."""
        if query is None:
            query = {}
        events = list(self.events.find(query))
        return self.to_json_serializable(events)
        
    def close(self):
        """Close the connection to MongoDB."""
        self.client.close()
        
    def __enter__(self):
        """Support for context manager."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure resources are cleaned up."""
        self.close()
        
    # 1. Advanced queries
    def find_events_by_time_range(self, start_date, end_date):
        """Find events within a specific time range."""
        start_str = start_date.isoformat()
        end_str = end_date.isoformat()
        query = {"timestamp": {"$gte": start_str, "$lte": end_str}}
        return list(self.events.find(query))

    def find_events_in_screen_zone(self, x_min, x_max, y_min, y_max):
        """Find events within a specific screen area."""
        query = {
            "x_pos": {"$gte": x_min, "$lte": x_max},
            "y_pos": {"$gte": y_min, "$lte": y_max}
        }
        return list(self.events.find(query))

    def aggregate_events_by_user(self):
        """Aggregate events by user."""
        pipeline = [
            {"$group": {
                "_id": "$user_id",
                "count": {"$sum": 1},
                "events": {"$push": "$$ROOT"}
            }}
        ]
        return list(self.events.aggregate(pipeline))

    def aggregate_events_by_page(self):
        """Aggregate events by page."""
        pipeline = [
            {"$group": {
                "_id": "$page",
                "count": {"$sum": 1},
                "events": {"$push": "$$ROOT"}
            }}
        ]
        return list(self.events.aggregate(pipeline))

    def aggregate_events_by_device(self):
        """Aggregate events by device type."""
        pipeline = [
            {"$group": {
                "_id": "$device",
                "count": {"$sum": 1},
                "events": {"$push": "$$ROOT"}
            }}
        ]
        return list(self.events.aggregate(pipeline))

    def update_event(self, event_id, update_data):
        """Update a single event."""
        result = self.events.update_one({"event_id": event_id}, {"$set": update_data})
        return result.modified_count

    def update_events_batch(self, event_ids, update_data):
        """Update multiple events in a single operation."""
        result = self.events.update_many({"event_id": {"$in": event_ids}}, {"$set": update_data})
        return result.modified_count

    def update_events_conditional(self, conditions, update_data):
        """Update events that match certain conditions."""
        result = self.events.update_many(conditions, {"$set": update_data})
        return result.modified_count

    def delete_event(self, event_id):
        """Delete a single event."""
        result = self.events.delete_one({"event_id": event_id})
        return result.deleted_count

    def delete_events_batch(self, event_ids):
        """Delete multiple events in a single operation."""
        result = self.events.delete_many({"event_id": {"$in": event_ids}})
        return result.deleted_count

    def delete_events_conditional(self, conditions):
        """Delete events that match certain conditions."""
        result = self.events.delete_many(conditions)
        return result.deleted_count