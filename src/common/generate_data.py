"""
Module to generate test data for database benchmarking.
"""
import random
import uuid
import time
from datetime import datetime

def generate_dataset(size):
    """
    Generates a dataset of mouse events for testing.
    
    Args:
        size (int): Number of events to generate
        
    Returns:
        list: List of mouse events
    """
    events = []
    
    # Possible pages
    pages = [f"/page_{i}" for i in range(1, 10)]
    
    # Possible event types
    event_types = ["click", "mousemove", "mouseover", "mouseout", "mousedown", "mouseup"]
    
    # Possible number of users
    user_count = max(min(size // 10, 100), 5)  # Min 5, max 100 users
    
    for i in range(size):
        # Generate random data for the event
        user_id = random.randint(1, user_count)
        page = random.choice(pages)
        event_type = random.choice(event_types)
        x_pos = random.randint(0, 1000)
        y_pos = random.randint(0, 800)
        
        # Generate a unique identifier for the event - IMPORTANT for update operations
        event_id = str(uuid.uuid4())
        
        # Create the event
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
    Generates an enriched dataset for advanced testing.
    
    Args:
        size (int): Number of events to generate
        
    Returns:
        list: List of enriched events
    """
    # Get the basic data
    basic_events = generate_dataset(size)
    
    # Enrich with additional data
    for event in basic_events:
        # Add duration information
        event['duration'] = round(random.uniform(0.1, 5.0), 2)
        
        # Add session information
        event['session_id'] = f"session_{random.randint(1, 20)}"
        
        # Add device information
        devices = ['desktop', 'mobile', 'tablet']
        event['device'] = random.choice(devices)
        
        # Add viewport information
        event['viewport_width'] = random.randint(320, 1920)
        event['viewport_height'] = random.randint(480, 1080)
    
    return basic_events