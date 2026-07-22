import json
import os
from datetime import datetime
from typing import Optional
import logging
log = logging.getLogger()

class State:
    def __init__(self, file_location: str):
        self.file_location = file_location
        self.state_file = os.path.join(file_location, "state.json")
        self._state = self._load_state()
        log.info("Qdrant state is initialized")
    
    def _load_state(self) -> dict:
        """Load state from JSON file or create default if doesn't exist"""
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                return json.load(f)
        else:
            # Create directory if it doesn't exist
            os.makedirs(self.file_location, exist_ok=True)
            # Default state
            default_state = {
                "bootstrapped_state": False,
                "latest_day": None,
                "start_day": None
            }
            self._save_state(default_state)
            return default_state
    
    def _save_state(self, state: dict):
        """Save state to JSON file"""
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def _update_state(self):
        """Update the state file with current state"""
        self._save_state(self._state)
    
    @property
    def bootstrapped_state(self) -> bool:
        """Get bootstrap state"""
        return self._state["bootstrapped_state"]
    
    @bootstrapped_state.setter
    def bootstrapped_state(self, value: bool):
        """Set bootstrapped state and update file"""
        self._state["bootstrapped_state"] = value
        self._update_state()
    
    @property
    def latest_day(self) -> Optional[datetime]:
        """Get latest day"""
        day_str = self._state["latest_day"]
        if day_str is None:
            return None
        return datetime.strptime(day_str, "%d-%m-%Y")
    
    @latest_day.setter
    def latest_day(self, value: datetime):
        """Set latest day and update file"""
        self._state["latest_day"] = value.strftime("%d-%m-%Y")
        self._update_state()
    
    @property
    def start_day(self) -> Optional[datetime]:
        """Get start day"""
        day_str = self._state["start_day"]
        if day_str is None:
            return None
        return datetime.strptime(day_str, "%d-%m-%Y")
    
    @start_day.setter
    def start_day(self, value: datetime):
        """Set start day and update file"""
        self._state["start_day"] = value.strftime("%d-%m-%Y")
        self._update_state()
    
    def set_latest_day_to_today(self):
        """Set latest day to current date"""
        self.latest_day = datetime.now()

