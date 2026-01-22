"""
Mode Controller Module
Handles the business logic and state management for different modes
"""

from typing import Optional, Dict, List
from datetime import datetime

from .. import config


class ModeController:
    """
    Controller class that manages the application state and business logic
    Separates business logic from GUI components (MVC pattern)
    """
    
    def __init__(self):
        """Initialize the controller"""
        self.current_mode: str = config.MODE_BATCH
        self.session_history: List[Dict] = []
        self.current_prediction: Optional[Dict] = None
        
    def set_mode(self, mode: str):
        """
        Set the current operating mode
        
        Args:
            mode: Mode to set (batch or realtime)
        """
        if mode not in [config.MODE_BATCH, config.MODE_REALTIME]:
            raise ValueError(f"Invalid mode: {mode}")
        
        self.current_mode = mode
        print(f"Mode switched to: {mode}")
        
    def get_mode(self) -> str:
        """
        Get the current mode
        
        Returns:
            Current mode string
        """
        return self.current_mode
        
    def add_to_history(self, prediction_data: Dict):
        """
        Add a prediction to the session history
        
        Args:
            prediction_data: Dictionary containing prediction information
        """
        # Add timestamp if not present
        if 'timestamp' not in prediction_data:
            prediction_data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        self.session_history.append(prediction_data)
        
        # Limit history size
        if len(self.session_history) > config.MAX_HISTORY_ENTRIES:
            self.session_history.pop(0)
        
    def get_history(self) -> List[Dict]:
        """
        Get the session history
        
        Returns:
            List of prediction dictionaries
        """
        return self.session_history
        
    def clear_history(self):
        """Clear the session history"""
        self.session_history = []
        
    def validate_csv_file(self, file_path: str) -> tuple[bool, str]:
        """
        Validate CSV file before processing
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # TODO: Implement actual validation
        # For now, return success
        return True, ""
        
    def process_batch_file(self, file_path: str) -> Dict:
        """
        Process a batch file and return prediction
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Dictionary containing prediction results
        """
        # TODO: Implement actual batch processing
        # This is a placeholder
        prediction_data = {
            'mode': 'batch',
            'file': file_path,
            'prediction': 'SAFE',
            'confidence': 0.85,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        self.current_prediction = prediction_data
        self.add_to_history(prediction_data)
        
        return prediction_data
        
    def process_realtime_data(self, trip_data: Dict) -> Dict:
        """
        Process real-time trip data and return prediction
        
        Args:
            trip_data: Dictionary containing trip information
            
        Returns:
            Dictionary containing prediction results
        """
        # TODO: Implement actual real-time processing
        # This is a placeholder
        prediction_data = {
            'mode': 'realtime',
            'booking_id': trip_data.get('booking_id', 'N/A'),
            'prediction': 'DANGEROUS',
            'confidence': 0.72,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        self.current_prediction = prediction_data
        self.add_to_history(prediction_data)
        
        return prediction_data
