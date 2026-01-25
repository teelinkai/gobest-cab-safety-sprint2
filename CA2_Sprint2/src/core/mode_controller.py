"""
Mode Controller Module - FIXED
Handles the business logic and state management for different modes
Integrated with DataProcessor and Predictor
"""

from typing import Optional, Dict, List
from datetime import datetime
import pandas as pd

from .. import config
from .data_processor import DataProcessor
from .predictor import Predictor


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
        
        # Initialize data processor and predictor
        print("🚀 Initializing GOBEST CAB Safety System...")
        self.processor = DataProcessor()
        self.predictor = Predictor()
        print("✅ System ready!\n")
        
    def set_mode(self, mode: str):
        """Set the current operating mode"""
        if mode not in [config.MODE_BATCH, config.MODE_REALTIME]:
            raise ValueError(f"Invalid mode: {mode}")
        
        self.current_mode = mode
        print(f"🔄 Mode switched to: {mode.upper()}")
        
    def get_mode(self) -> str:
        """Get the current mode"""
        return self.current_mode
        
    def add_to_history(self, prediction_data: Dict):
        """Add a prediction to the session history"""
        if 'timestamp' not in prediction_data:
            prediction_data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        self.session_history.append(prediction_data)
        
        # Limit history size
        if len(self.session_history) > config.MAX_HISTORY_ENTRIES:
            self.session_history.pop(0)
        
    def get_history(self) -> List[Dict]:
        """Get the session history"""
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
        from pathlib import Path
        return self.processor.validate_csv(Path(file_path))
        
    def process_batch_file(self, file_path: str) -> Dict:
        """
        Process a batch file and return predictions
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Dictionary containing prediction results
        """
        from pathlib import Path
        file_path = Path(file_path)
        
        print("\n" + "="*60)
        print(f"📊 BATCH PROCESSING: {file_path.name}")
        print("="*60)
        
        # 1. Validate file
        print("\n1️⃣  Validating file...")
        is_valid, error_msg = self.processor.validate_csv(file_path)
        if not is_valid:
            raise ValueError(f"Invalid CSV file: {error_msg}")
        print("   ✅ File validation passed")
        
        # 2. Detect dataset stage
        print("\n2️⃣  Detecting dataset stage...")
        stage = self.processor.detect_dataset_stage(file_path)
        print(f"   📋 Dataset stage: {stage}")
        
        # 3. Load and process based on stage
        if stage == "RAW_SENSOR":
            print("\n3️⃣  Processing RAW sensor data...")
            
            # Load raw sensor data
            raw_df = self.processor.load_csv_fast(file_path)
            
            # Extract features (OPTIMIZED - only 10 features!)
            features_df = self.processor.process_batch_data(raw_df)
            
        elif stage == "FEATURES_READY":
            print("\n3️⃣  Loading pre-computed features...")
            
            # Features already computed, just load
            features_df = pd.read_csv(file_path)
            print(f"   ✅ Loaded {len(features_df)} trips with features")
            
        else:
            raise ValueError(
                f"Unsupported dataset stage: {stage}\n"
                f"Expected: RAW_SENSOR or FEATURES_READY"
            )
        
        # 4. Make predictions
        print("\n4️⃣  Making predictions with Phase 1C model...")
        results_df = self.predictor.predict_batch(features_df)
        
        # 5. Calculate summary statistics
        print("\n5️⃣  Generating summary...")
        total_trips = len(results_df)
        dangerous_count = int((results_df['prediction'] == 1).sum())
        safe_count = total_trips - dangerous_count
        dangerous_pct = (dangerous_count / total_trips * 100) if total_trips > 0 else 0
        
        # Get average confidence for each class
        dangerous_trips = results_df[results_df['prediction'] == 1]
        safe_trips = results_df[results_df['prediction'] == 0]
        
        avg_confidence_dangerous = dangerous_trips['confidence'].mean() if len(dangerous_trips) > 0 else 0
        avg_confidence_safe = safe_trips['confidence'].mean() if len(safe_trips) > 0 else 0
        
        # Create prediction summary
        prediction_data = {
            'mode': 'batch',
            'file': file_path.name,
            'file_path': str(file_path),
            'total_trips': total_trips,
            'dangerous_count': dangerous_count,
            'safe_count': safe_count,
            'dangerous_pct': dangerous_pct,
            'safe_pct': 100 - dangerous_pct,
            'avg_confidence_dangerous': float(avg_confidence_dangerous),
            'avg_confidence_safe': float(avg_confidence_safe),
            'results_df': results_df,  # Full results for export
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Store current prediction
        self.current_prediction = prediction_data
        
        # Add to history (without full DataFrame to save memory)
        history_entry = prediction_data.copy()
        history_entry.pop('results_df', None)  # Don't store full results in history
        self.add_to_history(history_entry)
        
        # Print summary
        print("\n" + "="*60)
        print("✅ BATCH PROCESSING COMPLETE!")
        print("="*60)
        print(f"   Total Trips: {total_trips:,}")
        print(f"   🔴 Dangerous: {dangerous_count:,} ({dangerous_pct:.1f}%)")
        print(f"   🟢 Safe: {safe_count:,} ({100-dangerous_pct:.1f}%)")
        print(f"   Avg Confidence (Dangerous): {avg_confidence_dangerous:.1%}")
        print(f"   Avg Confidence (Safe): {avg_confidence_safe:.1%}")
        print("="*60 + "\n")
        
        return prediction_data
        
    def process_realtime_data(self, booking_id: str, trip_data_df: pd.DataFrame = None) -> Dict:
        """
        Process real-time trip data and return prediction
        
        Args:
            booking_id: Booking ID for the trip
            trip_data_df: Optional DataFrame with sensor data for the trip
            
        Returns:
            Dictionary containing prediction results
        """
        print("\n" + "="*60)
        print(f"🔴 REAL-TIME PROCESSING: {booking_id}")
        print("="*60)
        
        # Process the trip
        if trip_data_df is not None and not trip_data_df.empty:
            # Extract features
            features_df = self.processor.process_realtime_trip(booking_id, trip_data_df)
            
            # Predict
            prediction, confidence = self.predictor.predict_single(features_df)
        else:
            # Placeholder if no data provided
            print("⚠️  No trip data provided - using mock prediction")
            prediction = "SAFE"
            confidence = 0.85
        
        # Create prediction data
        prediction_data = {
            'mode': 'realtime',
            'booking_id': booking_id,
            'prediction': prediction,
            'confidence': float(confidence),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Store current prediction
        self.current_prediction = prediction_data
        self.add_to_history(prediction_data)
        
        # Print summary
        print("\n" + "="*60)
        print("✅ REAL-TIME PREDICTION COMPLETE!")
        print("="*60)
        print(f"   Booking ID: {booking_id}")
        print(f"   Prediction: {prediction}")
        print(f"   Confidence: {confidence:.1%}")
        print("="*60 + "\n")
        
        return prediction_data
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return self.predictor.model_info()
