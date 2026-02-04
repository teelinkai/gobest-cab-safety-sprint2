"""
Mode Controller Module - UPDATED WITH MULTI-FILE SUPPORT
Handles the business logic and state management for different modes
Integrated with DataProcessor and Predictor with Dask support
"""

from typing import Optional, Dict, List, Callable
from datetime import datetime
import pandas as pd
from pathlib import Path

from .. import config
from .data_processor import DataProcessor
from .predictor import Predictor


class ModeController:
    """
    Controller class that manages the application state and business logic
    Supports multi-file batch processing with memory-efficient merging
    """
    
    def __init__(self):
        """Initialize the controller"""
        self.current_mode: str = config.MODE_BATCH
        self.session_history: List[Dict] = []
        self.current_prediction: Optional[Dict] = None
        
        # Initialize data processor and predictor
        print("🚀 Initializing GOBEST CAB Safety System...")
        self.processor = DataProcessor(chunk_size=config.CHUNK_SIZE)
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
        return self.processor.validate_csv(Path(file_path))
        
    def process_batch_files(
        self, 
        file_paths: List[str],
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Dict:
        """
        Process multiple batch files with merging and deduplication
        
        Args:
            file_paths: List of paths to CSV files
            progress_callback: Optional callback for progress updates (progress, message)
            
        Returns:
            Dictionary containing prediction results
        """
        file_paths = [Path(fp) for fp in file_paths]
        
        print("\n" + "="*70)
        print(f"📊 BATCH PROCESSING: {len(file_paths)} file(s)")
        print("="*70)
        
        # 1. Validate all files
        print("\n1️⃣  Validating files...")
        if progress_callback:
            progress_callback(0.05, "Validating files...")
        
        for file_path in file_paths:
            is_valid, error_msg = self.processor.validate_csv(file_path)
            if not is_valid:
                raise ValueError(f"Invalid CSV file '{file_path.name}': {error_msg}")
        print("   ✅ All files validated")
        
        # 2. Detect dataset stage (use first file as reference)
        print("\n2️⃣  Detecting dataset stage...")
        stage = self.processor.detect_dataset_stage(file_paths[0])
        print(f"   📋 Dataset stage: {stage}")
        
        # 3. Load and merge files
        if stage == "RAW_SENSOR":
            print("\n3️⃣  Loading and merging RAW sensor data...")
            if progress_callback:
                progress_callback(0.1, "Loading and merging files...")
            
            # Load and merge with deduplication (Dask-powered)
            raw_df = self.processor.load_and_merge_csvs(
                file_paths, 
                progress_callback=progress_callback
            )
            
            # Extract features
            if progress_callback:
                progress_callback(0.5, "Extracting features...")
            
            features_df = self.processor.process_batch_data(
                raw_df,
                progress_callback=progress_callback
            )
            
        elif stage == "FEATURES_READY":
            print("\n3️⃣  Loading pre-computed features...")
            if progress_callback:
                progress_callback(0.3, "Loading features...")
            
            # Load all feature files and merge
            dfs = []
            for i, file_path in enumerate(file_paths):
                df = pd.read_csv(file_path)
                dfs.append(df)
                if progress_callback:
                    progress_callback(
                        0.3 + (0.3 * (i+1)/len(file_paths)),
                        f"Loaded {i+1}/{len(file_paths)} feature files"
                    )
            
            features_df = pd.concat(dfs, ignore_index=True)
            
            # Deduplicate by bookingID
            initial_count = len(features_df)
            features_df = features_df.drop_duplicates(subset=['bookingID'], keep='last')
            duplicates_removed = initial_count - len(features_df)
            
            print(f"   ✅ Loaded {len(features_df)} trips ({duplicates_removed} duplicates removed)")
            
        else:
            raise ValueError(
                f"Unsupported dataset stage: {stage}\n"
                f"Expected: RAW_SENSOR or FEATURES_READY"
            )
        
        # 4. Make predictions
        print("\n4️⃣  Making predictions with CA2 final model...")
        if progress_callback:
            progress_callback(0.85, "Making predictions...")
        
        results_df = self.predictor.predict_batch(features_df)
        
        # 5. Calculate summary statistics
        print("\n5️⃣  Generating summary...")
        if progress_callback:
            progress_callback(0.95, "Generating summary...")
        
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
        files_str = f"{file_paths[0].name}" if len(file_paths) == 1 else f"{len(file_paths)} files"
        
        prediction_data = {
            'mode': 'batch',
            'file': files_str,
            'file_paths': [str(f) for f in file_paths],
            'num_files': len(file_paths),
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
        history_entry.pop('results_df', None)
        self.add_to_history(history_entry)
        
        # Print summary
        print("\n" + "="*70)
        print("✅ BATCH PROCESSING COMPLETE!")
        print("="*70)
        print(f"   Files Processed: {len(file_paths)}")
        print(f"   Total Trips: {total_trips:,}")
        print(f"   🔴 Dangerous: {dangerous_count:,} ({dangerous_pct:.1f}%)")
        print(f"   🟢 Safe: {safe_count:,} ({100-dangerous_pct:.1f}%)")
        print(f"   Avg Confidence (Dangerous): {avg_confidence_dangerous:.1%}")
        print(f"   Avg Confidence (Safe): {avg_confidence_safe:.1%}")
        print("="*70 + "\n")
        
        if progress_callback:
            progress_callback(1.0, "✅ Complete!")
        
        return prediction_data
        
    def process_realtime_data(self, features_dict: Dict) -> Dict:
        """
        Process manually entered trip data
        
        Args:
            features_dict: Dictionary containing raw and calculated features
            
        Returns:
            Dictionary containing prediction results
        """
        booking_id = features_dict.get('bookingID', 'MANUAL')
        
        print("\n" + "="*60)
        print(f"🔴 REAL-TIME PROCESSING: {booking_id}")
        print("="*60)
        
        # Convert dictionary to DataFrame (single row)
        # We wrap values in lists [] to create the DataFrame
        features_df = pd.DataFrame([features_dict])
        
        # Predict using the existing predictor logic
        # The predictor handles scaling and feature selection automatically
        prediction, confidence = self.predictor.predict_single(features_df)
        
        # Create prediction data package
        prediction_data = {
            'mode': 'realtime',
            'booking_id': booking_id,
            'prediction': prediction,
            'confidence': float(confidence),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            # Pass input data back for display if needed
            'details': features_dict 
        }
        
        # Store current prediction
        self.current_prediction = prediction_data
        self.add_to_history(prediction_data)
        
        print(f"   Prediction: {prediction}")
        print(f"   Confidence: {confidence:.1%}")
        print("="*60 + "\n")
        
        return prediction_data
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return self.predictor.model_info()
    
    def generate_driver_risk_report(self, mapping_file_path: str) -> pd.DataFrame:
        """
        Generates a per-driver risk report by merging prediction results 
        with an uploaded ID mapping file.
        """
        # 1. Validation
        if not self.current_prediction or 'results_df' not in self.current_prediction:
            raise ValueError("No prediction results available to map.")
            
        # 2. Load the mapping file (bookingID, driver_id)
        # We read specifically columns 0 and 1 to be safe, or by name if standard
        try:
            mapping_df = pd.read_csv(mapping_file_path)
            # Normalize column names for easier merging
            mapping_df.columns = [c.strip() for c in mapping_df.columns]
            
            # Identify the driver column (handle 'driver_id', 'driverID', etc.)
            driver_col = next((c for c in mapping_df.columns if 'driver' in c.lower()), 'driver_id')
            booking_col = next((c for c in mapping_df.columns if 'booking' in c.lower()), 'bookingID')
            
        except Exception as e:
            raise ValueError(f"Could not read mapping file: {str(e)}")

        # 3. Get current prediction results
        results_df = self.current_prediction['results_df'].copy()
        
        # 4. Data Type Safety (Ensure bookingIDs are strings for merging)
        results_df['bookingID'] = results_df['bookingID'].astype(str)
        mapping_df[booking_col] = mapping_df[booking_col].astype(str)
        
        # 5. Merge Data
        merged_df = pd.merge(
            results_df, 
            mapping_df[[booking_col, driver_col]], 
            left_on='bookingID', 
            right_on=booking_col, 
            how='inner'
        )
        
        if merged_df.empty:
            raise ValueError("No matching bookingIDs found between results and mapping file.")

        # 6. Group by Driver and Calculate Stats
        driver_stats = merged_df.groupby(driver_col).agg(
            total_trips=('prediction', 'count'),
            dangerous_trips=('prediction', lambda x: (x == 1).sum()),
            safe_trips=('prediction', lambda x: (x == 0).sum()),
            avg_confidence=('confidence', 'mean')
        ).reset_index()
        
        # 7. Calculate Risk Percentage
        driver_stats['risk_score_pct'] = (driver_stats['dangerous_trips'] / driver_stats['total_trips']) * 100
        
        # 8. Sort: Most dangerous drivers at the top (by count, then by %)
        driver_stats = driver_stats.sort_values(
            by=['dangerous_trips', 'risk_score_pct'], 
            ascending=[False, False]
        )
        
        # Rename driver column for the final report
        driver_stats = driver_stats.rename(columns={driver_col: 'Driver ID'})
        
        return driver_stats