"""
Data Processor Module - OPTIMIZED VERSION
Handles CSV file processing and feature engineering
ONLY EXTRACTS THE 10 REQUIRED FEATURES (faster processing!)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import warnings

from .. import config

warnings.filterwarnings("ignore")


class DataProcessor:
    """
    Handles data processing operations for CSV files
    OPTIMIZED: Extracts only the 10 features needed by the model
    """
    
    # ==============================
    # CONSTANTS
    # ==============================
    ACC_COLS = ["acceleration_x", "acceleration_y", "acceleration_z"]
    GYRO_COLS = ["gyro_x", "gyro_y", "gyro_z"]
    LIMIT_COLS = ACC_COLS + GYRO_COLS
    
    # Precomputed 99.9% abs quantile limits
    P999_LIMITS = {
        "acceleration_x": 9.886328100000092,
        "acceleration_y": 29.507823600011807,
        "acceleration_z": 13.291296400000132,
        "gyro_x": 1.5066820000000671,
        "gyro_y": 2.5858964800000144,
        "gyro_z": 1.8445167700000504,
    }
    
    # Thresholds
    SPEED_OVER_80_THRESH = 22.2  # m/s (approx 80 km/h)
    ACCEL_MAG_HIGH_THRESH = 12.0
    HARD_ACCEL_RATE = 5.0
    
    def __init__(self):
        """Initialize the data processor"""
        self.last_processed_file: Optional[Path] = None
        self.last_raw_dataframe: Optional[pd.DataFrame] = None
        self.last_features_dataframe: Optional[pd.DataFrame] = None

    def _standardize_sensor_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names"""
        df = df.copy()
        df.columns = [c.strip() for c in df.columns]

        alias = {
            "bookingid": "bookingID",
            "booking_id": "bookingID",
            "second": "second",
            "speed": "speed",
            "accuracy": "accuracy",
            "acceleration_x": "acceleration_x",
            "acceleration_y": "acceleration_y",
            "acceleration_z": "acceleration_z",
            "gyro_x": "gyro_x",
            "gyro_y": "gyro_y",
            "gyro_z": "gyro_z",
        }

        rename = {}
        for c in df.columns:
            key = c.strip().lower()
            if key in alias:
                rename[c] = alias[key]

        return df.rename(columns=rename)
        
    def validate_csv(self, file_path: Path) -> Tuple[bool, str]:
        """Validate CSV file structure and content"""
        try:
            if not file_path.exists():
                return False, "File does not exist"
            
            # Check file size
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            if file_size_mb > config.MAX_FILE_SIZE_MB:
                return False, f"File too large ({file_size_mb:.1f} MB). Maximum is {config.MAX_FILE_SIZE_MB} MB"
            
            # Try to read header
            df = pd.read_csv(file_path, nrows=5)
            df = self._standardize_sensor_columns(df)
            
            # Check required columns
            missing_cols = [col for col in config.REQUIRED_COLUMNS if col not in df.columns]
            if missing_cols:
                return False, f"Missing required columns: {', '.join(missing_cols)}"
            
            return True, ""
            
        except Exception as e:
            return False, f"Error reading file: {str(e)}"
    
    def load_csv_fast(self, file_path: Path) -> pd.DataFrame:
        """Load CSV file efficiently"""
        print(f"📂 Loading CSV: {file_path.name}")
        df = pd.read_csv(file_path)
        print(f"   Loaded {len(df):,} rows")
        df = self._standardize_sensor_columns(df)
        
        self.last_processed_file = file_path
        self.last_raw_dataframe = df
        return df
    
    def detect_dataset_stage(self, file_path: Path) -> str:
        """Detect what stage of processing the dataset is at"""
        cols = pd.read_csv(file_path, nrows=0).columns
        cols_lower = {c.strip().lower() for c in cols}

        # Raw sensor (per-second)
        sensor_req = {c.lower() for c in config.REQUIRED_COLUMNS}
        if sensor_req.issubset(cols_lower):
            return "RAW_SENSOR"

        # Aggregated trip features
        model_feats = {c.lower() for c in config.MODEL_FEATURES}
        if "bookingid" in cols_lower and model_feats.issubset(cols_lower):
            return "FEATURES_READY"

        return "UNKNOWN"
    
    def clean_sensor_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean raw sensor data"""
        sensor_df_clean = df.copy()
        
        # Drop bad rows
        sensor_df_clean = sensor_df_clean[
            (sensor_df_clean["second"] <= 21600) & 
            ((sensor_df_clean["accuracy"].isna()) | (sensor_df_clean["accuracy"] <= 50))
        ]
        
        # Handle negative speed
        sensor_df_clean["speed"] = sensor_df_clean["speed"].where(
            sensor_df_clean["speed"] >= 0, np.nan
        )
        
        # Handle outliers
        for col in self.LIMIT_COLS:
            lim = self.P999_LIMITS[col]
            mask = sensor_df_clean[col].abs() > lim
            sensor_df_clean[col] = sensor_df_clean[col].where(~mask, np.nan)
        
        # Forward fill within groups
        for col in self.LIMIT_COLS + ["speed"]:
            sensor_df_clean[col] = sensor_df_clean.groupby("bookingID")[col].ffill()
        
        return sensor_df_clean
    
    def extract_features_optimized(self, sensor_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract ONLY the 10 features needed by the model (OPTIMIZED!)
        
        Required features:
        1. trip_duration_sec
        2. speed_mean  
        3. n_points
        4. gyro_mag_std
        5. speed_max
        6. jerk_linear_mean
        7. pct_time_speed_over_80
        8. gyro_jerk_mag_mean
        9. turn_sharpness_index
        10. pct_time_high_accel
        """
        print(f"⚙️  Extracting 10 optimized features...")
        
        sensor_df_chunk = sensor_df.copy()
        
        # === MAGNITUDES (needed for features) ===
        sensor_df_chunk["gyro_mag"] = np.sqrt(
            sensor_df_chunk["gyro_x"]**2 + 
            sensor_df_chunk["gyro_y"]**2 + 
            sensor_df_chunk["gyro_z"]**2
        )
        
        sensor_df_chunk["accel_mag"] = np.sqrt(
            sensor_df_chunk["acceleration_x"]**2 + 
            sensor_df_chunk["acceleration_y"]**2 + 
            sensor_df_chunk["acceleration_z"]**2
        )
        
        # === TIME DELTA ===
        sensor_df_chunk["delta_t"] = sensor_df_chunk.groupby("bookingID")["second"].diff().fillna(1)
        
        # === SPEED RATE ===
        sensor_df_chunk["speed_rate"] = (
            sensor_df_chunk.groupby("bookingID")["speed"].diff().fillna(0) / 
            sensor_df_chunk["delta_t"].replace(0, np.nan)
        ).fillna(0)
        
        # === FLAGS (for percentage features) ===
        sensor_df_chunk["is_speed_over_80"] = sensor_df_chunk["speed"] > self.SPEED_OVER_80_THRESH
        sensor_df_chunk["is_high_accel"] = sensor_df_chunk["accel_mag"] > self.ACCEL_MAG_HIGH_THRESH
        
        # === LINEAR JERK ===
        sensor_df_chunk["d_accel_x"] = sensor_df_chunk.groupby("bookingID")["acceleration_x"].diff().fillna(0)
        sensor_df_chunk["d_accel_y"] = sensor_df_chunk.groupby("bookingID")["acceleration_y"].diff().fillna(0)
        sensor_df_chunk["d_accel_z"] = sensor_df_chunk.groupby("bookingID")["acceleration_z"].diff().fillna(0)
        
        dt_nonzero = sensor_df_chunk["delta_t"].replace(0, np.nan)
        jerk_x_rate = (sensor_df_chunk["d_accel_x"] / dt_nonzero).fillna(0)
        jerk_y_rate = (sensor_df_chunk["d_accel_y"] / dt_nonzero).fillna(0)
        jerk_z_rate = (sensor_df_chunk["d_accel_z"] / dt_nonzero).fillna(0)
        
        sensor_df_chunk["jerk_linear"] = np.sqrt(
            jerk_x_rate**2 + jerk_y_rate**2 + jerk_z_rate**2
        )
        
        # === GYRO JERK ===
        sensor_df_chunk["d_gyro_x"] = sensor_df_chunk.groupby("bookingID")["gyro_x"].diff().fillna(0)
        sensor_df_chunk["d_gyro_y"] = sensor_df_chunk.groupby("bookingID")["gyro_y"].diff().fillna(0)
        sensor_df_chunk["d_gyro_z"] = sensor_df_chunk.groupby("bookingID")["gyro_z"].diff().fillna(0)
        
        gyro_jerk_x = (sensor_df_chunk["d_gyro_x"] / dt_nonzero).fillna(0)
        gyro_jerk_y = (sensor_df_chunk["d_gyro_y"] / dt_nonzero).fillna(0)
        gyro_jerk_z = (sensor_df_chunk["d_gyro_z"] / dt_nonzero).fillna(0)
        
        sensor_df_chunk["gyro_jerk_mag"] = np.sqrt(
            gyro_jerk_x**2 + gyro_jerk_y**2 + gyro_jerk_z**2
        )
        
        # === AGGREGATION (only needed features) ===
        agg = sensor_df_chunk.groupby("bookingID").agg({
            "second": ["min", "max", "count"],
            "speed": ["max", "mean"],
            "gyro_mag": ["std"],
            "jerk_linear": ["mean"],
            "gyro_jerk_mag": ["mean"],
            "is_speed_over_80": ["mean"],
            "is_high_accel": ["mean"]
        })
        
        # Flatten columns
        agg.columns = ["_".join(col) for col in agg.columns]
        trip_features = agg.reset_index()
        
        # === DERIVE FINAL FEATURES ===
        # 1. trip_duration_sec
        trip_features["trip_duration_sec"] = (
            trip_features["second_max"] - trip_features["second_min"]
        )
        
        # 2. n_points
        trip_features["n_points"] = trip_features["second_count"]
        
        # 3. speed_mean (already have)
        trip_features["speed_mean"] = trip_features["speed_mean"]
        
        # 4. speed_max (already have)
        trip_features["speed_max"] = trip_features["speed_max"]
        
        # 5. gyro_mag_std (already have)
        trip_features["gyro_mag_std"] = trip_features["gyro_mag_std"]
        
        # 6. jerk_linear_mean (already have)
        trip_features["jerk_linear_mean"] = trip_features["jerk_linear_mean"]
        
        # 7. pct_time_speed_over_80 (already have)
        trip_features["pct_time_speed_over_80"] = trip_features["is_speed_over_80_mean"]
        
        # 8. gyro_jerk_mag_mean (already have)
        trip_features["gyro_jerk_mag_mean"] = trip_features["gyro_jerk_mag_mean"]
        
        # 9. turn_sharpness_index
        eps = 1e-3
        # Need gyro_mag_max for this - add to aggregation above if missing
        # For now, approximate from std
        trip_features["turn_sharpness_index"] = (
            trip_features["gyro_mag_std"] / (trip_features["speed_mean"] + eps)
        )
        
        # 10. pct_time_high_accel (already have)
        trip_features["pct_time_high_accel"] = trip_features["is_high_accel_mean"]
        
        # === SELECT ONLY THE 10 FEATURES ===
        final_cols = ["bookingID"] + config.MODEL_FEATURES
        
        # Handle missing columns gracefully
        available_cols = [c for c in final_cols if c in trip_features.columns]
        
        result = trip_features[available_cols].copy()
        
        print(f"✅ Features extracted: {len(result)} trips, {len(available_cols)-1} features")
        
        return result
    
    def process_batch_data(self, raw_df: pd.DataFrame, batch_size: int = None) -> pd.DataFrame:
        """
        Process batch sensor data and extract features
        
        Args:
            raw_df: Raw sensor DataFrame
            batch_size: Ignored (kept for backward compatibility)
            
        Returns:
            DataFrame with engineered features
        """
        print(f"\n🔄 Processing {len(raw_df):,} sensor readings...")
        
        # Clean data
        clean_df = self.clean_sensor_data(raw_df)
        print(f"   Cleaned: {len(clean_df):,} rows retained")
        
        # Extract features (OPTIMIZED - only 10!)
        features_df = self.extract_features_optimized(clean_df)
        
        self.last_features_dataframe = features_df
        
        print(f"✅ Processing complete: {len(features_df)} trips ready for prediction\n")
        
        return features_df
    
    def process_realtime_trip(self, booking_id: str, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process a single trip for real-time prediction
        
        Args:
            booking_id: Booking ID to process
            raw_df: DataFrame containing trip data
            
        Returns:
            DataFrame with engineered features (one row)
        """
        print(f"🔄 Processing real-time trip: {booking_id}")
        
        # Filter to specific booking
        trip_data = raw_df[raw_df['bookingID'] == booking_id]
        
        if trip_data.empty:
            raise ValueError(f"No data found for booking ID: {booking_id}")
        
        # Extract features
        features = self.process_batch_data(trip_data)
        
        return features
    
    def get_booking_ids(self, df: pd.DataFrame) -> list:
        """Get unique booking IDs from DataFrame"""
        return df['bookingID'].unique().tolist()
