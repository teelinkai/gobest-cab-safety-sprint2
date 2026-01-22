"""
Data Processor Module
Handles CSV file processing and feature engineering
Extracted from etl_prefect.py - NO SQL DEPENDENCY
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
    Uses pure pandas - no SQL database required
    Extracts features directly from raw sensor data
    """
    
    # ==============================
    # CONSTANTS (from etl_prefect.py)
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
    CRUISE_MIN = 8.33
    CRUISE_MAX = 16.67
    ACCEL_MAG_HIGH_THRESH = 12.0
    GYRO_MAG_HIGH_THRESH = 1.0
    HARD_BRAKE_RATE = -5.0
    HARD_ACCEL_RATE = 5.0
    ZIGZAG_GYRO_Y_THRESH = 0.5
    SMOOTH_ACCEL_THRESH = 2.0
    SMOOTH_GYRO_THRESH = 0.5
    
    def __init__(self):
        """Initialize the data processor"""
        self.last_processed_file: Optional[Path] = None
        self.last_raw_dataframe: Optional[pd.DataFrame] = None
        self.last_features_dataframe: Optional[pd.DataFrame] = None

    def _standardize_sensor_columns(self, df: pd.DataFrame) -> pd.DataFrame:
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
        """
        Validate CSV file structure and content
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check file exists
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
        """
        Load CSV file efficiently for large files
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Pandas DataFrame (raw sensor data)
        """
        print(f"Loading CSV file: {file_path.name}")
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df):,} rows")
        df = self._standardize_sensor_columns(df)
        
        self.last_processed_file = file_path
        self.last_raw_dataframe = df
        return df
    
    def detect_dataset_stage(self, file_path: Path) -> str:
        cols = pd.read_csv(file_path, nrows=0).columns
        cols_lower = {c.strip().lower() for c in cols}

        # --- raw sensor (per-second) ---
        sensor_req = {c.lower() for c in config.REQUIRED_COLUMNS}
        if sensor_req.issubset(cols_lower):
            return "RAW_SENSOR"

        # --- aggregated trip features (your engineered columns) ---
        agg_markers = {"bookingid", "trip_duration_sec", "n_points", "speed_mean", "accel_mag_mean", "gyro_mag_mean"}
        if "bookingid" in cols_lower and len(agg_markers.intersection(cols_lower)) >= 3:
            return "AGG_TRIP"

        # --- already model-ready features (strongest check if you have model feature list) ---
        # If you know the model feature columns, put them in config.MODEL_FEATURES
        model_feats = getattr(config, "MODEL_FEATURES", None)
        if model_feats:
            mf = {c.lower() for c in model_feats}
            if "bookingid" in cols_lower and mf.issubset(cols_lower):
                return "FEATURES_READY"

        # --- labels-only ---
        if "bookingid" in cols_lower and "label" in cols_lower:
            return "SAFETY"

        # --- drivers-only ---
        if "id" in cols_lower and "name" in cols_lower:
            return "DRIVERS"

        return "UNKNOWN"
    
    def clean_sensor_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """
        Clean raw sensor data (from etl_prefect.py)
        
        Args:
            chunk: Raw sensor data chunk
            
        Returns:
            Cleaned DataFrame
        """
        sensor_df_clean = chunk.copy()
        
        # Drop bad rows
        sensor_df_clean = sensor_df_clean[
            (sensor_df_clean["second"] <= 21600) & 
            ((sensor_df_clean["accuracy"].isna()) | (sensor_df_clean["accuracy"] <= 50))
        ]
        
        # Handle negative speed
        sensor_df_clean["speed"] = sensor_df_clean["speed"].where(sensor_df_clean["speed"] >= 0, np.nan)
        
        # Handle Outliers
        for col in self.LIMIT_COLS:
            lim = self.P999_LIMITS[col]
            mask = sensor_df_clean[col].abs() > lim
            sensor_df_clean[col] = sensor_df_clean[col].where(~mask, np.nan)
            
        return sensor_df_clean
    
    def aggregate_sensor_chunk(self, sensor_df_chunk: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate raw sensor rows into one row per bookingID
        Full feature engineering (from etl_prefect.py)
        
        Args:
            sensor_df_chunk: Raw sensor data
            
        Returns:
            DataFrame with engineered features per trip
        """
        # 1. Clean
        sensor_df_chunk = self.clean_sensor_chunk(sensor_df_chunk)
        sensor_df_chunk = sensor_df_chunk.sort_values(["bookingID", "second"])
        
        # 2. Derived Features
        
        # Magnitudes
        sensor_df_chunk["accel_mag"] = np.sqrt(
            sensor_df_chunk["acceleration_x"].fillna(0)**2 + 
            sensor_df_chunk["acceleration_y"].fillna(0)**2 + 
            sensor_df_chunk["acceleration_z"].fillna(0)**2
        )
        sensor_df_chunk["gyro_mag"] = np.sqrt(
            sensor_df_chunk["gyro_x"].fillna(0)**2 + 
            sensor_df_chunk["gyro_y"].fillna(0)**2 + 
            sensor_df_chunk["gyro_z"].fillna(0)**2
        )
        
        # Deltas
        sensor_df_chunk["delta_t"] = sensor_df_chunk.groupby("bookingID")["second"].diff().fillna(0)
        sensor_df_chunk["delta_speed"] = sensor_df_chunk.groupby("bookingID")["speed"].diff().fillna(0)
        
        # Speed Rate (Accel/Brake)
        valid_dt = (sensor_df_chunk["delta_t"] > 0) & (sensor_df_chunk["delta_t"] <= 10)
        sensor_df_chunk["speed_rate"] = 0.0
        sensor_df_chunk.loc[valid_dt, "speed_rate"] = (
            sensor_df_chunk.loc[valid_dt, "delta_speed"] / sensor_df_chunk.loc[valid_dt, "delta_t"]
        )
        
        # Flags
        sensor_df_chunk["is_speed_over_80"] = sensor_df_chunk["speed"] > self.SPEED_OVER_80_THRESH
        sensor_df_chunk["is_cruising"] = sensor_df_chunk["speed"].between(self.CRUISE_MIN, self.CRUISE_MAX)
        sensor_df_chunk["is_high_accel"] = sensor_df_chunk["accel_mag"] > self.ACCEL_MAG_HIGH_THRESH
        sensor_df_chunk["is_hard_turn"] = sensor_df_chunk["gyro_mag"] > self.GYRO_MAG_HIGH_THRESH
        sensor_df_chunk["is_hard_brake"] = sensor_df_chunk["speed_rate"] < self.HARD_BRAKE_RATE
        sensor_df_chunk["is_hard_accel"] = sensor_df_chunk["speed_rate"] > self.HARD_ACCEL_RATE

        # ----- Linear jerk (rate of change of acceleration) -----
        sensor_df_chunk["d_accel_x"] = sensor_df_chunk.groupby("bookingID")["acceleration_x"].diff().fillna(0)
        sensor_df_chunk["d_accel_y"] = sensor_df_chunk.groupby("bookingID")["acceleration_y"].diff().fillna(0)
        sensor_df_chunk["d_accel_z"] = sensor_df_chunk.groupby("bookingID")["acceleration_z"].diff().fillna(0)

        dt_nonzero = sensor_df_chunk["delta_t"].replace(0, np.nan)
        jerk_x_rate = (sensor_df_chunk["d_accel_x"] / dt_nonzero).fillna(0)
        jerk_y_rate = (sensor_df_chunk["d_accel_y"] / dt_nonzero).fillna(0)
        jerk_z_rate = (sensor_df_chunk["d_accel_z"] / dt_nonzero).fillna(0)

        sensor_df_chunk["jerk_linear"] = np.sqrt(jerk_x_rate**2 + jerk_y_rate**2 + jerk_z_rate**2)

        # ----- Gyro jerk -----
        sensor_df_chunk["d_gyro_x"] = sensor_df_chunk.groupby("bookingID")["gyro_x"].diff().fillna(0)
        sensor_df_chunk["d_gyro_y"] = sensor_df_chunk.groupby("bookingID")["gyro_y"].diff().fillna(0)
        sensor_df_chunk["d_gyro_z"] = sensor_df_chunk.groupby("bookingID")["gyro_z"].diff().fillna(0)

        gyro_jerk_x = (sensor_df_chunk["d_gyro_x"] / dt_nonzero).fillna(0)
        gyro_jerk_y = (sensor_df_chunk["d_gyro_y"] / dt_nonzero).fillna(0)
        gyro_jerk_z = (sensor_df_chunk["d_gyro_z"] / dt_nonzero).fillna(0)

        sensor_df_chunk["gyro_jerk_mag"] = np.sqrt(gyro_jerk_x**2 + gyro_jerk_y**2 + gyro_jerk_z**2)

        # ----- Zig-zag detection on gyro_y -----
        sensor_df_chunk["gyro_y_sign"] = 0
        sensor_df_chunk.loc[sensor_df_chunk["gyro_y"] >  self.ZIGZAG_GYRO_Y_THRESH, "gyro_y_sign"] =  1
        sensor_df_chunk.loc[sensor_df_chunk["gyro_y"] < -self.ZIGZAG_GYRO_Y_THRESH, "gyro_y_sign"] = -1

        sensor_df_chunk["prev_gyro_y_sign"] = (
            sensor_df_chunk.groupby("bookingID")["gyro_y_sign"].shift().fillna(0).astype(int)
        )

        zigzag_mask = (
            (sensor_df_chunk["gyro_y_sign"] != 0) &
            (sensor_df_chunk["prev_gyro_y_sign"] != 0) &
            (sensor_df_chunk["gyro_y_sign"] * sensor_df_chunk["prev_gyro_y_sign"] == -1) &
            (sensor_df_chunk["delta_t"] <= 5)
        )
        sensor_df_chunk["is_zigzag"] = zigzag_mask

        # ----- Smooth driving flag -----
        sensor_df_chunk["is_smooth"] = (
            (sensor_df_chunk["accel_mag"] < self.SMOOTH_ACCEL_THRESH) &
            (sensor_df_chunk["gyro_mag"]  < self.SMOOTH_GYRO_THRESH)
        )

        # Label smooth segments
        grp = sensor_df_chunk.groupby("bookingID")
        prev_smooth = grp["is_smooth"].shift(fill_value=False)
        start_new_block = sensor_df_chunk["is_smooth"] & (~prev_smooth)
        sensor_df_chunk["smooth_block"] = start_new_block.astype(int).groupby(sensor_df_chunk["bookingID"]).cumsum()

        # Sum delta_t for smooth segments
        smooth_agg = (
            sensor_df_chunk.loc[sensor_df_chunk["smooth_block"] > 0]
            .groupby(["bookingID", "smooth_block"])["delta_t"]
            .sum()
            .reset_index(name="smooth_dur_sec")
        )

        if not smooth_agg.empty:
            longest_smooth = (
                smooth_agg.groupby("bookingID", as_index=False)["smooth_dur_sec"]
                .max()
                .rename(columns={"smooth_dur_sec": "longest_smooth_segment_sec"})
            )
        else:
            longest_smooth = pd.DataFrame(columns=["bookingID", "longest_smooth_segment_sec"])

        # 3. Aggregation
        agg = sensor_df_chunk.groupby("bookingID").agg({
            "second": ["min", "max", "count"],
            "speed": ["max", "mean", "std"],
            "accel_mag": ["max", "mean", "std"],
            "gyro_mag": ["max", "mean", "std"],
            "jerk_linear": ["mean"],
            "gyro_jerk_mag": ["mean"],
            "is_speed_over_80": ["mean"],
            "is_cruising": ["mean"],
            "is_high_accel": ["mean"],
            "is_hard_turn": ["sum", "mean"],
            "is_hard_brake": ["sum"],
            "is_hard_accel": ["sum"],
            "is_zigzag": ["sum"],
            "is_smooth": ["mean"]
        })
        
        # Flatten cols
        agg.columns = ["_".join(col) for col in agg.columns]
        trip_features = agg.reset_index()
        
        # Trip duration
        trip_features["trip_duration_sec"] = trip_features["second_max"] - trip_features["second_min"]
        trip_features["n_points"] = trip_features["second_count"]

        # Merge longest smooth segment
        trip_features = trip_features.merge(longest_smooth, on="bookingID", how="left")
        trip_features["longest_smooth_segment_sec"] = trip_features["longest_smooth_segment_sec"].fillna(0.0)

        # Turn sharpness
        eps = 1e-3
        trip_features["turn_sharpness_index"] = (
            trip_features["gyro_mag_max"] / (trip_features["speed_mean"] + eps)
        )

        # 4. Final Cleanup (Rename to clean names)
        trip_features = trip_features.rename(columns={
            "speed_max": "speed_max",
            "speed_mean": "speed_mean",
            "speed_std": "speed_std",
            "accel_mag_max": "accel_mag_max",
            "accel_mag_mean": "accel_mag_mean",
            "accel_mag_std": "accel_mag_std",
            "gyro_mag_max": "gyro_mag_max",
            "gyro_mag_mean": "gyro_mag_mean",
            "gyro_mag_std": "gyro_mag_std",
            "jerk_linear_mean": "jerk_linear_mean",
            "gyro_jerk_mag_mean": "gyro_jerk_mag_mean",
            "is_speed_over_80_mean": "pct_time_speed_over_80",
            "is_cruising_mean": "pct_time_cruising",
            "is_high_accel_mean": "pct_time_high_accel",
            "is_hard_turn_sum": "n_hard_turns",
            "is_hard_turn_mean": "pct_time_high_gyro",
            "is_hard_brake_sum": "n_hard_brakes",
            "is_hard_accel_sum": "n_hard_accels",
            "is_zigzag_sum": "n_zigzag_events",
            "is_smooth_mean": "pct_time_smooth",
        })

        # Select final columns
        final_cols = [
            "bookingID", "trip_duration_sec", "n_points",
            "speed_max", "speed_mean", "speed_std", "pct_time_speed_over_80", "pct_time_cruising",
            "accel_mag_max", "accel_mag_mean", "accel_mag_std", "jerk_linear_mean",
            "n_hard_accels", "n_hard_brakes", "pct_time_high_accel",
            "gyro_mag_max", "gyro_mag_mean", "gyro_mag_std", "gyro_jerk_mag_mean",
            "n_hard_turns", "pct_time_high_gyro", "n_zigzag_events", "turn_sharpness_index",
            "longest_smooth_segment_sec", "pct_time_smooth"
        ]
        
        # Filter to only existing columns (safeguard)
        existing_cols = [c for c in final_cols if c in trip_features.columns]
        return trip_features[existing_cols]
    
    def process_batch_data(self, raw_df: pd.DataFrame, batch_size: int = config.BATCH_SIZE) -> pd.DataFrame:
        print(f"Processing {len(raw_df):,} rows of sensor data...")
        # batch_size is kept for backward compatibility with existing calls
        features_df = self.aggregate_sensor_chunk(raw_df)
        print(f"Feature engineering complete: {len(features_df)} trips processed")
        self.last_features_dataframe = features_df
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
        print(f"Processing real-time trip: {booking_id}")
        
        # Filter to specific booking
        trip_data = raw_df[raw_df['bookingID'] == booking_id]
        
        if trip_data.empty:
            raise ValueError(f"No data found for booking ID: {booking_id}")
        
        # Extract features
        features = self.aggregate_sensor_chunk(trip_data)
        print(f"Real-time processing complete")
        
        return features
    
    def get_booking_ids(self, df: pd.DataFrame) -> list:
        """
        Get unique booking IDs from DataFrame
        
        Args:
            df: DataFrame to extract booking IDs from
            
        Returns:
            List of unique booking IDs
        """
        return df['bookingID'].unique().tolist()
