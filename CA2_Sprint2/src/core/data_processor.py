"""
Data Processor Module - OPTIMIZED VERSION WITH DASK
Handles CSV file processing and feature engineering with memory-efficient chunking
EXTRACTS THE 10 REQUIRED FEATURES for CA2 final model
"""

import pandas as pd
import dask.dataframe as dd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Callable
import warnings

warnings.filterwarnings("ignore")


class DataProcessor:
    """
    Handles data processing operations for CSV files with Dask for memory efficiency
    Supports multi-file merging with deduplication
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
    CRUISE_MIN = 8.33  # m/s (30 km/h)
    CRUISE_MAX = 16.67  # m/s (60 km/h)
    ACCEL_MAG_HIGH_THRESH = 12.0
    HARD_ACCEL_RATE = 5.0
    SMOOTH_ACCEL_THRESH = 2.0
    SMOOTH_GYRO_THRESH = 0.5
    
    def __init__(self, chunk_size: int = 1_000_000):
        """Initialize the data processor"""
        self.chunk_size = chunk_size
        self.last_processed_files: List[Path] = []
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
            from .. import config
            if file_size_mb > config.MAX_FILE_SIZE_MB:
                return False, f"File too large ({file_size_mb:.1f} MB). Maximum is {config.MAX_FILE_SIZE_MB} MB"
            
            # Try to read header
            df = pd.read_csv(file_path, nrows=5)
            df = self._standardize_sensor_columns(df)
            
            # Check required columns
            from .. import config
            missing_cols = [col for col in config.REQUIRED_COLUMNS if col not in df.columns]
            if missing_cols:
                return False, f"Missing required columns: {', '.join(missing_cols)}"
            
            return True, ""
            
        except Exception as e:
            return False, f"Error reading file: {str(e)}"
    
    def load_and_merge_csvs(
        self, 
        file_paths: List[Path],
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> pd.DataFrame:
        """
        Load and merge multiple CSV files with deduplication
        Uses Dask for memory-efficient processing
        
        Args:
            file_paths: List of CSV file paths to merge
            progress_callback: Optional callback for progress updates (progress, message)
            
        Returns:
            Merged DataFrame with deduplicated bookingIDs
        """
        print(f"\n📂 Loading and merging {len(file_paths)} file(s)...")
        
        if progress_callback:
            progress_callback(0.1, f"Loading {len(file_paths)} files...")
        
        # Use Dask to read all files efficiently
        dfs = []
        for i, file_path in enumerate(file_paths):
            print(f"   [{i+1}/{len(file_paths)}] Reading: {file_path.name}")
            
            # Read with Dask for memory efficiency
            dask_df = dd.read_csv(
                file_path,
                blocksize=f"{self.chunk_size // 1000}KB"
            )
            
            # Standardize columns
            dask_df = dask_df.rename(columns={
                col: self._get_standardized_name(col) 
                for col in dask_df.columns
            })
            
            dfs.append(dask_df)
            
            if progress_callback:
                progress_callback(0.1 + (0.2 * (i+1)/len(file_paths)), 
                                f"Loaded {i+1}/{len(file_paths)} files")
        
        # Concatenate all files
        print(f"\n🔗 Concatenating files...")
        if progress_callback:
            progress_callback(0.3, "Merging files...")
        
        merged_dask = dd.concat(dfs, axis=0, ignore_index=True)
        
        # Get total rows before dedup
        total_rows = len(merged_dask)
        print(f"   Total rows (before dedup): {total_rows:,}")
        
        # Convert to pandas for deduplication
        # For very large datasets, we process in chunks
        print(f"\n🔄 Converting to pandas and deduplicating...")
        if progress_callback:
            progress_callback(0.4, "Deduplicating bookingIDs...")
        
        merged_df = merged_dask.compute()
        
        # Deduplicate by bookingID + second (keep last occurrence)
        print(f"   Deduplicating by bookingID + second...")
        initial_rows = len(merged_df)
        merged_df = merged_df.drop_duplicates(
            subset=['bookingID', 'second'], 
            keep='last'
        ).reset_index(drop=True)
        
        duplicates_removed = initial_rows - len(merged_df)
        print(f"   Removed {duplicates_removed:,} duplicate rows")
        print(f"   Final rows: {len(merged_df):,}")
        
        # Get unique booking IDs
        unique_bookings = merged_df['bookingID'].nunique()
        print(f"   Unique bookingIDs: {unique_bookings:,}")
        
        if progress_callback:
            progress_callback(0.5, f"Merged: {len(merged_df):,} rows, {unique_bookings:,} trips")
        
        self.last_processed_files = file_paths
        self.last_raw_dataframe = merged_df
        
        return merged_df
    
    def _get_standardized_name(self, col: str) -> str:
        """Get standardized column name"""
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
        key = col.strip().lower()
        return alias.get(key, col)
    
    def detect_dataset_stage(self, file_path: Path) -> str:
        """Detect what stage of processing the dataset is at"""
        cols = pd.read_csv(file_path, nrows=0).columns
        cols_lower = {c.strip().lower() for c in cols}

        # Raw sensor (per-second)
        from .. import config
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
    
    def extract_features_optimized(
        self, 
        sensor_df: pd.DataFrame,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> pd.DataFrame:
        """
        Extract the 10 features needed by the CA2 final model
        
        Required features:
        1. trip_duration_sec
        2. speed_mean
        3. turn_sharpness_index
        4. pct_time_cruising
        5. gyro_accel_instability (INTERACTION)
        6. speed_max
        7. pct_time_high_accel
        8. jerk_linear_mean
        9. accel_risk_score (INTERACTION)
        10. longest_smooth_segment_sec
        """
        print(f"\n⚙️  Extracting 10 optimized features (CA2 final model)...")
        if progress_callback:
            progress_callback(0.6, "Extracting features...")
        
        sensor_df_chunk = sensor_df.copy()
        sensor_df_chunk = sensor_df_chunk.sort_values(["bookingID", "second"])
        
        # STEP 1: DERIVED FEATURES
        print(f"   Step 1/8: Computing magnitudes...")
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
        
        # Time delta
        sensor_df_chunk["delta_t"] = sensor_df_chunk.groupby("bookingID")["second"].diff().fillna(1)
        
        # STEP 2: FLAGS
        print(f"   Step 2/8: Creating behavioral flags...")
        sensor_df_chunk["is_cruising"] = sensor_df_chunk["speed"].between(self.CRUISE_MIN, self.CRUISE_MAX)
        sensor_df_chunk["is_high_accel"] = sensor_df_chunk["accel_mag"] > self.ACCEL_MAG_HIGH_THRESH
        
        # STEP 3: LINEAR JERK
        print(f"   Step 3/8: Computing linear jerk...")
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
        
        # STEP 4: SMOOTH DRIVING SEGMENTS
        print(f"   Step 4/8: Identifying smooth segments...")
        sensor_df_chunk["is_smooth"] = (
            (sensor_df_chunk["accel_mag"] < self.SMOOTH_ACCEL_THRESH) &
            (sensor_df_chunk["gyro_mag"] < self.SMOOTH_GYRO_THRESH)
        )
        
        grp = sensor_df_chunk.groupby("bookingID")
        prev_smooth = grp["is_smooth"].shift(fill_value=False)
        start_new_block = sensor_df_chunk["is_smooth"] & (~prev_smooth)
        sensor_df_chunk["smooth_block"] = start_new_block.astype(int).groupby(sensor_df_chunk["bookingID"]).cumsum()
        
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
        
        # STEP 5: HARD ACCELERATION EVENTS
        print(f"   Step 5/8: Detecting hard acceleration events...")
        sensor_df_chunk["delta_speed"] = sensor_df_chunk.groupby("bookingID")["speed"].diff().fillna(0)
        valid_dt = (sensor_df_chunk["delta_t"] > 0) & (sensor_df_chunk["delta_t"] <= 10)
        sensor_df_chunk["speed_rate"] = 0.0
        sensor_df_chunk.loc[valid_dt, "speed_rate"] = (
            sensor_df_chunk.loc[valid_dt, "delta_speed"] / sensor_df_chunk.loc[valid_dt, "delta_t"]
        )
        sensor_df_chunk["is_hard_accel"] = sensor_df_chunk["speed_rate"] > self.HARD_ACCEL_RATE
        
        # STEP 6: AGGREGATION
        print(f"   Step 6/8: Aggregating trip-level statistics...")
        agg = sensor_df_chunk.groupby("bookingID").agg({
            "second": ["min", "max"],
            "speed": ["max", "mean"],
            "accel_mag": ["max", "std"],
            "gyro_mag": ["max", "std"],
            "jerk_linear": ["mean"],
            "is_cruising": ["mean"],
            "is_high_accel": ["mean"],
            "is_hard_accel": ["sum"]
        })
        
        agg.columns = ["_".join(col) for col in agg.columns]
        trip_features = agg.reset_index()
        
        # STEP 7: DERIVE FINAL BASE FEATURES
        print(f"   Step 7/8: Computing derived features...")
        trip_features["trip_duration_sec"] = trip_features["second_max"] - trip_features["second_min"]
        trip_features["speed_mean"] = trip_features["speed_mean"]
        trip_features["speed_max"] = trip_features["speed_max"]
        
        eps = 1e-3
        trip_features["turn_sharpness_index"] = (
            trip_features["gyro_mag_max"] / (trip_features["speed_mean"] + eps)
        )
        
        trip_features["pct_time_cruising"] = trip_features["is_cruising_mean"]
        trip_features["pct_time_high_accel"] = trip_features["is_high_accel_mean"]
        trip_features["jerk_linear_mean"] = trip_features["jerk_linear_mean"]
        
        trip_features = trip_features.merge(longest_smooth, on="bookingID", how="left")
        trip_features["longest_smooth_segment_sec"] = trip_features["longest_smooth_segment_sec"].fillna(0.0)
        
        # STEP 8: INTERACTION FEATURES
        print(f"   Step 8/8: Creating interaction features...")
        trip_features["gyro_accel_instability"] = (
            trip_features["gyro_mag_std"] * trip_features["accel_mag_std"]
        )
        trip_features["accel_risk_score"] = (
            trip_features["accel_mag_max"] * trip_features["is_hard_accel_sum"]
        )
        
        # SELECT FINAL FEATURES
        from .. import config
        final_cols = ["bookingID"] + config.MODEL_FEATURES
        available_cols = [c for c in final_cols if c in trip_features.columns]
        
        result = trip_features[available_cols].copy()
        
        print(f"✅ Features extracted: {len(result):,} trips, {len(available_cols)-1} features")
        
        if progress_callback:
            progress_callback(0.8, f"Features ready: {len(result):,} trips")
        
        return result
    
    def process_batch_data(
        self, 
        raw_df: pd.DataFrame,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> pd.DataFrame:
        """
        Process batch sensor data and extract features
        
        Args:
            raw_df: Raw sensor DataFrame
            progress_callback: Optional callback for progress updates
            
        Returns:
            DataFrame with engineered features
        """
        print(f"\n📄 Processing {len(raw_df):,} sensor readings...")
        
        # Clean data
        if progress_callback:
            progress_callback(0.5, "Cleaning data...")
        
        clean_df = self.clean_sensor_data(raw_df)
        print(f"   Cleaned: {len(clean_df):,} rows retained")
        
        # Extract features
        features_df = self.extract_features_optimized(clean_df, progress_callback)
        
        self.last_features_dataframe = features_df
        
        print(f"✅ Processing complete: {len(features_df):,} trips ready for prediction\n")
        
        return features_df
    
    def get_booking_ids(self, df: pd.DataFrame) -> list:
        """Get unique booking IDs from DataFrame"""
        return df['bookingID'].unique().tolist()