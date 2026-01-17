# etl_prefect.py

from sqlalchemy import create_engine, text
import pymysql
import pandas as pd
import numpy as np
from prefect import flow, task
import warnings

warnings.filterwarnings("ignore")

# ==============================
# 1. DB CONNECTION
# ==============================
# UPDATED CREDENTIALS from your latest file
USERNAME = "root"
PASSWORD = "Maleniastets456" 
HOST     = "localhost"
DATABASE = "gobest_cab2"

connection_string = f"mysql+pymysql://{USERNAME}:{PASSWORD}@{HOST}/{DATABASE}"

engine = create_engine(
    connection_string,
    pool_pre_ping=True,
    connect_args={"local_infile": 1}
)

# ==============================
# 2. CONSTANTS
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

# Speed in m/s (22.2 m/s approx 80 km/h)
SPEED_OVER_80_THRESH = 22.2
CRUISE_MIN = 8.33
CRUISE_MAX = 16.67
ACCEL_MAG_HIGH_THRESH = 12.0
GYRO_MAG_HIGH_THRESH  = 1.0
HARD_BRAKE_RATE = -5.0
HARD_ACCEL_RATE =  5.0
ZIGZAG_GYRO_Y_THRESH = 0.5
SMOOTH_ACCEL_THRESH = 2.0
SMOOTH_GYRO_THRESH  = 0.5

# ==============================
# 3. HELPER FUNCTIONS
# ==============================

def load_drivers():
    return pd.read_sql("SELECT * FROM Driver", engine)

def load_safety():
    return pd.read_sql("SELECT * FROM SafetyLabel", engine)

def get_new_booking_ids():
    """
    Compare Source (SensorData) vs Destination (TripFeatures)
    to find which trips we haven't processed yet.
    """
    print("Checking for new data...")
    try:
        # Get what we already have
        existing_ids = pd.read_sql("SELECT DISTINCT bookingID FROM TripFeatures", engine)
        existing_set = set(existing_ids["bookingID"])
    except:
        # If table doesn't exist, we have nothing
        print("TripFeatures table not found. Processing ALL data.")
        existing_set = set()

    # Get all unique bookingIDs from the SOURCE
    # Note: If this is too slow, we can optimize with a LEFT JOIN query in SQL directly
    print("Fetching source IDs...")
    all_ids = pd.read_sql("SELECT DISTINCT bookingID FROM SensorData", engine)
    all_set = set(all_ids["bookingID"])
    
    new_ids = list(all_set - existing_set)
    print(f"Found {len(new_ids)} new trips to process.")
    return new_ids

def clean_drivers(drivers: pd.DataFrame) -> pd.DataFrame:
    drivers["date_of_birth"] = pd.to_datetime(drivers["date_of_birth"], errors="coerce")
    drivers["rating"] = drivers["rating"].astype(float)
    drivers["no_of_years_driving_exp"] = drivers["no_of_years_driving_exp"].astype("Int64")
    
    today = pd.Timestamp("today").normalize()
    drivers["age"] = (today - drivers["date_of_birth"]).dt.days // 365
    
    drivers["rating_bin"] = pd.cut(
        drivers["rating"],
        bins=[1.0, 2.5, 3.5, 4.5, 5.1],
        labels=["Very Low", "Low", "Medium", "High"],
    )
    drivers["exp_bin"] = pd.cut(
        drivers["no_of_years_driving_exp"],
        bins=[-1, 3, 5, 10, 20, 60],
        labels=["<3", "3-5", "5-10", "10-20", "20+"],
    )
    return drivers

def clean_safety(safety: pd.DataFrame) -> pd.DataFrame:
    safety["label"] = safety["label"].astype(int)
    safety["is_dangerous"] = safety["label"] == 1
    return safety

def clean_sensor_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Cleans a raw batch of sensor data"""
    sensor_df_clean = chunk.copy()
    
    # Drop bad rows
    sensor_df_clean = sensor_df_clean[
        (sensor_df_clean["second"] <= 21600) & 
        ((sensor_df_clean["accuracy"].isna()) | (sensor_df_clean["accuracy"] <= 50))
    ]
    
    # Handle negative speed
    sensor_df_clean["speed"] = sensor_df_clean["speed"].where(sensor_df_clean["speed"] >= 0, np.nan)
    
    # Handle Outliers
    for col in LIMIT_COLS:
        lim = P999_LIMITS[col]
        mask = sensor_df_clean[col].abs() > lim
        sensor_df_clean[col] = sensor_df_clean[col].where(~mask, np.nan)
        
    return sensor_df_clean

def aggregate_sensor_chunk(sensor_df_chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates raw sensor rows into one row per bookingID.
    Includes FULL feature engineering (Jerk, ZigZag, Smoothness).
    """
    # 1. Clean
    sensor_df_chunk = clean_sensor_chunk(sensor_df_chunk)
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
    sensor_df_chunk["is_speed_over_80"] = sensor_df_chunk["speed"] > SPEED_OVER_80_THRESH
    sensor_df_chunk["is_cruising"] = sensor_df_chunk["speed"].between(CRUISE_MIN, CRUISE_MAX)
    sensor_df_chunk["is_high_accel"] = sensor_df_chunk["accel_mag"] > ACCEL_MAG_HIGH_THRESH
    sensor_df_chunk["is_hard_turn"] = sensor_df_chunk["gyro_mag"] > GYRO_MAG_HIGH_THRESH
    sensor_df_chunk["is_hard_brake"] = sensor_df_chunk["speed_rate"] < HARD_BRAKE_RATE
    sensor_df_chunk["is_hard_accel"] = sensor_df_chunk["speed_rate"] > HARD_ACCEL_RATE

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
    sensor_df_chunk.loc[sensor_df_chunk["gyro_y"] >  ZIGZAG_GYRO_Y_THRESH, "gyro_y_sign"] =  1
    sensor_df_chunk.loc[sensor_df_chunk["gyro_y"] < -ZIGZAG_GYRO_Y_THRESH, "gyro_y_sign"] = -1

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
        (sensor_df_chunk["accel_mag"] < SMOOTH_ACCEL_THRESH) &
        (sensor_df_chunk["gyro_mag"]  < SMOOTH_GYRO_THRESH)
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

def build_trips_features_from_list(target_ids, batch_size=200) -> pd.DataFrame:
    """
    Process specific bookingIDs in batches.
    """
    all_aggs = []
    total = len(target_ids)
    
    if total == 0:
        return pd.DataFrame()

    print(f"Starting processing of {total} trips...")
    
    for i in range(0, total, batch_size):
        batch_ids = target_ids[i : i + batch_size]
        # Create comma-separated string for SQL IN clause
        ids_str = ",".join(map(str, batch_ids))
        
        query = f"SELECT * FROM SensorData WHERE bookingID IN ({ids_str})"
        chunk = pd.read_sql(query, engine)
        
        if not chunk.empty:
            agg = aggregate_sensor_chunk(chunk)
            all_aggs.append(agg)
            
        if i % 1000 == 0:
            print(f"Processed {i}/{total} trips...")

    if not all_aggs:
        return pd.DataFrame()

    return pd.concat(all_aggs, ignore_index=True)

# ==============================
# 4. PREFECT TASKS
# ==============================

@task
def t_get_new_ids():
    return get_new_booking_ids()

@task
def t_extract_drivers():
    return load_drivers()

@task
def t_extract_safety():
    return load_safety()

@task
def t_process_new_trips(new_ids):
    # Process in chunks of 500 trips at a time
    return build_trips_features_from_list(new_ids, batch_size=500)

@task
def t_update_trip_features(new_features_df):
    if not new_features_df.empty:
        print(f"Appending {len(new_features_df)} trips to TripFeatures Table...")
        # CRITICAL: Append mode ensures we don't overwrite previous work
        new_features_df.to_sql("TripFeatures", engine, if_exists="append", index=False)
    else:
        print("No new trips to append.")

@task
def t_refresh_master_views():
    print("Regenerating MasterTable (Joins)...")
    
    # Load FULL tables
    try:
        trips = pd.read_sql("SELECT * FROM TripFeatures", engine)
    except:
        print("TripFeatures table empty. Skipping MasterTable generation.")
        return

    drivers = clean_drivers(load_drivers())
    safety = clean_safety(load_safety())
    
    # Merge
    master = safety.merge(trips, on="bookingID", how="left") \
                   .merge(drivers, left_on="driver_id", right_on="id", how="left")
    
    # Save MasterTable
    master.to_sql("MasterTable", engine, if_exists="replace", index=False)
    master.to_csv("bi_dataset.csv", index=False)
    
    # (Optional) Re-generate summaries if needed, but MasterTable is key for CA2
    print("MasterTable refreshed and bi_dataset.csv saved.")

# ==============================
# 5. PREFECT FLOW
# ==============================

@flow(name="gobest-incremental-etl")
def etl_flow():
    # 1. Check for work
    new_ids = t_get_new_ids()
    
    # 2. Process only if needed
    if len(new_ids) > 0:
        new_features = t_process_new_trips(new_ids)
        t_update_trip_features(new_features)
    else:
        print("TripFeatures is already up to date. Skipping sensor processing.")
    
    # 3. Always refresh Master View (in case drivers/labels changed)
    t_refresh_master_views()

if __name__ == "__main__":
    etl_flow()