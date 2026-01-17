# etl_prefect.py

from sqlalchemy import create_engine, text
import pymysql
import pandas as pd
import numpy as np

from prefect import flow, task

import warnings
warnings.filterwarnings("ignore")
# ==============================
# DB CONNECTION
# ==============================

USERNAME = "root"
PASSWORD = "48835674#Ryleesucks"
HOST     = "localhost"
DATABASE = "gobest_cab"

connection_string = f"mysql+pymysql://{USERNAME}:{PASSWORD}@{HOST}/{DATABASE}"

engine = create_engine(
    connection_string,
    pool_pre_ping=True,
    connect_args={"local_infile": 1}
)

# Optional quick test (runs once when the script is imported)
with engine.connect() as conn:
    result = conn.execute(text("SELECT DATABASE();"))
    print("Connected to MySQL database:", result.scalar())

# ==============================
# CONSTANTS FOR SENSOR LIMITS
# ==============================

ACC_COLS = ["acceleration_x", "acceleration_y", "acceleration_z"]
GYRO_COLS = ["gyro_x", "gyro_y", "gyro_z"]
LIMIT_COLS = ACC_COLS + GYRO_COLS

# Precomputed 99.9% abs quantile limits from your notebook
P999_LIMITS = {
    "acceleration_x": 9.886328100000092,
    "acceleration_y": 29.507823600011807,
    "acceleration_z": 13.291296400000132,
    "gyro_x": 1.5066820000000671,
    "gyro_y": 2.5858964800000144,
    "gyro_z": 1.8445167700000504,
}

# Speed in m/s (22.2 ≈ 80 km/h)
SPEED_OVER_80_THRESH = 22.2
CRUISE_MIN = 8.33    # ~30 km/h
CRUISE_MAX = 16.67    # ~60 km/h

# High accel / gyro magnitude
ACCEL_MAG_HIGH_THRESH = 12.0
GYRO_MAG_HIGH_THRESH  = 1.0

# For jerk-based hard braking / accel, using speed_rate (Δspeed / Δt)
HARD_BRAKE_RATE = -5.0   # m/s², tune as you like
HARD_ACCEL_RATE =  5.0   # m/s²

# Zig-zag steering threshold on gyro_y
ZIGZAG_GYRO_Y_THRESH = 0.5

# Smooth driving thresholds
SMOOTH_ACCEL_THRESH = 2.0
SMOOTH_GYRO_THRESH  = 0.5

# ==============================
# RAW HELPER FUNCTIONS
# ==============================

def load_drivers():
    return pd.read_sql("SELECT * FROM Driver", engine)

def load_safety():
    return pd.read_sql("SELECT * FROM SafetyLabel", engine)

def load_sensor_chunk(limit=500_000, offset=0):
    query = f"""
        SELECT *
        FROM SensorData
        ORDER BY bookingID, second
        LIMIT {limit} OFFSET {offset}
    """
    return pd.read_sql(query, engine)

def clean_drivers(drivers: pd.DataFrame) -> pd.DataFrame:
    # types
    drivers["date_of_birth"] = pd.to_datetime(drivers["date_of_birth"], errors="coerce")
    drivers["rating"] = drivers["rating"].astype(float)
    drivers["no_of_years_driving_exp"] = drivers["no_of_years_driving_exp"].astype("Int64")

    # compute age
    today = pd.Timestamp("today").normalize()
    drivers["age"] = (today - drivers["date_of_birth"]).dt.days // 365

    # bins for later analysis
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
    sensor_df_clean = chunk.copy()

    # 1) Drop trips with insane 'second' values
    bad_trip_ids = sensor_df_clean.loc[
        sensor_df_clean["second"] > 21600, "bookingID"
    ].unique()

    if len(bad_trip_ids) > 0:
        print(f"Dropping {len(bad_trip_ids)} trips due to second > 21600")
        sensor_df_clean = sensor_df_clean[
            ~sensor_df_clean["bookingID"].isin(bad_trip_ids)
        ].copy()

    # 2) Drop rows with bad 'accuracy' (keep NaN or <= 50)
    sensor_df_clean = sensor_df_clean[
        (sensor_df_clean["accuracy"].isna()) |
        (sensor_df_clean["accuracy"] <= 50)
    ]

    # 3) Handle speed: negative speed -> NaN
    sensor_df_clean["speed"] = sensor_df_clean["speed"].where(
        sensor_df_clean["speed"] >= 0,
        np.nan,
    )

    # 4) Mark extreme acc/gyro as NaN using PRECOMPUTED 99.9% abs limits
    for col in LIMIT_COLS:
        lim = P999_LIMITS[col]
        mask = sensor_df_clean[col].abs() > lim
        sensor_df_clean[col] = sensor_df_clean[col].where(~mask, np.nan)

    return sensor_df_clean

def aggregate_sensor_chunk(sensor_df_chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the chunk, engineer essential features per row,
    then aggregate to one row per bookingID with the selected
    trip-level features.
    """
    # 1) Clean raw sensor rows
    sensor_df_chunk = clean_sensor_chunk(sensor_df_chunk)

    # Ensure rows are ordered per trip
    sensor_df_chunk = sensor_df_chunk.sort_values(["bookingID", "second"])

    # -----------------------------
    # 2) Per-row derived features
    # -----------------------------

    # Overall acceleration magnitude
    sensor_df_chunk["accel_mag"] = np.sqrt(
        sensor_df_chunk["acceleration_x"].fillna(0) ** 2 +
        sensor_df_chunk["acceleration_y"].fillna(0) ** 2 +
        sensor_df_chunk["acceleration_z"].fillna(0) ** 2
    )

    # Overall gyro magnitude
    sensor_df_chunk["gyro_mag"] = np.sqrt(
        sensor_df_chunk["gyro_x"].fillna(0) ** 2 +
        sensor_df_chunk["gyro_y"].fillna(0) ** 2 +
        sensor_df_chunk["gyro_z"].fillna(0) ** 2
    )

    # Δtime between consecutive readings in the same trip
    sensor_df_chunk["delta_t"] = (
        sensor_df_chunk
        .groupby("bookingID")["second"]
        .diff()
    )

    # Δspeed between consecutive readings in the same trip
    sensor_df_chunk["delta_speed"] = (
        sensor_df_chunk
        .groupby("bookingID")["speed"]
        .diff()
    )

    # First row in each trip has no previous row → set Δ=0
    sensor_df_chunk["delta_t"] = sensor_df_chunk["delta_t"].fillna(0)
    sensor_df_chunk["delta_speed"] = sensor_df_chunk["delta_speed"].fillna(0)

    # Valid rows where we trust Δ/Δt
    valid_dt = (sensor_df_chunk["delta_t"] > 0) & (sensor_df_chunk["delta_t"] <= 10)

    # Default: assume no event (rate = 0)
    sensor_df_chunk["speed_rate"] = 0.0
    sensor_df_chunk.loc[valid_dt, "speed_rate"] = (
        sensor_df_chunk.loc[valid_dt, "delta_speed"] /
        sensor_df_chunk.loc[valid_dt, "delta_t"]
    )

    # Basic behaviour flags
    sensor_df_chunk["is_speed_over_80"] = sensor_df_chunk["speed"] > SPEED_OVER_80_THRESH
    sensor_df_chunk["is_cruising"]      = sensor_df_chunk["speed"].between(CRUISE_MIN, CRUISE_MAX)

    # High accel / gyro flags (for proportions)
    sensor_df_chunk["is_high_accel"] = sensor_df_chunk["accel_mag"] > ACCEL_MAG_HIGH_THRESH
    sensor_df_chunk["is_hard_turn"]  = sensor_df_chunk["gyro_mag"]  > GYRO_MAG_HIGH_THRESH

    # Hard brake / accel based on rate of change of speed
    sensor_df_chunk["is_hard_brake"] = sensor_df_chunk["speed_rate"] < HARD_BRAKE_RATE
    sensor_df_chunk["is_hard_accel"] = sensor_df_chunk["speed_rate"] > HARD_ACCEL_RATE

    # ----- Linear jerk (rate of change of acceleration) -----
    sensor_df_chunk["d_accel_x"] = (
        sensor_df_chunk
        .groupby("bookingID")["acceleration_x"]
        .diff()
        .fillna(0)
    )
    sensor_df_chunk["d_accel_y"] = (
        sensor_df_chunk
        .groupby("bookingID")["acceleration_y"]
        .diff()
        .fillna(0)
    )
    sensor_df_chunk["d_accel_z"] = (
        sensor_df_chunk
        .groupby("bookingID")["acceleration_z"]
        .diff()
        .fillna(0)
    )

    dt_nonzero = sensor_df_chunk["delta_t"].replace(0, np.nan)

    jerk_x_rate = (sensor_df_chunk["d_accel_x"] / dt_nonzero).fillna(0)
    jerk_y_rate = (sensor_df_chunk["d_accel_y"] / dt_nonzero).fillna(0)
    jerk_z_rate = (sensor_df_chunk["d_accel_z"] / dt_nonzero).fillna(0)

    sensor_df_chunk["jerk_linear"] = np.sqrt(
        jerk_x_rate ** 2 + jerk_y_rate ** 2 + jerk_z_rate ** 2
    )

    # ----- Gyro jerk (rate of change of angular velocity) -----
    sensor_df_chunk["d_gyro_x"] = (
        sensor_df_chunk
        .groupby("bookingID")["gyro_x"]
        .diff()
        .fillna(0)
    )
    sensor_df_chunk["d_gyro_y"] = (
        sensor_df_chunk
        .groupby("bookingID")["gyro_y"]
        .diff()
        .fillna(0)
    )
    sensor_df_chunk["d_gyro_z"] = (
        sensor_df_chunk
        .groupby("bookingID")["gyro_z"]
        .diff()
        .fillna(0)
    )

    gyro_jerk_x = (sensor_df_chunk["d_gyro_x"] / dt_nonzero).fillna(0)
    gyro_jerk_y = (sensor_df_chunk["d_gyro_y"] / dt_nonzero).fillna(0)
    gyro_jerk_z = (sensor_df_chunk["d_gyro_z"] / dt_nonzero).fillna(0)

    sensor_df_chunk["gyro_jerk_mag"] = np.sqrt(
        gyro_jerk_x ** 2 + gyro_jerk_y ** 2 + gyro_jerk_z ** 2
    )

    # ----- Zig-zag detection on gyro_y -----
    sensor_df_chunk["gyro_y_sign"] = 0
    sensor_df_chunk.loc[sensor_df_chunk["gyro_y"] >  ZIGZAG_GYRO_Y_THRESH, "gyro_y_sign"] =  1
    sensor_df_chunk.loc[sensor_df_chunk["gyro_y"] < -ZIGZAG_GYRO_Y_THRESH, "gyro_y_sign"] = -1

    sensor_df_chunk["prev_gyro_y_sign"] = (
        sensor_df_chunk
        .groupby("bookingID")["gyro_y_sign"]
        .shift()
        .fillna(0)
        .astype(int)
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

    # Label smooth segments (for longest_smooth_segment_sec, using time)
    grp = sensor_df_chunk.groupby("bookingID")
    prev_smooth = grp["is_smooth"].shift(fill_value=False)

    start_new_block = sensor_df_chunk["is_smooth"] & (~prev_smooth)
    sensor_df_chunk["smooth_block"] = (
        start_new_block.astype(int)
        .groupby(sensor_df_chunk["bookingID"])
        .cumsum()
    )

    # For each smooth block, sum delta_t to get duration in seconds
    smooth_agg = (
        sensor_df_chunk
        .loc[sensor_df_chunk["smooth_block"] > 0]
        .groupby(["bookingID", "smooth_block"])["delta_t"]
        .sum()
        .reset_index(name="smooth_dur_sec")
    )

    if not smooth_agg.empty:
        longest_smooth = (
            smooth_agg
            .groupby("bookingID", as_index=False)["smooth_dur_sec"]
            .max()
            .rename(columns={"smooth_dur_sec": "longest_smooth_segment_sec"})
        )
    else:
        longest_smooth = pd.DataFrame(columns=["bookingID", "longest_smooth_segment_sec"])

    # -----------------------------
    # 3) Aggregate per bookingID
    # -----------------------------
    agg = sensor_df_chunk.groupby("bookingID").agg({
        # trip length
        "second": ["min", "max", "count"],   # -> duration & n_points

        # speed features
        "speed": ["max", "mean", "std"],

        # accel / gyro magnitudes
        "accel_mag": ["max", "mean", "std"],
        "gyro_mag":  ["max", "mean", "std"],

        # jerk features
        "jerk_linear":   ["mean"],
        "gyro_jerk_mag": ["mean"],

        # event / proportion flags
        "is_speed_over_80": ["mean"],       # -> pct_time_speed_over_80
        "is_cruising":      ["mean"],       # -> pct_time_cruising
        "is_high_accel":    ["mean"],       # -> pct_time_high_accel
        "is_hard_turn":     ["sum", "mean"],# -> n_hard_turns, pct_time_high_gyro
        "is_hard_brake":    ["sum"],        # -> n_hard_brakes
        "is_hard_accel":    ["sum"],        # -> n_hard_accels
        "is_zigzag":        ["sum"],        # -> n_zigzag_events
        "is_smooth":        ["mean"],       # -> pct_time_smooth
    })

    # Flatten multi-index columns: ('speed','max') -> 'speed_max'
    agg.columns = ["_".join(col) for col in agg.columns]
    trip_features_chunk = agg.reset_index()

    # Trip duration (sec.max - sec.min to handle irregular sampling)
    trip_features_chunk["trip_duration_sec"] = (
        trip_features_chunk["second_max"] - trip_features_chunk["second_min"]
    )

    # Number of sensor points
    trip_features_chunk["n_points"] = trip_features_chunk["second_count"].astype(int)

    # Merge longest smooth segment (sec)
    trip_features_chunk = trip_features_chunk.merge(
        longest_smooth,
        on="bookingID",
        how="left"
    )
    trip_features_chunk["longest_smooth_segment_sec"] = (
        trip_features_chunk["longest_smooth_segment_sec"]
        .fillna(0.0)
        .astype(float)
    )

    # -----------------------------
    # 4) Rename columns to requested names
    # -----------------------------
    trip_features_chunk = trip_features_chunk.rename(columns={
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
        "is_cruising_mean":      "pct_time_cruising",
        "is_high_accel_mean":    "pct_time_high_accel",
        "is_hard_turn_sum":      "n_hard_turns",
        "is_hard_turn_mean":     "pct_time_high_gyro",
        "is_hard_brake_sum":     "n_hard_brakes",
        "is_hard_accel_sum":     "n_hard_accels",
        "is_zigzag_sum":         "n_zigzag_events",
        "is_smooth_mean":        "pct_time_smooth",
    })

    # Turn sharpness index: strong turning vs average speed
    eps = 1e-3
    trip_features_chunk["turn_sharpness_index"] = (
        trip_features_chunk["gyro_mag_max"] /
        (trip_features_chunk["speed_mean"] + eps)
    )

    # -----------------------------
    # 5) Drop helper columns and keep ONLY requested ones
    # -----------------------------
    trip_features_chunk = trip_features_chunk.drop(
        columns=[c for c in trip_features_chunk.columns if c.startswith("second_")],
        errors="ignore",
    )

    final_cols = [
        "bookingID",
        "trip_duration_sec",
        "n_points",

        "speed_max",
        "speed_mean",
        "speed_std",
        "pct_time_speed_over_80",
        "pct_time_cruising",

        "accel_mag_max",
        "accel_mag_mean",
        "accel_mag_std",
        "jerk_linear_mean",
        "n_hard_accels",
        "n_hard_brakes",
        "pct_time_high_accel",

        "gyro_mag_max",
        "gyro_mag_mean",
        "gyro_mag_std",
        "gyro_jerk_mag_mean",
        "n_hard_turns",
        "pct_time_high_gyro",
        "n_zigzag_events",
        "turn_sharpness_index",

        "longest_smooth_segment_sec",
        "pct_time_smooth",
    ]

    trip_features_chunk = trip_features_chunk[final_cols]

    return trip_features_chunk


def build_trips_features_from_sensor(chunk_size=500_000) -> pd.DataFrame:
    """
    Read SensorData in chunks, clean each chunk,
    aggregate to trip-level, then concatenate.
    """
    all_aggs = []
    offset = 0

    while True:
        chunk = load_sensor_chunk(limit=chunk_size, offset=offset)
        if chunk.empty:
            print("No more rows. Stopping.")
            break

        print(f"Processing chunk with offset {offset}, rows: {len(chunk)}")
        agg_chunk = aggregate_sensor_chunk(chunk)
        all_aggs.append(agg_chunk)

        offset += chunk_size

    trips_features = pd.concat(all_aggs, ignore_index=True)

    # In case some bookingIDs appeared in multiple chunks
    trips_features = trips_features.drop_duplicates(subset=["bookingID"])

    print("Final TripFeatures shape:", trips_features.shape)
    return trips_features

# ==============================
# NON-PREFECT ONE-SHOT RUN (optional)
# ==============================

def run_etl_once():
    # Extract
    drivers_raw = load_drivers()
    safety_raw  = load_safety()

    # Transform
    drivers = clean_drivers(drivers_raw)
    safety  = clean_safety(safety_raw)

    trips_features = build_trips_features_from_sensor(chunk_size=500_000)

    master_df = (
        safety
        .merge(trips_features, on="bookingID", how="left")
        .merge(drivers, left_on="driver_id", right_on="id", how="left")
    )

    # DriverSummary
        # One row per driver
    driver_summary = (
        master_df
        .groupby("driver_id", as_index=False)
        .agg(
            name=("name", "first"),
            rating=("rating", "first"),
            no_of_years_driving_exp=("no_of_years_driving_exp", "first"),
            rating_bin=("rating_bin", "first"),
            exp_bin=("exp_bin", "first"),
            car_brand=("car_brand", "first"),
            car_model_year=("car_model_year", "first"),
            gender=("gender", "first"),

            total_trips=("bookingID", "nunique"),
            dangerous_trips=("is_dangerous", "sum"),
            dangerous_rate=("is_dangerous", "mean"),

            avg_speed_max=("speed_max", "mean"),
            avg_accel_mag_max=("accel_mag_max", "mean"),
            avg_gyro_mag_max=("gyro_mag_max", "mean"),
        )
    )


    # CarModelSummary
    car_model_summary = (
        master_df
        .groupby(
            ["car_brand", "car_model_year"],
            dropna=False,
        )
        .agg(
            total_trips=("bookingID", "count"),
            dangerous_trips=("is_dangerous", "sum"),
            dangerous_rate=("is_dangerous", "mean"),
            avg_speed_max=("speed_max", "mean"),
            avg_accel_mag_max=("accel_mag_max", "mean"),
            avg_gyro_mag_max=("gyro_mag_max", "mean"),
        )
        .reset_index()
    )

    # Load
    trips_features.to_sql("TripFeatures", engine, if_exists="replace", index=False)
    master_df.to_sql("MasterTable", engine, if_exists="replace", index=False)
    driver_summary.to_sql("DriverSummary", engine, if_exists="replace", index=False)
    car_model_summary.to_sql("CarModelSummary", engine, if_exists="replace", index=False)

    master_df.to_csv("bi_dataset.csv", index=False)
    driver_summary.to_csv("driver_summary.csv", index=False)
    car_model_summary.to_csv("car_model_summary.csv", index=False)

    print("ETL completed. MasterTable rows:", len(master_df))
    return master_df, trips_features, driver_summary, car_model_summary

# ==============================
# PREFECT TASKS & FLOW
# ==============================

@task
def t_extract_drivers():
    return load_drivers()

@task
def t_extract_safety():
    return load_safety()

@task
def t_build_trip_features():
    return build_trips_features_from_sensor(chunk_size=500_000)

@task
def t_transform_and_merge(drivers_raw, safety_raw, trips_features):
    drivers = clean_drivers(drivers_raw)
    safety  = clean_safety(safety_raw)

    master_df = (
        safety
        .merge(trips_features, on="bookingID", how="left")
        .merge(drivers, left_on="driver_id", right_on="id", how="left")
    )
    return master_df

@task
@task
def t_load_outputs(master_df, trips_features):
    # ---------- DriverSummary (one row per driver) ----------
    driver_summary = (
        master_df
        .groupby("driver_id", as_index=False)
        .agg(
            name=("name", "first"),
            rating=("rating", "first"),
            no_of_years_driving_exp=("no_of_years_driving_exp", "first"),
            rating_bin=("rating_bin", "first"),
            exp_bin=("exp_bin", "first"),
            car_brand=("car_brand", "first"),
            car_model_year=("car_model_year", "first"),
            gender=("gender", "first"),

            total_trips=("bookingID", "nunique"),
            dangerous_trips=("is_dangerous", "sum"),
            dangerous_rate=("is_dangerous", "mean"),

            avg_speed_max=("speed_max", "mean"),
            avg_accel_mag_max=("accel_mag_max", "mean"),
            avg_gyro_mag_max=("gyro_mag_max", "mean"),
        )
    )

    # ---------- CarModelSummary (this part is fine) ----------
    car_model_summary = (
        master_df
        .groupby(["car_brand", "car_model_year"], dropna=False)
        .agg(
            total_trips=("bookingID", "count"),
            dangerous_trips=("is_dangerous", "sum"),
            dangerous_rate=("is_dangerous", "mean"),
            avg_speed_max=("speed_max", "mean"),
            avg_accel_mag_max=("accel_mag_max", "mean"),
            avg_gyro_mag_max=("gyro_mag_max", "mean"),
        )
        .reset_index()
    )

    # ... rest (to_sql, to_csv) stays the same


    # Save all tables
    trips_features.to_sql("TripFeatures", engine, if_exists="replace", index=False)
    master_df.to_sql("MasterTable", engine, if_exists="replace", index=False)
    driver_summary.to_sql("DriverSummary", engine, if_exists="replace", index=False)
    car_model_summary.to_sql("CarModelSummary", engine, if_exists="replace", index=False)

    # CSVs for BI
    master_df.to_csv("bi_dataset.csv", index=False)
    driver_summary.to_csv("driver_summary.csv", index=False)
    car_model_summary.to_csv("car_model_summary.csv", index=False)

    print("Saved TripFeatures, MasterTable, DriverSummary, CarModelSummary and CSVs")

@flow(name="gobest-etl-flow")
def etl_flow():
    # Extract
    drivers_raw = t_extract_drivers()
    safety_raw  = t_extract_safety()
    trips_features = t_build_trip_features()

    # Transform
    master_df = t_transform_and_merge(drivers_raw, safety_raw, trips_features)

    # Load (and build summaries inside)
    t_load_outputs(master_df, trips_features)

if __name__ == "__main__":
    etl_flow()
