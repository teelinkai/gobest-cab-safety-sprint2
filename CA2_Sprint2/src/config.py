"""
Configuration file for GOBEST CAB Safety Prediction System
Contains all constants, paths, and settings
"""

from pathlib import Path

# ==================== PATHS ====================
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
ASSETS_DIR = BASE_DIR / "assets"
SRC_DIR = BASE_DIR / "src"
MODEL_DIR = BASE_DIR / "models"

# ==================== MODEL SETTINGS ====================
# Phase 1C Model Files
MODEL_PATH = MODEL_DIR / "phase1c_logistic_regression.pkl"
SCALER_PATH = MODEL_DIR / "phase1c_scaler.pkl"
FEATURES_PATH = MODEL_DIR / "phase1c_selected_features.txt"
CONFIG_PATH = MODEL_DIR / "phase1c_model_config.json"

# Model Features (ONLY THE 10 SELECTED FEATURES - for speed!)
MODEL_FEATURES = [
    'trip_duration_sec',
    'speed_mean',
    'n_points',
    'gyro_mag_std',
    'speed_max',
    'jerk_linear_mean',
    'pct_time_speed_over_80',
    'gyro_jerk_mag_mean',
    'turn_sharpness_index',
    'pct_time_high_accel'
]

# Prediction threshold (from Phase 1C)
PREDICTION_THRESHOLD = 0.20  # Lower threshold = higher recall (75%)

# ==================== GUI SETTINGS ====================
WINDOW_TITLE = "GOBEST CAB Safety Prediction System"
WINDOW_WIDTH = 850
WINDOW_HEIGHT = 600
WINDOW_GEOMETRY = f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}"

# Colors - Professional Theme
COLOR_PRIMARY = "#3498db"      # Blue
COLOR_PRIMARY_DARK = "#2980b9"
COLOR_SUCCESS = "#2ecc71"      # Green
COLOR_DANGER = "#e74c3c"       # Red
COLOR_WARNING = "#f39c12"      # Orange
COLOR_BACKGROUND = "#ecf0f1"   # Light gray
COLOR_TEXT = "#2c3e50"         # Dark blue-gray
COLOR_TEXT_LIGHT = "#7f8c8d"   # Gray

# Font Settings
FONT_FAMILY = "Segoe UI"
FONT_SIZE_TITLE = 28
FONT_SIZE_SUBTITLE = 14
FONT_SIZE_BUTTON = 11
FONT_SIZE_LABEL = 10

# ==================== MODE SETTINGS ====================
MODE_BATCH = "batch"
MODE_REALTIME = "realtime"

# ==================== PREDICTION SETTINGS ====================
BATCH_SIZE = 500000  # For processing large files in chunks
MAX_FILE_SIZE_MB = 500  # Maximum CSV file size in MB

# ==================== CSV PROCESSING ====================
# Required columns for RAW sensor data
REQUIRED_COLUMNS = [
    'bookingID', 'second', 'speed', 
    'acceleration_x', 'acceleration_y', 'acceleration_z',
    'gyro_x', 'gyro_y', 'gyro_z', 'accuracy'
]

# ==================== SESSION HISTORY ====================
MAX_HISTORY_ENTRIES = 50
