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
# Updated Model Files (CA2 Final Model)
MODEL_PATH = MODEL_DIR / "final_logistic_regression_tuned.pkl"
SCALER_PATH = MODEL_DIR / "feature_scaler.pkl"
CONFIG_PATH = MODEL_DIR / "model_metadata.json"

# Model Features (THE NEW 10 FEATURES - Updated from CA2 analysis)
MODEL_FEATURES = [
    'trip_duration_sec',
    'speed_mean',
    'turn_sharpness_index',
    'pct_time_cruising',
    'gyro_accel_instability',  # Interaction feature
    'speed_max',
    'pct_time_high_accel',
    'jerk_linear_mean',
    'accel_risk_score',         # Interaction feature
    'longest_smooth_segment_sec'
]

# Prediction threshold (from CA2 tuning)
PREDICTION_THRESHOLD = 0.4917  # Optimal F1 threshold

# ==================== GUI SETTINGS ====================
WINDOW_TITLE = "GOBEST CAB Safety Prediction System"
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 700
WINDOW_GEOMETRY = f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}"

# Colors - Modern Gradient Theme (Dark Blue to Purple)
COLOR_PRIMARY = "#667eea"          # Vibrant Purple-Blue
COLOR_PRIMARY_DARK = "#5a67d8"     # Darker Purple-Blue
COLOR_PRIMARY_LIGHT = "#7c3aed"    # Light Purple
COLOR_ACCENT = "#f093fb"           # Pink accent
COLOR_SUCCESS = "#10b981"          # Modern Green
COLOR_DANGER = "#ef4444"           # Modern Red
COLOR_WARNING = "#f59e0b"          # Modern Orange
COLOR_INFO = "#3b82f6"             # Modern Blue
COLOR_BACKGROUND = "#0f172a"       # Dark Navy (Modern Dark Mode)
COLOR_CARD = "#1e293b"             # Card background
COLOR_CARD_HOVER = "#334155"       # Card hover state
COLOR_TEXT = "#f1f5f9"             # Light text (for dark bg)
COLOR_TEXT_SECONDARY = "#94a3b8"   # Secondary text
COLOR_TEXT_MUTED = "#64748b"       # Muted text
COLOR_BORDER = "#334155"           # Border color
COLOR_SHADOW = "#000000"           # Shadow
COLOR_TEXT_LIGHT = "#f1f5f9"
COLOR_TEXT_PRIMARY = "#f1f5f9"
# Font Settings - Modern
FONT_FAMILY = "Segoe UI"
FONT_FAMILY_MONO = "Consolas"
FONT_SIZE_TITLE = 32
FONT_SIZE_SUBTITLE = 16
FONT_SIZE_HEADING = 18
FONT_SIZE_BUTTON = 12
FONT_SIZE_LABEL = 11
FONT_SIZE_SMALL = 9

# Animation Settings
ANIMATION_DURATION = 200  # milliseconds
HOVER_TRANSITION = 150    # milliseconds

# ==================== MODE SETTINGS ====================
MODE_BATCH = "batch"
MODE_REALTIME = "realtime"

# ==================== PREDICTION SETTINGS ====================
CHUNK_SIZE = 1_000_000  # Process 1 million rows at a time (Dask chunks)
MAX_FILE_SIZE_MB = 2000  # Increased to 2GB for large datasets

# ==================== DASK SETTINGS ====================
DASK_N_WORKERS = 4  # Number of parallel workers
DASK_THREADS_PER_WORKER = 2
DASK_MEMORY_LIMIT = '4GB'  # Per worker

# ==================== CSV PROCESSING ====================
# Required columns for RAW sensor data
REQUIRED_COLUMNS = [
    'bookingID', 'second', 'speed', 
    'acceleration_x', 'acceleration_y', 'acceleration_z',
    'gyro_x', 'gyro_y', 'gyro_z', 'accuracy'
]

# ==================== SESSION HISTORY ====================
MAX_HISTORY_ENTRIES = 50