"""
GOBEST CAB CYNOSURE CONFIGURATION
🗂️ RETRO TERMINAL SURVEILLANCE SYSTEM 🗂️
Inspired by Cynosure Corporation Technical Manuals
"""

from pathlib import Path

# ==================== PATHS ====================
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
ASSETS_DIR = BASE_DIR / "assets"
SRC_DIR = BASE_DIR / "src"
MODEL_DIR = BASE_DIR / "models"

# ==================== MODEL SETTINGS ====================
MODEL_PATH = MODEL_DIR / "final_logistic_regression_tuned.pkl"
SCALER_PATH = MODEL_DIR / "feature_scaler.pkl"
CONFIG_PATH = MODEL_DIR / "model_metadata.json"

MODEL_FEATURES = [
    'trip_duration_sec',
    'speed_mean',
    'turn_sharpness_index',
    'pct_time_cruising',
    'gyro_accel_instability',
    'speed_max',
    'pct_time_high_accel',
    'jerk_linear_mean',
    'accel_risk_score',
    'longest_smooth_segment_sec'
]

PREDICTION_THRESHOLD = 0.4917

# ==================== CYNOSURE UI THEME ====================
WINDOW_TITLE = "⚠️ GOBEST CAB :: CYNOSURE SAFETY TERMINAL ⚠️"
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
WINDOW_GEOMETRY = f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}"

# 🎨 CYNOSURE RETRO TERMINAL PALETTE (Beige Paper + Dark Text)
# Background Colors - Aged Paper/Terminal
COLOR_BG_PAPER = "#E8DCC0"            # Aged beige paper
COLOR_BG_DARKER = "#D4C4A8"           # Slightly darker beige
COLOR_BG_CARD = "#F0E8D8"             # Light paper card
COLOR_BG_ELEVATED = "#DED0B8"         # Elevated surface

# Text Colors - Dark on Light
COLOR_TEXT_PRIMARY = "#2B2416"        # Dark brown/black text
COLOR_TEXT_SECONDARY = "#4A3F2F"      # Medium brown
COLOR_TEXT_MUTED = "#6B5D4F"          # Light brown/gray
COLOR_TEXT_LABEL = "#8B7355"          # Label text

# Accent Colors - Technical Diagram Style
COLOR_ACCENT_BLUE = "#4A7BA7"         # Blueprint blue
COLOR_ACCENT_RED = "#C84C3C"          # Warning red
COLOR_ACCENT_GREEN = "#5A8A4F"        # Status green
COLOR_ACCENT_ORANGE = "#D47F3C"       # Alert orange
COLOR_ACCENT_PURPLE = "#7A5A8A"       # Section purple

# Status Colors
COLOR_SUCCESS = "#5A8A4F"             # Green (safe)
COLOR_DANGER = "#C84C3C"              # Red (dangerous)
COLOR_WARNING = "#D47F3C"             # Orange (warning)
COLOR_INFO = "#4A7BA7"                # Blue (info)

# Border & Line Colors
COLOR_BORDER_DARK = "#2B2416"         # Dark lines
COLOR_BORDER_MEDIUM = "#4A3F2F"       # Medium lines
COLOR_BORDER_LIGHT = "#8B7355"        # Light lines

# Section Colors (for cards/areas)
COLOR_SECTION_ALPHA = "#B8A8D8"       # Light purple (like diagram)
COLOR_SECTION_BRAVO = "#D8A8B8"       # Light pink
COLOR_SECTION_VICTOR = "#A8B8D8"      # Light blue
COLOR_SECTION_SIERRA = "#F0D8A8"      # Light orange
COLOR_SECTION_ADMIN = "#C8C8C8"       # Gray

# ==================== TYPOGRAPHY ====================
FONT_FAMILY = "Courier New"           # Monospace terminal font
FONT_FAMILY_DISPLAY = "Arial"         # Display font
FONT_SIZE_MEGA = 36                   # Large titles
FONT_SIZE_TITLE = 24                  # Section titles
FONT_SIZE_SUBTITLE = 14               # Subtitles
FONT_SIZE_HEADING = 16                # Headers
FONT_SIZE_BODY = 11                   # Body text
FONT_SIZE_BUTTON = 12                 # Buttons
FONT_SIZE_SMALL = 10                  # Small text
FONT_SIZE_TINY = 8                    # Labels/codes

# ==================== ANIMATION SETTINGS ====================
ANIMATION_FAST = 100      # ms
ANIMATION_NORMAL = 200    # ms
ANIMATION_SLOW = 400      # ms

# Progress Bar Settings
PROGRESS_BAR_HEIGHT = 30
PROGRESS_BAR_SEGMENTS = 20  # Number of segments for retro bar

# ==================== MODE SETTINGS ====================
MODE_BATCH = "batch"
MODE_REALTIME = "realtime"

# ==================== PREDICTION SETTINGS ====================
CHUNK_SIZE = 1_000_000
MAX_FILE_SIZE_MB = 2000

# ==================== DASK SETTINGS ====================
DASK_N_WORKERS = 4
DASK_THREADS_PER_WORKER = 2
DASK_MEMORY_LIMIT = '4GB'

# ==================== CSV PROCESSING ====================
REQUIRED_COLUMNS = [
    'bookingID', 'second', 'speed', 
    'acceleration_x', 'acceleration_y', 'acceleration_z',
    'gyro_x', 'gyro_y', 'gyro_z', 'accuracy'
]

# ==================== SESSION HISTORY ====================
MAX_HISTORY_ENTRIES = 50

# ==================== UI ELEMENT SIZES ====================
BUTTON_HEIGHT = 40
BUTTON_WIDTH_SMALL = 120
BUTTON_WIDTH_MEDIUM = 180
BUTTON_WIDTH_LARGE = 240

CARD_BORDER_WIDTH = 2
CARD_PADDING = 20

# ==================== ICONS & SYMBOLS ====================
ICON_BATCH = "📊"
ICON_REALTIME = "🔴"
ICON_FILE = "📄"
ICON_SUCCESS = "✓"
ICON_DANGER = "⚠"
ICON_LOADING = "⟳"
ICON_EXPORT = "💾"
ICON_CLEAR = "✕"
ICON_ADD = "+"
ICON_DATATERM = "█"

# ==================== DEPARTMENT CODES ====================
DEPT_ALPHA = "ALPHA"
DEPT_BRAVO = "BRAVO"
DEPT_VICTOR = "VICTOR"
DEPT_SIERRA = "SIERRA"
DEPT_ADMIN = "ADMINISTRATION"

# ==================== CYNOSURE BRANDING ====================
CORP_NAME = "CYNOSURE SYSTEMS"
SYSTEM_CODE = "GOBEST-CAB-SAFETY"
CLEARANCE_LEVEL = "AUTHORIZED PERSONNEL ONLY"
