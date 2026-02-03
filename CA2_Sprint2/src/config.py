"""
GOBEST CAB CYBERPUNK CONFIGURATION
🔥 NEON-POWERED SAFETY PREDICTION SYSTEM 🔥
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

# ==================== CYBERPUNK UI SETTINGS ====================
WINDOW_TITLE = "⚡ GOBEST CAB :: NEURAL SAFETY ANALYZER ⚡"
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
WINDOW_GEOMETRY = f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}"

# 🎨 CYBERPUNK COLOR PALETTE
# Primary Neons
COLOR_NEON_CYAN = "#00f0ff"           # Electric Cyan
COLOR_NEON_PINK = "#ff006e"           # Hot Pink
COLOR_NEON_PURPLE = "#bd00ff"         # Vivid Purple
COLOR_NEON_BLUE = "#0066ff"           # Electric Blue
COLOR_NEON_GREEN = "#00ff88"          # Matrix Green
COLOR_NEON_YELLOW = "#ffea00"         # Neon Yellow
COLOR_NEON_ORANGE = "#ff6600"         # Cyber Orange

# Dark Backgrounds (Cyberpunk Noir)
COLOR_BG_VOID = "#0a0a0f"             # Deep Space
COLOR_BG_DARK = "#121218"             # Dark Matter
COLOR_BG_CARD = "#1a1a24"             # Shadow Card
COLOR_BG_ELEVATED = "#22222f"         # Elevated Surface
COLOR_BG_GLASS = "#1a1a2480"          # Glass Effect (with alpha)

# Accent & Gradient Colors
COLOR_GRADIENT_START = "#bd00ff"      # Purple
COLOR_GRADIENT_MID = "#ff006e"        # Pink
COLOR_GRADIENT_END = "#00f0ff"        # Cyan

# Status Colors
COLOR_SUCCESS = "#00ff88"             # Matrix Green
COLOR_DANGER = "#ff006e"              # Hot Pink
COLOR_WARNING = "#ffea00"             # Neon Yellow
COLOR_INFO = "#00f0ff"                # Electric Cyan

# Text Colors
COLOR_TEXT_PRIMARY = "#ffffff"        # Pure White
COLOR_TEXT_SECONDARY = "#b8b8d4"      # Light Purple-Gray
COLOR_TEXT_MUTED = "#6666aa"          # Muted Purple
COLOR_TEXT_NEON = "#00f0ff"           # Cyan Highlight

# Border & Glow Effects
COLOR_BORDER_NEON = "#00f0ff"         # Cyan Border
COLOR_GLOW_PINK = "#ff006e40"         # Pink Glow (alpha)
COLOR_GLOW_CYAN = "#00f0ff40"         # Cyan Glow (alpha)
COLOR_GLOW_PURPLE = "#bd00ff40"       # Purple Glow (alpha)

# ==================== TYPOGRAPHY ====================
FONT_FAMILY = "Consolas"              # Monospace for cyber feel
FONT_FAMILY_DISPLAY = "Arial Black"   # Bold display font
FONT_SIZE_MEGA = 48                   # Huge titlesCyberMainWindow
FONT_SIZE_TITLE = 36                  # Main titles
FONT_SIZE_SUBTITLE = 18               # Subtitles
FONT_SIZE_HEADING = 16                # Section headers
FONT_SIZE_BODY = 12                   # Body text
FONT_SIZE_BUTTON = 14                 # Buttons
FONT_SIZE_SMALL = 10                  # Small text
FONT_SIZE_TINY = 8                    # Micro text

# ==================== ANIMATION SETTINGS ====================
ANIMATION_FAST = 100      # ms - Quick transitions
ANIMATION_NORMAL = 200    # ms - Standard animations
ANIMATION_SLOW = 400      # ms - Smooth animations
ANIMATION_EPIC = 800      # ms - Epic entrances

# Particle Effects
PARTICLE_COUNT = 30       # Number of background particles
PARTICLE_SPEED = 0.5      # Pixels per frame

# Glow Pulse
GLOW_PULSE_SPEED = 20     # ms per pulse update
GLOW_MIN_ALPHA = 30       # Minimum glow opacity
GLOW_MAX_ALPHA = 100      # Maximum glow opacity

# Scan Lines
SCANLINE_SPEED = 2        # Pixels per frame
SCANLINE_HEIGHT = 2       # Pixel height
SCANLINE_OPACITY = 15     # Alpha value

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
BUTTON_HEIGHT = 50
BUTTON_WIDTH_SMALL = 140
BUTTON_WIDTH_MEDIUM = 200
BUTTON_WIDTH_LARGE = 280

CARD_BORDER_RADIUS = 15
CARD_PADDING = 25

GLOW_OFFSET = 4  # Pixels for neon glow effect
BORDER_WIDTH = 2

# ==================== ICON SETS ====================
ICON_BATCH = "⚡"
ICON_REALTIME = "🔴"
ICON_FILE = "📁"
ICON_SUCCESS = "✓"
ICON_DANGER = "⚠"
ICON_LOADING = "⟳"
ICON_ROCKET = "🚀"
ICON_CHART = "📊"
ICON_EXPORT = "💾"
ICON_HISTORY = "📜"
ICON_CLEAR = "✕"
ICON_ADD = "+"
ICON_BRAIN = "🧠"
ICON_EYE = "👁"
ICON_SHIELD = "🛡"

# ==================== ASCII ART ====================
ASCII_LOGO = """
  ╔═══════════════════════════════════════╗
  ║   G O B E S T   C A B   S Y S T E M   ║
  ║     N E U R A L   A N A L Y Z E R     ║
  ╚═══════════════════════════════════════╝
"""

# ==================== SOUND SETTINGS (for future) ====================
ENABLE_SOUND_FX = False  # Toggle for sound effects
SOUND_CLICK = "click.wav"
SOUND_SUCCESS = "success.wav"
SOUND_ERROR = "error.wav"
