"""
GOBEST CAB Safety Prediction System
Main Application Entry Point

This is a GUI application for predicting cab safety using ML models.
It supports both Batch Processing and Real-Time modes.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.gui.main_window import MainWindow

def main():
    """Main application entry point"""
    app = MainWindow()
    app.run()

if __name__ == "__main__":
    main()
