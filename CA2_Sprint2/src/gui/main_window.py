"""
Main Window Module
Contains the primary application window and mode management
"""

import tkinter as tk
from typing import Optional

from .batch_mode_view import BatchModeView
from .realtime_mode_view import RealtimeModeView
from .results_view import ResultsView
from ..core.mode_controller import ModeController
from .. import config


class MainWindow:
    """
    Main application window that manages the GUI and navigation between modes
    """
    
    def __init__(self):
        """Initialize the main window"""
        self.root = tk.Tk()
        self.root.title(config.WINDOW_TITLE)
        self.root.geometry(config.WINDOW_GEOMETRY)
        self.root.configure(bg=config.COLOR_BACKGROUND)
        
        # Prevent window resizing for consistent layout
        self.root.resizable(False, False)
        
        # Initialize controller
        self.controller = ModeController()
        
        # Current view references
        self.current_view: Optional[tk.Frame] = None
        self.batch_view: Optional[BatchModeView] = None
        self.realtime_view: Optional[RealtimeModeView] = None
        self.results_view: Optional[ResultsView] = None
        
        # Setup UI
        self._setup_ui()
        self._show_batch_mode()
        
    def _setup_ui(self):
        """Setup the main UI structure"""
        # Main container frame
        self.main_container = tk.Frame(
            self.root,
            bg=config.COLOR_BACKGROUND
        )
        self.main_container.pack(fill=tk.BOTH, expand=True)
        
    def _clear_current_view(self):
        """Clear the current view from the window"""
        if self.current_view:
            self.current_view.pack_forget()
            self.current_view = None
    
    def _show_batch_mode(self):
        """Show the batch processing mode view"""
        self._clear_current_view()
        
        if not self.batch_view:
            self.batch_view = BatchModeView(
                self.main_container,
                self.controller,
                self._show_realtime_mode,
                self._show_results
            )
        
        self.current_view = self.batch_view
        self.current_view.pack(fill=tk.BOTH, expand=True)
        self.controller.set_mode(config.MODE_BATCH)
        
    def _show_realtime_mode(self):
        """Show the real-time mode view"""
        self._clear_current_view()
        
        if not self.realtime_view:
            self.realtime_view = RealtimeModeView(
                self.main_container,
                self.controller,
                self._show_batch_mode,
                self._show_results
            )
        
        self.current_view = self.realtime_view
        self.current_view.pack(fill=tk.BOTH, expand=True)
        self.controller.set_mode(config.MODE_REALTIME)
        
    def _show_results(self, prediction_data: dict):
        """
        Show the results view with prediction data
        
        Args:
            prediction_data: Dictionary containing prediction results
        """
        self._clear_current_view()
        
        # Create new results view with data
        self.results_view = ResultsView(
            self.main_container,
            prediction_data,
            self._show_batch_mode
        )
        
        self.current_view = self.results_view
        self.current_view.pack(fill=tk.BOTH, expand=True)
    
    def run(self):
        """Start the application main loop"""
        # Center window on screen
        self.root.update_idletasks()
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - config.WINDOW_WIDTH) // 2
        y = (screen_height - config.WINDOW_HEIGHT) // 2
        self.root.geometry(f"{config.WINDOW_WIDTH}x{config.WINDOW_HEIGHT}+{x}+{y}")
        
        # Start main loop
        self.root.mainloop()
