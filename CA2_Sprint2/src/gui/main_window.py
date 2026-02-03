"""
🗂️ CYNOSURE MAIN TERMINAL 🗂️
Main application window with retro paper theme
"""

import tkinter as tk
from pathlib import Path

from .. import config
from .batch_mode_view import CynosureBatchView
from .realtime_mode_view import CyberRealtimeView
from .results_view import CyberResultsView
from ..core.mode_controller import ModeController


class CyberMainWindow(tk.Tk):
    """
    CYNOSURE MAIN TERMINAL
    Retro paper-based interface
    """
    
    def __init__(self):
        super().__init__()
        
        # Window setup
        self.title(config.WINDOW_TITLE)
        self.geometry(config.WINDOW_GEOMETRY)
        self.configure(bg=config.COLOR_BG_PAPER)
        
        # Resizable
        self.resizable(True, True)
        self.minsize(1024, 768)
        
        # Center window
        self._center_window()
        
        # Initialize controller
        self.controller = ModeController()
        
        # View containers
        self.views = {}
        self.current_view = None
        self.current_view_name = None
        
        # Setup UI
        self._setup_ui()
        
        # Show initial view
        self.show_view("batch")
    
    def _center_window(self):
        """Center window on screen"""
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f'{config.WINDOW_WIDTH}x{config.WINDOW_HEIGHT}+{x}+{y}')
    
    def _setup_ui(self):
        """Setup main UI"""
        # Main container
        self.main_container = tk.Frame(self, bg=config.COLOR_BG_PAPER)
        self.main_container.pack(fill=tk.BOTH, expand=True)
        
        # Top header
        self._create_top_header()
        
        # Content area
        self.content_frame = tk.Frame(
            self.main_container,
            bg=config.COLOR_BG_PAPER
        )
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(80, 20))
        
        # Bottom status
        self._create_bottom_status()
        
        # Create views
        self._create_views()
    
    def _create_top_header(self):
        """Create header"""
        header = tk.Frame(self.main_container, bg=config.COLOR_BG_PAPER, height=70)
        header.pack(fill=tk.X, side=tk.TOP)
        header.pack_propagate(False)
        
        # Left border
        left_corner = tk.Canvas(header, width=40, height=70, bg=config.COLOR_BG_PAPER, highlightthickness=0)
        left_corner.pack(side=tk.LEFT)
        left_corner.create_line(5, 10, 30, 10, fill=config.COLOR_BORDER_DARK, width=3)
        left_corner.create_line(5, 10, 5, 40, fill=config.COLOR_BORDER_DARK, width=3)

        # Title
        title_frame = tk.Frame(header, bg=config.COLOR_BG_PAPER)
        title_frame.pack(side=tk.LEFT, padx=20)
        
        tk.Label(
            title_frame,
            text="GOBEST CAB",
            font=(config.FONT_FAMILY_DISPLAY, 22, "bold"),
            fg=config.COLOR_TEXT_PRIMARY,
            bg=config.COLOR_BG_PAPER
        ).pack(anchor="w")
        
        tk.Label(
            title_frame,
            text="SAFETY ANALYSIS TERMINAL v2.0 // CYNOSURE CORP.",
            font=(config.FONT_FAMILY, 8),
            fg=config.COLOR_TEXT_SECONDARY,
            bg=config.COLOR_BG_PAPER
        ).pack(anchor="w")
        
        # Status
        status_frame = tk.Frame(header, bg=config.COLOR_BG_PAPER)
        status_frame.pack(side=tk.RIGHT, padx=20)
        
        self._create_status_indicator(status_frame, "SYSTEM", "ONLINE", config.COLOR_SUCCESS)
        self._create_status_indicator(status_frame, "MODEL", "READY", config.COLOR_ACCENT_BLUE)
        
        # Right border
        right_corner = tk.Canvas(header, width=40, height=70, bg=config.COLOR_BG_PAPER, highlightthickness=0)
        right_corner.pack(side=tk.RIGHT)
        right_corner.create_line(10, 10, 35, 10, fill=config.COLOR_BORDER_DARK, width=3)
        right_corner.create_line(35, 10, 35, 40, fill=config.COLOR_BORDER_DARK, width=3)
        
        # Divider
        divider = tk.Frame(
            self.main_container,
            bg=config.COLOR_BORDER_DARK,
            height=2
        )
        divider.pack(fill=tk.X)
    
    def _create_status_indicator(self, parent, label: str, status: str, color: str):
        """Create status indicator"""
        frame = tk.Frame(parent, bg=config.COLOR_BG_PAPER)
        frame.pack(side=tk.LEFT, padx=10)
        
        tk.Label(
            frame,
            text=label,
            font=(config.FONT_FAMILY, 7),
            fg=config.COLOR_TEXT_MUTED,
            bg=config.COLOR_BG_PAPER
        ).pack()
        
        status_frame = tk.Frame(frame, bg=config.COLOR_BG_PAPER)
        status_frame.pack()
        
        # Dot
        dot = tk.Canvas(status_frame, width=8, height=8, bg=config.COLOR_BG_PAPER, highlightthickness=0)
        dot.pack(side=tk.LEFT, padx=(0, 5))
        dot.create_oval(2, 2, 6, 6, fill=color, outline="")
        
        tk.Label(
            status_frame,
            text=status,
            font=(config.FONT_FAMILY, 8, "bold"),
            fg=color,
            bg=config.COLOR_BG_PAPER
        ).pack(side=tk.LEFT)
    
    def _create_bottom_status(self):
        """Create bottom status bar"""
        bottom = tk.Frame(self.main_container, bg=config.COLOR_BG_PAPER, height=40)
        bottom.pack(fill=tk.X, side=tk.BOTTOM)
        bottom.pack_propagate(False)
        
        # Divider
        tk.Frame(bottom, bg=config.COLOR_BORDER_DARK, height=2).pack(fill=tk.X)
        
        # Status
        self.status_label = tk.Label(
            bottom,
            text="● SYSTEM READY",
            font=(config.FONT_FAMILY, 9),
            fg=config.COLOR_SUCCESS,
            bg=config.COLOR_BG_PAPER,
            anchor="w"
        )
        self.status_label.pack(side=tk.LEFT, padx=20, pady=5)
        
        # Model info
        model_info = tk.Label(
            bottom,
            text=f"MODEL: Logistic Regression | FEATURES: {len(config.MODEL_FEATURES)} | THRESHOLD: {config.PREDICTION_THRESHOLD:.4f}",
            font=(config.FONT_FAMILY, 7),
            fg=config.COLOR_TEXT_MUTED,
            bg=config.COLOR_BG_PAPER
        )
        model_info.pack(side=tk.RIGHT, padx=20, pady=5)
    
    def _create_views(self):
        """Create all views"""
        # Batch view
        self.views["batch"] = CynosureBatchView(
            self.content_frame,
            controller=self.controller,
            switch_to_realtime=lambda: self.show_view("realtime"),
            show_results=lambda data: self.show_view("results", data),
            show_history=lambda: self.show_view("batch")  # Placeholder
        )
        
        # Realtime view
        self.views["realtime"] = CyberRealtimeView(
            self.content_frame,
            controller=self.controller,
            switch_to_batch=lambda: self.show_view("batch"),
            show_results=lambda data: self.show_view("results", data)
        )
        
        # Results view
        self.views["results"] = CyberResultsView(
            self.content_frame,
            controller=self.controller,
            back_to_batch=lambda: self.show_view("batch"),
            back_to_realtime=lambda: self.show_view("realtime")
        )
    
    def show_view(self, view_name: str, data=None):
        """Show a view"""
        # Hide current
        if self.current_view:
            self.current_view.pack_forget()
        
        # Show new
        self.current_view = self.views[view_name]
        self.current_view_name = view_name
        
        # Pass data if needed
        if data and hasattr(self.current_view, 'display_results'):
            self.current_view.display_results(data)
        
        self.current_view.pack(fill=tk.BOTH, expand=True)
        
        # Update status
        self._update_status_for_view(view_name)
    
    def _update_status_for_view(self, view_name: str):
        """Update status bar"""
        status_messages = {
            "batch": "● BATCH MODE ACTIVE",
            "realtime": "● REAL-TIME MODE ACTIVE",
            "results": "● ANALYSIS COMPLETE"
        }
        
        status_colors = {
            "batch": config.COLOR_ACCENT_BLUE,
            "realtime": config.COLOR_ACCENT_ORANGE,
            "results": config.COLOR_SUCCESS
        }
        
        self.status_label.config(
            text=status_messages.get(view_name, "● SYSTEM READY"),
            fg=status_colors.get(view_name, config.COLOR_SUCCESS)
        )
