"""
🌟 CYBERPUNK MAIN WINDOW 🌟
Neural Safety Analyzer Interface with Particle Effects and Neon HUD
"""

import tkinter as tk
from tkinter import font as tkfont
from pathlib import Path

from .. import config
from .cyber_components import ParticleField, GlitchText
from .batch_mode_view import CyberBatchView
from .realtime_mode_view import CyberRealtimeView
from .results_view import CyberResultsView
from ..core.mode_controller import ModeController


class CyberMainWindow(tk.Tk):
    """
    🔥 MAIN CYBERPUNK APPLICATION WINDOW 🔥
    Features:
    - Animated particle background
    - Neon HUD elements
    - Smooth view transitions
    - Glitch effects
    """
    
    def __init__(self):
        super().__init__()
        
        # Window setup
        self.title(config.WINDOW_TITLE)
        self.geometry(config.WINDOW_GEOMETRY)
        self.configure(bg=config.COLOR_BG_VOID)
        
        # Make window resizable
        self.resizable(True, True)
        self.minsize(1024, 768)  # Set a reasonable minimum size
        
        # Center window on screen
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
        
        # Start intro animation
        self._play_intro_animation()
    
    def _center_window(self):
        """Center the window on screen"""
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f'{config.WINDOW_WIDTH}x{config.WINDOW_HEIGHT}+{x}+{y}')
    
    def _setup_ui(self):
        """Setup the main UI structure"""
        # Main container
        self.main_container = tk.Frame(self, bg=config.COLOR_BG_VOID)
        self.main_container.pack(fill=tk.BOTH, expand=True)
        
        # Animated particle background (behind everything)
        self.particle_field = ParticleField(
            self.main_container,
            width=config.WINDOW_WIDTH,
            height=config.WINDOW_HEIGHT
        )
        # FIX: Use relwidth/relheight to fill window on resize
        self.particle_field.place(x=0, y=0, relwidth=1, relheight=1)
        
        # Bind resize event to update particle field limits
        self.particle_field.bind('<Configure>', self._on_resize_particles)
        
        # HUD Container (on top of particles)
        self.hud_container = tk.Frame(
            self.main_container,
            bg=config.COLOR_BG_VOID
        )
        # FIX: Use relwidth/relheight to ensure HUD expands
        self.hud_container.place(x=0, y=0, relwidth=1, relheight=1)
        
        # Top HUD Bar
        self._create_top_hud()
        
        # Content area
        self.content_frame = tk.Frame(
            self.hud_container,
            bg=config.COLOR_BG_VOID
        )
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(80, 20))
        
        # Bottom status bar
        self._create_bottom_hud()
        
        # Create all views (hidden initially)
        self._create_views()
    
    def _on_resize_particles(self, event):
        """Update particle field dimensions when window resizes"""
        if hasattr(self, 'particle_field'):
            self.particle_field.width = event.width
            self.particle_field.height = event.height

    def _create_top_hud(self):
        """Create cyberpunk HUD header"""
        hud_top = tk.Frame(self.hud_container, bg=config.COLOR_BG_VOID, height=70)
        hud_top.pack(fill=tk.X, side=tk.TOP)
        hud_top.pack_propagate(False)
        
        # Left corner accent
        corner_left = tk.Canvas(hud_top, width=40, height=70, bg=config.COLOR_BG_VOID, highlightthickness=0)
        corner_left.pack(side=tk.LEFT)
        corner_left.create_line(5, 10, 30, 10, fill=config.COLOR_NEON_CYAN, width=3)
        corner_left.create_line(5, 10, 5, 40, fill=config.COLOR_NEON_CYAN, width=3)
        corner_left.create_line(5, 60, 30, 60, fill=config.COLOR_NEON_PINK, width=2)
        corner_left.create_line(5, 30, 5, 60, fill=config.COLOR_NEON_PINK, width=2)
        
        # Title section
        title_frame = tk.Frame(hud_top, bg=config.COLOR_BG_VOID)
        title_frame.pack(side=tk.LEFT, padx=20)
        
        # Main title with glitch effect
        self.title_label = GlitchText(
            title_frame,
            text="GOBEST CAB",
            font=(config.FONT_FAMILY_DISPLAY, 24, "bold"),
            fg=config.COLOR_NEON_CYAN,
            bg=config.COLOR_BG_VOID
        )
        self.title_label.pack(anchor="w")
        
        # Subtitle
        subtitle = tk.Label(
            title_frame,
            text="NEURAL SAFETY ANALYZER v2.0",
            font=(config.FONT_FAMILY, 9),
            fg=config.COLOR_TEXT_SECONDARY,
            bg=config.COLOR_BG_VOID
        )
        subtitle.pack(anchor="w")
        
        # Status indicators (right side)
        status_frame = tk.Frame(hud_top, bg=config.COLOR_BG_VOID)
        status_frame.pack(side=tk.RIGHT, padx=20)
        
        # System status
        self._create_status_indicator(status_frame, "NEURAL CORE", "ONLINE", config.COLOR_NEON_GREEN)
        self._create_status_indicator(status_frame, "SENSORS", "ACTIVE", config.COLOR_NEON_CYAN)
        
        # Right corner accent
        corner_right = tk.Canvas(hud_top, width=40, height=70, bg=config.COLOR_BG_VOID, highlightthickness=0)
        corner_right.pack(side=tk.RIGHT)
        corner_right.create_line(10, 10, 35, 10, fill=config.COLOR_NEON_CYAN, width=3)
        corner_right.create_line(35, 10, 35, 40, fill=config.COLOR_NEON_CYAN, width=3)
        corner_right.create_line(10, 60, 35, 60, fill=config.COLOR_NEON_PINK, width=2)
        corner_right.create_line(35, 30, 35, 60, fill=config.COLOR_NEON_PINK, width=2)
        
        # Horizontal divider line with glow
        divider = tk.Canvas(self.hud_container, height=3, bg=config.COLOR_BG_VOID, highlightthickness=0)
        divider.pack(fill=tk.X)
        
        # Bind resize for divider line
        def resize_line(e):
            divider.delete("all")
            divider.create_line(0, 1, e.width, 1, fill=config.COLOR_NEON_CYAN, width=2)
            divider.create_line(0, 1, e.width, 1, fill=config.COLOR_NEON_CYAN, stipple="gray50", width=6)
            
        divider.bind('<Configure>', resize_line)
    
    def _create_status_indicator(self, parent, label: str, status: str, color: str):
        """Create a small status indicator"""
        frame = tk.Frame(parent, bg=config.COLOR_BG_VOID)
        frame.pack(side=tk.LEFT, padx=10)
        
        tk.Label(
            frame,
            text=label,
            font=(config.FONT_FAMILY, 7),
            fg=config.COLOR_TEXT_MUTED,
            bg=config.COLOR_BG_VOID
        ).pack()
        
        status_frame = tk.Frame(frame, bg=config.COLOR_BG_VOID)
        status_frame.pack()
        
        # Blinking dot
        dot = tk.Canvas(status_frame, width=8, height=8, bg=config.COLOR_BG_VOID, highlightthickness=0)
        dot.pack(side=tk.LEFT, padx=(0, 5))
        dot.create_oval(2, 2, 6, 6, fill=color, outline=color)
        
        tk.Label(
            status_frame,
            text=status,
            font=(config.FONT_FAMILY, 8, "bold"),
            fg=color,
            bg=config.COLOR_BG_VOID
        ).pack(side=tk.LEFT)
    
    def _create_bottom_hud(self):
        """Create bottom status bar"""
        hud_bottom = tk.Frame(self.hud_container, bg=config.COLOR_BG_VOID, height=40)
        hud_bottom.pack(fill=tk.X, side=tk.BOTTOM)
        hud_bottom.pack_propagate(False)
        
        # Divider line
        divider = tk.Canvas(hud_bottom, height=2, bg=config.COLOR_BG_VOID, highlightthickness=0)
        divider.pack(fill=tk.X)
        
        def resize_bottom(e):
            divider.delete("all")
            divider.create_line(0, 0, e.width, 0, fill=config.COLOR_NEON_PINK, width=1)
            
        divider.bind('<Configure>', resize_bottom)
        
        # Status text
        self.status_label = tk.Label(
            hud_bottom,
            text="◉ SYSTEM READY",
            font=(config.FONT_FAMILY, 9),
            fg=config.COLOR_NEON_GREEN,
            bg=config.COLOR_BG_VOID,
            anchor="w"
        )
        self.status_label.pack(side=tk.LEFT, padx=20, pady=5)
        
        # Model info
        model_info = tk.Label(
            hud_bottom,
            text=f"MODEL: Logistic Regression | FEATURES: {len(config.MODEL_FEATURES)} | THRESHOLD: {config.PREDICTION_THRESHOLD:.4f}",
            font=(config.FONT_FAMILY, 8),
            fg=config.COLOR_TEXT_MUTED,
            bg=config.COLOR_BG_VOID
        )
        model_info.pack(side=tk.RIGHT, padx=20, pady=5)
    
    def _create_views(self):
        """Create all application views"""
        # Batch Processing View
        self.views["batch"] = CyberBatchView(
            self.content_frame,
            controller=self.controller,
            switch_to_realtime=lambda: self.show_view("realtime"),
            show_results=lambda data: self.show_view("results", data)
        )
        
        # Realtime Processing View
        self.views["realtime"] = CyberRealtimeView(
            self.content_frame,
            controller=self.controller,
            switch_to_batch=lambda: self.show_view("batch"),
            show_results=lambda data: self.show_view("results", data)
        )
        
        # Results View
        self.views["results"] = CyberResultsView(
            self.content_frame,
            controller=self.controller,
            back_to_batch=lambda: self.show_view("batch"),
            back_to_realtime=lambda: self.show_view("realtime")
        )
    
    def show_view(self, view_name: str, data=None):
        """
        Show a specific view with smooth transition
        
        Args:
            view_name: Name of the view to show
            data: Optional data to pass to the view
        """
        # Hide current view
        if self.current_view:
            self.current_view.pack_forget()
        
        # Show new view
        self.current_view = self.views[view_name]
        self.current_view_name = view_name
        
        # Pass data if needed (for results view)
        if data and hasattr(self.current_view, 'display_results'):
            self.current_view.display_results(data)
        
        # Animate entrance
        self.current_view.pack(fill=tk.BOTH, expand=True)
        self._animate_view_entrance()
        
        # Update status based on view
        self._update_status_for_view(view_name)
        
        # Trigger glitch effect on title
        self.title_label.start_glitch(300)
    
    def _animate_view_entrance(self):
        """Animate the view entrance"""
        # Fade in effect (simplified version)
        self.current_view.configure(bg=config.COLOR_BG_VOID)
    
    def _update_status_for_view(self, view_name: str):
        """Update bottom status bar based on current view"""
        status_messages = {
            "batch": "◉ BATCH MODE ACTIVE",
            "realtime": "◉ REAL-TIME MODE ACTIVE",
            "results": "◉ ANALYSIS COMPLETE"
        }
        
        status_colors = {
            "batch": config.COLOR_NEON_CYAN,
            "realtime": config.COLOR_NEON_PINK,
            "results": config.COLOR_NEON_GREEN
        }
        
        self.status_label.config(
            text=status_messages.get(view_name, "◉ SYSTEM READY"),
            fg=status_colors.get(view_name, config.COLOR_NEON_GREEN)
        )
    
    def _play_intro_animation(self):
        """Play intro animation on startup"""
        # Flash the title
        self.title_label.start_glitch(800)
        
        # Flash status indicators
        self.after(200, lambda: self._flash_element(self.status_label, 3))
    
    def _flash_element(self, element, times: int, current: int = 0):
        """Flash an element on and off"""
        if current >= times:
            return
        
        current_color = element.cget("fg")
        new_color = config.COLOR_BG_VOID if current_color != config.COLOR_BG_VOID else config.COLOR_NEON_CYAN
        element.config(fg=new_color)
        
        self.after(100, lambda: self._flash_element(element, times, current + 0.5))