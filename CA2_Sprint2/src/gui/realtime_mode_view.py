"""
Real-Time Mode View Module - ULTRA MODERN UI OVERHAUL
Aesthetic dashboard design with reactive inputs and gaming-inspired UI
"""

import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
from typing import Callable, Dict
from datetime import datetime

from .. import config

# --- AESTHETIC CONSTANTS ---
COLOR_BG = "#0f172a"          # Deep Slate (Main Background)
COLOR_CARD = "#1e293b"        # Lighter Slate (Card Background)
COLOR_ACCENT = "#38bdf8"      # Sky Blue (Glow effects)
COLOR_ACCENT_DIM = "#0ea5e9"  # Darker Blue (Borders)
COLOR_TEXT_MAIN = "#f1f5f9"   # White-ish
COLOR_TEXT_MUTED = "#94a3b8"  # Grey text
COLOR_INPUT_BG = "#0f172a"    # Dark input background


class CyberButton(tk.Canvas):
    """
    A massive, high-tech button with hover glow effects
    """
    def __init__(self, parent, text, command, width=200, height=50, bg_color=COLOR_ACCENT_DIM, fg_color="white", icon=""):
        super().__init__(parent, width=width, height=height, bg=COLOR_BG, highlightthickness=0, cursor="hand2")
        self.command = command
        self.bg_color = bg_color
        self.hover_color = COLOR_ACCENT
        self.text = text
        self.icon = icon
        self.width = width
        self.height = height
        self.is_disabled = False
        
        self.bind("<Button-1>", self._on_click)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self._draw()

    def _draw(self, color=None):
        if color is None: color = self.bg_color
        self.delete("all")
        
        # Tech border decoration
        # Main Rect
        self.create_rectangle(2, 2, self.width-2, self.height-2, fill=color, outline=color, tags="bg")
        
        # "Cut" corners effect
        self.create_polygon(
            0, 10, 10, 0, self.width, 0, self.width, self.height-10, self.width-10, self.height, 0, self.height,
            fill=color, outline=""
        )
        
        # Text & Icon
        display_text = f"{self.icon}  {self.text}" if self.icon else self.text
        self.create_text(self.width/2, self.height/2, text=display_text, 
                         font=(config.FONT_FAMILY, 11, "bold"), fill="white")

    def _on_enter(self, e):
        if not self.is_disabled:
            self._draw(self.hover_color)

    def _on_leave(self, e):
        if not self.is_disabled:
            self._draw(self.bg_color)

    def _on_click(self, e):
        if not self.is_disabled:
            self.command()

    def set_disabled(self, disabled):
        self.is_disabled = disabled
        color = COLOR_CARD if disabled else self.bg_color
        self._draw(color)

class ModernInput(tk.Canvas):
    """
    A custom input widget that looks like a high-tech capsule.
    Features: Rounded corners, border glow on focus, placeholder text support.
    """
    def __init__(self, parent, label_text, placeholder="", initial_value="", width=300, height=60):
        super().__init__(parent, width=width, height=height, bg=COLOR_CARD, highlightthickness=0)
        self.parent = parent
        self.placeholder = placeholder
        self.width = width
        self.height = height
        self.text_var = tk.StringVar(value=initial_value)
        self.is_placeholder = False  # <--- NEW FLAG added here
        
        # Draw Label
        self.create_text(10, 15, text=label_text.upper(), anchor="w", 
                         font=(config.FONT_FAMILY, 8, "bold"), fill=COLOR_ACCENT)

        # Create Entry widget (placed on top of canvas)
        self.entry = tk.Entry(
            self, 
            textvariable=self.text_var,
            font=(config.FONT_FAMILY, 10),
            bg=COLOR_INPUT_BG,
            fg=COLOR_TEXT_MAIN,
            insertbackground="white", # Cursor color
            relief=tk.FLAT,
            bd=0
        )
        
        # Place entry on canvas
        self.create_window(15, 40, window=self.entry, width=width-30, height=25, anchor="w")
        
        # Initial Draw
        self.draw_border(color=COLOR_TEXT_MUTED, width=1)
        
        # Bind Events for "Glow" effect
        self.entry.bind("<FocusIn>", lambda e: self.animate_focus(True))
        self.entry.bind("<FocusOut>", lambda e: self.animate_focus(False))
        
        # Add placeholder logic if needed (simplified here)
        if not initial_value and placeholder:
            self.entry.insert(0, placeholder)
            self.entry.config(fg=COLOR_TEXT_MUTED)
            self.is_placeholder = True  # <--- Mark as placeholder
            self.entry.bind("<FocusIn>", self._on_focus_in, add="+")
            self.entry.bind("<FocusOut>", self._on_focus_out, add="+")

    def draw_border(self, color, width=1):
        """Draws the rounded rectangle border"""
        self.delete("border")
        # Draw rounded rect using lines/arcs or a polygon. 
        x1, y1, x2, y2 = 2, 28, self.width-2, self.height-2
        radius = 10
        
        # We use a polygon to fake rounded corners or just lines for a "tech" look
        points = [
            x1, y2-10, x1, y2, x2, y2, x2, y2-10  # Bottom bracket
        ]
        self.create_line(points, fill=color, width=width, tags="border", capstyle=tk.ROUND)
        
        # Top-left indicator
        self.create_line(x1, y1+10, x1, y1, x1+10, y1, fill=color, width=width, tags="border")
        # Top-right indicator
        self.create_line(x2-10, y1, x2, y1, x2, y1+10, fill=color, width=width, tags="border")

    def animate_focus(self, focused):
        """Changes border color on focus"""
        color = COLOR_ACCENT if focused else COLOR_TEXT_MUTED
        width = 2 if focused else 1
        self.draw_border(color, width)

    def _on_focus_in(self, event):
        # <--- UPDATED LOGIC: check flag instead of text
        if self.is_placeholder:
            self.entry.delete(0, tk.END)
            self.entry.config(fg=COLOR_TEXT_MAIN)
            self.is_placeholder = False
        self.animate_focus(True)

    def _on_focus_out(self, event):
        if not self.entry.get():
            self.entry.insert(0, self.placeholder)
            self.entry.config(fg=COLOR_TEXT_MUTED)
            self.is_placeholder = True  # <--- Reset flag
        self.animate_focus(False)

    def get(self):
        # <--- UPDATED LOGIC: check flag instead of text
        if self.is_placeholder:
            return ""
        return self.entry.get()


class RealtimeModeView(tk.Frame):
    def __init__(self, parent, controller, switch_to_batch: Callable, show_results: Callable):
        super().__init__(parent, bg=COLOR_BG)
        self.controller = controller
        self.switch_to_batch = switch_to_batch
        self.show_results = show_results
        self.inputs = {}  # Store ModernInput instances
        
        self._setup_ui()
        
    def _setup_ui(self):
        """Setup the Cyberpunk Dashboard UI"""
        
        # --- HEADER SECTION ---
        header = tk.Frame(self, bg=COLOR_BG)
        header.pack(fill=tk.X, padx=40, pady=(30, 20))
        
        # Title Stack (Left)
        title_frame = tk.Frame(header, bg=COLOR_BG)
        title_frame.pack(side=tk.LEFT)
        
        tk.Label(title_frame, text="LIVE PREDICTION // DASHBOARD", 
                 font=(config.FONT_FAMILY, 24, "bold"),
                 bg=COLOR_BG, fg="white").pack(anchor="w")
        
        tk.Label(title_frame, text="MANUAL DATA ENTRY TERMINAL", 
                 font=(config.FONT_FAMILY, 10, "bold"),
                 bg=COLOR_BG, fg=COLOR_ACCENT_DIM).pack(anchor="w")

        # MASSIVE BACK BUTTON (Right)
        btn_back = CyberButton(
            header, 
            text="RETURN TO BATCH", 
            icon="↩",
            command=self.switch_to_batch,
            width=220, height=50,
            bg_color=COLOR_CARD
        )
        btn_back.pack(side=tk.RIGHT)

        # --- SCROLLABLE AREA ---
        # Main Canvas
        canvas = tk.Canvas(self, bg=COLOR_BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        
        # Inner Frame (The "Card Container")
        self.scroll_frame = tk.Frame(canvas, bg=COLOR_BG)
        
        # Configure Scrolling
        self.scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        window_id = canvas.create_window((0, 0), window=self.scroll_frame, anchor="n")
        
        # Auto-center content
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(window_id, width=e.width))
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack Scroll Area
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=0)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # --- CONTENT GRID ---
        # We will use a grid layout for the cards
        
        content_container = tk.Frame(self.scroll_frame, bg=COLOR_BG)
        content_container.pack(fill=tk.BOTH, expand=True, padx=40, pady=20)
        
        # Left Column: General Stats
        left_col = tk.Frame(content_container, bg=COLOR_BG)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 20))
        
        self._create_card(left_col, "TRIP IDENTIFICATION & METRICS", [
            ("Booking ID", "bookingID", "TRIP-2026-X"),
            ("Trip Duration (sec)", "trip_duration_sec", "1200"),
            ("Mean Speed (m/s)", "speed_mean", "15.5"),
            ("Max Speed (m/s)", "speed_max", "28.0"),
            ("Pct Time Cruising (0-1)", "pct_time_cruising", "0.45"),
        ])

        # Right Column: Dynamics & Raw Data
        right_col = tk.Frame(content_container, bg=COLOR_BG)
        right_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._create_card(right_col, "DRIVING BEHAVIOR & DYNAMICS", [
            ("Pct High Accel (0-1)", "pct_time_high_accel", "0.05"),
            ("Mean Linear Jerk", "jerk_linear_mean", "0.8"),
            ("Longest Smooth Segment", "longest_smooth_segment_sec", "45.0"),
        ])
        
        # Spacer
        tk.Frame(right_col, bg=COLOR_BG, height=20).pack()
        
        self._create_card(right_col, "SENSOR CALIBRATION DATA", [
            ("Max Accel Mag (m/s²)", "accel_mag_max", "12.5"),
            ("Count Hard Accels", "n_hard_accels", "3"),
            ("Max Gyro Mag (rad/s)", "gyro_mag_max", "0.8"),
            ("Std Dev Accel", "accel_mag_std", "1.2"),
            ("Std Dev Gyro", "gyro_mag_std", "0.1"),
        ])

        # --- PREDICT BUTTON (Floating at bottom or inline) ---
        # Let's put it at the bottom of the scroll frame content
        btn_container = tk.Frame(self.scroll_frame, bg=COLOR_BG, pady=40)
        btn_container.pack(fill=tk.X)
        
        self.predict_btn = tk.Button(
            btn_container, 
            text="INITIALIZE PREDICTION",
            font=(config.FONT_FAMILY, 14, "bold"),
            bg=COLOR_ACCENT_DIM,
            fg="white",
            activebackground="white",
            activeforeground=COLOR_ACCENT_DIM,
            relief=tk.FLAT,
            cursor="hand2",
            width=25,
            height=2,
            command=self._on_predict
        )
        self.predict_btn.pack()
        
        # Hover Effect for Button
        self.predict_btn.bind("<Enter>", lambda e: self.predict_btn.config(bg=COLOR_ACCENT))
        self.predict_btn.bind("<Leave>", lambda e: self.predict_btn.config(bg=COLOR_ACCENT_DIM))

    def _create_card(self, parent, title, fields):
        """Creates a stylized 'card' group"""
        # Card Container
        card = tk.Frame(parent, bg=COLOR_CARD, padx=20, pady=20)
        card.pack(fill=tk.X, pady=0)
        
        # Tech decoration line (Top border)
        tk.Frame(card, bg=COLOR_ACCENT, height=2).pack(fill=tk.X, pady=(0, 15))

        # Title
        tk.Label(card, text=title, font=(config.FONT_FAMILY, 9, "bold"),
                 fg=COLOR_ACCENT, bg=COLOR_CARD, anchor="w").pack(fill=tk.X, pady=(0, 15))

        # Fields
        for label, key, placeholder in fields:
            # We assume equal width for aesthetics
            inp = ModernInput(card, label, placeholder=placeholder, width=350)
            inp.pack(pady=2, fill=tk.X)
            self.inputs[key] = inp

    def _on_predict(self):
        """Gather data, animate, and predict"""
        try:
            # Simple "Loading" effect (Change button text)
            self.predict_btn.config(text="PROCESSING...", state="disabled", bg=COLOR_TEXT_MUTED)
            self.update_idletasks()
            
            raw_data = {}
            # 1. Gather Inputs from ModernInput widgets
            for key, widget in self.inputs.items():
                val = widget.get().strip()
                
                # Check empty
                if not val and key != "bookingID":
                    self.predict_btn.config(text="INITIALIZE PREDICTION", state="normal", bg=COLOR_ACCENT_DIM)
                    messagebox.showerror("Input Error", f"Missing value for {key}")
                    return
                
                if key != "bookingID":
                    try:
                        raw_data[key] = float(val)
                    except ValueError:
                        self.predict_btn.config(text="INITIALIZE PREDICTION", state="normal", bg=COLOR_ACCENT_DIM)
                        messagebox.showerror("Type Error", f"{key} must be a number.")
                        return
                else:
                    raw_data[key] = val if val else "MANUAL_ENTRY"

            # 2. Logic Calculations
            speed_mean = raw_data.get('speed_mean', 1.0)
            if speed_mean == 0: speed_mean = 0.001
            
            raw_data['turn_sharpness_index'] = raw_data['gyro_mag_max'] / speed_mean
            raw_data['accel_risk_score'] = raw_data['accel_mag_max'] * raw_data['n_hard_accels']
            raw_data['gyro_accel_instability'] = raw_data['gyro_mag_std'] * raw_data['accel_mag_std']

            # 3. Simulate slight delay for "Processing" feel
            self.after(500, lambda: self._finalize_prediction(raw_data))

        except Exception as e:
            self.predict_btn.config(text="INITIALIZE PREDICTION", state="normal", bg=COLOR_ACCENT_DIM)
            messagebox.showerror("System Error", str(e))

    def _finalize_prediction(self, raw_data):
        result = self.controller.process_realtime_data(raw_data)
        self.show_results(result)
        # Reset button state in case we come back
        self.predict_btn.config(text="INITIALIZE PREDICTION", state="normal", bg=COLOR_ACCENT_DIM)