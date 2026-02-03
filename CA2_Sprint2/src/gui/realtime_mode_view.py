"""
🔴 CYBERPUNK REAL-TIME VIEW 🔴
Single trip analysis with live feedback and manual data entry
"""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Callable, Dict

from .. import config
from .cyber_components import NeonButton

class ModernInput(tk.Canvas):
    """
    A custom input widget that looks like a high-tech capsule.
    Features: Rounded corners, border glow on focus, placeholder text support.
    Adapted for Cyberpunk Theme.
    """
    def __init__(self, parent, label_text, placeholder="", initial_value="", width=300, height=60):
        super().__init__(
            parent, 
            width=width, 
            height=height, 
            bg=config.COLOR_BG_CARD, 
            highlightthickness=0
        )
        self.parent = parent
        self.placeholder = placeholder
        self.width = width
        self.height = height
        self.text_var = tk.StringVar(value=initial_value)
        self.is_placeholder = False
        
        # Draw Label
        self.create_text(
            10, 15, 
            text=label_text.upper(), 
            anchor="w", 
            font=(config.FONT_FAMILY, 8, "bold"), 
            fill=config.COLOR_NEON_CYAN
        )

        # Create Entry widget
        self.entry = tk.Entry(
            self, 
            textvariable=self.text_var,
            font=(config.FONT_FAMILY, 10),
            bg=config.COLOR_BG_VOID,  # Darker background for input
            fg=config.COLOR_TEXT_PRIMARY,
            insertbackground=config.COLOR_NEON_CYAN, # Cursor color
            relief=tk.FLAT,
            bd=0
        )
        
        # Place entry on canvas
        self.create_window(15, 40, window=self.entry, width=width-30, height=25, anchor="w")
        
        # Initial Draw
        self.draw_border(color=config.COLOR_TEXT_MUTED, width=1)
        
        # Bind Events for "Glow" effect
        self.entry.bind("<FocusIn>", lambda e: self.animate_focus(True))
        self.entry.bind("<FocusOut>", lambda e: self.animate_focus(False))
        
        # Add placeholder logic
        if not initial_value and placeholder:
            self.entry.insert(0, placeholder)
            self.entry.config(fg=config.COLOR_TEXT_MUTED)
            self.is_placeholder = True
            self.entry.bind("<FocusIn>", self._on_focus_in, add="+")
            self.entry.bind("<FocusOut>", self._on_focus_out, add="+")

    def draw_border(self, color, width=1):
        """Draws the rounded rectangle border"""
        self.delete("border")
        x1, y1, x2, y2 = 2, 28, self.width-2, self.height-2
        
        points = [
            x1, y2-10, x1, y2, x2, y2, x2, y2-10
        ]
        self.create_line(points, fill=color, width=width, tags="border", capstyle=tk.ROUND)
        
        # Top-left indicator
        self.create_line(x1, y1+10, x1, y1, x1+10, y1, fill=color, width=width, tags="border")
        # Top-right indicator
        self.create_line(x2-10, y1, x2, y1, x2, y1+10, fill=color, width=width, tags="border")

    def animate_focus(self, focused):
        """Changes border color on focus"""
        color = config.COLOR_NEON_CYAN if focused else config.COLOR_TEXT_MUTED
        width = 2 if focused else 1
        self.draw_border(color, width)

    def _on_focus_in(self, event):
        if self.is_placeholder:
            self.entry.delete(0, tk.END)
            self.entry.config(fg=config.COLOR_TEXT_PRIMARY)
            self.is_placeholder = False
        self.animate_focus(True)

    def _on_focus_out(self, event):
        if not self.entry.get():
            self.entry.insert(0, self.placeholder)
            self.entry.config(fg=config.COLOR_TEXT_MUTED)
            self.is_placeholder = True
        self.animate_focus(False)

    def get(self):
        if self.is_placeholder:
            return ""
        return self.entry.get()


class CyberRealtimeView(tk.Frame):
    """
    Cyberpunk real-time processing view
    Features: Manual data entry, live feature calculation, prediction interface
    """
    
    def __init__(self, parent, controller, switch_to_batch: Callable, show_results: Callable):
        super().__init__(parent, bg=config.COLOR_BG_VOID)
        
        self.controller = controller
        self.switch_to_batch = switch_to_batch
        self.show_results = show_results
        self.inputs: Dict[str, ModernInput] = {}
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the Cyberpunk Dashboard UI"""
        
        # --- HEADER SECTION ---
        header = tk.Frame(self, bg=config.COLOR_BG_VOID)
        header.pack(fill=tk.X, padx=40, pady=(30, 20))
        
        # Title Stack (Left)
        title_frame = tk.Frame(header, bg=config.COLOR_BG_VOID)
        title_frame.pack(side=tk.LEFT)
        
        tk.Label(
            title_frame, 
            text="LIVE PREDICTION // DASHBOARD", 
            font=(config.FONT_FAMILY_DISPLAY, 24, "bold"),
            bg=config.COLOR_BG_VOID, 
            fg=config.COLOR_TEXT_PRIMARY
        ).pack(anchor="w")
        
        tk.Label(
            title_frame, 
            text="MANUAL DATA ENTRY TERMINAL", 
            font=(config.FONT_FAMILY, 10, "bold"),
            bg=config.COLOR_BG_VOID, 
            fg=config.COLOR_NEON_BLUE
        ).pack(anchor="w")

        # BACK BUTTON (Right)
        btn_back = NeonButton(
            header, 
            text="RETURN TO BATCH", 
            icon="↩",
            command=self.switch_to_batch,
            width=220, 
            height=50,
            neon_color=config.COLOR_NEON_PINK
        )
        btn_back.pack(side=tk.RIGHT)

        # --- SCROLLABLE AREA ---
        canvas = tk.Canvas(self, bg=config.COLOR_BG_VOID, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        
        self.scroll_frame = tk.Frame(canvas, bg=config.COLOR_BG_VOID)
        
        self.scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        window_id = canvas.create_window((0, 0), window=self.scroll_frame, anchor="n")
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(window_id, width=e.width))
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=0)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # --- BALANCED 2-COLUMN GRID ---
        content_container = tk.Frame(self.scroll_frame, bg=config.COLOR_BG_VOID)
        content_container.pack(fill=tk.BOTH, expand=True, padx=40, pady=20)
        
        # Left Column
        left_col = tk.Frame(content_container, bg=config.COLOR_BG_VOID)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Right Column
        right_col = tk.Frame(content_container, bg=config.COLOR_BG_VOID)
        right_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        # === LEFT COLUMN: 2 Cards ===
        self._create_card(left_col, "TRIP IDENTIFICATION", [
            ("Booking ID", "bookingID", "TRIP-2026-X"),
            ("Trip Duration (sec)", "trip_duration_sec", "1200"),
        ])
        
        tk.Frame(left_col, bg=config.COLOR_BG_VOID, height=15).pack()  # Spacer
        
        self._create_card(left_col, "SPEED METRICS", [
            ("Mean Speed (m/s)", "speed_mean", "15.5"),
            ("Max Speed (m/s)", "speed_max", "28.0"),
            ("Pct Time Cruising", "pct_time_cruising", "0.45"),
        ])

        # === RIGHT COLUMN: 2 Cards ===
        self._create_card(right_col, "DRIVING DYNAMICS", [
            ("Pct High Accel", "pct_time_high_accel", "0.05"),
            ("Mean Linear Jerk", "jerk_linear_mean", "0.8"),
            ("Longest Smooth Seg", "longest_smooth_segment_sec", "45.0"),
        ])
        
        tk.Frame(right_col, bg=config.COLOR_BG_VOID, height=15).pack()  # Spacer
        
        self._create_card(right_col, "SENSOR DATA", [
            ("Max Accel Mag", "accel_mag_max", "12.5"),
            ("Count Hard Accels", "n_hard_accels", "3"),
            ("Max Gyro Mag", "gyro_mag_max", "0.8"),
            ("Std Dev Accel", "accel_mag_std", "1.2"),
            ("Std Dev Gyro", "gyro_mag_std", "0.1"),
        ])

        # --- PREDICT BUTTON (Bottom of scroll frame) ---
        btn_container = tk.Frame(self.scroll_frame, bg=config.COLOR_BG_VOID, pady=40)
        btn_container.pack(fill=tk.X)
        
        self.predict_btn = NeonButton(
            btn_container,
            text="INITIALIZE PREDICTION",
            command=self._on_predict,
            neon_color=config.COLOR_NEON_GREEN,
            width=300,
            height=60,
            icon="⚡"
        )
        self.predict_btn.pack()

    def _create_card(self, parent, title, fields):
        """Creates a stylized 'card' group"""
        card = tk.Frame(parent, bg=config.COLOR_BG_CARD, padx=20, pady=20)
        card.pack(fill=tk.BOTH, expand=True)
        
        # Tech decoration line
        tk.Frame(card, bg=config.COLOR_NEON_CYAN, height=2).pack(fill=tk.X, pady=(0, 15))

        # Title
        tk.Label(
            card, 
            text=title, 
            font=(config.FONT_FAMILY, 9, "bold"),
            fg=config.COLOR_NEON_CYAN, 
            bg=config.COLOR_BG_CARD, 
            anchor="w"
        ).pack(fill=tk.X, pady=(0, 15))

        # Fields
        for label, key, placeholder in fields:
            inp = ModernInput(card, label, placeholder=placeholder, width=350)
            inp.pack(pady=2, fill=tk.X)
            self.inputs[key] = inp

    def _on_predict(self):
        """Gather data, animate, and predict"""
        try:
            # Loading effect
            self.predict_btn.set_disabled(True)
            self.update_idletasks()
            
            raw_data = {}
            # Gather Inputs
            for key, widget in self.inputs.items():
                val = widget.get().strip()
                
                # Check empty
                if not val and key != "bookingID":
                    self.predict_btn.set_disabled(False)
                    messagebox.showerror("Input Error", f"Missing value for {key}")
                    return
                
                if key != "bookingID":
                    try:
                        raw_data[key] = float(val)
                    except ValueError:
                        self.predict_btn.set_disabled(False)
                        messagebox.showerror("Type Error", f"{key} must be a number.")
                        return
                else:
                    raw_data[key] = val if val else "MANUAL_ENTRY"

            # Logic Calculations (interaction features)
            speed_mean = raw_data.get('speed_mean', 1.0)
            if speed_mean == 0: speed_mean = 0.001
            
            raw_data['turn_sharpness_index'] = raw_data['gyro_mag_max'] / speed_mean
            raw_data['accel_risk_score'] = raw_data['accel_mag_max'] * raw_data['n_hard_accels']
            raw_data['gyro_accel_instability'] = raw_data['gyro_mag_std'] * raw_data['accel_mag_std']

            # Simulate delay then predict
            self.after(500, lambda: self._finalize_prediction(raw_data))

        except Exception as e:
            self.predict_btn.set_disabled(False)
            messagebox.showerror("System Error", str(e))

    def _finalize_prediction(self, raw_data):
        """Execute prediction via controller and show results"""
        try:
            result = self.controller.process_realtime_data(raw_data)
            self.show_results(result)
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))
        finally:
            # Reset button state
            self.predict_btn.set_disabled(False)