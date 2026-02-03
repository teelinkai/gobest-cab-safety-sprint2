"""
🗂️ CYNOSURE REAL-TIME TERMINAL 🗂️
Manual data entry for single trip analysis
"""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Callable, Dict

from .. import config
from .cyber_components import RetroButton


class RetroInput(tk.Canvas):
    """Retro terminal input field"""
    
    def __init__(self, parent, label_text, placeholder="", width=300, height=60):
        super().__init__(
            parent,
            width=width,
            height=height,
            bg=config.COLOR_BG_CARD,
            highlightthickness=0
        )
        
        self.placeholder = placeholder
        self.width = width
        self.height = height
        self.is_placeholder = False
        
        # Label
        self.create_text(
            10, 15,
            text=label_text.upper(),
            anchor="w",
            font=(config.FONT_FAMILY, 8, "bold"),
            fill=config.COLOR_TEXT_SECONDARY
        )
        
        # Entry field
        self.entry = tk.Entry(
            self,
            font=(config.FONT_FAMILY, 10),
            bg=config.COLOR_BG_PAPER,
            fg=config.COLOR_TEXT_PRIMARY,
            insertbackground=config.COLOR_ACCENT_BLUE,
            relief=tk.FLAT,
            bd=2
        )
        
        self.create_window(15, 40, window=self.entry, width=width-30, height=25, anchor="w")
        
        # Border
        self.draw_border(config.COLOR_BORDER_LIGHT, 1)
        
        # Focus events
        self.entry.bind("<FocusIn>", lambda e: self.draw_border(config.COLOR_ACCENT_BLUE, 2))
        self.entry.bind("<FocusOut>", lambda e: self.draw_border(config.COLOR_BORDER_LIGHT, 1))
        
        # Placeholder
        if placeholder:
            self.entry.insert(0, placeholder)
            self.entry.config(fg=config.COLOR_TEXT_MUTED)
            self.is_placeholder = True
            self.entry.bind("<FocusIn>", self._on_focus_in, add="+")
            self.entry.bind("<FocusOut>", self._on_focus_out, add="+")
    
    def draw_border(self, color, width=1):
        """Draw border"""
        self.delete("border")
        self.create_rectangle(
            2, 28, self.width-2, self.height-2,
            outline=color,
            width=width,
            tags="border"
        )
    
    def _on_focus_in(self, event):
        if self.is_placeholder:
            self.entry.delete(0, tk.END)
            self.entry.config(fg=config.COLOR_TEXT_PRIMARY)
            self.is_placeholder = False
    
    def _on_focus_out(self, event):
        if not self.entry.get():
            self.entry.insert(0, self.placeholder)
            self.entry.config(fg=config.COLOR_TEXT_MUTED)
            self.is_placeholder = True
    
    def get(self):
        """Get value"""
        if self.is_placeholder:
            return ""
        return self.entry.get()


# Alias for compatibility
ModernInput = RetroInput


class CyberRealtimeView(tk.Frame):
    """
    CYNOSURE REAL-TIME TERMINAL
    Manual data entry interface
    """
    
    def __init__(self, parent, controller, switch_to_batch: Callable, show_results: Callable):
        super().__init__(parent, bg=config.COLOR_BG_PAPER)
        
        self.controller = controller
        self.switch_to_batch = switch_to_batch
        self.show_results = show_results
        self.inputs: Dict[str, RetroInput] = {}
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup UI"""
        
        # ===== HEADER =====
        header = tk.Frame(self, bg=config.COLOR_BG_PAPER)
        header.pack(fill=tk.X, padx=40, pady=(30, 20))

        # 👁️ BILL CIPHER LOGO
        from .cyber_components import BillCipherLogo
        logo = BillCipherLogo(header, size=90)
        logo.pack(side=tk.LEFT, padx=(0, 15))

        # Title
        title_frame = tk.Frame(header, bg=config.COLOR_BG_PAPER)
        title_frame.pack(side=tk.LEFT)
        
        tk.Label(
            title_frame,
            text="REAL-TIME PREDICTION TERMINAL",
            font=(config.FONT_FAMILY_DISPLAY, 20, "bold"),
            bg=config.COLOR_BG_PAPER,
            fg=config.COLOR_TEXT_PRIMARY
        ).pack(anchor="w")
        
        tk.Label(
            title_frame,
            text="MANUAL DATA ENTRY // SINGLE TRIP ANALYSIS",
            font=(config.FONT_FAMILY, 9),
            bg=config.COLOR_BG_PAPER,
            fg=config.COLOR_TEXT_SECONDARY
        ).pack(anchor="w")
        
        # Back button
        back_btn = RetroButton(
            header,
            text="← RETURN TO BATCH",
            command=self.switch_to_batch,
            width=180,
            height=45
        )
        back_btn.pack(side=tk.RIGHT)
        
        # ===== SCROLLABLE CONTENT =====
        canvas = tk.Canvas(self, bg=config.COLOR_BG_PAPER, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        
        scroll_frame = tk.Frame(canvas, bg=config.COLOR_BG_PAPER)
        
        scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        window_id = canvas.create_window((0, 0), window=scroll_frame, anchor="n")
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(window_id, width=e.width))
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Mouse wheel
        def on_wheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        scroll_frame.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", on_wheel))
        scroll_frame.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))
        
        # ===== CONTENT GRID =====
        content = tk.Frame(scroll_frame, bg=config.COLOR_BG_PAPER)
        content.pack(fill=tk.BOTH, expand=True, padx=40, pady=20)
        
        # Left column
        left_col = tk.Frame(content, bg=config.COLOR_BG_PAPER)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Right column
        right_col = tk.Frame(content, bg=config.COLOR_BG_PAPER)
        right_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        # Create cards
        self._create_card(left_col, "TRIP IDENTIFICATION", [
            ("Booking ID", "bookingID", "TRIP-2026-X"),
            ("Trip Duration (sec)", "trip_duration_sec", "1200"),
        ])
        
        tk.Frame(left_col, bg=config.COLOR_BG_PAPER, height=15).pack()
        
        self._create_card(left_col, "SPEED METRICS", [
            ("Mean Speed (m/s)", "speed_mean", "15.5"),
            ("Max Speed (m/s)", "speed_max", "28.0"),
            ("Pct Time Cruising", "pct_time_cruising", "0.45"),
        ])
        
        self._create_card(right_col, "DRIVING DYNAMICS", [
            ("Pct High Accel", "pct_time_high_accel", "0.05"),
            ("Mean Linear Jerk", "jerk_linear_mean", "0.8"),
            ("Longest Smooth Seg", "longest_smooth_segment_sec", "45.0"),
        ])
        
        tk.Frame(right_col, bg=config.COLOR_BG_PAPER, height=15).pack()
        
        self._create_card(right_col, "SENSOR DATA", [
            ("Max Accel Mag", "accel_mag_max", "12.5"),
            ("Count Hard Accels", "n_hard_accels", "3"),
            ("Max Gyro Mag", "gyro_mag_max", "0.8"),
            ("Std Dev Accel", "accel_mag_std", "1.2"),
            ("Std Dev Gyro", "gyro_mag_std", "0.1"),
        ])
        
        # ===== PREDICT BUTTON =====
        btn_container = tk.Frame(scroll_frame, bg=config.COLOR_BG_PAPER, pady=40)
        btn_container.pack(fill=tk.X)
        
        self.predict_btn = RetroButton(
            btn_container,
            text="▶ INITIATE PREDICTION",
            command=self._on_predict,
            width=280,
            height=55
        )
        self.predict_btn.pack()
    
    def _create_card(self, parent, title, fields):
        """Create input card"""
        card = tk.Frame(
            parent,
            bg=config.COLOR_BG_CARD,
            highlightbackground=config.COLOR_BORDER_DARK,
            highlightthickness=2
        )
        card.pack(fill=tk.BOTH, expand=True)
        
        # Top line
        tk.Frame(card, bg=config.COLOR_ACCENT_BLUE, height=3).pack(fill=tk.X)
        
        # Title
        tk.Label(
            card,
            text=title,
            font=(config.FONT_FAMILY, 10, "bold"),
            fg=config.COLOR_TEXT_PRIMARY,
            bg=config.COLOR_BG_CARD
        ).pack(pady=(15, 10))
        
        # Fields
        for label, key, placeholder in fields:
            inp = RetroInput(card, label, placeholder=placeholder, width=350)
            inp.pack(pady=5, padx=15)
            self.inputs[key] = inp
        
        # Bottom padding
        tk.Frame(card, bg=config.COLOR_BG_CARD, height=15).pack()
    
    def _on_predict(self):
        """Make prediction"""
        try:
            self.predict_btn.set_disabled(True)
            self.update_idletasks()
            
            raw_data = {}
            
            # Gather inputs
            for key, widget in self.inputs.items():
                val = widget.get().strip()
                
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
            
            # Calculate interaction features
            speed_mean = raw_data.get('speed_mean', 1.0)
            if speed_mean == 0:
                speed_mean = 0.001
            
            raw_data['turn_sharpness_index'] = raw_data['gyro_mag_max'] / speed_mean
            raw_data['accel_risk_score'] = raw_data['accel_mag_max'] * raw_data['n_hard_accels']
            raw_data['gyro_accel_instability'] = raw_data['gyro_mag_std'] * raw_data['accel_mag_std']
            
            # Predict
            self.after(300, lambda: self._finalize_prediction(raw_data))
            
        except Exception as e:
            self.predict_btn.set_disabled(False)
            messagebox.showerror("System Error", str(e))
    
    def _finalize_prediction(self, raw_data):
        """Execute prediction"""
        try:
            result = self.controller.process_realtime_data(raw_data)
            self.show_results(result)
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))
        finally:
            self.predict_btn.set_disabled(False)
