"""
Batch Mode View Module - PREMIUM CYBERPUNK EDITION
High-tech interface for bulk data processing with full page scrolling
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
from typing import Callable, List
import threading

from .. import config

# --- THEME CONSTANTS ---
COLOR_BG = "#0f172a"          # Deep Slate
COLOR_CARD = "#1e293b"        # Lighter Slate
COLOR_ACCENT = "#38bdf8"      # Sky Blue
COLOR_ACCENT_DIM = "#0ea5e9"  # Darker Blue
COLOR_SUCCESS = "#10b981"     # Emerald Green
COLOR_WARNING = "#f59e0b"     # Amber
COLOR_DANGER = "#ef4444"      # Red
COLOR_TEXT_MAIN = "#f1f5f9"
COLOR_TEXT_MUTED = "#94a3b8"

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
        self.create_rectangle(2, 2, self.width-2, self.height-2, fill=color, outline=color, tags="bg")
        self.create_polygon(
            0, 10, 10, 0, self.width, 0, self.width, self.height-10, self.width-10, self.height, 0, self.height,
            fill=color, outline=""
        )
        
        display_text = f"{self.icon}  {self.text}" if self.icon else self.text
        self.create_text(self.width/2, self.height/2, text=display_text, 
                         font=(config.FONT_FAMILY, 11, "bold"), fill="white")

    def _on_enter(self, e):
        if not self.is_disabled: self._draw(self.hover_color)

    def _on_leave(self, e):
        if not self.is_disabled: self._draw(self.bg_color)

    def _on_click(self, e):
        if not self.is_disabled: self.command()

    def set_disabled(self, disabled):
        self.is_disabled = disabled
        color = COLOR_CARD if disabled else self.bg_color
        self._draw(color)


class BatchModeView(tk.Frame):
    def __init__(self, parent, controller, switch_to_realtime: Callable, show_results: Callable):
        super().__init__(parent, bg=COLOR_BG)
        self.controller = controller
        self.switch_to_realtime = switch_to_realtime
        self.show_results = show_results
        self.selected_files: List[Path] = []
        self.is_processing = False
        
        self._setup_ui()
        
    def _setup_ui(self):
        # --- 1. MAIN SCROLLABLE CONTAINER ---
        # This ensures the whole page scrolls when minimized
        main_canvas = tk.Canvas(self, bg=COLOR_BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=main_canvas.yview)
        
        # The frame that holds all content
        self.content_frame = tk.Frame(main_canvas, bg=COLOR_BG)
        
        # Configure scrolling
        self.content_frame.bind(
            "<Configure>", 
            lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all"))
        )
        
        window_id = main_canvas.create_window((0, 0), window=self.content_frame, anchor="nw")
        
        # Auto-width
        main_canvas.bind("<Configure>", lambda e: main_canvas.itemconfig(window_id, width=e.width))
        main_canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack Layout
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        main_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # --- 2. HEADER SECTION ---
        header_frame = tk.Frame(self.content_frame, bg=COLOR_BG)
        header_frame.pack(fill=tk.X, padx=40, pady=(30, 20))

        # Title Stack
        title_col = tk.Frame(header_frame, bg=COLOR_BG)
        title_col.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        tk.Label(title_col, text="BATCH PROCESSOR // CORE", 
                 font=(config.FONT_FAMILY, 28, "bold"),
                 bg=COLOR_BG, fg="white").pack(anchor="w")
        
        tk.Label(title_col, text="LARGE DATASET INGESTION & ANALYSIS SYSTEM", 
                 font=(config.FONT_FAMILY, 10, "bold"),
                 bg=COLOR_BG, fg=COLOR_TEXT_MUTED).pack(anchor="w")

        # Switch Button
        self.switch_btn = CyberButton(
            header_frame, 
            text="SWITCH TO REAL-TIME", 
            icon="⚡",
            command=self.switch_to_realtime,
            width=220, height=50,
            bg_color=COLOR_CARD
        )
        self.switch_btn.pack(side=tk.RIGHT)

        # --- 3. DATA INGESTION PORT (Redesigned) ---
        ingest_container = tk.Frame(self.content_frame, bg=COLOR_BG)
        ingest_container.pack(fill=tk.X, padx=40, pady=10)

        # The Card
        self.drop_zone = tk.Frame(ingest_container, bg=COLOR_CARD, bd=1, relief="solid")
        self.drop_zone.pack(fill=tk.X, ipady=20)
        
        # Decoration Line
        tk.Frame(self.drop_zone, bg=COLOR_ACCENT, height=2).pack(fill=tk.X, side=tk.TOP)

        # Center Content
        center_frame = tk.Frame(self.drop_zone, bg=COLOR_CARD)
        center_frame.pack(expand=True)

        tk.Label(center_frame, text="DATA INGESTION PORT", 
                 font=(config.FONT_FAMILY, 16, "bold"), 
                 bg=COLOR_CARD, fg=COLOR_ACCENT).pack(pady=(10, 5))
        
        tk.Label(center_frame, text="Select .CSV sensor logs for bulk processing", 
                 font=(config.FONT_FAMILY, 10), 
                 bg=COLOR_CARD, fg=COLOR_TEXT_MUTED).pack(pady=(0, 20))
        
        # Styled Button
        self.btn_add = tk.Button(
            center_frame, text="+ SELECT FILES", 
            bg=COLOR_ACCENT_DIM, fg="white", 
            font=(config.FONT_FAMILY, 11, "bold"),
            relief="flat", padx=30, pady=10, cursor="hand2",
            activebackground="white", activeforeground=COLOR_ACCENT_DIM,
            command=self._add_files
        )
        self.btn_add.pack(pady=5)

        # --- 4. FILE QUEUE ---
        queue_frame = tk.Frame(self.content_frame, bg=COLOR_BG)
        queue_frame.pack(fill=tk.BOTH, expand=True, padx=40, pady=20)

        tk.Label(queue_frame, text="QUEUED DATASETS:", 
                 font=(config.FONT_FAMILY, 10, "bold"), 
                 bg=COLOR_BG, fg=COLOR_TEXT_MUTED).pack(anchor="w", pady=(0, 5))

        # Inner list container (Non-scrollable, just a stack since outer page scrolls)
        self.file_list_frame = tk.Frame(queue_frame, bg=COLOR_BG)
        self.file_list_frame.pack(fill=tk.BOTH, expand=True)
        
        # Empty State
        self.empty_msg = tk.Label(self.file_list_frame, text="NO DATA LOADED // WAITING FOR INPUT", 
                                  bg=COLOR_BG, fg="#334155", font=(config.FONT_FAMILY, 12, "bold"))
        self.empty_msg.pack(pady=40)

        # --- 5. ACTION FOOTER ---
        footer = tk.Frame(self.content_frame, bg=COLOR_BG)
        footer.pack(fill=tk.X, padx=40, pady=30)
        
        # Stats
        self.stats_label = tk.Label(footer, text="READY TO PROCESS", 
                                    font=(config.FONT_FAMILY, 12), bg=COLOR_BG, fg=COLOR_TEXT_MUTED)
        self.stats_label.pack(side=tk.LEFT)
        
        # Clear Button
        self.clear_btn = tk.Button(footer, text="CLEAR QUEUE", 
                                   command=self._clear_files,
                                   bg=COLOR_BG, fg=COLOR_DANGER, bd=0, 
                                   font=(config.FONT_FAMILY, 10, "bold"), cursor="hand2")
        self.clear_btn.pack(side=tk.LEFT, padx=20)

        # Execute Button
        self.process_btn = CyberButton(
            footer, text="INITIATE ANALYSIS", 
            command=self._process_files,
            width=300, height=60,
            bg_color=COLOR_SUCCESS,
            icon="▶"
        )
        self.process_btn.pack(side=tk.RIGHT)

        # Progress Bar
        self.progress_bar = ttk.Progressbar(self.content_frame, orient="horizontal", mode="determinate")

    def _add_files(self):
        if self.is_processing: return
        file_paths = filedialog.askopenfilenames(
            title="Select Sensor Logs", filetypes=[("CSV Files", "*.csv")]
        )
        if not file_paths: return
        
        existing = {str(p.resolve()) for p in self.selected_files}
        for p in file_paths:
            path_obj = Path(p)
            if str(path_obj.resolve()) not in existing:
                self.selected_files.append(path_obj)
        
        self._update_file_list()

    def _update_file_list(self):
        for w in self.file_list_frame.winfo_children(): w.destroy()
        
        if not self.selected_files:
            self.empty_msg = tk.Label(self.file_list_frame, text="NO DATA LOADED // WAITING FOR INPUT", 
                                      bg=COLOR_BG, fg="#334155", font=(config.FONT_FAMILY, 12, "bold"))
            self.empty_msg.pack(pady=40)
            self.stats_label.config(text="READY TO PROCESS")
            return

        for i, f in enumerate(self.selected_files):
            card = tk.Frame(self.file_list_frame, bg=COLOR_CARD, pady=10, padx=15)
            card.pack(fill=tk.X, pady=2)
            
            tk.Label(card, text="📄", bg=COLOR_CARD, fg="white", font=("Arial", 14)).pack(side=tk.LEFT)
            
            info_frame = tk.Frame(card, bg=COLOR_CARD)
            info_frame.pack(side=tk.LEFT, padx=15)
            tk.Label(info_frame, text=f.name, bg=COLOR_CARD, fg="white", font=(config.FONT_FAMILY, 10, "bold")).pack(anchor="w")
            try:
                size_txt = f"{f.stat().st_size / 1024:.1f} KB"
            except:
                size_txt = "Unknown Size"
            tk.Label(info_frame, text=size_txt, bg=COLOR_CARD, fg=COLOR_TEXT_MUTED, font=(config.FONT_FAMILY, 8)).pack(anchor="w")
            
            tk.Label(card, text="QUEUED", bg=COLOR_CARD, fg=COLOR_ACCENT, font=(config.FONT_FAMILY, 8, "bold")).pack(side=tk.RIGHT)

        total_mb = sum(f.stat().st_size for f in self.selected_files) / (1024*1024)
        self.stats_label.config(text=f"{len(self.selected_files)} FILES SELECTED ({total_mb:.2f} MB)")

    def _clear_files(self):
        if self.is_processing: return
        self.selected_files = []
        self._update_file_list()

    def _process_files(self):
        if not self.selected_files:
            messagebox.showwarning("No Data", "Please inject data files first.")
            return
        
        self.is_processing = True
        self.process_btn.set_disabled(True)
        self.switch_btn.set_disabled(True)
        self.btn_add.config(state="disabled")
        self.clear_btn.config(state="disabled")
        
        self.progress_bar.pack(fill=tk.X, side=tk.BOTTOM, pady=10)
        self.progress_bar.start(10)
        
        thread = threading.Thread(target=self._run_process, daemon=True)
        thread.start()

    def _run_process(self):
        try:
            data = self.controller.process_batch_files([str(f) for f in self.selected_files])
            self.after(0, lambda: self._finish_process(data))
        except Exception as e:
            self.after(0, lambda: self._error_process(str(e)))

    def _finish_process(self, data):
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.is_processing = False
        self.process_btn.set_disabled(False)
        self.switch_btn.set_disabled(False)
        self.btn_add.config(state="normal")
        self.clear_btn.config(state="normal")
        self.show_results(data)

    def _error_process(self, msg):
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.is_processing = False
        self.process_btn.set_disabled(False)
        self.switch_btn.set_disabled(False)
        self.btn_add.config(state="normal")
        self.clear_btn.config(state="normal")
        messagebox.showerror("Processing Error", msg)