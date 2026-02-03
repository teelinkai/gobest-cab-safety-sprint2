"""
🗂️ CYNOSURE RESULTS TERMINAL 🗂️
Display prediction results with retro styling
"""

import tkinter as tk
from tkinter import messagebox, filedialog, ttk
from typing import Callable
from datetime import datetime
from pathlib import Path

from .. import config
from .cyber_components import RetroButton


class CyberResultsView(tk.Frame):
    """
    CYNOSURE RESULTS DISPLAY
    Shows prediction statistics and export functionality
    """
    
    def __init__(self, parent, controller, back_to_batch: Callable, back_to_realtime: Callable):
        super().__init__(parent, bg=config.COLOR_BG_PAPER)
        
        self.controller = controller
        self.back_to_batch = back_to_batch
        self.back_to_realtime = back_to_realtime
        self.current_data = None
        
        self._setup_main_scroll()
    
    def _setup_main_scroll(self):
        """Setup scrollable area"""
        self.main_canvas = tk.Canvas(self, bg=config.COLOR_BG_PAPER, highlightthickness=0)
        self.main_scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.main_canvas.yview)
        
        self.scroll_frame = tk.Frame(self.main_canvas, bg=config.COLOR_BG_PAPER)
        
        self.scroll_frame.bind(
            "<Configure>",
            lambda e: self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))
        )
        
        self.window_id = self.main_canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        
        self.main_canvas.bind(
            "<Configure>",
            lambda e: self.main_canvas.itemconfig(self.window_id, width=e.width)
        )
        
        self.main_canvas.configure(yscrollcommand=self.main_scrollbar.set)
        
        self.main_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.main_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Mouse wheel
        def on_wheel(event):
            self.main_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        self.scroll_frame.bind("<Enter>", lambda e: self.main_canvas.bind_all("<MouseWheel>", on_wheel))
        self.scroll_frame.bind("<Leave>", lambda e: self.main_canvas.unbind_all("<MouseWheel>"))
        
        self._create_ui(self.scroll_frame)
    
    def _create_ui(self, parent):
        """Create UI elements"""
        # Header
        self._create_header(parent)
        
        # Stats container
        self.stats_container = tk.Frame(parent, bg=config.COLOR_BG_PAPER)
        self.stats_container.pack(fill=tk.BOTH, expand=True, pady=20)
        
        # Action buttons
        self._create_action_buttons(parent)
    
    def _create_header(self, parent):
        header = tk.Frame(parent, bg=config.COLOR_BG_PAPER)
        header.pack(fill=tk.X, pady=(20, 30), padx=40)
        
        # 👁️ BILL CIPHER LOGO
        from .cyber_components import BillCipherLogo
        
        logo_title_frame = tk.Frame(header, bg=config.COLOR_BG_PAPER)
        logo_title_frame.pack()
        
        logo = BillCipherLogo(logo_title_frame, size=90)
        logo.pack(side=tk.LEFT, padx=(0, 15))
        
        title_container = tk.Frame(logo_title_frame, bg=config.COLOR_BG_PAPER)
        title_container.pack(side=tk.LEFT)
        
        tk.Label(
            title_container,
            text="📊 ANALYSIS RESULTS",
            font=(config.FONT_FAMILY_DISPLAY, 28, "bold"),
            fg=config.COLOR_TEXT_PRIMARY,
            bg=config.COLOR_BG_PAPER
        ).pack()
        
        tk.Label(
            header,
            text="PREDICTION COMPLETE // CYNOSURE SYSTEMS",
            font=(config.FONT_FAMILY, 10),
            fg=config.COLOR_TEXT_SECONDARY,
            bg=config.COLOR_BG_PAPER
        ).pack()
    
    def _create_action_buttons(self, parent):
        """Create action buttons"""
        btn_frame = tk.Frame(parent, bg=config.COLOR_BG_PAPER)
        btn_frame.pack(pady=40)
        
        self.export_btn = RetroButton(
            btn_frame,
            text="💾 EXPORT RESULTS",
            command=self._export_results,
            width=180,
            height=50
        )
        self.export_btn.pack(side=tk.LEFT, padx=10)
        
        self.back_btn = RetroButton(
            btn_frame,
            text="← NEW ANALYSIS",
            command=self.back_to_batch,
            width=180,
            height=50
        )
        self.back_btn.pack(side=tk.LEFT, padx=10)
    
    def display_results(self, prediction_data: dict):
        """Display results"""
        self.current_data = prediction_data
        
        # Clear previous stats
        for widget in self.stats_container.winfo_children():
            widget.destroy()
        
        # Create stats grid
        grid = tk.Frame(self.stats_container, bg=config.COLOR_BG_PAPER)
        grid.pack(expand=True)
        
        # Top row
        top_row = tk.Frame(grid, bg=config.COLOR_BG_PAPER)
        top_row.pack(pady=15)
        
        self._create_stat_card(
            top_row,
            "TOTAL TRIPS",
            str(prediction_data['total_trips']),
            config.COLOR_ACCENT_BLUE,
            "📊"
        ).pack(side=tk.LEFT, padx=15)
        
        self._create_stat_card(
            top_row,
            "FILES PROCESSED",
            str(prediction_data.get('num_files', 1)),
            config.COLOR_ACCENT_PURPLE,
            "📄"
        ).pack(side=tk.LEFT, padx=15)
        
        # Bottom row
        bottom_row = tk.Frame(grid, bg=config.COLOR_BG_PAPER)
        bottom_row.pack(pady=15)
        
        self._create_stat_card(
            bottom_row,
            "DANGEROUS",
            f"{prediction_data['dangerous_count']} ({prediction_data['dangerous_pct']:.1f}%)",
            config.COLOR_DANGER,
            "⚠"
        ).pack(side=tk.LEFT, padx=15)
        
        self._create_stat_card(
            bottom_row,
            "SAFE",
            f"{prediction_data['safe_count']} ({prediction_data['safe_pct']:.1f}%)",
            config.COLOR_SUCCESS,
            "✓"
        ).pack(side=tk.LEFT, padx=15)
        
        # Confidence row
        conf_row = tk.Frame(grid, bg=config.COLOR_BG_PAPER)
        conf_row.pack(pady=15)
        
        self._create_stat_card(
            conf_row,
            "AVG CONFIDENCE (DANGEROUS)",
            f"{prediction_data['avg_confidence_dangerous']:.1%}",
            config.COLOR_ACCENT_ORANGE,
            "🎯"
        ).pack(side=tk.LEFT, padx=15)
        
        self._create_stat_card(
            conf_row,
            "AVG CONFIDENCE (SAFE)",
            f"{prediction_data['avg_confidence_safe']:.1%}",
            config.COLOR_ACCENT_GREEN,
            "🛡"
        ).pack(side=tk.LEFT, padx=15)
        
        # Reset scroll
        self.main_canvas.yview_moveto(0)
    
    def _create_stat_card(self, parent, label: str, value: str, color: str, icon: str):
        """Create stat card"""
        card = tk.Frame(
            parent,
            bg=config.COLOR_BG_CARD,
            highlightbackground=color,
            highlightthickness=3,
            width=250,
            height=120
        )
        card.pack_propagate(False)
        
        # Icon
        tk.Label(
            card,
            text=icon,
            font=("Arial", 24),
            bg=config.COLOR_BG_CARD,
            fg=color
        ).pack(pady=(15, 5))
        
        # Label
        tk.Label(
            card,
            text=label,
            font=(config.FONT_FAMILY, 8),
            bg=config.COLOR_BG_CARD,
            fg=config.COLOR_TEXT_SECONDARY
        ).pack()
        
        # Value
        tk.Label(
            card,
            text=value,
            font=(config.FONT_FAMILY, 18, "bold"),
            bg=config.COLOR_BG_CARD,
            fg=color
        ).pack(pady=(5, 10))
        
        return card
    
    def _export_results(self):
        """Export results"""
        if not self.current_data:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"Safety_Predictions_{timestamp}.csv"
        
        file_path = filedialog.asksaveasfilename(
            title="EXPORT RESULTS",
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
            initialfile=default_name
        )
        
        if not file_path:
            return
        
        try:
            results_df = self.current_data['results_df']
            results_df.to_csv(file_path, index=False)
            
            messagebox.showinfo(
                "EXPORT SUCCESSFUL",
                f"Results exported successfully!\n\n{Path(file_path).name}"
            )
        except Exception as e:
            messagebox.showerror(
                "EXPORT ERROR",
                f"Failed to export results:\n\n{str(e)}"
            )
