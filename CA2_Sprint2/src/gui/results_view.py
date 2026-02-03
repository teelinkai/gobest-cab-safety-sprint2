"""
📊 CYBERPUNK RESULTS VIEW 📊
Epic data visualization with neon charts and export functionality
"""

import tkinter as tk
from tkinter import messagebox, filedialog
from typing import Callable
from datetime import datetime
from pathlib import Path

from .. import config
from .cyber_components import NeonButton, CyberCard


class CyberResultsView(tk.Frame):
    """
    🌟 CYBERPUNK RESULTS DISPLAY 🌟
    Beautiful visualization of prediction results
    """
    
    def __init__(self, parent, controller, back_to_batch: Callable, back_to_realtime: Callable):
        super().__init__(parent, bg=config.COLOR_BG_VOID)
        
        self.controller = controller
        self.back_to_batch = back_to_batch
        self.back_to_realtime = back_to_realtime
        self.current_data = None
        
        self._create_ui()
    
    def _create_ui(self):
        """Create results UI"""
        # Header
        self._create_header()
        
        # Stats container
        self.stats_container = tk.Frame(self, bg=config.COLOR_BG_VOID)
        self.stats_container.pack(fill=tk.BOTH, expand=True, pady=20)
        
        # Action buttons
        self._create_action_buttons()
    
    def _create_header(self):
        """Create header section"""
        header = tk.Frame(self, bg=config.COLOR_BG_VOID)
        header.pack(fill=tk.X, pady=(0, 30))
        
        tk.Label(
            header,
            text="📊 ANALYSIS RESULTS",
            font=(config.FONT_FAMILY_DISPLAY, 32, "bold"),
            fg=config.COLOR_NEON_GREEN,
            bg=config.COLOR_BG_VOID
        ).pack()
        
        tk.Label(
            header,
            text="NEURAL PREDICTION COMPLETE",
            font=(config.FONT_FAMILY, 11),
            fg=config.COLOR_TEXT_SECONDARY,
            bg=config.COLOR_BG_VOID
        ).pack()
    
    def _create_action_buttons(self):
        """Create action buttons"""
        btn_frame = tk.Frame(self, bg=config.COLOR_BG_VOID)
        btn_frame.pack(pady=20)
        
        self.export_btn = NeonButton(
            btn_frame,
            text="EXPORT RESULTS",
            command=self._export_results,
            neon_color=config.COLOR_NEON_CYAN,
            width=200,
            height=50,
            icon="💾"
        )
        self.export_btn.pack(side=tk.LEFT, padx=10)
        
        self.back_btn = NeonButton(
            btn_frame,
            text="NEW ANALYSIS",
            command=self.back_to_batch,
            neon_color=config.COLOR_NEON_PINK,
            width=200,
            height=50,
            icon="↩"
        )
        self.back_btn.pack(side=tk.LEFT, padx=10)
    
    def display_results(self, prediction_data: dict):
        """Display prediction results"""
        self.current_data = prediction_data
        
        # Clear previous
        for widget in self.stats_container.winfo_children():
            widget.destroy()
        
        # Create stats grid
        grid = tk.Frame(self.stats_container, bg=config.COLOR_BG_VOID)
        grid.pack(expand=True)
        
        # Top row - main stats
        top_row = tk.Frame(grid, bg=config.COLOR_BG_VOID)
        top_row.pack(pady=20)
        
        self._create_stat_card(
            top_row,
            "TOTAL TRIPS",
            str(prediction_data['total_trips']),
            config.COLOR_NEON_CYAN,
            "📊"
        ).pack(side=tk.LEFT, padx=15)
        
        self._create_stat_card(
            top_row,
            "FILES PROCESSED",
            str(prediction_data.get('num_files', 1)),
            config.COLOR_NEON_PURPLE,
            "📁"
        ).pack(side=tk.LEFT, padx=15)
        
        # Bottom row - safety stats
        bottom_row = tk.Frame(grid, bg=config.COLOR_BG_VOID)
        bottom_row.pack(pady=20)
        
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
        
        # Confidence stats
        confidence_row = tk.Frame(grid, bg=config.COLOR_BG_VOID)
        confidence_row.pack(pady=20)
        
        self._create_stat_card(
            confidence_row,
            "AVG CONFIDENCE (DANGEROUS)",
            f"{prediction_data['avg_confidence_dangerous']:.1%}",
            config.COLOR_NEON_ORANGE,
            "🧠"
        ).pack(side=tk.LEFT, padx=15)
        
        self._create_stat_card(
            confidence_row,
            "AVG CONFIDENCE (SAFE)",
            f"{prediction_data['avg_confidence_safe']:.1%}",
            config.COLOR_NEON_GREEN,
            "🛡"
        ).pack(side=tk.LEFT, padx=15)
    
    def _create_stat_card(self, parent, label: str, value: str, color: str, icon: str):
        """Create a stat display card"""
        card_frame = tk.Frame(parent, bg=config.COLOR_BG_VOID)
        
        # Card canvas
        card = tk.Canvas(
            card_frame,
            width=250,
            height=120,
            bg=config.COLOR_BG_VOID,
            highlightthickness=0
        )
        card.pack()
        
        # Background
        self._draw_rounded_rect(
            card, 5, 5, 245, 115, 12,
            fill=config.COLOR_BG_CARD,
            outline=""
        )
        
        # Neon border
        self._draw_rounded_rect(
            card, 5, 5, 245, 115, 12,
            fill="",
            outline=color,
            width=2
        )
        
        # Glow effect (Fixed transparency crash)
        for i in range(3):
            self._draw_rounded_rect(
                card, 3-i, 3-i, 247+i, 117+i, 12,
                fill="",
                outline=color,
                stipple="gray25",  # Use stipple instead of hex alpha
                width=1
            )
        
        # Icon
        card.create_text(
            30, 35,
            text=icon,
            font=(config.FONT_FAMILY, 24),
            fill=color
        )
        
        # Label
        card.create_text(
            125, 30,
            text=label,
            font=(config.FONT_FAMILY, 9),
            fill=config.COLOR_TEXT_SECONDARY
        )
        
        # Value
        card.create_text(
            125, 70,
            text=value,
            font=(config.FONT_FAMILY, 20, "bold"),
            fill=color
        )
        
        return card_frame
    
    def _draw_rounded_rect(self, canvas, x1, y1, x2, y2, radius, **kwargs):
        """Draw rounded rectangle on canvas"""
        points = [
            x1+radius, y1,
            x1+radius, y1,
            x2-radius, y1,
            x2-radius, y1,
            x2, y1,
            x2, y1+radius,
            x2, y1+radius,
            x2, y2-radius,
            x2, y2-radius,
            x2, y2,
            x2-radius, y2,
            x2-radius, y2,
            x1+radius, y2,
            x1+radius, y2,
            x1, y2,
            x1, y2-radius,
            x1, y2-radius,
            x1, y1+radius,
            x1, y1+radius,
            x1, y1
        ]
        return canvas.create_polygon(points, smooth=True, **kwargs)
    
    def _export_results(self):
        """Export results to CSV"""
        if not self.current_data:
            return
        
        # Ask for save location
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"Safety_Predictions_{timestamp}.csv"
        
        file_path = filedialog.asksaveasfilename(
            title="SAVE RESULTS",
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
            initialfile=default_name
        )
        
        if not file_path:
            return
        
        try:
            # Export DataFrame
            results_df = self.current_data['results_df']
            results_df.to_csv(file_path, index=False)
            
            messagebox.showinfo(
                "SUCCESS",
                f"Results exported successfully!\n\n{Path(file_path).name}"
            )
        except Exception as e:
            messagebox.showerror(
                "ERROR",
                f"Failed to export results:\n\n{str(e)}"
            )