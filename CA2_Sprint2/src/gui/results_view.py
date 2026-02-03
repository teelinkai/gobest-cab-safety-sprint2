"""
📊 CYBERPUNK RESULTS VIEW 📊
Epic data visualization with neon charts and export functionality
"""

import tkinter as tk
from tkinter import messagebox, filedialog, ttk
from typing import Callable
from datetime import datetime
from pathlib import Path

from .. import config
from .cyber_components import NeonButton, CyberCard


class CyberResultsView(tk.Frame):
    """
    🌟 CYBERPUNK RESULTS DISPLAY 🌟
    Beautiful visualization of prediction results (Now Scrollable!)
    """
    
    def __init__(self, parent, controller, back_to_batch: Callable, back_to_realtime: Callable):
        super().__init__(parent, bg=config.COLOR_BG_VOID)
        
        self.controller = controller
        self.back_to_batch = back_to_batch
        self.back_to_realtime = back_to_realtime
        self.current_data = None
        
        # Setup the scrollable wrapper immediately
        self._setup_main_scroll()
    
    def _setup_main_scroll(self):
        """Setup the main scrollable area"""
        # 1. Create Canvas & Scrollbar
        self.main_canvas = tk.Canvas(self, bg=config.COLOR_BG_VOID, highlightthickness=0)
        self.main_scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.main_canvas.yview)
        
        # 2. Create the Frame that will hold all the content
        self.scroll_frame = tk.Frame(self.main_canvas, bg=config.COLOR_BG_VOID)
        
        # 3. Configure Scrolling Logic
        self.scroll_frame.bind(
            "<Configure>",
            lambda e: self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))
        )
        
        # 4. Create the window inside the canvas
        self.window_id = self.main_canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        
        # 5. Ensure the inner frame expands to fill the width of the canvas
        self.main_canvas.bind(
            "<Configure>",
            lambda e: self.main_canvas.itemconfig(self.window_id, width=e.width)
        )
        
        # 6. Link scrollbar to canvas
        self.main_canvas.configure(yscrollcommand=self.main_scrollbar.set)
        
        # 7. Pack the scroll components (Hide Scrollbar)
        self.main_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        # self.main_scrollbar.pack(side=tk.RIGHT, fill=tk.Y) <-- HIDDEN
        
        # 8. Mouse Scroll Binding
        def _on_mousewheel(event):
            self.main_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            
        self.scroll_frame.bind("<Enter>", lambda e: self.main_canvas.bind_all("<MouseWheel>", _on_mousewheel))
        self.scroll_frame.bind("<Leave>", lambda e: self.main_canvas.unbind_all("<MouseWheel>"))
        
        # 9. Build the actual UI inside the scroll_frame
        self._create_ui(self.scroll_frame)
    
    def _create_ui(self, parent):
        """Create results UI elements inside the parent frame"""
        # Header
        self._create_header(parent)
        
        # Stats container
        self.stats_container = tk.Frame(parent, bg=config.COLOR_BG_VOID)
        self.stats_container.pack(fill=tk.BOTH, expand=True, pady=20)
        
        # Action buttons
        self._create_action_buttons(parent)
    
    def _create_header(self, parent):
        """Create header section"""
        header = tk.Frame(parent, bg=config.COLOR_BG_VOID)
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
    
    def _create_action_buttons(self, parent):
        """Create action buttons"""
        btn_frame = tk.Frame(parent, bg=config.COLOR_BG_VOID)
        btn_frame.pack(pady=40)  # Added extra padding at bottom
        
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
        
        # Clear previous stats
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
        
        # Reset scroll to top
        self.main_canvas.yview_moveto(0)
    
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
        
        # Glow effect (Stippled to prevent crash)
        for i in range(3):
            self._draw_rounded_rect(
                card, 3-i, 3-i, 247+i, 117+i, 12,
                fill="",
                outline=color,
                stipple="gray25",
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