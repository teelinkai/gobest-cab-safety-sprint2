"""
📜 CYBERPUNK HISTORY VIEW 📜
File explorer for past predictions with integrated CSV viewer
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
from pathlib import Path
import os

from .. import config
from .cyber_components import RetroButton

# Default directory for history
# In a real app, this might come from config.py
HISTORY_DIR = Path("saved_results")

class HistoryCard(tk.Canvas):
    """
    Small clickable card representing a saved CSV file
    """
    def __init__(self, parent, file_path: Path, on_click):
        super().__init__(parent, height=50, bg=config.COLOR_BG_CARD, highlightthickness=0)
        self.file_path = file_path
        self.on_click = on_click
        self.bind("<Button-1>", self._on_click)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        
        # Draw background
        self.bg = self.create_rectangle(0, 0, 1000, 50, fill=config.COLOR_BG_CARD, outline="")
        
        # Icon
        self.create_text(25, 25, text="📄", font=("Arial", 16), fill=config.COLOR_NEON_PURPLE)
        
        # Filename
        self.create_text(
            50, 25, 
            text=file_path.name, 
            anchor="w", 
            font=(config.FONT_FAMILY, 10, "bold"),
            fill=config.COLOR_TEXT_PRIMARY
        )
        
        # Date (modified time)
        mod_time = datetime_str = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime('%Y-%m-%d %H:%M')
        self.create_text(
            350, 25,
            text=mod_time,
            anchor="e",
            font=(config.FONT_FAMILY, 9),
            fill=config.COLOR_TEXT_MUTED
        )

    def _on_enter(self, e):
        self.itemconfig(self.bg, fill=config.COLOR_BG_ELEVATED)
        self.config(cursor="hand2")

    def _on_leave(self, e):
        self.itemconfig(self.bg, fill=config.COLOR_BG_CARD)
        self.config(cursor="")

    def _on_click(self, e):
        self.on_click(self.file_path)

from datetime import datetime

class CyberHistoryView(tk.Frame):
    """
    History Page: Lists CSV files and displays contents
    """
    def __init__(self, parent, controller, back_to_main):
        super().__init__(parent, bg=config.COLOR_BG_VOID)
        self.controller = controller
        self.back_to_main = back_to_main
        self.current_df = None
        
        # Ensure directory exists
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        
        self._setup_ui()
        
    def _setup_ui(self):
        """Build the split-pane layout"""
        # --- HEADER ---
        header = tk.Frame(self, bg=config.COLOR_BG_VOID)
        header.pack(fill=tk.X, padx=20, pady=(20, 10))
        
        tk.Label(
            header, 
            text="📂 PREDICTION ARCHIVE", 
            font=(config.FONT_FAMILY_DISPLAY, 24, "bold"),
            fg=config.COLOR_NEON_PURPLE,
            bg=config.COLOR_BG_VOID
        ).pack(side=tk.LEFT)
        
        # Action Buttons
        btn_frame = tk.Frame(header, bg=config.COLOR_BG_VOID)
        btn_frame.pack(side=tk.RIGHT)
        
        NeonButton(
            btn_frame, 
            text="OPEN FOLDER", 
            command=self._open_explorer,
            width=140, height=40,
            neon_color=config.COLOR_NEON_CYAN
        ).pack(side=tk.LEFT, padx=5)
        
        NeonButton(
            btn_frame, 
            text="BACK", 
            command=self.back_to_main,
            width=100, height=40,
            neon_color=config.COLOR_NEON_PINK
        ).pack(side=tk.LEFT, padx=5)

        # --- MAIN CONTENT (SPLIT VIEW) ---
        content = tk.Frame(self, bg=config.COLOR_BG_VOID)
        content.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # LEFT: File List
        left_pane = tk.Frame(content, bg=config.COLOR_BG_CARD, width=300)
        left_pane.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_pane.pack_propagate(False)
        
        tk.Label(
            left_pane, text="SAVED FILES", 
            bg=config.COLOR_BG_CARD, fg=config.COLOR_TEXT_MUTED,
            font=(config.FONT_FAMILY, 10, "bold")
        ).pack(pady=10)
        
        self.file_list_frame = tk.Frame(left_pane, bg=config.COLOR_BG_CARD)
        self.file_list_frame.pack(fill=tk.BOTH, expand=True)
        
        # RIGHT: Data Preview (Treeview)
        right_pane = tk.Frame(content, bg=config.COLOR_BG_VOID)
        right_pane.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Style the Treeview
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Cyber.Treeview",
            background=config.COLOR_BG_CARD,
            foreground=config.COLOR_TEXT_PRIMARY,
            fieldbackground=config.COLOR_BG_CARD,
            borderwidth=0,
            font=(config.FONT_FAMILY, 9)
        )
        style.configure(
            "Cyber.Treeview.Heading",
            background=config.COLOR_BG_ELEVATED,
            foreground=config.COLOR_NEON_CYAN,
            font=(config.FONT_FAMILY, 9, "bold"),
            relief="flat"
        )
        style.map("Cyber.Treeview", background=[('selected', config.COLOR_NEON_PURPLE)])
        
        # Treeview Container
        self.tree = ttk.Treeview(right_pane, style="Cyber.Treeview", selectmode="browse")
        
        # Scrollbars for table
        y_scroll = ttk.Scrollbar(right_pane, orient="vertical", command=self.tree.yview)
        x_scroll = ttk.Scrollbar(right_pane, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        y_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        x_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Initial Load
        self.refresh_file_list()

    def refresh_file_list(self):
        """Scan directory and populate left pane"""
        for widget in self.file_list_frame.winfo_children():
            widget.destroy()
            
        files = list(HISTORY_DIR.glob("*.csv"))
        files.sort(key=os.path.getmtime, reverse=True)
        
        if not files:
            tk.Label(
                self.file_list_frame, 
                text="No files found in\nsaved_results/", 
                bg=config.COLOR_BG_CARD, 
                fg=config.COLOR_TEXT_MUTED
            ).pack(pady=20)
            return

        for f in files:
            card = HistoryCard(self.file_list_frame, f, self._load_csv)
            card.pack(fill=tk.X, pady=1, padx=2)

    def _load_csv(self, file_path):
        """Read CSV and display in Treeview"""
        try:
            df = pd.read_csv(file_path)
            self._display_dataframe(df)
        except Exception as e:
            messagebox.showerror("Error", f"Could not load file:\n{e}")

    def _display_dataframe(self, df):
        """Render Pandas DF to Treeview"""
        # Clear existing
        self.tree.delete(*self.tree.get_children())
        
        # Set columns
        cols = list(df.columns)
        self.tree["columns"] = cols
        self.tree["show"] = "headings"
        
        for col in cols:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=120, anchor="center")
            
        # Insert rows
        for _, row in df.iterrows():
            values = [row[col] for col in cols]
            self.tree.insert("", "end", values=values)

    def _open_explorer(self):
        """Open the history folder in file explorer"""
        path = str(HISTORY_DIR.absolute())
        try:
            os.startfile(path)  # Windows only
        except AttributeError:
            import subprocess
            subprocess.Popen(['xdg-open', path])  # Linux
        except Exception:
            messagebox.showinfo("Info", f"Files are saved at:\n{path}")