"""
🗂️ CYNOSURE BATCH PROCESSING TERMINAL 🗂️
Retro paper-based interface with technical diagram aesthetics
✅ FIXED: Entire page is now scrollable
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
from typing import Callable, List
import threading

from .. import config


class CynosureProgressBar(tk.Canvas):
    """
    Retro segmented progress bar with percentage
    Styled like old terminal displays
    """
    
    def __init__(self, parent, width=600, height=40):
        super().__init__(
            parent,
            width=width,
            height=height,
            bg=config.COLOR_BG_PAPER,
            highlightthickness=0
        )
        self.width = width
        self.height = height
        self.progress = 0.0
        self.message = ""
        self.segments = config.PROGRESS_BAR_SEGMENTS
        
        self._draw()
    
    def _draw(self):
        """Draw the retro progress bar"""
        self.delete("all")
        
        # Background box
        self.create_rectangle(
            5, 5, self.width-5, self.height-5,
            outline=config.COLOR_BORDER_DARK,
            width=2,
            fill=config.COLOR_BG_CARD
        )
        
        # Calculate filled segments
        segment_width = (self.width - 20) / self.segments
        filled_segments = int(self.progress * self.segments)
        
        # Draw segments
        for i in range(self.segments):
            x1 = 10 + (i * segment_width)
            x2 = x1 + segment_width - 2
            y1 = 10
            y2 = self.height - 10
            
            if i < filled_segments:
                # Filled segment
                self.create_rectangle(
                    x1, y1, x2, y2,
                    fill=config.COLOR_ACCENT_BLUE,
                    outline=config.COLOR_BORDER_DARK,
                    width=1
                )
            else:
                # Empty segment
                self.create_rectangle(
                    x1, y1, x2, y2,
                    fill=config.COLOR_BG_DARKER,
                    outline=config.COLOR_BORDER_LIGHT,
                    width=1
                )
        
        # Percentage text
        percentage = int(self.progress * 100)
        self.create_text(
            self.width / 2,
            self.height / 2,
            text=f"{percentage}%",
            font=(config.FONT_FAMILY, 12, "bold"),
            fill=config.COLOR_TEXT_PRIMARY
        )
    
    def set_progress(self, progress: float, message: str = ""):
        """Update progress (0.0 to 1.0)"""
        self.progress = max(0.0, min(1.0, progress))
        self.message = message
        self._draw()
    
    def update(self):
        """Force update"""
        super().update()


class RetroButton(tk.Canvas):
    """Retro paper button with hover effect"""
    
    def __init__(self, parent, text, command, width=160, height=40, **kwargs):
        super().__init__(
            parent,
            width=width,
            height=height,
            bg=config.COLOR_BG_PAPER,
            highlightthickness=0,
            cursor="hand2"
        )
        
        self.text = text
        self.command = command
        self.width = width
        self.height = height
        self.is_disabled = False
        self.is_hovered = False
        
        self.bind("<Button-1>", self._on_click)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        
        self._draw()
    
    def _draw(self):
        """Draw button"""
        self.delete("all")
        
        if self.is_disabled:
            bg = config.COLOR_BG_DARKER
            fg = config.COLOR_TEXT_MUTED
            border = config.COLOR_BORDER_LIGHT
        elif self.is_hovered:
            bg = config.COLOR_ACCENT_BLUE
            fg = config.COLOR_BG_PAPER
            border = config.COLOR_BORDER_DARK
        else:
            bg = config.COLOR_BG_CARD
            fg = config.COLOR_TEXT_PRIMARY
            border = config.COLOR_BORDER_DARK
        
        # Button rectangle
        self.create_rectangle(
            2, 2, self.width-2, self.height-2,
            fill=bg,
            outline=border,
            width=2
        )
        
        # Text
        self.create_text(
            self.width/2, self.height/2,
            text=self.text,
            font=(config.FONT_FAMILY, config.FONT_SIZE_BUTTON, "bold"),
            fill=fg
        )
    
    def _on_click(self, event):
        if not self.is_disabled and self.command:
            self.command()
    
    def _on_enter(self, event):
        if not self.is_disabled:
            self.is_hovered = True
            self._draw()
    
    def _on_leave(self, event):
        self.is_hovered = False
        self._draw()
    
    def set_disabled(self, disabled: bool):
        """Enable/disable button"""
        self.is_disabled = disabled
        self._draw()


class CynosureBatchView(tk.Frame):
    """
    CYNOSURE BATCH PROCESSING TERMINAL
    ✅ FIXED: Entire page is scrollable
    """
    
    def __init__(self, parent, controller, switch_to_realtime: Callable, 
                 show_results: Callable, show_history: Callable):
        super().__init__(parent, bg=config.COLOR_BG_PAPER)
        
        self.controller = controller
        self.switch_to_realtime = switch_to_realtime
        self.show_results = show_results
        self.show_history = show_history
        self.selected_files: List[Path] = []
        self.is_processing = False
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup UI with FULL PAGE SCROLLING"""
        
        # ===== CREATE SCROLLABLE WRAPPER =====
        # Main canvas
        self.main_canvas = tk.Canvas(
            self,
            bg=config.COLOR_BG_PAPER,
            highlightthickness=0
        )
        
        # Scrollbar
        scrollbar = tk.Scrollbar(
            self,
            orient="vertical",
            command=self.main_canvas.yview
        )
        
        # Scrollable frame
        self.scroll_content = tk.Frame(
            self.main_canvas,
            bg=config.COLOR_BG_PAPER
        )
        
        # Configure scrolling
        self.scroll_content.bind(
            "<Configure>",
            lambda e: self.main_canvas.configure(
                scrollregion=self.main_canvas.bbox("all")
            )
        )
        
        # Create window in canvas
        self.window_id = self.main_canvas.create_window(
            (0, 0),
            window=self.scroll_content,
            anchor="nw"
        )
        
        # Auto-adjust width
        self.main_canvas.bind(
            "<Configure>",
            lambda e: self.main_canvas.itemconfig(
                self.window_id,
                width=e.width
            )
        )
        
        self.main_canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas and scrollbar
        self.main_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Mouse wheel support
        def on_wheel(event):
            self.main_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        self.scroll_content.bind(
            "<Enter>",
            lambda e: self.main_canvas.bind_all("<MouseWheel>", on_wheel)
        )
        self.scroll_content.bind(
            "<Leave>",
            lambda e: self.main_canvas.unbind_all("<MouseWheel>")
        )
        
        # ===== NOW BUILD ALL CONTENT INSIDE scroll_content =====
        
        # ===== HEADER =====
        header = tk.Frame(self.scroll_content, bg=config.COLOR_BG_PAPER)
        header.pack(fill=tk.X, padx=30, pady=(20, 10))

        # 👁️ BILL CIPHER LOGO
        from .cyber_components import BillCipherLogo
        logo = BillCipherLogo(header, size=90)
        logo.pack(side=tk.LEFT, padx=(0, 15))

        # Title section
        title_frame = tk.Frame(header, bg=config.COLOR_BG_PAPER)
        title_frame.pack(side=tk.LEFT)
        
        tk.Label(
            title_frame,
            text="GOBEST CAB SAFETY TERMINAL",
            font=(config.FONT_FAMILY_DISPLAY, 20, "bold"),
            bg=config.COLOR_BG_PAPER,
            fg=config.COLOR_TEXT_PRIMARY
        ).pack(anchor="w")
        
        tk.Label(
            title_frame,
            text="BATCH PROCESSING SYSTEM // CYNOSURE CORP.",
            font=(config.FONT_FAMILY, 9),
            bg=config.COLOR_BG_PAPER,
            fg=config.COLOR_TEXT_SECONDARY
        ).pack(anchor="w")
        
        tk.Label(
            title_frame,
            text="⚠ AUTHORIZED PERSONNEL ONLY ⚠",
            font=(config.FONT_FAMILY, 8, "bold"),
            bg=config.COLOR_BG_PAPER,
            fg=config.COLOR_DANGER
        ).pack(anchor="w", pady=(5, 0))
        
        # Mode switch button
        self.switch_btn = RetroButton(
            header,
            text="→ REAL-TIME MODE",
            command=self.switch_to_realtime,
            width=180,
            height=45
        )
        self.switch_btn.pack(side=tk.RIGHT)
        
        # ===== UPLOAD SECTION =====
        upload_frame = tk.Frame(self.scroll_content, bg=config.COLOR_BG_PAPER)
        upload_frame.pack(fill=tk.X, padx=30, pady=10)
        
        # Upload card
        upload_card = tk.Frame(
            upload_frame,
            bg=config.COLOR_BG_CARD,
            highlightbackground=config.COLOR_BORDER_DARK,
            highlightthickness=2
        )
        upload_card.pack(fill=tk.X, ipady=15)
        
        # Top line
        tk.Frame(
            upload_card,
            bg=config.COLOR_ACCENT_BLUE,
            height=3
        ).pack(fill=tk.X)
        
        # Content
        tk.Label(
            upload_card,
            text="📄 DATA INGESTION PORT",
            font=(config.FONT_FAMILY, 14, "bold"),
            bg=config.COLOR_BG_CARD,
            fg=config.COLOR_TEXT_PRIMARY
        ).pack(pady=(10, 5))
        
        tk.Label(
            upload_card,
            text="SELECT .CSV SENSOR LOG FILES FOR ANALYSIS",
            font=(config.FONT_FAMILY, 9),
            bg=config.COLOR_BG_CARD,
            fg=config.COLOR_TEXT_SECONDARY
        ).pack()
        
        # Add button
        self.btn_add = RetroButton(
            upload_card,
            text="+ SELECT FILES",
            command=self._add_files,
            width=160,
            height=35
        )
        self.btn_add.pack(pady=10)
        
        # ===== FILE QUEUE (Nested scrollable) =====
        queue_frame = tk.Frame(self.scroll_content, bg=config.COLOR_BG_PAPER)
        queue_frame.pack(fill=tk.X, padx=30, pady=10)
        
        # Label
        tk.Label(
            queue_frame,
            text="QUEUED DATASETS:",
            font=(config.FONT_FAMILY, 10, "bold"),
            bg=config.COLOR_BG_PAPER,
            fg=config.COLOR_TEXT_PRIMARY
        ).pack(anchor="w", pady=(0, 5))
        
        # Container with fixed height for file list
        list_container = tk.Frame(
            queue_frame,
            bg=config.COLOR_BG_CARD,
            highlightbackground=config.COLOR_BORDER_DARK,
            highlightthickness=2,
            height=250
        )
        list_container.pack(fill=tk.X)
        list_container.pack_propagate(False)
        
        # Canvas + scrollbar for file list
        self.file_canvas = tk.Canvas(
            list_container,
            bg=config.COLOR_BG_CARD,
            highlightthickness=0
        )
        
        file_scrollbar = tk.Scrollbar(
            list_container,
            orient="vertical",
            command=self.file_canvas.yview
        )
        
        self.file_list_frame = tk.Frame(self.file_canvas, bg=config.COLOR_BG_CARD)
        
        # Update scrollregion when content changes
        self.file_list_frame.bind(
            "<Configure>",
            lambda e: self.file_canvas.configure(
                scrollregion=self.file_canvas.bbox("all")
            )
        )
        
        # === FIX STARTS HERE ===
        # Capture the window ID so we can resize it
        self.file_window_id = self.file_canvas.create_window(
            (0, 0),
            window=self.file_list_frame,
            anchor="nw"
        )
        
        # Force the inner frame width to match the canvas width
        # This makes the list items expand to fill the container (centering them)
        self.file_canvas.bind(
            "<Configure>",
            lambda e: self.file_canvas.itemconfig(self.file_window_id, width=e.width)
        )
        # === FIX ENDS HERE ===
        
        self.file_canvas.configure(yscrollcommand=file_scrollbar.set)
        
        # Pack file list canvas
        self.file_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        file_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Empty state
        self._show_empty_state()
        
        # ===== PROGRESS BAR (Hidden initially) =====
        self.progress_container = tk.Frame(
            self.scroll_content,
            bg=config.COLOR_BG_PAPER
        )
        
        self.progress_bar = CynosureProgressBar(
            self.progress_container,
            width=700,
            height=40
        )
        self.progress_bar.pack(pady=10)
        
        self.progress_label = tk.Label(
            self.progress_container,
            text="",
            font=(config.FONT_FAMILY, 10),
            bg=config.COLOR_BG_PAPER,
            fg=config.COLOR_TEXT_SECONDARY
        )
        self.progress_label.pack()
        
        # ===== ACTION FOOTER =====
        footer = tk.Frame(self.scroll_content, bg=config.COLOR_BG_PAPER)
        footer.pack(fill=tk.X, padx=30, pady=(20, 40))
        
        # Stats
        self.stats_label = tk.Label(
            footer,
            text="SYSTEM READY",
            font=(config.FONT_FAMILY, 10, "bold"),
            bg=config.COLOR_BG_PAPER,
            fg=config.COLOR_TEXT_PRIMARY
        )
        self.stats_label.pack(side=tk.LEFT)
        
        # Clear button
        self.clear_btn = RetroButton(
            footer,
            text="✕ CLEAR QUEUE",
            command=self._clear_files,
            width=140,
            height=40
        )
        self.clear_btn.pack(side=tk.LEFT, padx=15)
        
        # Process button
        self.process_btn = RetroButton(
            footer,
            text="▶ INITIATE ANALYSIS",
            command=self._process_files,
            width=220,
            height=50
        )
        self.process_btn.pack(side=tk.RIGHT)
    
    def _show_empty_state(self):
        """Show empty state"""
        for widget in self.file_list_frame.winfo_children():
            widget.destroy()
        
        tk.Label(
            self.file_list_frame,
            text="NO DATA LOADED\n\nAWAITING FILE INPUT...",
            font=(config.FONT_FAMILY, 11),
            bg=config.COLOR_BG_CARD,
            fg=config.COLOR_TEXT_MUTED,
            justify=tk.CENTER
        ).pack(pady=50)
    
    def _add_files(self):
        """Add files to queue"""
        if self.is_processing:
            return
        
        file_paths = filedialog.askopenfilenames(
            title="Select Sensor Data Files",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        
        if not file_paths:
            return
        
        # Add new files
        existing = {str(p.resolve()) for p in self.selected_files}
        for p in file_paths:
            path_obj = Path(p)
            if str(path_obj.resolve()) not in existing:
                self.selected_files.append(path_obj)
        
        self._update_file_list()
    
    def _update_file_list(self):
        """Update file list"""
        for widget in self.file_list_frame.winfo_children():
            widget.destroy()
        
        if not self.selected_files:
            self._show_empty_state()
            self.stats_label.config(text="SYSTEM READY")
            return
        
        # Show files
        for i, f in enumerate(self.selected_files):
            file_frame = tk.Frame(
                self.file_list_frame,
                bg=config.COLOR_BG_DARKER,
                highlightbackground=config.COLOR_BORDER_LIGHT,
                highlightthickness=1
            )
            file_frame.pack(fill=tk.X, pady=2, padx=5)
            
            # Number
            tk.Label(
                file_frame,
                text=f"{i+1:02d}.",
                font=(config.FONT_FAMILY, 10, "bold"),
                bg=config.COLOR_BG_DARKER,
                fg=config.COLOR_TEXT_SECONDARY,
                width=4
            ).pack(side=tk.LEFT, padx=(10, 5))
            
            # Icon
            tk.Label(
                file_frame,
                text="📄",
                bg=config.COLOR_BG_DARKER,
                font=("Arial", 12)
            ).pack(side=tk.LEFT)
            
            # Info
            info_frame = tk.Frame(file_frame, bg=config.COLOR_BG_DARKER)
            info_frame.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
            
            tk.Label(
                info_frame,
                text=f.name,
                font=(config.FONT_FAMILY, 10, "bold"),
                bg=config.COLOR_BG_DARKER,
                fg=config.COLOR_TEXT_PRIMARY,
                anchor="w"
            ).pack(fill=tk.X)
            
            # Size
            try:
                size_mb = f.stat().st_size / (1024 * 1024)
                size_txt = f"{size_mb:.2f} MB"
            except:
                size_txt = "SIZE UNKNOWN"
            
            tk.Label(
                info_frame,
                text=size_txt,
                font=(config.FONT_FAMILY, 8),
                bg=config.COLOR_BG_DARKER,
                fg=config.COLOR_TEXT_SECONDARY,
                anchor="w"
            ).pack(fill=tk.X)
            
            # Status
            tk.Label(
                file_frame,
                text="[QUEUED]",
                font=(config.FONT_FAMILY, 9, "bold"),
                bg=config.COLOR_BG_DARKER,
                fg=config.COLOR_ACCENT_BLUE
            ).pack(side=tk.RIGHT, padx=10)
        
        # Update stats
        total_mb = sum(f.stat().st_size for f in self.selected_files) / (1024 * 1024)
        self.stats_label.config(
            text=f"FILES QUEUED: {len(self.selected_files)} ({total_mb:.2f} MB)"
        )
    
    def _clear_files(self):
        """Clear queue"""
        if self.is_processing:
            return
        
        self.selected_files = []
        self._update_file_list()
    
    def _process_files(self):
        """Start processing"""
        if not self.selected_files:
            messagebox.showwarning(
                "No Data",
                "Please select data files first.\n\nNo files in queue."
            )
            return
        
        self.is_processing = True
        self.process_btn.set_disabled(True)
        self.switch_btn.set_disabled(True)
        self.btn_add.set_disabled(True)
        self.clear_btn.set_disabled(True)
        
        # Show progress
        self.progress_container.pack(before=self.stats_label.master, padx=30, pady=15)
        self.progress_bar.set_progress(0.0, "INITIALIZING...")
        self.progress_label.config(text="INITIALIZING ANALYSIS SEQUENCE...")
        
        # Start thread
        thread = threading.Thread(target=self._run_process, daemon=True)
        thread.start()
    
    def _run_process(self):
        """Process in background with progress updates"""
        try:
            def progress_callback(progress, message):
                self.after(0, lambda: self._update_progress(progress, message))
            
            # Process
            data = self.controller.process_batch_files(
                [str(f) for f in self.selected_files],
                progress_callback=progress_callback
            )
            
            self.after(0, lambda: self._finish_process(data))
        except Exception as e:
            error_msg = str(e)   # capture NOW before Python deletes 'e'
            self.after(0, lambda: self._error_process(error_msg))
    
    def _update_progress(self, progress: float, message: str):
        """Update progress bar"""
        self.progress_bar.set_progress(progress, message)
        self.progress_label.config(text=message.upper())
        self.progress_bar.update()
    
    def _finish_process(self, data):
        """Finish processing"""
        self.progress_container.pack_forget()
        self.is_processing = False
        self.process_btn.set_disabled(False)
        self.switch_btn.set_disabled(False)
        self.btn_add.set_disabled(False)
        self.clear_btn.set_disabled(False)
        self.show_results(data)
    
    def _error_process(self, msg):
        """Handle error"""
        self.progress_container.pack_forget()
        self.is_processing = False
        self.process_btn.set_disabled(False)
        self.switch_btn.set_disabled(False)
        self.btn_add.set_disabled(False)
        self.clear_btn.set_disabled(False)
        messagebox.showerror("Processing Error", f"ERROR:\n\n{msg}")