"""
Batch Mode View Module - MODERNIZED
Beautiful UI with multi-file support and progress tracking
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
from typing import Callable, List
import threading

from .. import config


class ModernButton(tk.Canvas):
    """Custom modern button with hover effects"""
    
    def __init__(self, parent, text, command, bg_color, fg_color="white", 
                 width=150, height=45, **kwargs):
        super().__init__(parent, width=width, height=height, 
                        bg=config.COLOR_BACKGROUND, highlightthickness=0, **kwargs)
        
        self.text = text
        self.command = command
        self.bg_color = bg_color
        self.hover_color = config.COLOR_PRIMARY_DARK
        self.fg_color = fg_color
        self.width = width
        self.height = height
        self.is_hovered = False
        self.is_disabled = False
        
        self._draw()
        self.bind("<Button-1>", self._on_click)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
    
    def _draw(self):
        """Draw the button"""
        self.delete("all")
        
        # Determine color
        if self.is_disabled:
            color = config.COLOR_TEXT_MUTED
        elif self.is_hovered:
            color = self.hover_color
        else:
            color = self.bg_color
        
        # Draw rounded rectangle
        self.create_rounded_rectangle(
            5, 5, self.width-5, self.height-5,
            radius=10, fill=color, outline=""
        )
        
        # Draw text
        self.create_text(
            self.width/2, self.height/2,
            text=self.text,
            fill=self.fg_color,
            font=(config.FONT_FAMILY, config.FONT_SIZE_BUTTON, "bold")
        )
    
    def create_rounded_rectangle(self, x1, y1, x2, y2, radius=25, **kwargs):
        """Create a rounded rectangle"""
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
        return self.create_polygon(points, smooth=True, **kwargs)
    
    def _on_click(self, event):
        if not self.is_disabled and self.command:
            self.command()
    
    def _on_enter(self, event):
        if not self.is_disabled:
            self.is_hovered = True
            self._draw()
            self.config(cursor="hand2")
    
    def _on_leave(self, event):
        self.is_hovered = False
        self._draw()
        self.config(cursor="")
    
    def set_disabled(self, disabled: bool):
        """Enable/disable the button"""
        self.is_disabled = disabled
        self._draw()


class BatchModeView(tk.Frame):
    """
    Modernized Batch Processing View with multi-file support
    """
    
    def __init__(
        self, 
        parent: tk.Widget, 
        controller,
        switch_to_realtime: Callable,
        show_results: Callable
    ):
        super().__init__(parent, bg=config.COLOR_BACKGROUND)
        
        self.controller = controller
        self.switch_to_realtime = switch_to_realtime
        self.show_results = show_results
        self.selected_files: List[Path] = []
        self.is_processing = False
        
        self._setup_ui()
        
    def _setup_ui(self):
        """Setup the modern UI"""
        # Main container with padding
        main_container = tk.Frame(self, bg=config.COLOR_BACKGROUND)
        main_container.pack(fill=tk.BOTH, expand=True, padx=40, pady=30)
        
        # Header
        self._create_header(main_container)
        
        # File selection card
        self._create_file_card(main_container)
        
        # Status and progress
        self._create_status_section(main_container)
        
        # Action button
        self._create_action_button(main_container)
        
    def _create_header(self, parent):
        """Create modern header"""
        header_frame = tk.Frame(parent, bg=config.COLOR_BACKGROUND)
        header_frame.pack(fill=tk.X, pady=(0, 30))
        
        # Mode switcher (left)
        mode_btn = ModernButton(
            header_frame,
            text="⟲ Real-Time",
            command=self.switch_to_realtime,
            bg_color=config.COLOR_CARD,
            width=120,
            height=40
        )
        mode_btn.pack(side=tk.LEFT)
        
        # Title (center)
        title_frame = tk.Frame(header_frame, bg=config.COLOR_BACKGROUND)
        title_frame.pack(side=tk.LEFT, expand=True)
        
        title = tk.Label(
            title_frame,
            text="BATCH PROCESSING",
            font=(config.FONT_FAMILY, config.FONT_SIZE_TITLE, "bold"),
            bg=config.COLOR_BACKGROUND,
            fg=config.COLOR_TEXT
        )
        title.pack()
        
        subtitle = tk.Label(
            title_frame,
            text="Process multiple sensor data files efficiently",
            font=(config.FONT_FAMILY, config.FONT_SIZE_LABEL),
            bg=config.COLOR_BACKGROUND,
            fg=config.COLOR_TEXT_SECONDARY
        )
        subtitle.pack()
        
    def _create_file_card(self, parent):
        """Create file selection card"""
        card = tk.Frame(
            parent,
            bg=config.COLOR_CARD,
            highlightbackground=config.COLOR_BORDER,
            highlightthickness=1
        )
        card.pack(fill=tk.X, pady=20)
        
        # Card header
        card_header = tk.Frame(card, bg=config.COLOR_CARD)
        card_header.pack(fill=tk.X, padx=25, pady=(20, 10))
        
        tk.Label(
            card_header,
            text="📂 Select Sensor Data Files",
            font=(config.FONT_FAMILY, config.FONT_SIZE_HEADING, "bold"),
            bg=config.COLOR_CARD,
            fg=config.COLOR_TEXT
        ).pack(side=tk.LEFT)
        
        # File list area
        self.file_list_frame = tk.Frame(card, bg=config.COLOR_CARD)
        self.file_list_frame.pack(fill=tk.BOTH, expand=True, padx=25, pady=10)
        
        # Initial empty state
        self.empty_label = tk.Label(
            self.file_list_frame,
            text="No files selected\nClick 'Add Files' to begin",
            font=(config.FONT_FAMILY, config.FONT_SIZE_LABEL),
            bg=config.COLOR_CARD,
            fg=config.COLOR_TEXT_MUTED,
            justify=tk.CENTER
        )
        self.empty_label.pack(pady=40)
        
        # Buttons row
        btn_frame = tk.Frame(card, bg=config.COLOR_CARD)
        btn_frame.pack(fill=tk.X, padx=25, pady=(10, 20))
        
        self.add_btn = ModernButton(
            btn_frame,
            text="+ Add Files",
            command=self._add_files,
            bg_color=config.COLOR_PRIMARY,
            width=130,
            height=40
        )
        self.add_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.clear_btn = ModernButton(
            btn_frame,
            text="✕ Clear All",
            command=self._clear_files,
            bg_color=config.COLOR_WARNING,
            width=130,
            height=40
        )
        self.clear_btn.pack(side=tk.LEFT)
        
        # Stats
        self.stats_label = tk.Label(
            btn_frame,
            text="",
            font=(config.FONT_FAMILY, config.FONT_SIZE_SMALL),
            bg=config.COLOR_CARD,
            fg=config.COLOR_TEXT_SECONDARY
        )
        self.stats_label.pack(side=tk.RIGHT)
        
    def _create_status_section(self, parent):
        """Create status and progress section"""
        self.status_frame = tk.Frame(parent, bg=config.COLOR_BACKGROUND)
        self.status_frame.pack(fill=tk.X, pady=20)
        
        # Status text
        self.status_label = tk.Label(
            self.status_frame,
            text="",
            font=(config.FONT_FAMILY, config.FONT_SIZE_LABEL),
            bg=config.COLOR_BACKGROUND,
            fg=config.COLOR_TEXT_SECONDARY,
            wraplength=900
        )
        self.status_label.pack(pady=(0, 10))
        
        # Modern progress bar
        self.progress_canvas = tk.Canvas(
            self.status_frame,
            width=800,
            height=8,
            bg=config.COLOR_BACKGROUND,
            highlightthickness=0
        )
        
    def _create_action_button(self, parent):
        """Create main action button"""
        btn_frame = tk.Frame(parent, bg=config.COLOR_BACKGROUND)
        btn_frame.pack(pady=30)
        
        self.process_btn = ModernButton(
            btn_frame,
            text="▶ PROCESS FILES",
            command=self._process_files,
            bg_color=config.COLOR_PRIMARY,
            width=250,
            height=60
        )
        self.process_btn.pack()
        
    def _add_files(self):
        """Add files to selection"""
        if self.is_processing:
            return
        
        file_paths = filedialog.askopenfilenames(
            title="Select SensorData CSV File(s)",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not file_paths:
            return
        
        # Add new files (avoid duplicates)
        new_files = [Path(p) for p in file_paths]
        existing = {str(p.resolve()) for p in self.selected_files}
        
        for p in new_files:
            try:
                rp = str(p.resolve())
            except Exception:
                rp = str(p)
            if rp not in existing:
                self.selected_files.append(p)
                existing.add(rp)
        
        self._update_file_list()
        
    def _clear_files(self):
        """Clear all selected files"""
        if self.is_processing:
            return
        
        self.selected_files = []
        self._update_file_list()
        self._update_status("", config.COLOR_TEXT_SECONDARY)
        
    def _update_file_list(self):
        """Update the file list display"""
        # Clear existing
        for widget in self.file_list_frame.winfo_children():
            widget.destroy()
        
        if not self.selected_files:
            # Show empty state
            self.empty_label = tk.Label(
                self.file_list_frame,
                text="No files selected\nClick 'Add Files' to begin",
                font=(config.FONT_FAMILY, config.FONT_SIZE_LABEL),
                bg=config.COLOR_CARD,
                fg=config.COLOR_TEXT_MUTED,
                justify=tk.CENTER
            )
            self.empty_label.pack(pady=40)
            self.stats_label.config(text="")
        else:
            # Show file list
            for i, file_path in enumerate(self.selected_files):
                file_frame = tk.Frame(
                    self.file_list_frame,
                    bg=config.COLOR_CARD_HOVER,
                    highlightbackground=config.COLOR_BORDER,
                    highlightthickness=1
                )
                file_frame.pack(fill=tk.X, pady=3, padx=5)
                
                # File icon and name
                tk.Label(
                    file_frame,
                    text=f"📄 {file_path.name}",
                    font=(config.FONT_FAMILY, config.FONT_SIZE_LABEL),
                    bg=config.COLOR_CARD_HOVER,
                    fg=config.COLOR_TEXT,
                    anchor="w"
                ).pack(side=tk.LEFT, padx=15, pady=10)
                
                # File size
                size_mb = file_path.stat().st_size / (1024 * 1024)
                tk.Label(
                    file_frame,
                    text=f"{size_mb:.1f} MB",
                    font=(config.FONT_FAMILY, config.FONT_SIZE_SMALL),
                    bg=config.COLOR_CARD_HOVER,
                    fg=config.COLOR_TEXT_MUTED
                ).pack(side=tk.RIGHT, padx=15)
            
            # Update stats
            total_size = sum(f.stat().st_size for f in self.selected_files) / (1024 * 1024)
            self.stats_label.config(
                text=f"{len(self.selected_files)} file(s) • {total_size:.1f} MB total"
            )
            
            self._update_status(
                f"✓ {len(self.selected_files)} file(s) ready to process",
                config.COLOR_SUCCESS
            )
        
    def _update_status(self, message: str, color: str = None):
        """Update status message"""
        self.status_label.config(text=message)
        if color:
            self.status_label.config(fg=color)
            
    def _show_progress(self, progress: float, message: str = ""):
        """Update modern progress bar"""
        if not self.progress_canvas.winfo_ismapped():
            self.progress_canvas.pack(pady=5)
        
        self.progress_canvas.delete("all")
        
        # Background
        self.progress_canvas.create_rectangle(
            0, 0, 800, 8,
            fill=config.COLOR_CARD,
            outline=""
        )
        
        # Progress
        width = int(798 * progress)
        
        # Gradient effect (simplified - two colors)
        self.progress_canvas.create_rectangle(
            1, 1, width, 7,
            fill=config.COLOR_PRIMARY,
            outline=""
        )
        
        # Update status with message
        if message:
            self._update_status(message, config.COLOR_INFO)
            
    def _hide_progress(self):
        """Hide progress bar"""
        self.progress_canvas.pack_forget()
        
    def _process_files(self):
        """Process all selected files"""
        if self.is_processing:
            return
        
        if not self.selected_files:
            messagebox.showerror("Error", "Please select at least one CSV file!")
            return
        
        # Validate files exist
        missing = [p for p in self.selected_files if not p.exists()]
        if missing:
            messagebox.showerror("Error", "One or more selected files no longer exist.")
            return
        
        # Start processing in thread
        self.is_processing = True
        self.process_btn.set_disabled(True)
        self.add_btn.set_disabled(True)
        self.clear_btn.set_disabled(True)
        
        thread = threading.Thread(target=self._process_files_thread, daemon=True)
        thread.start()
        
    def _process_files_thread(self):
        """Process files in background thread"""
        try:
            def progress_callback(progress, message):
                self.after(0, lambda: self._show_progress(progress, message))
            
            # Update UI
            self.after(0, lambda: self._update_status("🚀 Starting batch processing...", config.COLOR_INFO))
            
            # Process using controller (which now handles multi-file merging)
            prediction_data = self.controller.process_batch_files(
                [str(f) for f in self.selected_files],
                progress_callback=progress_callback
            )
            
            # Complete
            self.after(0, lambda: self._show_progress(1.0, "✅ Processing complete!"))
            self.after(500, lambda: self._navigate_to_results(prediction_data))
            
        except Exception as e:
            error_msg = str(e)
            self.after(0, lambda: self._on_error(error_msg))
            
    def _navigate_to_results(self, prediction_data):
        """Navigate to results view"""
        self._hide_progress()
        self.is_processing = False
        self.process_btn.set_disabled(False)
        self.add_btn.set_disabled(False)
        self.clear_btn.set_disabled(False)
        
        self.show_results(prediction_data)
        
    def _on_error(self, error_message: str):
        """Handle errors"""
        self._hide_progress()
        self._update_status(f"❌ Error: {error_message}", config.COLOR_DANGER)
        self.is_processing = False
        self.process_btn.set_disabled(False)
        self.add_btn.set_disabled(False)
        self.clear_btn.set_disabled(False)
        
        messagebox.showerror(
            "Processing Error",
            f"Failed to process files:\n\n{error_message}"
        )