"""
Batch Mode View Module
Handles the UI for batch processing mode
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
from typing import Callable
import threading

from .. import config


class BatchModeView(tk.Frame):
    """
    View for Batch Processing Mode
    Allows users to upload CSV files for batch prediction
    """
    
    def __init__(
        self, 
        parent: tk.Widget, 
        controller,
        switch_to_realtime: Callable,
        show_results: Callable
    ):
        """
        Initialize Batch Mode View
        
        Args:
            parent: Parent widget
            controller: Mode controller instance
            switch_to_realtime: Callback to switch to realtime mode
            show_results: Callback to show results view
        """
        super().__init__(parent, bg=config.COLOR_BACKGROUND)
        
        self.controller = controller
        self.switch_to_realtime = switch_to_realtime
        self.show_results = show_results
        self.selected_files: list[Path] = []
        
        # Status tracking
        self.is_processing = False
        
        self._setup_ui()
        
    def _setup_ui(self):
        """Setup the UI components"""
        # Top bar with switch mode and history buttons
        self._create_top_bar()
        
        # Title
        self._create_title()
        
        # File input section
        self._create_file_input()
        
        # Status label (for progress updates)
        self._create_status_label()
        
        # Progress bar
        self._create_progress_bar()
        
        # Enter button
        self._create_enter_button()
        
    def _create_top_bar(self):
        """Create the top bar with control buttons"""
        top_bar = tk.Frame(self, bg=config.COLOR_BACKGROUND)
        top_bar.pack(fill=tk.X, padx=20, pady=10)
        
        # Switch Mode button (left)
        switch_btn = tk.Button(
            top_bar,
            text="⇄",
            font=(config.FONT_FAMILY, config.FONT_SIZE_BUTTON, "bold"),
            bg=config.COLOR_PRIMARY,
            fg="white",
            width=3,
            height=1,
            relief=tk.RAISED,
            cursor="hand2",
            command=self.switch_to_realtime
        )
        switch_btn.pack(side=tk.LEFT)
        
        # Tooltip for switch button
        self._create_tooltip(switch_btn, "Switch to Real-Time Mode")
        
        # History button (right)
        history_btn = tk.Button(
            top_bar,
            text="📋",
            font=(config.FONT_FAMILY, config.FONT_SIZE_BUTTON, "bold"),
            bg=config.COLOR_PRIMARY,
            fg="white",
            width=3,
            height=1,
            relief=tk.RAISED,
            cursor="hand2",
            command=self._show_history
        )
        history_btn.pack(side=tk.RIGHT)
        
        # Tooltip for history button
        self._create_tooltip(history_btn, "View History")
        
    def _create_title(self):
        """Create the title section"""
        title_frame = tk.Frame(self, bg=config.COLOR_BACKGROUND)
        title_frame.pack(pady=(50, 30))
        
        title_label = tk.Label(
            title_frame,
            text="BATCH PROCESSING MODE",
            font=(config.FONT_FAMILY, config.FONT_SIZE_TITLE, "bold"),
            bg=config.COLOR_BACKGROUND,
            fg=config.COLOR_TEXT
        )
        title_label.pack()
        
    def _create_file_input(self):
        """Create file input section"""
        file_frame = tk.Frame(self, bg=config.COLOR_BACKGROUND)
        file_frame.pack(pady=20)
        
        # File entry with browse button
        self.file_entry = tk.Entry(
            file_frame,
            width=40,
            font=(config.FONT_FAMILY, config.FONT_SIZE_LABEL),
            relief=tk.SOLID,
            borderwidth=1
        )
        self.file_entry.pack(side=tk.LEFT, padx=(0, 10), ipady=5)
        self.file_entry.insert(0, "Select SensorData.csv file")
        self.file_entry.config(state='readonly', fg=config.COLOR_TEXT_LIGHT)
        
        # Browse button
        browse_btn = tk.Button(
            file_frame,
            text="Browse",
            font=(config.FONT_FAMILY, config.FONT_SIZE_BUTTON),
            bg=config.COLOR_PRIMARY,
            fg="white",
            padx=15,
            pady=5,
            relief=tk.RAISED,
            cursor="hand2",
            command=self._browse_file
        )
        browse_btn.pack(side=tk.LEFT)

        clear_btn = tk.Button(
            file_frame,
            text="Clear",
            font=(config.FONT_FAMILY, config.FONT_SIZE_BUTTON),
            bg=config.COLOR_WARNING,
            fg="white",
            padx=15,
            pady=5,
            relief=tk.RAISED,
            cursor="hand2",
            command=self._clear_files
        )
        clear_btn.pack(side=tk.LEFT, padx=(10, 0))

    def _clear_files(self):
        if self.is_processing:
            return
        self.selected_files = []
        self.file_entry.config(state='normal', fg=config.COLOR_TEXT_LIGHT)
        self.file_entry.delete(0, tk.END)
        self.file_entry.insert(0, "Select SensorData.csv file(s)")
        self.file_entry.config(state='readonly')
        self._update_status("Selection cleared.", config.COLOR_TEXT_LIGHT)
        
    def _create_status_label(self):
        """Create status label for showing progress messages"""
        self.status_label = tk.Label(
            self,
            text="",
            font=(config.FONT_FAMILY, config.FONT_SIZE_LABEL),
            bg=config.COLOR_BACKGROUND,
            fg=config.COLOR_TEXT,
            height=2,
            wraplength=760,
            justify="left"
        )
        self.status_label.pack(pady=(10, 5))
        
    def _create_progress_bar(self):
        """Create progress bar"""
        self.progress_frame = tk.Frame(self, bg=config.COLOR_BACKGROUND)
        self.progress_frame.pack(pady=5)
        
        # Create canvas for progress bar
        self.progress_canvas = tk.Canvas(
            self.progress_frame,
            width=400,
            height=25,
            bg='white',
            highlightthickness=1,
            highlightbackground=config.COLOR_TEXT_LIGHT
        )
        self.progress_canvas.pack()
        
        # Initially hide the progress bar
        self.progress_frame.pack_forget()
        
    def _create_enter_button(self):
        """Create the main Enter button"""
        self.enter_btn = tk.Button(
            self,
            text="Process File",
            font=(config.FONT_FAMILY, config.FONT_SIZE_BUTTON + 2, "bold"),
            bg=config.COLOR_PRIMARY,
            fg="white",
            width=15,
            height=2,
            relief=tk.RAISED,
            cursor="hand2",
            command=self._process_file
        )
        self.enter_btn.pack(pady=30)
        
        # Bind hover effects
        self.enter_btn.bind('<Enter>', lambda e: self.enter_btn.config(bg=config.COLOR_PRIMARY_DARK) if not self.is_processing else None)
        self.enter_btn.bind('<Leave>', lambda e: self.enter_btn.config(bg=config.COLOR_PRIMARY) if not self.is_processing else None)
        
    def _browse_file(self):
        """Open file dialog to select one or more CSV files (append to selection)."""
        if self.is_processing:
            return

        file_paths = filedialog.askopenfilenames(
            title="Select SensorData CSV File(s)",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if not file_paths:
            return

        # Append new files, avoid duplicates
        new_paths = [Path(p) for p in file_paths]
        existing = {str(p.resolve()) for p in self.selected_files}

        for p in new_paths:
            try:
                rp = str(p.resolve())
            except Exception:
                rp = str(p)
            if rp not in existing:
                self.selected_files.append(p)
                existing.add(rp)

        # Update entry display
        if len(self.selected_files) == 1:
            display_text = self.selected_files[0].name
        else:
            display_text = f"{len(self.selected_files)} files selected (e.g., {self.selected_files[0].name})"

        self.file_entry.config(state='normal', fg=config.COLOR_TEXT)
        self.file_entry.delete(0, tk.END)
        self.file_entry.insert(0, display_text)
        self.file_entry.config(state='readonly')

        self._update_status(
            f"{len(self.selected_files)} file(s) selected. You can click Browse again to add more.",
            config.COLOR_TEXT
        )
            
    def _update_status(self, message: str, color: str = None):
        """Update status label with message"""
        self.status_label.config(text=message)
        if color:
            self.status_label.config(fg=color)
            
    def _show_progress(self, progress: float):
        """
        Update progress bar
        
        Args:
            progress: Progress value between 0 and 1
        """
        # Show progress bar if hidden
        if not self.progress_frame.winfo_ismapped():
            self.progress_frame.pack(pady=5)
        
        # Clear canvas
        self.progress_canvas.delete("all")
        
        # Draw background
        self.progress_canvas.create_rectangle(
            0, 0, 400, 25,
            fill='white',
            outline=''
        )
        
        # Draw progress
        progress_width = int(398 * progress)
        self.progress_canvas.create_rectangle(
            1, 1, progress_width, 24,
            fill=config.COLOR_PRIMARY,
            outline=''
        )
        
        # Draw percentage text
        percentage = int(progress * 100)
        self.progress_canvas.create_text(
            200, 12,
            text=f"{percentage}%",
            font=(config.FONT_FAMILY, 10, "bold"),
            fill=config.COLOR_TEXT
        )
        
    def _hide_progress(self):
        """Hide progress bar"""
        self.progress_frame.pack_forget()
        self.progress_canvas.delete("all")
        
    def _process_file(self):
        """Process the selected file(s)"""
        if self.is_processing:
            return

        if not self.selected_files:
            messagebox.showerror("Error", "Please select at least one CSV file first!")
            return

        # quick existence check
        missing = [p for p in self.selected_files if not p.exists()]
        if missing:
            messagebox.showerror("Error", "One or more selected files no longer exist.")
            return

        self.is_processing = True
        self.enter_btn.config(state='disabled', bg=config.COLOR_TEXT_LIGHT)
        thread = threading.Thread(target=self._process_file_thread, daemon=True)
        thread.start()
        
    def _process_file_thread(self):
        """Process files in background thread - FIXED VERSION"""
        try:
            # Update status
            self.after(0, lambda: self._update_status("🔄 Processing files...", config.COLOR_PRIMARY))
            self.after(0, lambda: self._show_progress(0.1))
            
            # Process using controller
            file_path = self.selected_files[0]  # Use first file
            
            self.after(0, lambda: self._update_status(f"📂 Loading {file_path.name}...", config.COLOR_PRIMARY))
            self.after(0, lambda: self._show_progress(0.3))
            
            # Call controller to do all the work
            prediction_data = self.controller.process_batch_file(str(file_path))
            
            self.after(0, lambda: self._show_progress(1.0))
            self.after(0, lambda: self._update_status("✅ Processing complete!", config.COLOR_SUCCESS))
            
            # Navigate to results view
            self.after(500, lambda: self._navigate_to_results(prediction_data))
            
        except Exception as e:
            error_msg = str(e)
            self.after(0, lambda: self._on_error(error_msg))
            
    def _on_error(self, error_message: str):
        """Handle processing error"""
        self._hide_progress()
        self._update_status(f"❌ {error_message}", config.COLOR_DANGER)
        self.is_processing = False
        self.enter_btn.config(state='normal', bg=config.COLOR_PRIMARY)
        
        messagebox.showinfo("Results", result_msg.strip())
        self._update_status("Ready to process another file", config.COLOR_TEXT)
        
    def _show_history(self):
        """Show history dialog"""
        messagebox.showinfo("History", "History feature coming soon!")
        
    def _create_tooltip(self, widget, text):
        """
        Create a simple tooltip for a widget
        
        Args:
            widget: Widget to attach tooltip to
            text: Tooltip text
        """
        def on_enter(event):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
            label = tk.Label(
                tooltip, 
                text=text, 
                background="yellow", 
                relief=tk.SOLID, 
                borderwidth=1,
                font=(config.FONT_FAMILY, 8)
            )
            label.pack()
            widget.tooltip = tooltip
            
        def on_leave(event):
            if hasattr(widget, 'tooltip'):
                widget.tooltip.destroy()
                
        widget.bind('<Enter>', on_enter)
        widget.bind('<Leave>', on_leave)

    def _navigate_to_results(self, prediction_data):
        """Navigate to results view"""
        self._hide_progress()
        self.is_processing = False
        self.enter_btn.config(state='normal', bg=config.COLOR_PRIMARY)
        
        # Show results view
        self.show_results(prediction_data)

    def _on_error(self, error_message: str):
        """Handle errors"""
        self._hide_progress()
        self._update_status(f"❌ Error: {error_message}", config.COLOR_DANGER)
        self.is_processing = False
        self.enter_btn.config(state='normal', bg=config.COLOR_PRIMARY)
        
        messagebox.showerror(
            "Processing Error",
            f"Failed to process file:\n\n{error_message}"
        )