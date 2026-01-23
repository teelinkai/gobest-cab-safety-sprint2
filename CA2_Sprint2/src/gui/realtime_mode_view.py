"""
Real-Time Mode View Module
Handles the UI for real-time prediction mode
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
from typing import Callable

from .. import config


class RealtimeModeView(tk.Frame):
    """
    View for Real-Time Mode
    Allows users to input trip data for real-time prediction
    """
    
    def __init__(
        self, 
        parent: tk.Widget, 
        controller,
        switch_to_batch: Callable,
        show_results: Callable
    ):
        """
        Initialize Real-Time Mode View
        
        Args:
            parent: Parent widget
            controller: Mode controller instance
            switch_to_batch: Callback to switch to batch mode
            show_results: Callback to show results view
        """
        super().__init__(parent, bg=config.COLOR_BACKGROUND)
        
        self.controller = controller
        self.switch_to_batch = switch_to_batch
        self.show_results = show_results
        self.selected_file_path: Path = None
        
        self._setup_ui()
        
    def _setup_ui(self):
        """Setup the UI components"""
        # Top bar with switch mode and history buttons
        self._create_top_bar()
        
        # Title
        self._create_title()
        
        # File input section (optional)
        self._create_file_input()
        
        # Form details section
        self._create_form_details()
        
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
            command=self.switch_to_batch
        )
        switch_btn.pack(side=tk.LEFT)
        
        # Tooltip for switch button
        self._create_tooltip(switch_btn, "Switch to Batch Mode")
        
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
            text="REAL TIME MODE",
            font=(config.FONT_FAMILY, config.FONT_SIZE_TITLE, "bold"),
            bg=config.COLOR_BACKGROUND,
            fg=config.COLOR_TEXT
        )
        title_label.pack()
        
    def _create_file_input(self):
        """Create optional file input section"""
        file_frame = tk.Frame(self, bg=config.COLOR_BACKGROUND)
        file_frame.pack(pady=10)
        
        # File entry with browse button
        self.file_entry = tk.Entry(
            file_frame,
            width=40,
            font=(config.FONT_FAMILY, config.FONT_SIZE_LABEL),
            relief=tk.SOLID,
            borderwidth=1
        )
        self.file_entry.pack(side=tk.LEFT, padx=(0, 10), ipady=5)
        self.file_entry.insert(0, "Enter File (Optional?)")
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
        
    def _create_form_details(self):
        """Create the form details section"""
        form_frame = tk.LabelFrame(
            self,
            text="Trip Details",
            font=(config.FONT_FAMILY, config.FONT_SIZE_LABEL, "bold"),
            bg=config.COLOR_BACKGROUND,
            fg=config.COLOR_TEXT,
            padx=20,
            pady=15
        )
        form_frame.pack(pady=20, padx=50, fill=tk.BOTH, expand=True)
        
        # Create form fields
        fields = [
            ("Booking ID:", "booking_id"),
            ("Driver ID:", "driver_id"),
            ("Trip Duration (sec):", "trip_duration"),
        ]
        
        self.form_entries = {}
        
        for idx, (label_text, field_name) in enumerate(fields):
            row_frame = tk.Frame(form_frame, bg=config.COLOR_BACKGROUND)
            row_frame.pack(fill=tk.X, pady=5)
            
            label = tk.Label(
                row_frame,
                text=label_text,
                font=(config.FONT_FAMILY, config.FONT_SIZE_LABEL),
                bg=config.COLOR_BACKGROUND,
                fg=config.COLOR_TEXT,
                width=20,
                anchor='w'
            )
            label.pack(side=tk.LEFT)
            
            entry = tk.Entry(
                row_frame,
                font=(config.FONT_FAMILY, config.FONT_SIZE_LABEL),
                relief=tk.SOLID,
                borderwidth=1
            )
            entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=3)
            
            self.form_entries[field_name] = entry
        
        # Note about data input
        note_label = tk.Label(
            form_frame,
            text="Note: For real-time mode, you can either load a partial trip\nor let the system simulate real-time sensor data.",
            font=(config.FONT_FAMILY, 8, "italic"),
            bg=config.COLOR_BACKGROUND,
            fg=config.COLOR_TEXT_LIGHT,
            justify=tk.LEFT
        )
        note_label.pack(pady=(15, 0))
        
    def _create_enter_button(self):
        """Create the main Enter button"""
        enter_btn = tk.Button(
            self,
            text="Enter",
            font=(config.FONT_FAMILY, config.FONT_SIZE_BUTTON + 2, "bold"),
            bg=config.COLOR_PRIMARY,
            fg="white",
            width=15,
            height=2,
            relief=tk.RAISED,
            cursor="hand2",
            command=self._process_realtime
        )
        enter_btn.pack(pady=20)
        
        # Bind hover effects
        enter_btn.bind('<Enter>', lambda e: enter_btn.config(bg=config.COLOR_PRIMARY_DARK))
        enter_btn.bind('<Leave>', lambda e: enter_btn.config(bg=config.COLOR_PRIMARY))
        
    def _browse_file(self):
        """Open file dialog to select CSV file"""
        file_path = filedialog.askopenfilename(
            title="Select CSV File (Optional)",
            filetypes=[
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.selected_file_path = Path(file_path)
            self.file_entry.config(state='normal', fg=config.COLOR_TEXT)
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, self.selected_file_path.name)
            self.file_entry.config(state='readonly')
            
    def _process_realtime(self):
        """Process real-time prediction (placeholder for now)"""
        # Get form data
        booking_id = self.form_entries['booking_id'].get()
        driver_id = self.form_entries['driver_id'].get()
        
        if not booking_id and not self.selected_file_path:
            messagebox.showwarning(
                "Missing Data", 
                "Please either enter a Booking ID or select a file!"
            )
            return
            
        # TODO: Implement actual real-time processing
        messagebox.showinfo(
            "Processing", 
            "Real-time prediction processing...\n\nThis is a placeholder - actual implementation pending."
        )
        
        # Mock prediction data for testing
        prediction_data = {
            'mode': 'realtime',
            'booking_id': booking_id or 'N/A',
            'prediction': 'SAFE',  # or 'DANGEROUS'
            'confidence': 0.92,
            'timestamp': '2026-01-22 14:35:00'
        }
        
        # Show results (commented out for now during layout testing)
        # self.show_results(prediction_data)
        
    def _show_history(self):
        """Show history dialog (placeholder)"""
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
