"""
Results View Module
Displays prediction results with history
"""

import tkinter as tk
from tkinter import ttk
from typing import Callable, Dict

from .. import config


class ResultsView(tk.Frame):
    """
    View for displaying prediction results
    Shows whether the trip is SAFE or DANGEROUS with confidence
    """
    
    def __init__(
        self, 
        parent: tk.Widget, 
        prediction_data: Dict,
        on_predict_again: Callable
    ):
        """
        Initialize Results View
        
        Args:
            parent: Parent widget
            prediction_data: Dictionary containing prediction results
            on_predict_again: Callback to return to prediction mode
        """
        super().__init__(parent, bg=config.COLOR_BACKGROUND)
        
        self.prediction_data = prediction_data
        self.on_predict_again = on_predict_again
        
        self._setup_ui()
        
    def _setup_ui(self):
        """Setup the UI components"""
        # Title with prediction result
        self._create_result_title()
        
        # Details section
        self._create_details_section()
        
        # History section
        self._create_history_section()
        
        # Predict Again button
        self._create_predict_again_button()
        
    def _create_result_title(self):
        """Create the result title with color coding"""
        title_frame = tk.Frame(self, bg=config.COLOR_BACKGROUND)
        title_frame.pack(pady=(50, 20))
        
        prediction = self.prediction_data.get('prediction', 'UNKNOWN')
        
        if prediction == 'DANGEROUS':
            color = config.COLOR_DANGER
            icon = "⚠️"
        elif prediction == 'SAFE':
            color = config.COLOR_SUCCESS
            icon = "✓"
        else:
            color = config.COLOR_TEXT
            icon = "?"
        
        # Icon and text
        result_label = tk.Label(
            title_frame,
            text=f"{icon} {prediction}",
            font=(config.FONT_FAMILY, config.FONT_SIZE_TITLE + 6, "bold"),
            bg=config.COLOR_BACKGROUND,
            fg=color
        )
        result_label.pack()
        
        # Confidence
        confidence = self.prediction_data.get('confidence', 0) * 100
        confidence_label = tk.Label(
            title_frame,
            text=f"Confidence: {confidence:.1f}%",
            font=(config.FONT_FAMILY, config.FONT_SIZE_SUBTITLE),
            bg=config.COLOR_BACKGROUND,
            fg=config.COLOR_TEXT_LIGHT
        )
        confidence_label.pack(pady=(5, 0))
        
    def _create_details_section(self):
        """Create the details section showing prediction info"""
        details_frame = tk.LabelFrame(
            self,
            text="Prediction Details",
            font=(config.FONT_FAMILY, config.FONT_SIZE_LABEL, "bold"),
            bg=config.COLOR_BACKGROUND,
            fg=config.COLOR_TEXT,
            padx=20,
            pady=15
        )
        details_frame.pack(pady=10, padx=50, fill=tk.X)
        
        # Create detail rows
        details = [
            ("Mode:", self.prediction_data.get('mode', 'N/A').upper()),
            ("File/Booking ID:", self.prediction_data.get('file') or self.prediction_data.get('booking_id', 'N/A')),
            ("Timestamp:", self.prediction_data.get('timestamp', 'N/A')),
        ]
        
        for label_text, value_text in details:
            row_frame = tk.Frame(details_frame, bg=config.COLOR_BACKGROUND)
            row_frame.pack(fill=tk.X, pady=3)
            
            label = tk.Label(
                row_frame,
                text=label_text,
                font=(config.FONT_FAMILY, config.FONT_SIZE_LABEL, "bold"),
                bg=config.COLOR_BACKGROUND,
                fg=config.COLOR_TEXT,
                width=15,
                anchor='w'
            )
            label.pack(side=tk.LEFT)
            
            value = tk.Label(
                row_frame,
                text=value_text,
                font=(config.FONT_FAMILY, config.FONT_SIZE_LABEL),
                bg=config.COLOR_BACKGROUND,
                fg=config.COLOR_TEXT_LIGHT,
                anchor='w'
            )
            value.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
    def _create_history_section(self):
        """Create the session history section"""
        history_frame = tk.LabelFrame(
            self,
            text="History for this session",
            font=(config.FONT_FAMILY, config.FONT_SIZE_LABEL, "bold"),
            bg=config.COLOR_BACKGROUND,
            fg=config.COLOR_TEXT,
            padx=20,
            pady=15
        )
        history_frame.pack(pady=10, padx=50, fill=tk.BOTH, expand=True)
        
        # Create scrollable history list
        history_canvas = tk.Canvas(
            history_frame,
            bg='white',
            highlightthickness=1,
            highlightbackground=config.COLOR_TEXT_LIGHT
        )
        scrollbar = ttk.Scrollbar(
            history_frame,
            orient="vertical",
            command=history_canvas.yview
        )
        scrollable_frame = tk.Frame(history_canvas, bg='white')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: history_canvas.configure(scrollregion=history_canvas.bbox("all"))
        )
        
        history_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        history_canvas.configure(yscrollcommand=scrollbar.set)
        
        # Add placeholder history items
        # TODO: Connect to actual history data from controller
        placeholder_text = tk.Label(
            scrollable_frame,
            text="No previous predictions in this session",
            font=(config.FONT_FAMILY, config.FONT_SIZE_LABEL, "italic"),
            bg='white',
            fg=config.COLOR_TEXT_LIGHT,
            pady=20
        )
        placeholder_text.pack()
        
        history_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
    def _create_predict_again_button(self):
        """Create the Predict Again button"""
        predict_again_btn = tk.Button(
            self,
            text="Predict Again",
            font=(config.FONT_FAMILY, config.FONT_SIZE_BUTTON + 2, "bold"),
            bg=config.COLOR_PRIMARY,
            fg="white",
            width=15,
            height=2,
            relief=tk.RAISED,
            cursor="hand2",
            command=self.on_predict_again
        )
        predict_again_btn.pack(pady=20)
        
        # Bind hover effects
        predict_again_btn.bind('<Enter>', lambda e: predict_again_btn.config(bg=config.COLOR_PRIMARY_DARK))
        predict_again_btn.bind('<Leave>', lambda e: predict_again_btn.config(bg=config.COLOR_PRIMARY))
