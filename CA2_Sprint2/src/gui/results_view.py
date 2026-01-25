"""
Results View Module
Displays prediction results with history
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Callable, Dict
from pathlib import Path
from datetime import datetime

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
        
        # Check if batch mode (has total_trips) or realtime mode (has prediction)
        mode = self.prediction_data.get('mode', 'unknown')
        
        if mode == 'batch':
            # Batch mode - show summary
            total = self.prediction_data.get('total_trips', 0)
            dangerous = self.prediction_data.get('dangerous_count', 0)
            dangerous_pct = self.prediction_data.get('dangerous_pct', 0)
            
            result_label = tk.Label(
                title_frame,
                text=f"📊 BATCH RESULTS",
                font=(config.FONT_FAMILY, config.FONT_SIZE_TITLE, "bold"),
                bg=config.COLOR_BACKGROUND,
                fg=config.COLOR_PRIMARY
            )
            result_label.pack()
            
            summary_label = tk.Label(
                title_frame,
                text=f"{total:,} trips analyzed",
                font=(config.FONT_FAMILY, config.FONT_SIZE_SUBTITLE),
                bg=config.COLOR_BACKGROUND,
                fg=config.COLOR_TEXT_LIGHT
            )
            summary_label.pack(pady=(5, 0))
            
            # Show dangerous count prominently
            dangerous_label = tk.Label(
                title_frame,
                text=f"⚠️ {dangerous:,} DANGEROUS ({dangerous_pct:.1f}%)",
                font=(config.FONT_FAMILY, 18, "bold"),
                bg=config.COLOR_BACKGROUND,
                fg=config.COLOR_DANGER
            )
            dangerous_label.pack(pady=(10, 0))
            
        else:
            # Realtime mode - show single prediction
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
        
        mode = self.prediction_data.get('mode', 'N/A')
        
        if mode == 'batch':
            # Batch mode - show summary stats
            total = self.prediction_data.get('total_trips', 0)
            dangerous = self.prediction_data.get('dangerous_count', 0)
            safe = self.prediction_data.get('safe_count', 0)
            dangerous_pct = self.prediction_data.get('dangerous_pct', 0)
            safe_pct = self.prediction_data.get('safe_pct', 0)
            
            details = [
                ("Mode:", "BATCH"),
                ("File:", self.prediction_data.get('file', 'N/A')),
                ("Total Trips:", f"{total:,}"),
                ("🔴 Dangerous:", f"{dangerous:,} ({dangerous_pct:.1f}%)"),
                ("🟢 Safe:", f"{safe:,} ({safe_pct:.1f}%)"),
                ("Timestamp:", self.prediction_data.get('timestamp', 'N/A')),
            ]
        else:
            # Realtime mode - show single prediction
            details = [
                ("Mode:", mode.upper()),
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
                width=20,
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
        
        # Add export button for batch mode
        if mode == 'batch' and 'results_df' in self.prediction_data:
            export_btn = tk.Button(
                details_frame,
                text="📤 Export Detailed Results",
                font=(config.FONT_FAMILY, config.FONT_SIZE_BUTTON, "bold"),
                bg=config.COLOR_SUCCESS,
                fg="white",
                padx=20,
                pady=8,
                relief=tk.RAISED,
                cursor="hand2",
                command=self._export_results
            )
            export_btn.pack(pady=(15, 5))
            
            # Hover effects
            export_btn.bind('<Enter>', lambda e: export_btn.config(bg="#27ae60"))
            export_btn.bind('<Leave>', lambda e: export_btn.config(bg=config.COLOR_SUCCESS))
            
            # Info label
            info_label = tk.Label(
                details_frame,
                text="Export full predictions with bookingIDs and confidence scores",
                font=(config.FONT_FAMILY, 8, "italic"),
                bg=config.COLOR_BACKGROUND,
                fg=config.COLOR_TEXT_LIGHT
            )
            info_label.pack()
        
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
    
    def _export_results(self):
        """Export detailed prediction results to CSV in Safety Predictions folder"""
        try:
            results_df = self.prediction_data.get('results_df')
            
            if results_df is None:
                messagebox.showerror(
                    "Export Error",
                    "No detailed results available to export!"
                )
                return
            
            # Create Safety Predictions folder in project directory
            base_dir = Path(__file__).parent.parent.parent
            export_dir = base_dir / "Safety Predictions"
            export_dir.mkdir(exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            input_file = self.prediction_data.get('file', 'unknown')
            input_name = Path(input_file).stem if input_file else 'predictions'
            
            default_filename = f"{input_name}_predictions_{timestamp}.csv"
            default_path = export_dir / default_filename
            
            # Ask user for save location (defaulting to Safety Predictions folder)
            file_path = filedialog.asksaveasfilename(
                title="Export Prediction Results",
                initialdir=export_dir,
                initialfile=default_filename,
                defaultextension=".csv",
                filetypes=[
                    ("CSV files", "*.csv"),
                    ("All files", "*.*")
                ]
            )
            
            if file_path:
                # Export to CSV
                results_df.to_csv(file_path, index=False)
                
                # Get summary stats
                total = len(results_df)
                dangerous = (results_df['prediction'] == 1).sum()
                
                messagebox.showinfo(
                    "Export Successful",
                    f"✅ Results exported successfully!\n\n"
                    f"Location: {file_path}\n\n"
                    f"Summary:\n"
                    f"  • Total trips: {total:,}\n"
                    f"  • Dangerous: {dangerous:,}\n"
                    f"  • Safe: {total - dangerous:,}\n\n"
                    f"File contains:\n"
                    f"  • bookingID\n"
                    f"  • prediction (0=Safe, 1=Dangerous)\n"
                    f"  • prediction_label (SAFE/DANGEROUS)\n"
                    f"  • probability_dangerous\n"
                    f"  • probability_safe\n"
                    f"  • confidence"
                )
                
                print(f"✅ Results exported to: {file_path}")
                
        except Exception as e:
            messagebox.showerror(
                "Export Error",
                f"Failed to export results:\n\n{str(e)}"
            )
            print(f"❌ Export error: {e}")
