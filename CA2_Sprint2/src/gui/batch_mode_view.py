"""
⚡ CYBERPUNK BATCH PROCESSING VIEW ⚡
Multi-file selection with neon cards and epic animations
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
from typing import Callable, List
import threading

from .. import config
from .cyber_components import NeonButton, CyberCard, CyberProgressBar


class FileCard(tk.Frame):
    """Neon file card with hover effects"""
    
    def __init__(self, parent, file_path: Path, on_remove: Callable):
        super().__init__(parent, bg=config.COLOR_BG_VOID)
        
        self.file_path = file_path
        self.on_remove = on_remove
        self.is_hovered = False
        
        self._create_ui()
        
    def _create_ui(self):
        """Create the file card UI"""
        # Main card canvas
        self.card = tk.Canvas(
            self,
            width=900,
            height=60,
            bg=config.COLOR_BG_VOID,
            highlightthickness=0
        )
        self.card.pack(pady=5)
        
        # Bind hover events
        self.card.bind("<Enter>", self._on_enter)
        self.card.bind("<Leave>", self._on_leave)
        
        self._draw()
    
    def _draw(self):
        """Draw the card"""
        self.card.delete("all")
        
        # Background
        bg_color = config.COLOR_BG_ELEVATED if self.is_hovered else config.COLOR_BG_CARD
        self._draw_rounded_rect(5, 5, 895, 55, 8, fill=bg_color, outline="")
        
        # Border with glow on hover
        border_color = config.COLOR_NEON_CYAN if self.is_hovered else config.COLOR_TEXT_MUTED
        if self.is_hovered:
            # Glow effect - using stipple instead of alpha
            for i in range(3):
                self._draw_rounded_rect(
                    3-i, 3-i, 897+i, 57+i, 8,
                    fill="", outline=border_color, stipple="gray25", width=1
                )
        
        self._draw_rounded_rect(5, 5, 895, 55, 8, fill="", outline=border_color, width=2)
        
        # File icon
        self.card.create_text(
            25, 30,
            text="📁",
            font=(config.FONT_FAMILY, 20),
            fill=config.COLOR_NEON_CYAN
        )
        
        # File name
        self.card.create_text(
            60, 22,
            text=self.file_path.name,
            anchor="w",
            font=(config.FONT_FAMILY, 11, "bold"),
            fill=config.COLOR_TEXT_PRIMARY
        )
        
        # File size
        size_mb = self.file_path.stat().st_size / (1024 * 1024)
        size_text = f"{size_mb:.2f} MB"
        self.card.create_text(
            60, 40,
            text=size_text,
            anchor="w",
            font=(config.FONT_FAMILY, 9),
            fill=config.COLOR_TEXT_MUTED
        )
        
        # Remove button
        remove_x = 850
        if self.is_hovered:
            # Hover state - show red X
            self.card.create_oval(
                remove_x-15, 15, remove_x+15, 45,
                fill=config.COLOR_DANGER,
                outline=""
            )
            self.card.create_text(
                remove_x, 30,
                text="✕",
                font=(config.FONT_FAMILY, 14, "bold"),
                fill=config.COLOR_TEXT_PRIMARY
            )
            
            # Make clickable
            self.card.tag_bind("remove", "<Button-1>", lambda e: self.on_remove(self.file_path))
            self.card.addtag_withtag("remove", "all")
    
    def _draw_rounded_rect(self, x1, y1, x2, y2, radius, **kwargs):
        """Draw rounded rectangle"""
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
        return self.card.create_polygon(points, smooth=True, **kwargs)
    
    def _on_enter(self, event):
        self.is_hovered = True
        self._draw()
        self.card.config(cursor="hand2")
    
    def _on_leave(self, event):
        self.is_hovered = False
        self._draw()
        self.card.config(cursor="")


class CyberBatchView(tk.Frame):
    """
    🌟 CYBERPUNK BATCH PROCESSING VIEW 🌟
    Multi-file processing with epic neon animations
    """
    
    def __init__(self, parent, controller, switch_to_realtime: Callable, show_results: Callable, show_history: Callable):
        super().__init__(parent, bg=config.COLOR_BG_VOID)
        
        self.controller = controller
        self.switch_to_realtime = switch_to_realtime
        self.show_results = show_results
        self.show_history = show_history
        self.selected_files: List[Path] = []
        self.is_processing = False
        
        # Setup outer scrollable wrapper
        self._setup_main_scroll()
    
    def _setup_main_scroll(self):
        """Setup the main scrollable area for the entire view"""
        # Canvas and Scrollbar
        self.main_canvas = tk.Canvas(self, bg=config.COLOR_BG_VOID, highlightthickness=0)
        self.main_scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.main_canvas.yview)
        
        # Scrollable Frame
        self.scroll_frame = tk.Frame(self.main_canvas, bg=config.COLOR_BG_VOID)
        self.scroll_frame.bind(
            "<Configure>",
            lambda e: self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))
        )
        
        # Window in canvas
        self.window_id = self.main_canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        
        # Resize internal frame to match canvas width
        self.main_canvas.bind("<Configure>", lambda e: self.main_canvas.itemconfig(self.window_id, width=e.width))
        self.main_canvas.configure(yscrollcommand=self.main_scrollbar.set)
        
        # Pack only the canvas (Hide the scrollbar visual, but keep functionality)
        self.main_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Bind MouseWheel for scrolling
        self.bind_mouse_scroll(self.main_canvas, self.scroll_frame)
        
        # Build the actual UI inside the scroll_frame
        self._setup_ui(self.scroll_frame)
    
    def bind_mouse_scroll(self, canvas, frame):
        """Bind mouse wheel events to canvas"""
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        # Bind when hovering over the frame
        frame.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _on_mousewheel))
        frame.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

    def _setup_ui(self, parent):
        """Setup the epic UI inside the scrollable frame"""
        # Header section
        self._create_header(parent)
        
        # File selection zone
        self._create_file_zone(parent)
        
        # Progress section
        self._create_progress_section(parent)
        
        # Action buttons
        self._create_action_section(parent)
    
    def _create_header(self, parent):
        """Create header with mode switcher"""
        header = tk.Frame(parent, bg=config.COLOR_BG_VOID)
        header.pack(fill=tk.X, pady=(0, 20))
        
        # Button Container
        btn_frame = tk.Frame(header, bg=config.COLOR_BG_VOID)
        btn_frame.pack(side=tk.LEFT)
        
        # Mode switcher button
        mode_btn = NeonButton(
            btn_frame,
            text="REAL-TIME MODE",
            command=self.switch_to_realtime,
            neon_color=config.COLOR_NEON_PINK,
            width=180,
            height=45,
            icon="🔴"
        )
        mode_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # History Button (NEW)
        history_btn = NeonButton(
            btn_frame,
            text="HISTORY",
            command=self.show_history,
            neon_color=config.COLOR_NEON_PURPLE,
            width=140,
            height=45,
            icon="📂"
        )
        history_btn.pack(side=tk.LEFT)
        
        # Title
        title_frame = tk.Frame(header, bg=config.COLOR_BG_VOID)
        title_frame.pack(side=tk.RIGHT, expand=True)
        
        title = tk.Label(
            title_frame,
            text="⚡ BATCH PROCESSING",
            font=(config.FONT_FAMILY_DISPLAY, 28, "bold"),
            fg=config.COLOR_NEON_CYAN,
            bg=config.COLOR_BG_VOID
        )
        title.pack(anchor="e")
        
        subtitle = tk.Label(
            title_frame,
            text="MULTI-FILE NEURAL ANALYSIS SYSTEM",
            font=(config.FONT_FAMILY, 10),
            fg=config.COLOR_TEXT_SECONDARY,
            bg=config.COLOR_BG_VOID
        )
        subtitle.pack(anchor="e")
    
    def _create_file_zone(self, parent):
        """Create file selection zone"""
        # Container with neon border
        zone_container = tk.Frame(parent, bg=config.COLOR_BG_VOID)
        zone_container.pack(fill=tk.BOTH, expand=True, pady=10, padx=20)
        
        # Header bar
        header_bar = tk.Frame(zone_container, bg=config.COLOR_BG_CARD, height=50)
        header_bar.pack(fill=tk.X)
        header_bar.pack_propagate(False)
        
        # Create neon border effect
        border_canvas = tk.Canvas(
            header_bar,
            height=50,
            bg=config.COLOR_BG_CARD,
            highlightthickness=0
        )
        border_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Draw top border
        border_canvas.create_line(
            0, 0, 1200, 0,
            fill=config.COLOR_NEON_CYAN,
            width=2
        )
        
        # Stipple effect
        border_canvas.create_line(
            0, 0, 1200, 0,
            fill=config.COLOR_NEON_CYAN,
            stipple="gray50",
            width=6
        )
        
        # Header text
        border_canvas.create_text(
            20, 25,
            text="📂 FILE QUEUE",
            anchor="w",
            font=(config.FONT_FAMILY, 12, "bold"),
            fill=config.COLOR_NEON_CYAN
        )
        
        # Stats
        self.stats_text = border_canvas.create_text(
            1100, 25,
            text="0 FILES",
            anchor="e",
            font=(config.FONT_FAMILY, 10),
            fill=config.COLOR_TEXT_SECONDARY
        )
        
        # File list with scrollbar
        list_container = tk.Frame(zone_container, bg=config.COLOR_BG_CARD)
        list_container.pack(fill=tk.BOTH, expand=True)
        
        # Canvas for scrolling
        self.file_canvas = tk.Canvas(
            list_container,
            bg=config.COLOR_BG_CARD,
            highlightthickness=0,
            height=300
        )
        self.file_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Internal scrollbar for file list (Removed visual, kept logic)
        scrollbar = tk.Scrollbar(
            list_container,
            orient=tk.VERTICAL,
            command=self.file_canvas.yview
        )
        
        self.file_canvas.configure(yscrollcommand=scrollbar.set)
        
        # Frame inside canvas
        self.file_list_frame = tk.Frame(self.file_canvas, bg=config.COLOR_BG_CARD)
        self.file_canvas_window = self.file_canvas.create_window(
            (0, 0),
            window=self.file_list_frame,
            anchor="nw"
        )
        
        # Update scroll region when frame changes
        self.file_list_frame.bind(
            "<Configure>",
            lambda e: self.file_canvas.configure(scrollregion=self.file_canvas.bbox("all"))
        )
        
        # Resize internal window to canvas width
        self.file_canvas.bind(
            "<Configure>",
            lambda e: self.file_canvas.itemconfig(self.file_canvas_window, width=e.width)
        )
        
        # Mouse scroll for file list
        self.bind_mouse_scroll(self.file_canvas, self.file_list_frame)

        # Empty state
        self.empty_label = tk.Label(
            self.file_list_frame,
            text="⚡ NO FILES LOADED ⚡\n\nCLICK 'ADD FILES' TO BEGIN ANALYSIS",
            font=(config.FONT_FAMILY, 12),
            fg=config.COLOR_TEXT_MUTED,
            bg=config.COLOR_BG_CARD,
            justify=tk.CENTER,
            pady=80
        )
        self.empty_label.pack(fill=tk.BOTH, expand=True)
        
        # Bottom border
        bottom_border = tk.Canvas(
            zone_container,
            height=2,
            bg=config.COLOR_BG_CARD,
            highlightthickness=0
        )
        bottom_border.pack(fill=tk.X)
        bottom_border.create_line(
            0, 0, 1200, 0,
            fill=config.COLOR_NEON_PINK,
            width=2
        )
        
        # Button row
        btn_row = tk.Frame(zone_container, bg=config.COLOR_BG_CARD, height=70)
        btn_row.pack(fill=tk.X)
        btn_row.pack_propagate(False)
        
        btn_container = tk.Frame(btn_row, bg=config.COLOR_BG_CARD)
        btn_container.pack(pady=10)
        
        self.add_btn = NeonButton(
            btn_container,
            text="ADD FILES",
            command=self._add_files,
            neon_color=config.COLOR_NEON_CYAN,
            width=160,
            icon="+"
        )
        self.add_btn.pack(side=tk.LEFT, padx=5)
        
        self.clear_btn = NeonButton(
            btn_container,
            text="CLEAR ALL",
            command=self._clear_files,
            neon_color=config.COLOR_NEON_ORANGE,
            width=160,
            icon="✕"
        )
        self.clear_btn.pack(side=tk.LEFT, padx=5)
    
    def _create_progress_section(self, parent):
        """Create progress display section"""
        self.progress_container = tk.Frame(parent, bg=config.COLOR_BG_VOID)
        
        # Status message
        self.status_label = tk.Label(
            self.progress_container,
            text="",
            font=(config.FONT_FAMILY, 11),
            fg=config.COLOR_TEXT_SECONDARY,
            bg=config.COLOR_BG_VOID
        )
        self.status_label.pack(pady=(10, 5))
        
        # Progress bar
        self.progress_bar = CyberProgressBar(
            self.progress_container,
            width=800,
            height=35
        )
    
    def _create_action_section(self, parent):
        """Create main action button"""
        action_frame = tk.Frame(parent, bg=config.COLOR_BG_VOID)
        action_frame.pack(pady=30)
        
        self.process_btn = NeonButton(
            action_frame,
            text="INITIATE ANALYSIS",
            command=self._process_files,
            neon_color=config.COLOR_NEON_PINK,
            width=320,
            height=60,
            icon="⚡"
        )
        self.process_btn.pack()
    
    def _add_files(self):
        """Add files to queue"""
        if self.is_processing:
            return
        
        file_paths = filedialog.askopenfilenames(
            title="SELECT SENSOR DATA FILES",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        
        if not file_paths:
            return
        
        # Add files (avoid duplicates)
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
        """Clear all files"""
        if self.is_processing:
            return
        
        self.selected_files = []
        self._update_file_list()
    
    def _update_file_list(self):
        """Update the file list display"""
        # Clear existing
        for widget in self.file_list_frame.winfo_children():
            widget.destroy()
        
        if not self.selected_files:
            # Show empty state
            self.empty_label = tk.Label(
                self.file_list_frame,
                text="⚡ NO FILES LOADED ⚡\n\nCLICK 'ADD FILES' TO BEGIN ANALYSIS",
                font=(config.FONT_FAMILY, 12),
                fg=config.COLOR_TEXT_MUTED,
                bg=config.COLOR_BG_CARD,
                justify=tk.CENTER,
                pady=80
            )
            self.empty_label.pack(fill=tk.BOTH, expand=True)
            
            # Update stats (using canvas)
            self.file_canvas.itemconfig(self.stats_text, text="0 FILES")
        else:
            # Show file cards
            for file_path in self.selected_files:
                card = FileCard(
                    self.file_list_frame,
                    file_path,
                    on_remove=self._remove_file
                )
                card.pack(fill=tk.X, padx=10, pady=2)
            
            # Update stats
            total_size = sum(f.stat().st_size for f in self.selected_files) / (1024 * 1024)
            stats_text = f"{len(self.selected_files)} FILES • {total_size:.1f} MB"
            self.file_canvas.itemconfig(self.stats_text, text=stats_text)
    
    def _remove_file(self, file_path: Path):
        """Remove a file from the list"""
        if self.is_processing:
            return
        
        self.selected_files = [f for f in self.selected_files if f != file_path]
        self._update_file_list()
    
    def _process_files(self):
        """Process all files"""
        if self.is_processing:
            return
        
        if not self.selected_files:
            messagebox.showerror(
                "NO FILES",
                "Please add at least one file to the queue!"
            )
            return
        
        # Start processing
        self.is_processing = True
        self.process_btn.set_disabled(True)
        self.add_btn.set_disabled(True)
        self.clear_btn.set_disabled(True)
        
        # Show progress
        self.progress_container.pack(before=self.process_btn.master, pady=20)
        
        # Start in thread
        thread = threading.Thread(target=self._process_thread, daemon=True)
        thread.start()
    
    def _process_thread(self):
        """Process files in background"""
        try:
            def progress_callback(progress, message):
                self.after(0, lambda: self._update_progress(progress, message))
            
            # Process
            prediction_data = self.controller.process_batch_files(
                [str(f) for f in self.selected_files],
                progress_callback=progress_callback
            )
            
            # Complete
            self.after(0, lambda: self._on_complete(prediction_data))
            
        except Exception as e:
            self.after(0, lambda: self._on_error(str(e)))
    
    def _update_progress(self, progress: float, message: str):
        """Update progress display"""
        self.progress_bar.set_progress(progress)
        self.status_label.config(text=message, fg=config.COLOR_NEON_CYAN)
    
    def _on_complete(self, prediction_data):
        """Handle completion"""
        self.progress_bar.set_progress(1.0)
        self.status_label.config(
            text="✓ ANALYSIS COMPLETE",
            fg=config.COLOR_NEON_GREEN
        )
        
        self.after(500, lambda: self._navigate_to_results(prediction_data))
    
    def _navigate_to_results(self, prediction_data):
        """Navigate to results"""
        self.progress_container.pack_forget()
        self.is_processing = False
        self.process_btn.set_disabled(False)
        self.add_btn.set_disabled(False)
        self.clear_btn.set_disabled(False)
        
        self.show_results(prediction_data)
    
    def _on_error(self, error_msg: str):
        """Handle error"""
        self.progress_container.pack_forget()
        self.is_processing = False
        self.process_btn.set_disabled(False)
        self.add_btn.set_disabled(False)
        self.clear_btn.set_disabled(False)
        
        messagebox.showerror("ERROR", f"Processing failed:\n\n{error_msg}")