"""
🗂️ CYNOSURE RETRO COMPONENTS 🗂️
Reusable UI components for the Cynosure terminal theme
Includes Bill Cipher logo widget
"""

import tkinter as tk
from PIL import Image, ImageTk
from pathlib import Path
from .. import config


class BillCipherLogo(tk.Label):
    """Bill Cipher logo widget with transparent background"""
    
    def __init__(self, parent, size=80, **kwargs):
        super().__init__(parent, bg=config.COLOR_BG_PAPER, **kwargs)
        
        # Path to logo
        logo_path = config.ASSETS_DIR / "bill_cipher_logo.png"
        
        # If logo doesn't exist, show text fallback
        if not logo_path.exists():
            self.config(
                text="👁️",
                font=("Arial", size),
                fg=config.COLOR_ACCENT_BLUE
            )
            return
        
        try:
            # Load and resize logo
            img = Image.open(logo_path)
            img.thumbnail((size, size), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            self.photo = ImageTk.PhotoImage(img)
            self.config(image=self.photo)
            
        except Exception as e:
            # Fallback to emoji if image fails
            self.config(
                text="👁️",
                font=("Arial", size),
                fg=config.COLOR_ACCENT_BLUE
            )


class RetroButton(tk.Canvas):
    """Retro paper-style button"""
    
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


# Alias for compatibility
NeonButton = RetroButton


class CyberCard(tk.Frame):
    """Retro card container"""
    
    def __init__(self, parent, **kwargs):
        super().__init__(
            parent,
            bg=config.COLOR_BG_CARD,
            highlightbackground=config.COLOR_BORDER_DARK,
            highlightthickness=2,
            **kwargs
        )


class ParticleField(tk.Canvas):
    """Empty placeholder - no particles in retro theme"""
    
    def __init__(self, parent, width, height, **kwargs):
        super().__init__(
            parent,
            width=width,
            height=height,
            bg=config.COLOR_BG_PAPER,
            highlightthickness=0
        )
        self.width = width
        self.height = height


class GlitchText(tk.Label):
    """Simple text label - no glitch in retro theme"""
    
    def __init__(self, parent, text, **kwargs):
        super().__init__(parent, text=text, **kwargs)
    
    def start_glitch(self, duration):
        """Placeholder - no animation"""
        pass
