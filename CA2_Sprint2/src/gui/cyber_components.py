"""
🔥 CYBERPUNK ANIMATED COMPONENTS 🔥
Custom Tkinter widgets with neon effects, animations, and particle systems
"""

import tkinter as tk
from tkinter import Canvas
import random
import math
from typing import Callable, List, Tuple, Optional

from .. import config


class NeonButton(Canvas):
    """
    🌟 Cyberpunk Neon Button with glow effects
    Features: Hover animations, pulsing glow, click feedback
    """
    
    def __init__(self, parent, text: str, command: Callable = None,
                 neon_color: str = None, width: int = 200, height: int = 50,
                 icon: str = "", **kwargs):
        super().__init__(
            parent, 
            width=width, 
            height=height,
            bg=config.COLOR_BG_VOID,
            highlightthickness=0,
            **kwargs
        )
        
        self.text = text
        self.icon = icon
        self.command = command
        self.neon_color = neon_color or config.COLOR_NEON_CYAN
        self.width = width
        self.height = height
        self.is_hovered = False
        self.is_pressed = False
        self.is_disabled = False
        self.glow_alpha = 0
        self.glow_direction = 1
        self.animation_id = None
        
        self._draw()
        self._bind_events()
        
    def _draw(self):
        """Draw the neon button with glow effects"""
        self.delete("all")
        
        # Determine colors based on state
        if self.is_disabled:
            base_color = config.COLOR_TEXT_MUTED
            glow_color = config.COLOR_BG_ELEVATED
        else:
            base_color = self.neon_color
            glow_color = self.neon_color
        
        # Draw outer glow (Simulated with stipple instead of alpha)
        if self.is_hovered and not self.is_disabled:
            glow_size = 8 if not self.is_pressed else 4
            for i in range(3):
                self.create_rounded_rect(
                    10 - glow_size + i*2, 10 - glow_size + i*2,
                    self.width - 10 + glow_size - i*2, self.height - 10 + glow_size - i*2,
                    radius=12,
                    fill=glow_color,
                    stipple="gray25", 
                    outline=""
                )
        
        # Main button background
        bg_color = config.COLOR_BG_ELEVATED if not self.is_pressed else config.COLOR_BG_CARD
        self.create_rounded_rect(
            10, 10, self.width - 10, self.height - 10,
            radius=10,
            fill=bg_color,
            outline=""
        )
        
        # Neon border
        border_width = 3 if self.is_hovered else 2
        self.create_rounded_rect(
            10, 10, self.width - 10, self.height - 10,
            radius=10,
            fill="",
            outline=base_color,
            width=border_width
        )
        
        # Inner highlight (top edge)
        if not self.is_pressed:
            self.create_line(
                20, 15, self.width - 20, 15,
                fill=base_color, 
                width=1
            )
        
        # Text with icon
        text_color = base_color if not self.is_disabled else config.COLOR_TEXT_MUTED
        full_text = f"{self.icon} {self.text}" if self.icon else self.text
        
        y_offset = 2 if self.is_pressed else 0
        self.create_text(
            self.width / 2, self.height / 2 + y_offset,
            text=full_text,
            fill=text_color,
            font=(config.FONT_FAMILY, config.FONT_SIZE_BUTTON, "bold")
        )
        
        # Add scanline effect when hovered
        if self.is_hovered and not self.is_disabled:
            for i in range(0, self.height, 4):
                self.create_line(
                    15, i, self.width - 15, i,
                    fill=config.COLOR_TEXT_PRIMARY,
                    stipple="gray12",
                    width=1
                )
    
    def create_rounded_rect(self, x1, y1, x2, y2, radius=25, **kwargs):
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
    
    def _bind_events(self):
        """Bind mouse events"""
        self.bind("<Button-1>", self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
    
    def _on_press(self, event):
        if not self.is_disabled:
            self.is_pressed = True
            self._draw()
    
    def _on_release(self, event):
        if not self.is_disabled:
            self.is_pressed = False
            self._draw()
            if self.command:
                self.command()
    
    def _on_enter(self, event):
        if not self.is_disabled:
            self.is_hovered = True
            self._draw()
            self.config(cursor="hand2")
            self._start_glow_pulse()
    
    def _on_leave(self, event):
        self.is_hovered = False
        self._draw()
        self.config(cursor="")
        self._stop_glow_pulse()
    
    def _start_glow_pulse(self):
        """Start pulsing glow animation"""
        if self.animation_id is None:
            self._pulse_glow()
    
    def _pulse_glow(self):
        """Animate the glow effect"""
        if not self.is_hovered:
            return
        
        self.glow_alpha += self.glow_direction * 5
        if self.glow_alpha >= 100:
            self.glow_direction = -1
        elif self.glow_alpha <= 30:
            self.glow_direction = 1
        
        self._draw()
        self.animation_id = self.after(config.GLOW_PULSE_SPEED, self._pulse_glow)
    
    def _stop_glow_pulse(self):
        """Stop glow animation"""
        if self.animation_id:
            self.after_cancel(self.animation_id)
            self.animation_id = None
    
    def set_disabled(self, disabled: bool):
        """Enable/disable the button"""
        self.is_disabled = disabled
        self._draw()


class CyberCard(Canvas):
    """
    🎴 Cyberpunk Card with animated border and glow
    """
    
    def __init__(self, parent, width: int, height: int, 
                 border_color: str = None, **kwargs):
        super().__init__(
            parent,
            width=width,
            height=height,
            bg=config.COLOR_BG_VOID,
            highlightthickness=0,
            **kwargs
        )
        
        self.width = width
        self.height = height
        self.border_color = border_color or config.COLOR_NEON_CYAN
        self.scan_offset = 0
        self.animation_id = None
        
        self._draw()
        self.bind("<Enter>", lambda e: self._start_scan())
        self.bind("<Leave>", lambda e: self._stop_scan())
    
    def _draw(self):
        """Draw the card"""
        self.delete("all")
        
        # Background
        self.create_rounded_rect(
            5, 5, self.width - 5, self.height - 5,
            radius=config.CARD_BORDER_RADIUS,
            fill=config.COLOR_BG_CARD,
            outline=""
        )
        
        # Neon border
        self.create_rounded_rect(
            5, 5, self.width - 5, self.height - 5,
            radius=config.CARD_BORDER_RADIUS,
            fill="",
            outline=self.border_color,
            width=2
        )
        
        # Corner accents (cyberpunk style)
        corner_size = 15
        corners = [
            (10, 10),  # Top-left
            (self.width - 10 - corner_size, 10),  # Top-right
            (10, self.height - 10 - corner_size),  # Bottom-left
            (self.width - 10 - corner_size, self.height - 10 - corner_size)  # Bottom-right
        ]
        
        for x, y in corners:
            # Horizontal line
            self.create_line(x, y, x + corner_size, y, 
                           fill=self.border_color, width=3)
            # Vertical line
            self.create_line(x, y, x, y + corner_size,
                           fill=self.border_color, width=3)
    
    def create_rounded_rect(self, x1, y1, x2, y2, radius=25, **kwargs):
        """Create rounded rectangle"""
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
    
    def _start_scan(self):
        """Start scanning animation"""
        if self.animation_id is None:
            self._animate_scan()
    
    def _animate_scan(self):
        """Animate scanning line"""
        self.scan_offset = (self.scan_offset + 2) % self.height
        
        # Draw scan line
        self.delete("scanline")
        self.create_line(
            10, self.scan_offset, self.width - 10, self.scan_offset,
            fill=self.border_color, 
            stipple="gray50",
            width=2,
            tags="scanline"
        )
        
        self.animation_id = self.after(20, self._animate_scan)
    
    def _stop_scan(self):
        """Stop scanning animation"""
        if self.animation_id:
            self.after_cancel(self.animation_id)
            self.animation_id = None
        self.delete("scanline")


class ParticleField(Canvas):
    """
    ✨ Animated particle background (Matrix/Cyberpunk style)
    """
    
    def __init__(self, parent, width: int, height: int, **kwargs):
        super().__init__(
            parent,
            width=width,
            height=height,
            bg=config.COLOR_BG_VOID,
            highlightthickness=0,
            **kwargs
        )
        
        self.width = width
        self.height = height
        self.particles: List[dict] = []
        self.animation_id = None
        
        self._create_particles()
        self._animate()
    
    def _create_particles(self):
        """Create particle objects"""
        for _ in range(config.PARTICLE_COUNT):
            particle = {
                'x': random.randint(0, self.width),
                'y': random.randint(0, self.height),
                'vx': random.uniform(-0.5, 0.5),
                'vy': random.uniform(0.5, 2),
                'size': random.randint(1, 3),
                'color': random.choice([
                    config.COLOR_NEON_CYAN,
                    config.COLOR_NEON_PINK,
                    config.COLOR_NEON_PURPLE
                ]),
                'alpha': random.randint(30, 80)
            }
            self.particles.append(particle)
    
    def _animate(self):
        """Animate particles"""
        self.delete("all")
        
        for particle in self.particles:
            # Update position
            particle['x'] += particle['vx']
            particle['y'] += particle['vy']
            
            # Wrap around edges
            if particle['y'] > self.height:
                particle['y'] = 0
                particle['x'] = random.randint(0, self.width)
            if particle['x'] < 0 or particle['x'] > self.width:
                particle['x'] = random.randint(0, self.width)
            
            # Build color safely (strip alpha for Tkinter)
            base_color = particle['color'][:7]  # keep only #RRGGBB
            self.create_oval(
                particle['x'], particle['y'],
                particle['x'] + particle['size'], particle['y'] + particle['size'],
                fill=base_color,
                outline=""
            )
            
            # Draw trail (also strip alpha)
            if particle['size'] > 1:
                trail_color = particle['color'][:7]  # strip alpha
                self.create_oval(
                    particle['x'], particle['y'] - 5,
                    particle['x'] + 1, particle['y'],
                    fill=trail_color,
                    outline=""
                )
        
        # Connect nearby particles
        self._draw_connections()
        
        self.animation_id = self.after(30, self._animate)
    
    def _draw_connections(self):
        """Draw lines between nearby particles"""
        for i, p1 in enumerate(self.particles):
            for p2 in self.particles[i+1:]:
                dist = math.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)
                if dist < 100:
                    # Removed alpha calculation to prevent crash
                    self.create_line(
                        p1['x'], p1['y'], p2['x'], p2['y'],
                        fill=config.COLOR_NEON_CYAN,
                        width=1
                    )


class CyberProgressBar(Canvas):
    """
    ⚡ Cyberpunk animated progress bar with neon effects
    """
    
    def __init__(self, parent, width: int = 600, height: int = 30, **kwargs):
        super().__init__(
            parent,
            width=width,
            height=height,
            bg=config.COLOR_BG_VOID,
            highlightthickness=0,
            **kwargs
        )
        
        self.width = width
        self.height = height
        self.progress = 0.0
        self.animation_offset = 0
        self.animation_id = None
        
        self._draw()
    
    def set_progress(self, progress: float):
        """Set progress (0.0 to 1.0)"""
        self.progress = max(0.0, min(1.0, progress))
        self._draw()
        
        if self.progress > 0 and self.progress < 1.0:
            if self.animation_id is None:
                self._animate_stripes()
        else:
            self._stop_animation()
    
    def _draw(self):
        """Draw the progress bar"""
        self.delete("all")
        
        # Background track
        self.create_rectangle(
            0, 0, self.width, self.height,
            fill=config.COLOR_BG_CARD,
            outline=config.COLOR_BORDER_NEON,
            width=2
        )
        
        # Progress fill
        if self.progress > 0:
            fill_width = int(self.width * self.progress)
            
            # Gradient effect (simplified - three segments)
            segments = 3
            segment_width = fill_width / segments
            
            colors = [
                config.COLOR_NEON_PURPLE,
                config.COLOR_NEON_PINK,
                config.COLOR_NEON_CYAN
            ]
            
            for i in range(segments):
                x1 = i * segment_width
                x2 = min((i + 1) * segment_width, fill_width)
                if x2 > x1:
                    self.create_rectangle(
                        x1, 2, x2, self.height - 2,
                        fill=colors[i],
                        outline=""
                    )
            
            # Animated stripes
            stripe_spacing = 20
            for i in range(-10, int(fill_width / stripe_spacing) + 10):
                x = i * stripe_spacing + self.animation_offset
                if 0 <= x <= fill_width:
                    self.create_line(
                        x, 0, x + 10, self.height,
                        fill=config.COLOR_TEXT_PRIMARY,
                        stipple="gray25",
                        width=2
                    )
            
            # Glow effect on leading edge
            if fill_width < self.width:
                for i in range(5):
                    self.create_line(
                        fill_width + i, 0,
                        fill_width + i, self.height,
                        fill=config.COLOR_NEON_CYAN,
                        width=1
                    )
        
        # Percentage text
        pct_text = f"{int(self.progress * 100)}%"
        self.create_text(
            self.width / 2, self.height / 2,
            text=pct_text,
            fill=config.COLOR_TEXT_PRIMARY,
            font=(config.FONT_FAMILY, config.FONT_SIZE_BODY, "bold")
        )
    
    def _animate_stripes(self):
        """Animate the diagonal stripes"""
        self.animation_offset = (self.animation_offset + 1) % 20
        self._draw()
        self.animation_id = self.after(50, self._animate_stripes)
    
    def _stop_animation(self):
        """Stop stripe animation"""
        if self.animation_id:
            self.after_cancel(self.animation_id)
            self.animation_id = None


class GlitchText(tk.Label):
    """
    👾 Glitchy text effect for cyberpunk aesthetic
    """
    
    def __init__(self, parent, text: str, **kwargs):
        super().__init__(parent, text=text, **kwargs)
        self.original_text = text
        self.is_glitching = False
        self.glitch_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
        
    def start_glitch(self, duration: int = 500):
        """Start glitch effect for specified duration"""
        if not self.is_glitching:
            self.is_glitching = True
            self._glitch_step(0, duration)
    
    def _glitch_step(self, elapsed: int, duration: int):
        """Single glitch animation step"""
        if elapsed >= duration:
            self.config(text=self.original_text)
            self.is_glitching = False
            return
        
        # Create glitched text
        glitched = ""
        for char in self.original_text:
            if random.random() < 0.3:
                glitched += random.choice(self.glitch_chars)
            else:
                glitched += char
        
        self.config(text=glitched)
        self.after(50, lambda: self._glitch_step(elapsed + 50, duration))