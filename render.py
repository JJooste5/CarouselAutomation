#!/usr/bin/env python3
# Batch render Instagram carousels (1080x1080) from CSV/XLSX using Pillow.
# - Title slide: one line + logo, both centered. Logo above text.
# - Content slides: heading, subheading, body.
# - Random background image per slide from images_dir. Each used image is moved to used_dir.
# - Layouts are defined in layouts.json and selected per slide or random within type.
# - Input format is long-form: one row per slide.
#
# Usage:
#   python render.py --content content.csv --layouts layouts.json --config config.json
#
# Requirements:
#   pip install -r requirements.txt
#
import argparse
import csv
import json
import os
import random
import shutil
import sys
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

from PIL import Image, ImageDraw, ImageFont, ImageColor, ImageOps, ImageFilter, ImageEnhance

# =============== CUSTOMIZABLE PARAMETERS ===============

# Canvas/Image dimensions
CANVAS_W = 1080
CANVAS_H = 1080

# Shadow parameters
SHADOW_BLUR_RADIUS = 5
SHADOW_OFFSET_TEXT = 6  # Shadow offset for text
SHADOW_OFFSET_LOGO_LARGE = 5  # Shadow offset for large logos (title slide)
SHADOW_OFFSET_LOGO_SMALL = 4  # Shadow offset for small logos (content slides)
SHADOW_COLOR_TEXT = (0, 0, 0, 45)  # Semi-transparent black for text shadows
SHADOW_OPACITY_LOGO_LARGE = 0.6  # Shadow opacity for large logos
SHADOW_OPACITY_LOGO_SMALL = 0.4  # Shadow opacity for small logos

# Background image processing parameters
BG_BRIGHTNESS_FACTOR = 0.65  # Reduce brightness (0.75 = 75% of original)
BG_CONTRAST_FACTOR = 1.2  # Reduce contrast (0.9 = 90% of original)
BG_SATURATION_FACTOR = 1.3  # Increase saturation (1.2 = 120% saturation)
BG_WARM_COLOR = (255, 235, 205)  # Warm beige/golden color for temperature adjustment
BG_WARM_BLEND_FACTOR = 0.2  # How much warm color to blend (12%)
BG_RED_BOOST_FACTOR = 1.05  # Boost red channel slightly
BG_BLUE_REDUCE_FACTOR = 0.95  # Reduce blue channel slightly

# Dimming overlays
DIM_SOFT_OPACITY = 0.15  # 15% black overlay for soft dim
DIM_HARD_OPACITY = 0.35  # 35% black overlay for hard dim

# Logo parameters
LOGO_COLOR = (255, 255, 255)  # Force logos to white
LOGO_DEFAULT_WIDTH_TITLE = 160  # Default width for title slide logo
LOGO_DEFAULT_WIDTH_CONTENT = 80  # Default width for content slide logo
LOGO_DEFAULT_GAP = 24  # Default gap between logo and text on title slide
LOGO_DEFAULT_X = 40  # Default X position for content slide logo
LOGO_DEFAULT_Y = 40  # Default Y position for content slide logo

# Font sizes (these override whatever is in layouts.json)
FONT_SIZE_TITLE = 64  # Font size for title slides
FONT_SIZE_HEADING = 56  # Font size for all content headings
FONT_SIZE_SUBHEADING = 34  # Font size for all content subheadings
FONT_SIZE_BODY = 26  # Font size for all content body text

# Text spacing between elements (in pixels)
GAP_HEADING_TO_SUBHEADING = 20  # Gap between heading and subheading
GAP_SUBHEADING_TO_BODY = 15     # Gap between subheading and body

# Text fitting parameters
MIN_FONT_SIZE_CONTENT = 18  # Minimum font size for content text
MIN_FONT_SIZE_TITLE = 24  # Minimum font size for title text
FONT_SIZE_SHRINK_STEP = 2  # How much to reduce font size when shrinking to fit

# Output parameters
JPEG_QUALITY = 95  # JPEG compression quality
JPEG_SUBSAMPLING = 1  # JPEG subsampling (1 = 4:4:4, best quality)
JPEG_OPTIMIZE = True  # Optimize JPEG encoding

# Default text parameters
DEFAULT_TEXT_COLOR = "#FFFFFF"  # White text
DEFAULT_LINE_SPACING = 1.25  # Line spacing multiplier
DEFAULT_ALIGN = "left"  # Text alignment

# Image file extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# =============== END OF CUSTOMIZABLE PARAMETERS ===============

# --------------- Helpers ---------------

@dataclass
class Frame:
    field: str
    x: int
    y: int
    w: int
    h: int
    font: str
    size: int
    color: str = DEFAULT_TEXT_COLOR
    align: str = DEFAULT_ALIGN
    line_spacing: float = DEFAULT_LINE_SPACING
    shrink_to_fit: bool = True
    max_lines: Optional[int] = None

def load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(path, size)
    except Exception as e:
        raise RuntimeError(f"Failed to load font '{path}' size {size}. {e}")

def measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> Tuple[int, int]:
    # PIL textbbox gives (left, top, right, bottom)
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]

def wrap_text_to_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
    if not text:
        return [""]
    lines: List[str] = []
    for paragraph in text.split("\n"):
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        line = words[0]
        for w in words[1:]:
            trial = line + " " + w
            tw, _ = measure_text(draw, trial, font)
            if tw <= max_width:
                line = trial
            else:
                lines.append(line)
                line = w
        lines.append(line)
    return lines

def fit_text_block(draw: ImageDraw.ImageDraw, text: str, font_path: str, max_font_size: int,
                   box_w: int, box_h: int, line_spacing: float, align: str,
                   min_font_size: int = MIN_FONT_SIZE_CONTENT, max_lines: Optional[int] = None) -> Tuple[ImageFont.FreeTypeFont, List[str], int]:
    size = max_font_size
    while size >= min_font_size:
        font = load_font(font_path, size)
        lines = wrap_text_to_width(draw, text, font, box_w)
        if max_lines is not None and len(lines) > max_lines:
            lines = lines[:max_lines]
        ascent, descent = font.getmetrics()
        line_h = ascent + descent
        total_h = int(round(len(lines) * line_h * line_spacing))
        if total_h <= box_h:
            return font, lines, int(round(line_h * line_spacing))
        size -= FONT_SIZE_SHRINK_STEP
    # fallback at min size
    font = load_font(font_path, min_font_size)
    lines = wrap_text_to_width(draw, text, font, box_w)
    if max_lines is not None and len(lines) > max_lines:
        lines = lines[:max_lines]
    ascent, descent = font.getmetrics()
    line_h = ascent + descent
    return font, lines, int(round(line_h * line_spacing))

def paste_centered(base: Image.Image, child: Image.Image, y: int) -> int:
    x = (base.width - child.width) // 2
    base.paste(child, (x, y), child if child.mode == "RGBA" else None)
    return x

# --------------- Background image handling ---------------

def list_images(images_dir: str) -> List[str]:
    return [os.path.join(images_dir, f) for f in os.listdir(images_dir)
            if os.path.splitext(f.lower())[1] in IMAGE_EXTENSIONS and os.path.isfile(os.path.join(images_dir, f))]

def take_random_image(images_dir: str, used_dir: str, rng: random.Random) -> Optional[str]:
    pool = list_images(images_dir)
    if not pool:
        return None
    path = rng.choice(pool)
    # Move to used_dir and return the moved path (new location) for record, but we open it before moving
    os.makedirs(used_dir, exist_ok=True)
    dest = os.path.join(used_dir, os.path.basename(path))
    # Ensure unique filename if already exists in used
    base, ext = os.path.splitext(dest)
    i = 1
    while os.path.exists(dest):
        dest = f"{base}_{i}{ext}"
        i += 1
    shutil.move(path, dest)
    return dest

def place_background(canvas: Image.Image, bg_path: Optional[str], fit_mode: str = "cover",
                     opacity: float = 1.0, dim: str = "none") -> None:
    if not bg_path or not os.path.exists(bg_path):
        # Solid color background preserved by canvas init
        return
    bg = Image.open(bg_path).convert("RGB")
    bg = ImageOps.exif_transpose(bg)
    
    # Apply exposure and contrast adjustments
    # Reduce exposure (brightness)
    brightness_enhancer = ImageEnhance.Brightness(bg)
    bg = brightness_enhancer.enhance(BG_BRIGHTNESS_FACTOR)
    
    # Reduce contrast
    contrast_enhancer = ImageEnhance.Contrast(bg)
    bg = contrast_enhancer.enhance(BG_CONTRAST_FACTOR)

    # Increase saturation for more vibrant colors
    saturation_enhancer = ImageEnhance.Color(bg)
    bg = saturation_enhancer.enhance(BG_SATURATION_FACTOR)
    
    # Add warm color temperature adjustment
    # Create a warm overlay
    warm_overlay = Image.new("RGB", bg.size, BG_WARM_COLOR)
    bg = Image.blend(bg, warm_overlay, BG_WARM_BLEND_FACTOR)
    
    r, g, b = bg.split()
    r = r.point(lambda i: min(255, int(i * BG_RED_BOOST_FACTOR)))
    b = b.point(lambda i: int(i * BG_BLUE_REDUCE_FACTOR))
    bg = Image.merge("RGB", (r, g, b))
    
    if fit_mode == "cover":
        # Calculate scaling to ensure image covers entire canvas
        ratio = max(canvas.width / bg.width, canvas.height / bg.height)
        new_width = int(bg.width * ratio)
        new_height = int(bg.height * ratio)
        
        # Add a small buffer to ensure no gaps due to rounding
        new_width = max(new_width, canvas.width + 2)
        new_height = max(new_height, canvas.height + 2)
        
        bg = bg.resize((new_width, new_height), Image.LANCZOS)
        
        # Center crop
        x = (bg.width - canvas.width) // 2
        y = (bg.height - canvas.height) // 2
        bg = bg.crop((x, y, x + canvas.width, y + canvas.height))
    else:  # contain
        ratio = min(canvas.width / bg.width, canvas.height / bg.height)
        new_w, new_h = int(bg.width * ratio), int(bg.height * ratio)
        bg = bg.resize((new_w, new_h), Image.LANCZOS)
        # Create background with canvas background color instead of black
        pad = Image.new("RGB", (canvas.width, canvas.height), canvas.getpixel((0, 0)))
        pad.paste(bg, ((canvas.width - new_w) // 2, (canvas.height - new_h) // 2))
        bg = pad
        
    # Ensure the background is exactly canvas size
    if bg.size != (canvas.width, canvas.height):
        bg = bg.resize((canvas.width, canvas.height), Image.LANCZOS)
    
    if dim == "soft":
        overlay = Image.new("RGBA", (canvas.width, canvas.height), (0, 0, 0, int(255 * DIM_SOFT_OPACITY)))
        bg = Image.alpha_composite(bg.convert("RGBA"), overlay).convert("RGB")
    elif dim == "hard":
        overlay = Image.new("RGBA", (canvas.width, canvas.height), (0, 0, 0, int(255 * DIM_HARD_OPACITY)))
        bg = Image.alpha_composite(bg.convert("RGBA"), overlay).convert("RGB")
    
    if opacity < 1.0:
        base = Image.new("RGB", (canvas.width, canvas.height), canvas.getpixel((0, 0)))
        bg = Image.blend(base, bg, opacity)
    
    canvas.paste(bg, (0, 0))

def analyze_image_brightness(img_path: str) -> Dict[str, float]:
    """Analyze brightness of 5 regions in the image corresponding to layout positions."""
    if not img_path or not os.path.exists(img_path):
        return {}
    
    img = Image.open(img_path).convert("RGB")
    img = ImageOps.exif_transpose(img)
    
    # Resize to standard canvas size for consistent analysis
    img = img.resize((CANVAS_W, CANVAS_H), Image.LANCZOS)
    
    # Define regions based on your layouts
    regions = {
        "content_top_right": (520, 80, 960, 310),      # x, y, x+w, y+h
        "content_mid_right": (520, 320, 960, 540),
        "content_bottom_right": (520, 620, 960, 1040),
        "content_bottom_left": (120, 620, 560, 1040),
        "content_center": (180, 240, 900, 910)
    }
    
    brightness = {}
    for name, (x1, y1, x2, y2) in regions.items():
        # Crop region
        region = img.crop((x1, y1, x2, y2))
        # Convert to grayscale and calculate mean brightness
        gray = region.convert("L")
        brightness[name] = sum(gray.getdata()) / (gray.width * gray.height)
    
    return brightness

def draw_overlay(canvas: Image.Image, overlay_spec: Dict[str, Any]) -> None:
    if not overlay_spec:
        return
    color = ImageColor.getrgb(overlay_spec.get("color", "#000000"))
    alpha = int(overlay_spec.get("alpha", 120))
    rect = overlay_spec.get("rect", [0, 0, canvas.width, canvas.height])
    layer = Image.new("RGBA", (canvas.width, canvas.height), (0, 0, 0, 0))
    box = Image.new("RGBA", (rect[2], rect[3]), color + (alpha,))
    layer.paste(box, (rect[0], rect[1]))
    canvas.alpha_composite(layer)

# --------------- Rendering ---------------

def render_title_slide(canvas: Image.Image,
                       draw: ImageDraw.ImageDraw,
                       layout: Dict[str, Any],
                       title_text: str,
                       logo_path: Optional[str]) -> None:
    tf = layout["title_frame"]

    # Fit text inside the frame box - use FONT_SIZE_TITLE instead of layout value
    font, lines, step = fit_text_block(
        draw,
        title_text or "",
        tf["font"],
        FONT_SIZE_TITLE,  # Use constant instead of tf["size"]
        tf["w"],
        tf["h"],
        tf.get("line_spacing", DEFAULT_LINE_SPACING),
        tf.get("align", "center"),
        min_font_size=MIN_FONT_SIZE_TITLE,
        max_lines=tf.get("max_lines", 3),
    )

    # Measure block height
    text_lines = lines if lines else [""]
    text_block_h = step * len(text_lines)

    # Prepare logo (forced to white)
    logo_block = layout.get("logo", {})
    logo_img = None
    logo_h = 0
    gap = int(logo_block.get("gap", LOGO_DEFAULT_GAP))

    if logo_path and os.path.exists(logo_path):
        try:
            raw_logo = Image.open(logo_path).convert("RGBA")
            target_w = int(logo_block.get("width", LOGO_DEFAULT_WIDTH_TITLE))
            ratio = target_w / max(1, raw_logo.width)
            raw_logo = raw_logo.resize((target_w, int(raw_logo.height * ratio)), Image.LANCZOS)

            # Recolor to pure white, preserve alpha
            alpha = raw_logo.split()[-1]
            logo_img = Image.new("RGBA", raw_logo.size, LOGO_COLOR + (0,))
            logo_img.putalpha(alpha)

            logo_h = logo_img.height
        except Exception:
            logo_img = None
            logo_h = 0
            gap = 0
    else:
        gap = 0

    # Compute vertical placement inside the title_frame box
    combined_h = (logo_h + (gap if logo_img and text_lines else 0) + text_block_h)
    start_y = tf["y"] + max(0, (tf["h"] - combined_h) // 2)

    # Place logo centered within the title_frame width
    if logo_img:
        x_logo = tf["x"] + (tf["w"] - logo_img.width) // 2
        
        # Create shadow for logo
        shadow_offset = SHADOW_OFFSET_LOGO_LARGE
        shadow_opacity = SHADOW_OPACITY_LOGO_LARGE
        
        # Create shadow version of logo
        shadow_logo = Image.new("RGBA", logo_img.size, (0, 0, 0, 0))
        shadow_alpha = logo_img.split()[-1]  # Get alpha channel
        # Apply shadow color with reduced opacity
        shadow_alpha_adjusted = Image.eval(shadow_alpha, lambda x: int(x * shadow_opacity))
        shadow_logo.putalpha(shadow_alpha_adjusted)
        shadow_logo = shadow_logo.filter(ImageFilter.GaussianBlur(SHADOW_BLUR_RADIUS))
        
        # Paste shadow first
        canvas.paste(shadow_logo, (x_logo + shadow_offset, start_y + shadow_offset), shadow_logo)
        
        # Paste main logo
        canvas.paste(logo_img, (x_logo, start_y), logo_img)
        start_y += logo_h + (gap if text_lines else 0)

    # Draw text lines, aligned within the title_frame, in white
    align = tf.get("align", "center")
    color = DEFAULT_TEXT_COLOR
    shadow_offset = SHADOW_OFFSET_TEXT
    shadow_color = SHADOW_COLOR_TEXT
    
    y = start_y
    for ln in text_lines:
        lw, _ = measure_text(draw, ln, font)
        if align == "right":
            x = tf["x"] + tf["w"] - lw
        elif align == "center":
            x = tf["x"] + (tf["w"] - lw) // 2
        else:  # left
            x = tf["x"]
        
        # Draw shadow first
        shadow_img = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow_img)
        shadow_draw.text((x + shadow_offset, y + shadow_offset), ln, font=font, fill=shadow_color)
        shadow_img = shadow_img.filter(ImageFilter.GaussianBlur(SHADOW_BLUR_RADIUS))
        canvas.alpha_composite(shadow_img)
        
        # Draw main text
        draw.text((x, y), ln, font=font, fill=color)
        y += step

def draw_text_block(canvas: Image.Image, frame: Dict[str, Any], value: str, field_type: str = None) -> int:
    """Draw text block and return the actual height used."""
    draw = ImageDraw.Draw(canvas)
    
    # Determine font size based on field type
    if field_type == "heading":
        font_size = FONT_SIZE_HEADING
    elif field_type == "subheading":
        font_size = FONT_SIZE_SUBHEADING
    elif field_type == "body":
        font_size = FONT_SIZE_BODY
    else:
        font_size = frame["size"]  # Fallback to layout value if no field type
    
    font, lines, step = fit_text_block(draw, value, frame["font"], font_size,
                                       frame["w"], frame["h"], frame.get("line_spacing", DEFAULT_LINE_SPACING),
                                       frame.get("align", DEFAULT_ALIGN), min_font_size=MIN_FONT_SIZE_CONTENT,
                                       max_lines=frame.get("max_lines"))
    
    # Shadow settings
    shadow_offset = SHADOW_OFFSET_TEXT
    shadow_color = SHADOW_COLOR_TEXT
    
    y = frame["y"]
    for ln in lines:
        tw, th = measure_text(draw, ln, font)
        align = frame.get("align", DEFAULT_ALIGN)
        if align == "center":
            x = frame["x"] + (frame["w"] - tw) // 2
        elif align == "right":
            x = frame["x"] + frame["w"] - tw
        else:
            x = frame["x"]
        
        # Draw shadow first
        shadow_img = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow_img)
        shadow_draw.text((x + shadow_offset, y + shadow_offset), ln, font=font, fill=shadow_color)
        shadow_img = shadow_img.filter(ImageFilter.GaussianBlur(SHADOW_BLUR_RADIUS))
        canvas.alpha_composite(shadow_img)
        
        # Draw main text
        draw.text((x, y), ln, font=font, fill=frame.get("color", DEFAULT_TEXT_COLOR))
        y += step
    
    # Return the total height used
    return len(lines) * step if lines else 0

def render_content_slide(canvas: Image.Image, layout: Dict[str, Any], heading: str, subheading: str, body: str, logo_path: Optional[str]) -> None:
    # Optional overlay
    if "overlay" in layout:
        draw_overlay(canvas, layout["overlay"])
    
    # Add logo to top left if available
    logo_config = layout.get("logo", {})
    if logo_path and os.path.exists(logo_path) and logo_config:
        try:
            logo = Image.open(logo_path).convert("RGBA")
            target_w = int(logo_config.get("width", LOGO_DEFAULT_WIDTH_CONTENT))
            ratio = target_w / max(1, logo.width)
            logo = logo.resize((target_w, int(logo.height * ratio)), Image.LANCZOS)
            
            # Recolor to pure white, preserve alpha
            alpha = logo.split()[-1]
            white_logo = Image.new("RGBA", logo.size, LOGO_COLOR + (0,))
            white_logo.putalpha(alpha)
            
            # Position from config
            x_pos = int(logo_config.get("x", LOGO_DEFAULT_X))
            y_pos = int(logo_config.get("y", LOGO_DEFAULT_Y))
            
            # Create shadow for logo
            shadow_offset = SHADOW_OFFSET_LOGO_SMALL
            shadow_opacity = SHADOW_OPACITY_LOGO_SMALL
            
            # Create shadow version
            shadow_logo = Image.new("RGBA", white_logo.size, (0, 0, 0, 0))
            shadow_alpha = white_logo.split()[-1]
            shadow_alpha_adjusted = Image.eval(shadow_alpha, lambda x: int(x * shadow_opacity))
            shadow_logo.putalpha(shadow_alpha_adjusted)
            shadow_logo = shadow_logo.filter(ImageFilter.GaussianBlur(SHADOW_BLUR_RADIUS))
            
            # Paste shadow first
            canvas.paste(shadow_logo, (x_pos + shadow_offset, y_pos + shadow_offset), shadow_logo)
            
            # Paste main logo
            canvas.paste(white_logo, (x_pos, y_pos), white_logo)
            
        except Exception as e:
            print(f"Warning: Failed to add logo to content slide: {e}")
    
    # Get frame configurations
    frames_by_field = {}
    for frame in layout["frames"]:
        frames_by_field[frame["field"]] = frame.copy()
    
    # Start with heading position from layout
    current_y = frames_by_field.get("heading", {}).get("y", 120)
    
    # Draw heading if present
    if heading and "heading" in frames_by_field:
        frame = frames_by_field["heading"].copy()
        frame["y"] = current_y
        height = draw_text_block(canvas, frame, heading, "heading")
        current_y += height + GAP_HEADING_TO_SUBHEADING
    
    # Draw subheading if present
    if subheading and "subheading" in frames_by_field:
        frame = frames_by_field["subheading"].copy()
        frame["y"] = current_y
        height = draw_text_block(canvas, frame, subheading, "subheading")
        current_y += height + GAP_SUBHEADING_TO_BODY
    
    # Draw body if present
    if body and "body" in frames_by_field:
        frame = frames_by_field["body"].copy()
        frame["y"] = current_y
        draw_text_block(canvas, frame, body, "body")

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def read_rows(content_path: str) -> List[Dict[str, str]]:
    _, ext = os.path.splitext(content_path.lower())
    if ext == ".csv":
        with open(content_path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    elif ext in (".xlsx", ".xls"):
        try:
            import pandas as pd  # type: ignore
        except Exception as e:
            raise RuntimeError("Reading Excel requires pandas and openpyxl. Install from requirements.txt.") from e
        df = pd.read_excel(content_path)
        return [dict(row) for _, row in df.iterrows()]
    else:
        raise RuntimeError("Unsupported content file. Use .csv or .xlsx")

def choose_layout(layouts: Dict[str, Any], slide_type: str, requested: Optional[str], 
                  rng: random.Random, brightness_data: Optional[Dict[str, float]] = None,
                  last_used_layout: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    # Filter layouts by prefix
    prefix = "title_" if slide_type == "title" else "content_"
    if requested:
        name = requested
        if name not in layouts:
            raise KeyError(f"Layout '{name}' not found in layouts.json")
        return name, layouts[name]
    
    pool = [name for name in layouts.keys() if name.startswith(prefix)]
    if not pool:
        raise RuntimeError(f"No layouts found with prefix '{prefix}'")

    if last_used_layout and last_used_layout in pool and len(pool) > 1:
        pool = [name for name in pool if name != last_used_layout]
    
    # For content slides with brightness data, choose region by darkness
    if slide_type == "content" and brightness_data:
        # Filter to only layouts we have brightness data for
        valid_layouts = [name for name in pool if name in brightness_data]
        
        # Remove last used layout if possible
        if last_used_layout and last_used_layout in valid_layouts and len(valid_layouts) > 1:
            valid_layouts = [name for name in valid_layouts if name != last_used_layout]
        
        if valid_layouts:
            # Calculate weights - darker regions get higher weights
            # Invert brightness (255 - brightness) and square for stronger preference
            weights = []
            for layout in valid_layouts:
                darkness = 255 - brightness_data[layout]
                weight = darkness ** 2  # Square to favor darker regions more
                weights.append(weight)
            
            # Normalize weights
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
                
                # Choose weighted random
                chosen = rng.choices(valid_layouts, weights=weights, k=1)[0]
                return chosen, layouts[chosen]
    
    # Fallback to random selection
    name = rng.choice(pool)
    return name, layouts[name]

def main():
    content_file = "content.csv"  # Change this to your content file
    layouts_file = "layouts.json"
    config_file = "config.json"
    
    with open(config_file, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    with open(layouts_file, "r", encoding="utf-8") as f:
        layouts = json.load(f)

    out_dir = cfg.get("out_dir", "out")
    images_dir = cfg.get("images_dir", "images")
    used_dir = cfg.get("used_dir", "images_used")
    logo_path = cfg.get("logo_path", None)
    bg_fit = cfg.get("background_fit", "cover")
    bg_opacity = float(cfg.get("background_opacity", 1.0))
    bg_dim = cfg.get("background_dim", "soft")
    rng = random.Random(cfg.get("seed", None))

    ensure_dir(out_dir)
    ensure_dir(images_dir)
    ensure_dir(used_dir)

    rows = read_rows(content_file)

    # Normalize keys to lower
    norm_rows: List[Dict[str, str]] = []
    for r in rows:
        nr = {str(k).strip().lower(): ("" if r[k] is None else str(r[k])) for k in r.keys()}
        norm_rows.append(nr)

    # Group by post_id and order by slide
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in norm_rows:
        pid = r.get("post_id", "").strip()
        grouped[pid].append(r)  # keep CSV order per post_id


    # Render each slide
    for pid, slides in grouped.items():
        if not pid:
            print("Skipping row with empty post_id")
            continue
        
        last_layout_name = None
        for idx, r in enumerate(slides, start=1):
            slide_num = idx
            # Force first slide to be title; others default to content unless specified
            slide_type = "title" if slide_num == 1 else r.get("slide_type", "content").strip().lower()

            # Get background image first for content slides
            bg_path = take_random_image(images_dir, used_dir, rng)

            # Analyze brightness for content slides
            brightness_data = None
            if slide_type == "content" and bg_path:
                brightness_data = analyze_image_brightness(bg_path)

            # Choose layout based on brightness for content, or use requested/random for title
            # Choose layout based on brightness for content, or use requested/random for title
            layout_name_req = (r.get("layout", "").strip() or None) if slide_type == "title" else None
            layout_name, layout = choose_layout(layouts, slide_type, layout_name_req, rng, 
                                            brightness_data, last_layout_name)
            
            # Update last used layout
            last_layout_name = layout_name
            # Create canvas
            canvas_bg = layout.get("canvas", {}).get("bg", "#000000")
            canvas = Image.new("RGBA", (CANVAS_W, CANVAS_H), canvas_bg)

            place_background(canvas, bg_path,
                            fit_mode=layout.get("background_image", {}).get("fit", bg_fit),
                            opacity=layout.get("background_image", {}).get("opacity", bg_opacity),
                            dim=layout.get("background_image", {}).get("dim", bg_dim))

            # Optional overlay before text
            if "overlay" in layout:
                draw_overlay(canvas, layout["overlay"])

            draw = ImageDraw.Draw(canvas)
            heading = r.get("heading", "").strip()
            subheading = r.get("subheading", "").strip()
            body = r.get("body", "").strip()
            if slide_type == "title":
                render_title_slide(canvas, draw, layout, heading, logo_path)
            else:
                render_content_slide(canvas, layout, heading, subheading, body, logo_path)

            basename = f"{pid}_{idx}.jpg"
            out_path = os.path.join(out_dir, basename)
            canvas.convert("RGB").save(out_path, quality=JPEG_QUALITY, subsampling=JPEG_SUBSAMPLING, optimize=JPEG_OPTIMIZE)
            print(f"Saved {out_path}")

    print("Done.")
    print(f"Output dir: {out_dir}")
    print(f"Images moved to used dir: {used_dir}")

if __name__ == "__main__":
    main()