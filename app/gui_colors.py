"""Label -> colour (shared by the Qt preview and the annotated-media export)."""
import colorsys
import zlib


def rgb_color(label: str) -> tuple:
    h = (zlib.crc32(label.encode()) % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.78, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def bgr_color(label: str) -> tuple:
    r, g, b = rgb_color(label)
    return b, g, r
