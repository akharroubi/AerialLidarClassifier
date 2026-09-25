"""Generate the Aerial LiDAR Classifier logo: SVG master + PNG exports.

Design ("classified profile"): a cross-section of a scene the plugin
classifies, in the plugin's own ASPRS palette: a brown ground band, a green
tree canopy on a dark trunk, a red building with a gable roof, an orange
power line strung from a magenta pole to the roof, all textured with a
grid of pale dots so that at 64 px and above the shapes read as point
clouds while at 16 px they collapse to three coloured masses (ground, tree,
building). A pale blue scan fan from the top-left corner says "aerial
sensor" without adding a hard shape.

Everything is defined once in unit coordinates (0..1) and drawn twice: as
SVG text (assets/logo.svg) and with PIL at 2048 px, then downsampled to
the exported sizes. Run:  python generate_logo.py [--variant plain|fan]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

PLUGIN_DIR = Path(__file__).resolve().parent / "Aerial_LiDAR_Classifier"
ASSETS = PLUGIN_DIR / "assets"
SIZES = (16, 24, 32, 48, 64, 128, 256, 512)
SUPER = 2048  # render size before downsampling

# Palette: the plugin's ASPRS class colours (config.DEFAULT_CLASS_MAPPING),
# with the building red deepened a little for contrast on white.
GROUND = "#A87E55"
GROUND_DARK = "#7A5A3A"
CANOPY = "#228B22"
CANOPY_LIGHT = "#3CA83C"
BUILDING = "#D62828"
ROOF = "#9E1B1B"
WIRE = "#FFA500"
POLE = "#FF00FF"
FAN = (91, 155, 213, 58)  # pale blue, translucent
DOT = (255, 255, 255, 56)  # point-cloud texture

# Geometry in unit coordinates ---------------------------------------------
GROUND_TOP = 0.78
GROUND_BOTTOM = 0.95
GROUND_RADIUS = 0.08
TRUNK = (0.215, 0.60, 0.265, GROUND_TOP + 0.02)      # x0, y0, x1, y1
CANOPY_A = (0.24, 0.44, 0.21)                        # cx, cy, r
CANOPY_B = (0.32, 0.53, 0.14)
BODY = (0.52, 0.42, 0.86, GROUND_TOP + 0.02)
ROOF_PTS = [(0.485, 0.43), (0.69, 0.20), (0.895, 0.43)]
POLE_RECT = (0.915, 0.24, 0.95, GROUND_TOP + 0.02)
CROSSARM = (0.885, 0.27, 0.98, 0.30)
# one span of line strung across the scene, behind tree and roof
WIRE_PTS = [(0.03, 0.29), (0.25, 0.355), (0.50, 0.365), (0.75, 0.325), (0.9325, 0.275)]
# scan cone from a sensor above the scene, kept inside the canvas so no
# blurred tint reaches the image border (transparent edges are a
# requirement for a toolbar icon)
FAN_PTS = [(0.50, 0.03), (0.07, GROUND_TOP + 0.02), (0.93, GROUND_TOP + 0.02)]
PLANE_CENTER = (0.50, 0.075)
PLANE = "#3B4A5A"
DOT_STEP = 0.046
DOT_R = 0.0085


def _px(v: float, s: int) -> float:
    return v * s


def draw_png(size: int, variant: str) -> Image.Image:
    s = SUPER
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    if variant in ("fan", "plane"):
        fan = Image.new("RGBA", (s, s), (0, 0, 0, 0))
        ImageDraw.Draw(fan).polygon([(_px(x, s), _px(y, s)) for x, y in FAN_PTS], fill=FAN)
        fan = fan.filter(ImageFilter.GaussianBlur(s * 0.012))
        # fade the cone out over the outer 4 % so the border stays fully
        # transparent whatever the blur radius does
        border = Image.new("L", (s, s), 0)
        ImageDraw.Draw(border).rectangle(
            [(int(s * 0.04), int(s * 0.04)), (int(s * 0.96), int(s * 0.96))], fill=255,
        )
        border = border.filter(ImageFilter.GaussianBlur(s * 0.015))
        np = __import__("numpy")
        fan.putalpha(Image.fromarray(
            (np.asarray(fan.getchannel("A")).astype("uint16")
             * np.asarray(border).astype("uint16") // 255).astype("uint8")))
        img.alpha_composite(fan)
        d = ImageDraw.Draw(img)

    shapes = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shapes)
    if variant == "plane":
        cx, cy = PLANE_CENTER
        sd.rounded_rectangle([(_px(cx - 0.09, s), _px(cy - 0.012, s)),
                              (_px(cx + 0.09, s), _px(cy + 0.012, s))],
                             radius=_px(0.012, s), fill=PLANE)          # wings
        sd.ellipse([(_px(cx - 0.045, s), _px(cy - 0.022, s)),
                    (_px(cx + 0.055, s), _px(cy + 0.022, s))], fill=PLANE)  # fuselage
        sd.polygon([(_px(cx - 0.045, s), _px(cy - 0.045, s)),
                    (_px(cx - 0.02, s), _px(cy - 0.045, s)),
                    (_px(cx - 0.01, s), _px(cy, s))], fill=PLANE)       # tail fin
    # ground band with rounded bottom corners
    sd.rounded_rectangle(
        [(_px(0.04, s), _px(GROUND_TOP, s)), (_px(0.96, s), _px(GROUND_BOTTOM, s))],
        radius=_px(GROUND_RADIUS, s), fill=GROUND,
    )
    sd.rectangle([(_px(0.04, s), _px(GROUND_TOP, s)), (_px(0.96, s), _px(GROUND_TOP + 0.06, s))], fill=GROUND)
    # pole with cross-arm, and one span of line behind everything else
    x0, y0, x1, y1 = POLE_RECT
    sd.rounded_rectangle([(_px(x0, s), _px(y0, s)), (_px(x1, s), _px(y1, s))],
                         radius=_px(0.012, s), fill=POLE)
    x0, y0, x1, y1 = CROSSARM
    sd.rounded_rectangle([(_px(x0, s), _px(y0, s)), (_px(x1, s), _px(y1, s))],
                         radius=_px(0.01, s), fill=POLE)
    sd.line([(_px(x, s), _px(y, s)) for x, y in WIRE_PTS], fill=WIRE,
            width=int(_px(0.024, s)), joint="curve")
    # tree
    x0, y0, x1, y1 = TRUNK
    sd.rectangle([(_px(x0, s), _px(y0, s)), (_px(x1, s), _px(y1, s))], fill=GROUND_DARK)
    for (cx, cy, r), col in ((CANOPY_A, CANOPY), (CANOPY_B, CANOPY_LIGHT)):
        sd.ellipse([(_px(cx - r, s), _px(cy - r, s)), (_px(cx + r, s), _px(cy + r, s))], fill=col)
    # building
    x0, y0, x1, y1 = BODY
    sd.rectangle([(_px(x0, s), _px(y0, s)), (_px(x1, s), _px(y1, s))], fill=BUILDING)
    sd.polygon([(_px(x, s), _px(y, s)) for x, y in ROOF_PTS], fill=ROOF)

    # point-cloud texture: pale dots clipped to the shapes
    dots = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    dd = ImageDraw.Draw(dots)
    n = int(1 / DOT_STEP) + 2
    for i in range(n):
        for j in range(n):
            cx = (i + 0.5 * (j % 2)) * DOT_STEP
            cy = j * DOT_STEP
            dd.ellipse([(_px(cx - DOT_R, s), _px(cy - DOT_R, s)),
                        (_px(cx + DOT_R, s), _px(cy + DOT_R, s))], fill=DOT)
    mask = shapes.getchannel("A")
    dots.putalpha(Image.fromarray(
        (__import__("numpy").asarray(dots.getchannel("A")).astype("uint16")
         * __import__("numpy").asarray(mask).astype("uint16") // 255).astype("uint8")))
    shapes.alpha_composite(dots)
    img.alpha_composite(shapes)

    out = img.resize((size, size), Image.LANCZOS)
    if size <= 32:
        # a touch of sharpening keeps the three masses crisp on the toolbar
        out = out.filter(ImageFilter.UnsharpMask(radius=1, percent=60, threshold=2))
    return out


def svg_text(variant: str) -> str:
    def p(v):
        return f"{v * 100:.2f}"
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="512" height="512">',
        "  <title>Aerial LiDAR Classifier</title>",
        "  <defs>",
        f'    <pattern id="pts" width="{DOT_STEP * 100:.2f}" height="{DOT_STEP * 200:.2f}" patternUnits="userSpaceOnUse">',
        f'      <circle cx="0" cy="0" r="{DOT_R * 100:.2f}" fill="#fff" fill-opacity="0.25"/>',
        f'      <circle cx="{DOT_STEP * 50:.2f}" cy="{DOT_STEP * 100:.2f}" r="{DOT_R * 100:.2f}" fill="#fff" fill-opacity="0.25"/>',
        "    </pattern>",
        # The dot texture is confined to the scene through a mask: a copy of
        # the shapes turned white by a flood filter. A clipPath cannot
        # reference a group, browsers render it as empty.
        '    <filter id="white"><feFlood flood-color="#fff"/><feComposite in2="SourceAlpha" operator="in"/></filter>',
        '    <mask id="scene" maskUnits="userSpaceOnUse" x="0" y="0" width="100" height="100">',
        '      <use href="#shapes" filter="url(#white)"/>',
        "    </mask>",
        '    <filter id="soft" x="-10%" y="-10%" width="120%" height="120%"><feGaussianBlur stdDeviation="1.2"/></filter>',
        "  </defs>",
    ]
    if variant in ("fan", "plane"):
        pts = " ".join(f"{p(x)},{p(y)}" for x, y in FAN_PTS)
        parts.append(f'  <polygon points="{pts}" fill="rgb(91,155,213)" fill-opacity="0.23" filter="url(#soft)"/>')
    x0, y0, x1, y1 = POLE_RECT
    ax0, ay0, ax1, ay1 = CROSSARM
    tx0, ty0, tx1, ty1 = TRUNK
    bx0, by0, bx1, by1 = BODY
    roof = " ".join(f"{p(x)},{p(y)}" for x, y in ROOF_PTS)
    wire = " ".join(f"{p(x)},{p(y)}" for x, y in WIRE_PTS)
    parts += ['  <g id="shapes">']
    if variant == "plane":
        cx, cy = PLANE_CENTER
        parts += [
            f'    <rect x="{p(cx - 0.09)}" y="{p(cy - 0.012)}" width="18" height="2.4" rx="1.2" fill="{PLANE}"/>',
            f'    <ellipse cx="{p(cx + 0.005)}" cy="{p(cy)}" rx="5" ry="2.2" fill="{PLANE}"/>',
            f'    <polygon points="{p(cx - 0.045)},{p(cy - 0.045)} {p(cx - 0.02)},{p(cy - 0.045)} {p(cx - 0.01)},{p(cy)}" fill="{PLANE}"/>',
        ]
    parts += [
        f'    <rect x="4" y="{p(GROUND_TOP)}" width="92" height="{p(GROUND_BOTTOM - GROUND_TOP)}" rx="{GROUND_RADIUS * 100:.1f}" fill="{GROUND}"/>',
        f'    <rect x="4" y="{p(GROUND_TOP)}" width="92" height="6" fill="{GROUND}"/>',
        f'    <rect x="{p(x0)}" y="{p(y0)}" width="{p(x1 - x0)}" height="{p(y1 - y0)}" rx="1.2" fill="{POLE}"/>',
        f'    <rect x="{p(ax0)}" y="{p(ay0)}" width="{p(ax1 - ax0)}" height="{p(ay1 - ay0)}" rx="1" fill="{POLE}"/>',
        f'    <polyline points="{wire}" fill="none" stroke="{WIRE}" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>',
        f'    <rect x="{p(tx0)}" y="{p(ty0)}" width="{p(tx1 - tx0)}" height="{p(ty1 - ty0)}" fill="{GROUND_DARK}"/>',
        f'    <circle cx="{p(CANOPY_A[0])}" cy="{p(CANOPY_A[1])}" r="{p(CANOPY_A[2])}" fill="{CANOPY}"/>',
        f'    <circle cx="{p(CANOPY_B[0])}" cy="{p(CANOPY_B[1])}" r="{p(CANOPY_B[2])}" fill="{CANOPY_LIGHT}"/>',
        f'    <rect x="{p(bx0)}" y="{p(by0)}" width="{p(bx1 - bx0)}" height="{p(by1 - by0)}" fill="{BUILDING}"/>',
        f'    <polygon points="{roof}" fill="{ROOF}"/>',
        "  </g>",
        '  <rect x="0" y="0" width="100" height="100" fill="url(#pts)" mask="url(#scene)"/>',
        "</svg>",
        "",
    ]
    return "\n".join(parts)


def contact_sheet(variant: str, out_path: Path) -> None:
    """Icon at toolbar sizes on a white and a dark bar, with reference boxes."""
    sizes = (16, 24, 32, 64, 128)
    pad = 24
    width = sum(s + pad for s in sizes) + pad + 160
    sheet = Image.new("RGBA", (width, 2 * (128 + 2 * pad)), (255, 255, 255, 255))
    d = ImageDraw.Draw(sheet)
    d.rectangle([(0, 128 + 2 * pad), (width, sheet.height)], fill=(43, 43, 43, 255))
    for row, bg in enumerate(((255, 255, 255), (43, 43, 43))):
        x = pad
        y_base = row * (128 + 2 * pad) + pad
        for s in sizes:
            icon = draw_png(s, variant)
            sheet.alpha_composite(icon, (x, y_base + (128 - s) // 2))
            x += s + pad
        # generic neighbour icons for scale (a grey square glyph and a circle)
        for k in range(2):
            gx = x + k * 40
            col = (120, 120, 120, 255) if bg[0] == 255 else (200, 200, 200, 255)
            if k == 0:
                d.rectangle([(gx, y_base + 56), (gx + 16, y_base + 72)], outline=col, width=2)
            else:
                d.ellipse([(gx, y_base + 56), (gx + 16, y_base + 72)], outline=col, width=2)
    sheet.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("plain", "fan", "plane"), default="plane")
    parser.add_argument("--out", default=str(ASSETS))
    parser.add_argument("--sheet-only", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    contact_sheet(args.variant, out / f"contact_sheet_{args.variant}.png")
    if args.sheet_only:
        return
    (out / "logo.svg").write_text(svg_text(args.variant), encoding="utf-8")
    for s in SIZES:
        draw_png(s, args.variant).save(out / f"logo_{s}.png", optimize=True)
    draw_png(128, args.variant).save(PLUGIN_DIR / "icon.png", optimize=True)
    print("wrote", out / "logo.svg", "+ PNG", SIZES, "and", PLUGIN_DIR / "icon.png")


if __name__ == "__main__":
    main()
