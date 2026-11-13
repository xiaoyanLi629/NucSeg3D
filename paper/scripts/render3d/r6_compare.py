"""Before/after zoom crops for the R6 Fig. 2 revision: both PDFs rasterised at 1200 dpi (pdftoppm); crops of the same content
(mouse-embryo S1 tile of the 3D grid, centre of the cube panel), centres given in mm from the panel geometry of each layout."""
import sys
from PIL import Image, ImageDraw, ImageFont
S, OUT = sys.argv[1], sys.argv[2]
DPI = 1200; mm = DPI / 25.4
A = Image.open(f"{S}/z_R5-Fig2-cube.png"); B = Image.open(f"{S}/z_R6-Fig2.png")
regs = [("grid, mouse embryo / S1 tile, 12 x 12 mm", (96.0, 127.0), (93.8, 122.9), 12), ("cube panel centre, 14 x 14 mm", (159.0, 110.0), (159.0, 105.0), 14)]
font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 44)
rows = []
for name, ca, cb, w in regs:
    cr = lambda im, c: im.crop((int((c[0] - w / 2) * mm), int((c[1] - w / 2) * mm), int((c[0] + w / 2) * mm), int((c[1] + w / 2) * mm)))  # noqa: E731
    a, b = cr(A, ca), cr(B, cb)
    c = Image.new("RGB", (a.width + b.width + 60, a.height + 140), "white"); c.paste(a, (0, 140)); c.paste(b, (a.width + 60, 140))
    d = ImageDraw.Draw(c); d.text((0, 8), name + " (PDF rasterised at 1200 dpi)", fill="black", font=font)
    d.text((0, 75), "R5 (before)", fill="black", font=font); d.text((a.width + 60, 75), "R6 (after)", fill="black", font=font)
    rows.append(c)
o = Image.new("RGB", (max(r.width for r in rows), sum(r.height for r in rows) + 40), "white"); y = 0
for r in rows:
    o.paste(r, (0, y)); y += r.height + 40
o.save(OUT); print(OUT, o.size)
