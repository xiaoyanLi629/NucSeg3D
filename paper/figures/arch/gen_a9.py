"""Fig. 1 A9: A8-d panels a (architecture, insets i-v) and b (training schedule) on top of the A8-b lower panels
("How z information enters the flow", "3D training target", "Inference on one test volume"; nc-arch2's gen_a7.panel_cmp7 /
panel_target7 / panel_infer7 as used in gen_a8.layout_b8), relettered c, d, e.

gen_a7.py and gen_a8.py are imported read-only. To give the whole page one container style, this file installs thin wrappers for
gen_a7.card and gen_a7.step_card at run time (the A8-d style: borderless warm-grey #EDECE9 blocks with a grey tab and a coloured accent
bar; step cards white with a tinted header strip and a numbered circle, on one warm-grey block) and draws the panel letters/titles with
the A8-d plabel. Panel contents, data and positions inside the cards are unchanged.
Usage: python gen_a9.py
"""
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_fig import LW, PAL, Fig, export, tint  # noqa: E402
import gen_a7 as G7  # noqa: E402  (nc-arch2, read-only)
import gen_a7d as GD  # noqa: E402
import gen_a8d as A8  # noqa: E402
from gen_a7d import ACC2D, BLOCK, F_S, INK, LOSS, MUTE, RED, RULE, ncirc, plabel, section  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ACC = {"I": ACC2D, "II": RED, "III": LOSS}


def card9(f, x, y, w, h, fam, tab=None, sub=None):
    """Replacement for gen_a7.card: A8-d section block (no border, grey tab, accent bar)."""
    section(f, x, y, w, h, BLOCK, tab, tab_note=sub, accent=ACC.get(fam, MUTE))


def step_card9(f, x, y, w, h, fam, k, title):
    """Replacement for gen_a7.step_card: white card, tinted header strip, numbered circle (A8-d panel e style)."""
    acc = ACC.get(fam, MUTE)
    hs = 4.6
    f.rect(x, y, w, h, rx=1.2, fill="#ffffff", stroke=RULE, sw=0.7 * LW, chk=False)
    f.add(f'<path d="M{x:.2f},{y + hs:.2f} V{y + 1.2:.2f} Q{x:.2f},{y:.2f} {x + 1.2:.2f},{y:.2f} H{x + w - 1.2:.2f} '
          f'Q{x + w:.2f},{y:.2f} {x + w:.2f},{y + 1.2:.2f} V{y + hs:.2f} Z" fill="{tint(acc, 0.22)}" stroke="none"/>')
    ncirc(f, x + 2.4, y + hs / 2, k, acc)
    f.text(x + 4.6, y + hs / 2 + 0.75, title, F_S, 700, c=INK)


def layout_a9():
    G7.STYLE["v"] = "b"
    G7.card = card9
    G7.step_card = step_card9
    f = Fig(183.0, 247.0)
    f.plabel = lambda x, y, letter, title: plabel(f, x, y + 0.4, letter, title)
    ya = A8.panel_arch8(f, 0.0, 0.0, letter="a")
    ob = ya + 1.4
    A8.panel_sched8(f, 0.0, ob, 183.0, letter="b")
    oc = ob + 5.4 + 21.0 + 1.4
    G7.panel_cmp7(f, 0.0, oc, 120.5, letter="c")
    G7.panel_target7(f, 122.0, oc, 60.5, letter="d")
    oe = oc + 42.2
    bottom = oe + 5.2 + 33.8 + 0.6
    f.H = round(bottom, 1)
    # one warm-grey block behind the seven white step cards (drawn first, so it sits behind)
    n0 = len(f.el)
    G7.panel_infer7(f, 0.0, oe, 183.0, letter="e")
    f.el.insert(n0, f'<rect x="0.00" y="{oe + 4.4:.2f}" width="183.00" height="{bottom - (oe + 4.4):.2f}" rx="1.4" fill="{BLOCK}" stroke="none"/>')
    return f


def main():
    cid = "A9"
    GD.ensure_crops()
    f = layout_a9()
    f.el = [e for e in f.el if not (e.startswith("<text") and e.count("<text") == 1 and e.split(">", 1)[1] == "</text>")]
    # absolute-value bars: draw "∣" (U+2223, set as a placeholder in the labels) as a 1.35x-tall bar so it cannot read as I or l
    f.el = [e.replace("∣", f'<tspan font-size="{5.0 * 1.3 * 0.3528:.3f}" font-weight="300">|</tspan>') for e in f.el]
    res = export(f, cid)
    print(cid, "issues:", len(res.get("issues", [])), "min font", res.get("min_font_pt"), "size", f.W, "x", f.H)
    for s_ in res.get("issues", [])[:80]:
        print("   ", s_)
    p8 = json.load(open(os.path.join(HERE, "A8-d.provenance.json")))
    json.dump(dict(figure=cid, size_mm=[f.W, f.H], code_snapshot=p8["code_snapshot"], model=p8["model"],
                   palette="paper/figures/palette.json; blocks #EDECE9 (3D-viewer scheme)",
                   generator="paper/figures/arch/gen_a9.py: panels a, b = gen_a8d.panel_arch8 / panel_sched8; panels c, d, e = gen_a7.panel_cmp7 / "
                             "panel_target7 / panel_infer7 (nc-arch2, imported read-only, as in gen_a8.layout_b8) with gen_a7.card and gen_a7.step_card "
                             "replaced at run time by the A8-d container style (card9, step_card9)",
                   new_assets=p8.get("new_assets"), references_web=p8.get("references_web"), references_style=p8.get("references_style"),
                   items=f.prov),
              open(os.path.join(HERE, f"{cid}.provenance.json"), "w"), indent=1, ensure_ascii=False)
    gal = os.path.join(HERE, "..", "gallery", "assets")
    for ext in ("pdf", "png", "provenance.json"):
        shutil.copy(os.path.join(HERE, f"{cid}.{ext}"), os.path.join(gal, f"{cid}.{ext}"))


if __name__ == "__main__":
    main()
