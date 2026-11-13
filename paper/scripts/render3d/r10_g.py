"""Fig. 3g, round 10 (user 2026-11-11: keep the 3D exploded stack with pulled planes, add zoomed callouts, four parts across the width).
White page; dark only inside the image tiles. Layout, left to right:
  1 xy plane (z = 39 of the R5 crop) pulled from the stack;
  2 exploded stack of seven raw xy planes (r5_panels.stack_render, light variant);
  3 xz plane (y = 148) pulled from the stack, with three numbered boxes;
  4 three zoomed xz callouts of single nuclei, chosen by the rule registered for round 9 (AGENTS.md "FIG3G 事后描述",
    cellposeft_orx commit f5e0cb9; post hoc, single volume, descriptive): among GT nuclei in the shown xz crop matched by both models,
    the nucleus nearest the 10th, 50th and 90th percentile of dIoU = IoU(S1) - IoU(2D), labelled with both IoUs.
Masks, crop and planes are those of R7 (r5_panels); callout selection and full-plane cut-outs reuse r9_g."""
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Rectangle

import r5_panels as P5
import r9_g as G9
import rstyle as R

BOX = "#f0b429"   # callout box colour (amber; model colours are reserved)


def draw_g(fig, W, H, x, y, w, h):
    up = lambda a: np.repeat(np.repeat(a, 4, 0), 4, 1)  # noqa: E731  (4x nearest, embedded unresampled)
    mm = lambda xx, yy, ww, hh: fig.add_axes(P5._mm(fig, W, H, xx, yy, ww, hh))  # noqa: E731
    tc, mc = R.FG, R.MUTED
    s = P5.s
    top, bot = 5.0, 3.2
    side = h - top - bot
    # widths
    CW, CG = 15.2, 2.0
    callw = 3 * CW + 2 * CG
    g_xy_st, g_st_xz, g_xz_c = 3.0, 3.0, 5.0
    stack_w = w - 2 * side - callw - g_xy_st - g_st_xz - g_xz_c
    x_xy = x
    x_st = x_xy + side + g_xy_st
    x_xz = x_st + stack_w + g_st_xz
    x_c = x_xz + side + g_xz_c
    # 1 xy plane
    axr = mm(x_xy, y + top, side, side)
    labs = [(P5.gt[P5.kmid], P5.GTC), (P5.d2[P5.kmid], R.D2), (P5.s1[P5.kmid], R.S1)]
    axr.imshow(up(P5._rgb(P5.img[P5.kmid], labs)), interpolation="none", extent=(-0.5, s - 0.5, s - 0.5, -0.5))
    # 3 xz plane with boxes
    rgb = G9._rgb_xz()
    axl = mm(x_xz, y + top, side, side)
    axl.imshow(up(rgb), interpolation="none", extent=(-0.5, s - 0.5, rgb.shape[0] - 0.5, -0.5))
    axl.set_xlim(-0.5, s - 0.5); axl.set_ylim(rgb.shape[0] - 0.5, -0.5)
    picks, pinfo = G9.callout_picks()
    ysec = P5.ymid
    for k, pk in enumerate(picks):
        sec = G9.GT[:, ysec + P5.y0] == pk["gt"]
        zz, xx = np.nonzero(sec)
        m = 6
        x1, x2 = xx.min() - P5.x0 - m, xx.max() - P5.x0 + m
        z1, z2 = (zz.min() - P5.z0 - 0.5) * G9.RAT - m, (zz.max() - P5.z0 + 0.5) * G9.RAT + m
        pk["box_px"] = [int(x1), int(x2), float(z1), float(z2)]
        axl.add_patch(Rectangle((x1, z1), x2 - x1, z2 - z1, fill=False, ec=BOX, lw=0.7, clip_on=True))
        axl.text(x1 + 2, z1 - 3, str(k + 1), color=BOX, fontsize=5.6, fontweight="bold", va="bottom",
                 path_effects=[pe.withStroke(linewidth=1.0, foreground="black")])
    for a_, ttl in ((axl, f"xz plane (y = {P5.ymid + P5.y0})"), (axr, f"xy plane (z = {P5.kmid + P5.z0})")):
        a_.set_xticks([]); a_.set_yticks([])
        for sp in a_.spines.values():
            sp.set_color(tc); sp.set_linewidth(0.6)
        a_.set_title(ttl, fontsize=6.0, fontweight="bold", color=tc, pad=2)
    P5._scalebar(axr, 10 / P5.UM_PX, "10 µm", "#ffffff", s, s)
    P5._scalebar(axl, 10 / P5.UM_PX, "10 µm", "#ffffff", rgb.shape[0], s)
    # 2 stack (light render), centred in its slot
    stack, cn = P5.stack_render(False)
    sw, sh = stack.size
    avail = h - bot - 0.6
    ch = min(avail, stack_w * sh / sw); cw2 = ch * sw / sh
    axs = mm(x_st + (stack_w - cw2) / 2, y + 0.3 + (avail - ch) / 2, cw2, ch)
    axs.imshow(stack, interpolation="none"); axs.axis("off")
    # connectors: xy plane of the stack -> xy panel (left), xz cut of the stack -> xz panel (right)
    pts = [cn["xy_a"], cn["xy_b"]]
    left = sorted(pts, key=lambda q: q[1])
    for (px, py), (tx, ty) in zip(left, ((1, 1), (1, 0))):
        fig.add_artist(ConnectionPatch(xyA=(px, py), coordsA=axs.transData, xyB=(tx, ty), coordsB=axr.transAxes, color=tc, lw=0.45))
    for cname, (tx, ty) in (("xz_top", (0, 1)), ("xz_bot", (0, 0))):
        fig.add_artist(ConnectionPatch(xyA=cn[cname], coordsA=axs.transData, xyB=(tx, ty), coordsB=axl.transAxes, color=P5.XZC[False], lw=0.45))
    # 4 callouts
    full = G9._rgb_xz_full()
    ytop_c = y + top + (side - (CW + 2.6 + 5.0)) / 2 + 2.6
    for k, pk in enumerate(picks):
        x1, x2, z1, z2 = pk["box_px"]
        X1, X2, Zz1, Zz2 = x1 + P5.x0, x2 + P5.x0, z1 + P5.z0 * G9.RAT, z2 + P5.z0 * G9.RAT
        sz = max(X2 - X1, Zz2 - Zz1)
        xc_, zc_ = (X1 + X2) / 2, (Zz1 + Zz2) / 2
        cx = x_c + k * (CW + CG)
        ax = mm(cx, ytop_c, CW, CW)
        ax.imshow(up(full), interpolation="none", extent=(-0.5, full.shape[1] - 0.5, full.shape[0] - 0.5, -0.5))
        ax.set_xlim(xc_ - sz / 2, xc_ + sz / 2); ax.set_ylim(zc_ + sz / 2, zc_ - sz / 2)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(BOX); sp.set_linewidth(0.8)
        ax.set_title(f"{k + 1}  ({pk['percentile']}th pct ΔIoU)", fontsize=5.4, color=tc, pad=1.5)
        L = 2.0 / P5.UM_PX
        ax.plot([xc_ - sz / 2 + sz * 0.07, xc_ - sz / 2 + sz * 0.07 + L], [zc_ + sz / 2 - sz * 0.08] * 2, color="#ffffff", lw=0.9, solid_capstyle="butt")
        cxm = (cx + CW / 2) / W
        fig.text(cxm, 1 - (ytop_c + CW + 0.8) / H, f"2D  IoU {pk['iou_2D']:.2f}", ha="center", va="top", fontsize=5.4, color=R.PALETTE["models"]["2D_dark"])
        fig.text(cxm, 1 - (ytop_c + CW + 3.0) / H, f"S1  IoU {pk['iou_S1']:.2f}", ha="center", va="top", fontsize=5.4, color=R.PALETTE["models"]["S1_dark"])
        if k == 0:   # one leader line from box 1 to the callout row (avoids clutter; numbers identify the others)
            fig.add_artist(ConnectionPatch(xyA=(x2, (z1 + z2) / 2), coordsA=axl.transData, xyB=(0, 0.5), coordsB=ax.transAxes,
                                           color=BOX, lw=0.5, ls=(0, (2, 1.2))))
    fig.text((x_c + callw / 2) / W, 1 - (ytop_c + CW + 5.6) / H, "callouts from the xz plane; bar 2 µm",
             fontsize=5.0, color=mc, va="top", ha="center")
    # legend + source line
    leg = [Line2D([], [], color=P5.GTC, lw=1.2, label=R.LAB["gt"], path_effects=[pe.withStroke(linewidth=2.2, foreground=R.FG)]),
           Line2D([], [], color=R.D2, lw=1.2, label=R.LAB["2D"]), Line2D([], [], color=R.S1, lw=1.2, label=R.LAB["S1"])]
    fig.legend(handles=leg, loc="upper center", bbox_to_anchor=((x_c + callw / 2) / W, 1 - (y + top + 0.6) / H), ncol=3, fontsize=5.6,
               frameon=False, labelcolor=tc, handlelength=1.4, columnspacing=0.8, borderaxespad=0)
    fig.text((x + w / 2) / W, 1 - (y + h + 0.2) / H,
             f"{P5.SOURCE['dataset']} test volume {P5.SOURCE['volume']} (median rule), seed 0; stack: raw planes z = {", ".join(str(int(k + P5.z0)) for k in P5.ks)} (spacing 7–8)",
             fontsize=5.0, color=mc, ha="center", va="bottom")
    return {"panel": "g (round 10): xy plane | exploded stack | xz plane with boxes | three dIoU-percentile xz callouts; white page",
            "registration": "callout rule: AGENTS.md FIG3G 事后描述, cellposeft_orx commit f5e0cb9 (post hoc, single volume, descriptive)",
            "callouts": [{k: v for k, v in p.items() if k != "box_px"} for p in picks], "callout_pool": pinfo,
            "planes": [int(k + P5.z0) for k in P5.ks], "xy_pulled_z": int(P5.kmid + P5.z0), "xz_pulled_y": int(P5.ymid + P5.y0),
            "outlines": {"GT": P5.GTC, "2D": R.D2, "S1": R.S1}, "rasters": "4x nearest upsampling, embedded unresampled", **P5.SOURCE}
