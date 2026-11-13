"""Automated text-overlap check for matplotlib figures: pairwise overlap of the bounding boxes of all texts that are actually drawn
(figure texts, axes texts, titles, axis labels, current tick labels, legend texts). Stale tick objects are ignored."""


def _texts(fig):
    out = list(fig.texts)
    for lg in getattr(fig, "legends", []):
        out += list(lg.get_texts()) + ([lg.get_title()] if lg.get_title() else [])
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        out += list(ax.texts) + [ax.title, ax._left_title, ax._right_title] + ([ax.xaxis.label, ax.yaxis.label] if ax.axison else [])
        if ax.axison:
            for t in ax.get_xticklabels():
                if ax.xaxis.get_visible():
                    t._fc_rot_group = id(ax) if t.get_rotation() % 180 else None
                    out.append(t)
            out += [t for t in ax.get_yticklabels() if ax.yaxis.get_visible()]
        lg = ax.get_legend()
        if lg is not None:
            out += list(lg.get_texts()) + ([lg.get_title()] if lg.get_title() else [])
    return out


MIN_PT = 5.0


def check(fig, pad=0.5):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    items, seen, small = [], set(), []
    for t in _texts(fig):
        if id(t) in seen or not t.get_visible():
            continue
        seen.add(id(t))
        s = t.get_text()
        if not s or not s.strip():
            continue
        if t.axes is not None and t in t.axes.get_xticklabels() + t.axes.get_yticklabels():
            # skip tick labels outside the view limits
            ax = t.axes
            x, y = t.get_position()
            if t in ax.get_xticklabels() and not (min(ax.get_xlim()) - 1e-9 <= x <= max(ax.get_xlim()) + 1e-9):
                continue
            if t in ax.get_yticklabels() and not (min(ax.get_ylim()) - 1e-9 <= y <= max(ax.get_ylim()) + 1e-9):
                continue
        bb = t.get_window_extent(r)
        if bb.width <= 0:
            continue
        items.append((s.replace("\n", " ")[:40], bb, getattr(t, "_fc_rot_group", None)))
        if t.get_fontsize() < MIN_PT - 1e-6:
            small.append(f"FONT {t.get_fontsize():.1f} pt: '{s[:30]}'")
    out = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if items[i][2] is not None and items[i][2] == items[j][2]:
                continue   # neighbouring rotated tick labels of one axis: their axis-aligned boxes overlap by construction
            a, b = items[i][1], items[j][1]
            if a.x0 + pad < b.x1 and b.x0 + pad < a.x1 and a.y0 + pad < b.y1 and b.y0 + pad < a.y1:
                out.append(f"'{items[i][0]}' x '{items[j][0]}'")
    return sorted(set(out)) + sorted(set(small))
