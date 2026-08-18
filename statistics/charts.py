# The four charts, drawn as inline SVG from the measured numbers.

from common import RESULTS, esc, fmt, path_from

AX = "var(--rule)"
MUTED = "var(--ink-3)"


def axis_text(x, y, text, anchor="middle", cls="tick"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
            f'class="{cls}">{esc(text)}</text>')


def chart_training():
    tr = RESULTS["training"]["main"]
    train = tr["series"]["train_acc"]
    val = tr["series"]["val_acc"]
    boundary = tr["phase_boundary_epoch"]
    best = tr["best_val_epoch"]

    W, H = 740, 320
    L, R, T, B = 46, 142, 18, 44
    pw, ph = W - L - R, H - T - B

    lo, hi = 0.25, 1.0
    n = len(train)

    def px(epoch):
        return L + (epoch - 1) / (n - 1) * pw

    def py(value):
        return T + (hi - value) / (hi - lo) * ph

    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" '
             'aria-label="Training and validation accuracy across 25 epochs" '
             'class="chart">']

    for value in (0.25, 0.4, 0.6, 0.8, 1.0):
        y = py(value)
        parts.append(f'<line x1="{L}" y1="{y:.1f}" x2="{L + pw}" y2="{y:.1f}" '
                     f'stroke="{AX}" stroke-width="1"/>')
        parts.append(axis_text(L - 10, y + 4, f"{value:.0%}", "end"))

    bx = px(boundary)
    parts.append(f'<line x1="{bx:.1f}" y1="{T}" x2="{bx:.1f}" y2="{T + ph}" '
                 'stroke="var(--accent)" stroke-width="1"/>')
    parts.append(f'<text x="{bx + 7:.1f}" y="{T + 14}" class="note-in" '
                 'text-anchor="start">fine-tuning</text>')
    parts.append(f'<text x="{bx - 7:.1f}" y="{T + 14}" class="note-in" '
                 'text-anchor="end">head only</text>')

    for epoch in (1, 5, 10, 15, 20, 25):
        parts.append(axis_text(px(epoch), T + ph + 24, str(epoch)))
    parts.append(axis_text(L + pw / 2, T + ph + 40, "epoch", "middle", "axis-title"))

    for series, slot in ((train, 1), (val, 2)):
        pts = [(px(i + 1), py(v)) for i, v in enumerate(series)]
        parts.append(f'<path d="{path_from(pts)}" fill="none" '
                     f'stroke="var(--s{slot})" stroke-width="2" '
                     'stroke-linejoin="round" stroke-linecap="round"/>')
        ex, ey = pts[-1]
        parts.append(f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="4" '
                     f'fill="var(--s{slot})" stroke="var(--surface)" '
                     'stroke-width="2"/>')

    parts.append(f'<text x="{px(n) + 12:.1f}" y="{py(train[-1]) + 4:.1f}" '
                 'class="end-label" text-anchor="start">training 98.9%</text>')
    parts.append(f'<text x="{px(n) + 12:.1f}" y="{py(val[-1]) + 4:.1f}" '
                 'class="end-label" text-anchor="start">validation 76.1%</text>')

    parts.append(f'<circle cx="{px(best):.1f}" cy="{py(val[best - 1]):.1f}" '
                 'r="6" fill="none" stroke="var(--ink-2)" stroke-width="1.5"/>')
    parts.append(f'<text x="{px(best):.1f}" y="{py(val[best - 1]) - 16:.1f}" '
                 'class="note-in" text-anchor="middle">weights kept: 78.3%</text>')

    parts.append("</svg>")
    return "".join(parts)


def chart_backbones():
    var = RESULTS["variance"]["backbones_batch_32"]
    specs = {b["architecture"]: b for b in RESULTS["backbones"]["backbones"]}
    order = sorted(var, key=lambda a: -specs[a]["parameters"])

    W = 760
    row_h, top = 34, 54
    H = top + row_h * len(order) + 10
    label_w = 152
    ax0, bx0, bar_max = 164, 474, 172
    bar_h = 13

    max_params = max(specs[a]["parameters"] for a in order)
    max_ms = max(var[a]["per_image_ms_mean"] for a in order)

    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" '
             'aria-label="Backbone parameter count against measured latency" '
             'class="chart">']

    parts.append(f'<text x="{ax0}" y="22" class="panel-head">parameters</text>')
    parts.append(f'<text x="{bx0}" y="22" class="panel-head">'
                 'milliseconds per image</text>')
    parts.append(f'<line x1="{ax0}" y1="34" x2="{ax0 + bar_max}" y2="34" '
                 f'stroke="{AX}" stroke-width="1"/>')
    parts.append(f'<line x1="{bx0}" y1="34" x2="{bx0 + bar_max}" y2="34" '
                 f'stroke="{AX}" stroke-width="1"/>')

    for index, arch in enumerate(order):
        y = top + index * row_h
        mid = y + bar_h / 2 + 4

        parts.append(f'<text x="{label_w}" y="{mid:.1f}" text-anchor="end" '
                     f'class="row-label">{esc(arch)}</text>')

        pw = specs[arch]["parameters"] / max_params * bar_max
        parts.append(f'<rect x="{ax0}" y="{y}" width="{pw:.1f}" '
                     f'height="{bar_h}" rx="3" fill="var(--s1)"/>')
        parts.append(f'<text x="{ax0 + pw + 8:.1f}" y="{mid:.1f}" '
                     f'class="bar-value">'
                     f'{fmt(specs[arch]["parameters"] / 1e6, 2)}M</text>')

        mw = var[arch]["per_image_ms_mean"] / max_ms * bar_max
        parts.append(f'<rect x="{bx0}" y="{y}" width="{mw:.1f}" '
                     f'height="{bar_h}" rx="3" fill="var(--s2)"/>')
        parts.append(f'<text x="{bx0 + mw + 8:.1f}" y="{mid:.1f}" '
                     f'class="bar-value">'
                     f'{var[arch]["per_image_ms_mean"]:.1f}</text>')

    parts.append("</svg>")
    return "".join(parts)


def chart_profiles():
    scans = RESULTS["profiles"]["scans"]
    resolutions = ["VGA", "phone", "1080p"]
    profiles = ["full", "balanced", "edge"]
    slot = {"full": 1, "balanced": 2, "edge": 3}

    def value(res, prof):
        return next(s for s in scans if s["resolution"] == res
                    and s["profile"] == prof)

    W, H = 740, 330
    L, R, T, B = 60, 20, 22, 54
    pw, ph = W - L - R, H - T - B
    top_ms = 5000

    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" '
             'aria-label="Scan latency by profile and frame size" class="chart">']

    for ms in (0, 1000, 2000, 3000, 4000, 5000):
        y = T + ph - ms / top_ms * ph
        parts.append(f'<line x1="{L}" y1="{y:.1f}" x2="{L + pw}" y2="{y:.1f}" '
                     f'stroke="{AX}" stroke-width="1"/>')
        parts.append(axis_text(L - 10, y + 4, f"{ms:,}", "end"))

    label_y = T + ph / 2
    parts.append(f'<text x="16" y="{label_y:.1f}" class="axis-title" '
                 f'text-anchor="middle" '
                 f'transform="rotate(-90 16 {label_y:.1f})">milliseconds</text>')

    group_w = pw / len(resolutions)
    bar_w, gap = 44, 12

    for gi, res in enumerate(resolutions):
        gx = L + gi * group_w
        block = len(profiles) * bar_w + (len(profiles) - 1) * gap
        start = gx + (group_w - block) / 2

        for pi, prof in enumerate(profiles):
            entry = value(res, prof)
            h = entry["headline_ms"] / top_ms * ph
            x = start + pi * (bar_w + gap)
            y = T + ph - h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" '
                         f'height="{h:.1f}" rx="3" '
                         f'fill="var(--s{slot[prof]})"/>')
            parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{y - 8:.1f}" '
                         f'class="bar-value" text-anchor="middle">'
                         f'{entry["windows"]}w</text>')

        parts.append(axis_text(gx + group_w / 2, T + ph + 26, res))

    parts.append(f'<line x1="{L}" y1="{T + ph:.1f}" x2="{L + pw}" '
                 f'y2="{T + ph:.1f}" stroke="var(--axis)" stroke-width="1"/>')
    parts.append("</svg>")
    return "".join(parts)


def chart_threads():
    rows = RESULTS["threads"]["measurements"]

    W, H = 600, 300
    L, R, T, B = 46, 116, 18, 50
    pw, ph = W - L - R, H - T - B
    top = 6.0

    def px(threads):
        return L + (threads - 1) / 5 * pw

    def py(speed):
        return T + ph - (speed - 1) / (top - 1) * ph

    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" '
             'aria-label="Speedup against thread count, measured versus linear" '
             'class="chart">']

    for value in (1, 2, 3, 4, 5, 6):
        y = py(value)
        parts.append(f'<line x1="{L}" y1="{y:.1f}" x2="{L + pw}" y2="{y:.1f}" '
                     f'stroke="{AX}" stroke-width="1"/>')
        parts.append(axis_text(L - 10, y + 4, f"{value}x", "end"))

    for threads in (1, 2, 4, 6):
        parts.append(axis_text(px(threads), T + ph + 24, str(threads)))
    parts.append(axis_text(L + pw / 2, T + ph + 42, "threads", "middle",
                           "axis-title"))

    ideal = [(px(1), py(1)), (px(6), py(6))]
    parts.append(f'<path d="{path_from(ideal)}" fill="none" stroke="{MUTED}" '
                 'stroke-width="2" stroke-linecap="round"/>')
    parts.append(f'<text x="{px(6) + 10:.1f}" y="{py(6) + 4:.1f}" '
                 'class="end-label muted-label" text-anchor="start">'
                 'linear</text>')

    pts = [(px(r["threads"]), py(r["speedup_vs_one_thread"])) for r in rows]
    parts.append(f'<path d="{path_from(pts)}" fill="none" stroke="var(--s1)" '
                 'stroke-width="2" stroke-linejoin="round" '
                 'stroke-linecap="round"/>')
    for x, y in pts:
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" '
                     'fill="var(--s1)" stroke="var(--surface)" '
                     'stroke-width="2"/>')
    parts.append(f'<text x="{pts[-1][0] + 10:.1f}" y="{pts[-1][1] + 4:.1f}" '
                 'class="end-label" text-anchor="start">measured 3.6x</text>')

    parts.append("</svg>")
    return "".join(parts)
