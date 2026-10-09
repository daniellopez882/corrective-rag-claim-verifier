"""Reliability diagram rendered as a standalone SVG (no plotting library)."""

from __future__ import annotations

import math
from html import escape
from typing import Any

SERIES_COLOURS = {"raw": "#8a8f98", "platt": "#2b6cb0", "isotonic": "#c05621"}


def reliability_svg(methods: dict[str, dict[str, Any]], title: str, size: int = 420) -> str:
    """One curve per calibration method: observed frequency against mean confidence per bin.

    Marker area grows with the number of claims in the bin; empty bins are skipped.
    """
    pad_left, pad_right, pad_top, pad_bottom = 56, 132, 40, 52
    plot = size - pad_left - pad_bottom
    width = pad_left + plot + pad_right
    height = pad_top + plot + pad_bottom

    def px(v: float) -> float:
        return pad_left + v * plot

    def py(v: float) -> float:
        return pad_top + (1.0 - v) * plot

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Helvetica, Arial, sans-serif" '
        'font-size="12">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{pad_left}" y="22" font-size="13" fill="#1a202c">{escape(title)}</text>',
    ]
    for i in range(11):
        v = i / 10
        parts.append(
            f'<line x1="{px(v):.1f}" y1="{py(0):.1f}" x2="{px(v):.1f}" y2="{py(1):.1f}" '
            'stroke="#edf0f4"/>'
        )
        parts.append(
            f'<line x1="{px(0):.1f}" y1="{py(v):.1f}" x2="{px(1):.1f}" y2="{py(v):.1f}" '
            'stroke="#edf0f4"/>'
        )
        if i % 2 == 0:
            parts.append(
                f'<text x="{px(v):.1f}" y="{py(0) + 16:.1f}" text-anchor="middle" '
                f'fill="#4a5568">{v:.1f}</text>'
            )
            parts.append(
                f'<text x="{px(0) - 8:.1f}" y="{py(v) + 4:.1f}" text-anchor="end" '
                f'fill="#4a5568">{v:.1f}</text>'
            )
    parts.append(
        f'<rect x="{px(0)}" y="{py(1)}" width="{plot}" height="{plot}" fill="none" '
        'stroke="#a0aec0"/>'
    )
    parts.append(
        f'<line x1="{px(0)}" y1="{py(0)}" x2="{px(1)}" y2="{py(1)}" stroke="#a0aec0" '
        'stroke-dasharray="4 4"/>'
    )
    parts.append(
        f'<text x="{px(0.5):.1f}" y="{height - 12}" text-anchor="middle" fill="#2d3748">'
        "mean predicted probability</text>"
    )
    parts.append(
        f'<text transform="translate(16 {py(0.5):.1f}) rotate(-90)" text-anchor="middle" '
        'fill="#2d3748">observed frequency</text>'
    )
    legend_y = pad_top + 8
    for name, report in methods.items():
        colour = SERIES_COLOURS.get(name, "#2d3748")
        points = [
            (b["mean_confidence"], b["observed_frequency"], b["count"])
            for b in report["reliability"]
            if b["count"]
        ]
        if len(points) > 1:
            path = " ".join(f"{px(c):.1f},{py(f):.1f}" for c, f, _ in points)
            parts.append(
                f'<polyline points="{path}" fill="none" stroke="{colour}" stroke-width="1.8"/>'
            )
        for c, f, n in points:
            r = 2.5 + 1.2 * math.sqrt(n)
            parts.append(
                f'<circle cx="{px(c):.1f}" cy="{py(f):.1f}" r="{min(r, 14):.1f}" fill="{colour}" '
                'fill-opacity="0.55" stroke="#ffffff" stroke-width="0.8"/>'
            )
        ece = report.get("ece_10bins")
        label = f"{name} (ECE {ece:.3f})" if isinstance(ece, float) else name
        lx = px(1) + 14
        parts.append(f'<rect x="{lx}" y="{legend_y - 9}" width="10" height="10" fill="{colour}"/>')
        parts.append(f'<text x="{lx + 16}" y="{legend_y}" fill="#2d3748">{escape(label)}</text>')
        legend_y += 20
    parts.append("</svg>")
    return "\n".join(parts) + "\n"
