"""Render the three documented sequence diagrams as local SVG/PNG assets.

Uses a small renderer for the sequenceDiagram subset used by the source document,
plus system librsvg/cairo. No remote rendering service or network is required.
"""
from pathlib import Path
import ctypes as C
import ctypes.util
import html
import re

from PIL import ImageFont

OUT = Path(__file__).resolve().parent
DOC = OUT.parent.parent / "NATIVE_SERIAL_SEQUENCE.md"
FONT = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 23)
WIDTH, MARGIN = 2240, 100
TITLES = ["01  Prefill：KV 生产、提交与后台写回", "02  单层 Decode：从 QKV 到下一层", "03  父 Execute：冻结、功能执行与目标等待"]
NAMES = ["01-prefill", "02-decode-layer", "03-execute-internal"]


def esc(s):
    return html.escape(s, quote=True)


def wrap(s, width):
    result = []
    for paragraph in s.replace("<br/>", "\n").replace("<br>", "\n").splitlines():
        line = ""
        for c in paragraph:
            if line and FONT.getlength(line + c) > width:
                result.append(line)
                line = c
            else:
                line += c
        result.append(line)
    return result or [""]


def text(x, y, lines, size=23, color="#20344c", anchor="start", bold=False):
    weight = ' font-weight="700"' if bold else ""
    return f'<text x="{x}" y="{y}" fill="{color}" font-size="{size}" text-anchor="{anchor}"{weight}>' + ''.join(
        f'<tspan x="{x}" dy="{0 if i == 0 else 34}">{esc(s)}</tspan>' for i, s in enumerate(lines)
    ) + '</text>'


def rect(x, y, w, h, fill, stroke="none", radius=8):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}"/>'


def render(source, title, path):
    rows = [s.strip() for s in source.splitlines() if s.strip()]
    participants = []
    for s in rows:
        m = re.match(r'participant (\w+) as (.*)', s)
        if m:
            participants.append(m.groups())
    n = len(participants)
    spacing = (WIDTH - 2 * MARGIN - 300) / (n - 1)
    xs = {key: MARGIN + 150 + i * spacing for i, (key, _) in enumerate(participants)}
    body, frames, stack = [], [], []
    y = 340
    for s in rows:
        if s == "sequenceDiagram" or s.startswith("participant "):
            continue
        m = re.match(r'(loop|opt|alt|par|rect)\s*(.*)', s)
        if m:
            kind, label = m.groups()
            left = 44 + len(stack) * 17
            label = "冻结区间 · Host 继续执行，QEMU 虚拟时间停止" if kind == "rect" else {
                "loop": "循环", "opt": "条件", "alt": "分支", "par": "并行分支"
            }[kind] + "：" + label
            stack.append((kind, left, y, label))
            body.append(text(left + 16, y + 31, [label], size=22, bold=True))
            y += 66
            continue
        if s == "end":
            kind, left, top, label = stack.pop()
            fill = "#e9f2ff" if kind == "rect" else "#f6f8fb"
            frames.append((top, len(stack), rect(left, top, WIDTH - 2 * left, y - top + 8, fill, "#cad7e5")))
            y += 28
            continue
        m = re.match(r'(else|and)\s*(.*)', s)
        if m:
            left = stack[-1][1]
            body.append(f'<path d="M {left} {y} H {WIDTH-left}" stroke="#a9bacd" stroke-dasharray="8 6"/>')
            label = ("另一并行分支：" if m[1] == "and" else "否则：") + m[2]
            body.append(text(left + 16, y + 32, [label], size=22, bold=True))
            y += 66
            continue
        m = re.match(r'Note over ([\w,]+):\s*(.*)', s, re.I)
        if m:
            ids, label = m.groups()
            endpoints = ids.split(',')
            if len(endpoints) == 1:
                w = min(spacing - 20, 510)
                left = max(75, min(xs[endpoints[0]] - w / 2, WIDTH - 75 - w))
            else:
                a, b = sorted(xs[k] for k in endpoints)
                left, w = a - 115, b - a + 230
            lines = wrap(label, w - 32)
            h = 34 * len(lines) + 24
            body.append(rect(left, y, w, h, "#fff8df", "#e5d8a4"))
            body.append(text(left + 16, y + 32, lines, color="#705315"))
            y += h + 22
            continue
        m = re.match(r'(\w+)(-->>|->>|-\))(\w+):\s*(.*)', s)
        if not m:
            raise ValueError(f"Unsupported diagram line: {s}")
        a, arrow, b, label = m.groups()
        x1, x2 = xs[a], xs[b]
        if a == b:
            rightward = x1 < WIDTH * .70
            available = spacing - 48 if n > 4 else 480
            w = max(260, min(available, 520))
            lines = wrap(label, w)
            tx = x1 + 22 if rightward else x1 - w - 22
            body.append(text(tx, y + 24, lines))
            yy = y + len(lines) * 34 + 16
            dx = 95 if rightward else -95
            body.append(f'<path d="M {x1} {yy} h {dx} v 25 h {-dx}" fill="none" stroke="#315d86" stroke-width="2.3" marker-end="url(#arrow)"/>')
            y = yy + 52
        else:
            lines = wrap(label, abs(x2 - x1) - 32)
            cx = (x1 + x2) / 2
            body.append(text(cx, y + 24, lines, anchor="middle"))
            yy = y + len(lines) * 34 + 15
            dash = ' stroke-dasharray="9 6"' if arrow == '-->>' else ''
            marker = 'openarrow' if arrow == '-)' else 'arrow'
            body.append(f'<path d="M {x1} {yy} H {x2}" stroke="#315d86" stroke-width="2.3"{dash} marker-end="url(#{marker})"/>')
            y = yy + 35
    if stack:
        raise ValueError("Unclosed diagram blocks")
    h = y + 130
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{h}" viewBox="0 0 {WIDTH} {h}">',
           '<style>text {font-family:"Noto Sans CJK SC",sans-serif;}</style>',
           '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="5" orient="auto"><path d="M0,0 L10,5 L0,10 Z" fill="#315d86"/></marker><marker id="openarrow" markerWidth="10" markerHeight="10" refX="9" refY="5" orient="auto"><path d="M0,0 L10,5 L0,10" fill="none" stroke="#315d86" stroke-width="1.5"/></marker></defs>',
           rect(0, 0, WIDTH, h, "#ffffff", radius=0),
           text(70, 76, [title], size=41, bold=True),
           text(72, 124, ["CEMU · native KV / autonomous SparF · 单 microbatch 串行路径"], size=24, color="#60738b"),
           text(72, 168, ["T = 执行区间   W = 等待区间   R / D = 模型预算 / deadline；未填写实测值，纵向长度不表示时间比例。"], size=23),
           text(72, 207, ["G：Guest 调用时间   H：Host 实际时间   V：QEMU 虚拟时间   C：GPU stream 时间"], size=23, color="#60738b")]
    svg.extend(x[2] for x in sorted(frames, key=lambda t: (t[1], t[0])))
    for key, label in participants:
        x = xs[key]
        svg.append(f'<path d="M {x} 313 V {y+20}" stroke="#bdc9d7" stroke-width="1.8" stroke-dasharray="7 8"/>')
        w = min(spacing - 14, 400)
        svg.append(rect(x - w / 2, 243, w, 70, "#183c60" if label.startswith("Host") else "#2b6b7c"))
        lines = wrap(label, w - 24)
        svg.append(text(x, 269 if len(lines)>1 else 287, lines, size=23, color="white", anchor="middle", bold=True))
    svg.extend(body)
    svg.append(text(70, h - 43, ["来源：NATIVE_SERIAL_SEQUENCE.md · 2026-09-29 · 箭头为调用/完成关系；嵌套和重叠时间不可直接相加。"], size=21, color="#60738b"))
    svg.append('</svg>')
    path.write_text('\n'.join(svg))
    return WIDTH, h


def png(svg_path, size):
    rsvg = C.CDLL(ctypes.util.find_library('rsvg-2'))
    cairo = C.CDLL(ctypes.util.find_library('cairo'))
    gobject = C.CDLL(ctypes.util.find_library('gobject-2.0'))
    rsvg.rsvg_handle_new_from_file.argtypes = [C.c_char_p, C.POINTER(C.c_void_p)]
    rsvg.rsvg_handle_new_from_file.restype = C.c_void_p
    rsvg.rsvg_handle_render_cairo.argtypes = [C.c_void_p, C.c_void_p]
    rsvg.rsvg_handle_render_cairo.restype = C.c_int
    cairo.cairo_image_surface_create.argtypes = [C.c_int, C.c_int, C.c_int]
    cairo.cairo_image_surface_create.restype = C.c_void_p
    cairo.cairo_create.argtypes = [C.c_void_p]
    cairo.cairo_create.restype = C.c_void_p
    cairo.cairo_surface_write_to_png.argtypes = [C.c_void_p, C.c_char_p]
    cairo.cairo_surface_write_to_png.restype = C.c_int
    cairo.cairo_destroy.argtypes = [C.c_void_p]
    cairo.cairo_surface_destroy.argtypes = [C.c_void_p]
    gobject.g_object_unref.argtypes = [C.c_void_p]
    error = C.c_void_p()
    handle = rsvg.rsvg_handle_new_from_file(str(svg_path).encode(), C.byref(error))
    if not handle:
        raise RuntimeError(f"SVG load failed: {svg_path}")
    surface = cairo.cairo_image_surface_create(0, *size)
    ctx = cairo.cairo_create(surface)
    if not rsvg.rsvg_handle_render_cairo(handle, ctx):
        raise RuntimeError("SVG render failed")
    status = cairo.cairo_surface_write_to_png(surface, str(svg_path.with_suffix('.png')).encode())
    cairo.cairo_destroy(ctx)
    cairo.cairo_surface_destroy(surface)
    gobject.g_object_unref(handle)
    if status:
        raise RuntimeError(f"PNG write failed: {status}")


if __name__ == '__main__':
    diagrams = re.findall(r'```mermaid\n(.*?)```', DOC.read_text(), re.S)
    assert len(diagrams) == 3
    for source, name, title in zip(diagrams, NAMES, TITLES):
        (OUT / f'{name}.mmd').write_text(source)
        dest = OUT / f'{name}.svg'
        dimensions = render(source, title, dest)
        png(dest, dimensions)
        print(name, dimensions)
