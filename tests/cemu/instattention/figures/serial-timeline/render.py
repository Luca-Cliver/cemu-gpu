"""Export horizontal timelines. Structural charts have no invented durations.

The measured chart uses one saved historical Host trace, never Guest statistics
from another run. Uses the sibling renderer's local librsvg/cairo PNG exporter.
"""
from pathlib import Path
import csv
import html
import importlib.util
import json

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[5]
SOURCE = ROOT / 'native-batch-device-timeline.jsonl'
spec = importlib.util.spec_from_file_location('sequence_renderer', OUT.parent / 'serial-sequence/render.py')
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)

W = 2600
INK, MUTED, LINE = '#172b43', '#5c6e83', '#d6e0eb'
BLUE, GREEN, PURPLE, AMBER = '#dcecff', '#d7f0e5', '#eadefa', '#fff0cc'
BG = '#f5f8fc'


class SVG:
    def __init__(self, height, title, subtitle):
        self.height = height
        self.items = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{height}" viewBox="0 0 {W} {height}">',
                      '<style>text{font-family:"Noto Sans CJK SC",sans-serif;}</style>',
                      '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#58708c"/></marker></defs>']
        self.rect(0, 0, W, height, 'white', radius=0)
        self.text(65, 65, title, 37, bold=True)
        self.text(65, 110, subtitle, 23, MUTED)

    def text(self, x, y, text, size=23, color=INK, bold=False, anchor='start', leading=None):
        lines = text.split('\n')
        attrs = f'x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="{anchor}"'
        if bold:
            attrs += ' font-weight="700"'
        self.items.append(f'<text {attrs}>' + ''.join(f'<tspan x="{x}" dy="{0 if i == 0 else leading or size * 1.45}">{html.escape(s)}</tspan>' for i, s in enumerate(lines)) + '</text>')

    def rect(self, x, y, w, h, fill=BG, stroke='none', radius=9):
        self.items.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}"/>')

    def line(self, x, y, xx, yy, dash=False, arrow=False, color=LINE):
        self.items.append(f'<path d="M{x},{y} L{xx},{yy}" stroke="{color}" stroke-width="2" fill="none"' + (' stroke-dasharray="7 6"' if dash else '') + (' marker-end="url(#arrow)"' if arrow else '') + '/>')

    def box(self, x, y, w, h, label, fill=BLUE, size=23):
        self.rect(x, y, w, h, fill)
        lines = label.count('\n') + 1
        self.text(x + w / 2, y + h / 2 - (lines - 1) * size * .725 + size * .34, label, size, anchor='middle')

    def bracket(self, x, xx, y, label, color=MUTED):
        self.line(x, y - 8, x, y, color=color)
        self.line(x, y, xx, y, color=color)
        self.line(xx, y - 8, xx, y, color=color)
        self.text((x + xx) / 2, y + 31, label, 22, color, anchor='middle')

    def save(self, name):
        self.text(65, self.height - 30, 'CEMU · native KV / autonomous SparF · 单 microbatch 串行路径 · 图表生成 2026-09-29', 19, MUTED)
        self.items.append('</svg>')
        path = OUT / (name + '.svg')
        path.write_text('\n'.join(self.items))
        renderer.png(path, (W, self.height))
        print(path.with_suffix('.png'))


def prefill():
    s = SVG(1350, '01  完整生成与 Prefill：沿时间轴看谁在做事、谁在等待',
            '结构时间轴：从左到右表示先后，图块宽度不代表耗时。未取得同次运行的 Guest 逐段记录，以下时长均待测。')
    s.text(65, 167, '完整生成', 27, bold=True)
    overview = [(65, 340, '初始化 / 注册\n输入与 embedding'), (425, 560, 'Prefill：逐层 Attention + MLP\n输出头产生第一个 token'),
                (1005, 1020, 'Decode：逐 token、逐层重复图 02\n每轮末尾输出头产生下一个 token'), (2045, 490, '最终 BUFFER_DRAIN\n剩余 tail 写完并 sealed')]
    for i, (x, w, label) in enumerate(overview):
        s.box(x, 192, w, 105, label, GREEN if i == 1 else BLUE, 24)
        if i < 3:
            s.line(x + w + 2, 245, overview[i + 1][0] - 3, 245, arrow=True, color=MUTED)
    s.text(65, 337, '下方放大 Prefill 相邻两层：writer 为单线程；只画一种可能的重叠布局，实际重叠量取决于运行。', 24, MUTED)
    x0, unit = 300, 220
    x = lambda a: x0 + a * unit
    s.line(x0, 380, 2520, 380, arrow=True, color=MUTED)
    s.text(2515, 367, '时间 →', 21, MUTED, anchor='end')
    for yy, label in [(445, 'Guest 主线程\n与 GPU'), (600, 'Guest KV writer\n单线程'), (763, 'Host tail / NAND\n后台 worker')]:
        s.text(65, yy + 25, label, 24, bold=True)
        s.line(x0, yy + 37, 2520, yy + 37)
    blocks = [(0, 2, '层 l：load_layer\nprefill_attention()', BLUE), (2, 1.0, '提交 l 写任务\n等 l−1 writer', AMBER),
              (3, 1.1, '层 l MLP\nrun_mlp()', BLUE), (4.1, 2, '层 l+1\nload + Attention', BLUE),
              (6.1, 1.2, '提交 l+1\n等 l writer', AMBER), (7.3, 1.0, '层 l+1\nMLP', BLUE), (8.3, 1.8, '…末层 / 输出头\n等待最后 writer', AMBER)]
    for a, b, label, color in blocks:
        s.box(x(a) + 3, 425, b * unit - 6, 107, label, color, 22)
    s.box(x(2) + 3, 579, 4.6 * unit - 6, 112, '层 l：_kv() 导出 →（打包 → APPEND）循环 → _update_mean()\nGuest GPU → Guest CPU → ioctl(/dev/ng0n1) → Host tail', GREEN, 23)
    s.box(x(6.6) + 3, 579, 3.5 * unit - 6, 112, '层 l+1 writer（随后执行）\n上一层 writer 完成后才可处理下一任务', GREEN, 22)
    s.line(x(2.03), 532, x(2.03), 576, arrow=True, color=MUTED)
    s.line(x(6.6), 579, x(6.6), 535, arrow=True, color=MUTED)
    s.text(x(3.2), 563, '与主线程 MLP、下一层 Attention 可以重叠', 21, '#24765d')
    s.box(x(2.5), 748, 6.9 * unit, 108, 'tail 满组 → pack_batch() → bb_native_kv_batch(write) → 等写 deadline → 推进 committed\n后台与上方重叠；尾部不足组可以留在 tail，Attention 可直接从 tail 读取', PURPLE, 23)
    s.line(x(2.5), 708, x(9.4), 708, dash=True, color=MUTED)
    s.text(65, 911, '这个阶段的“等多久”', 27, bold=True)
    rows = [
        ('GPU 计算', 'prefill_attention()：LN / QKV / causal QK / softmax / PV / Wo；run_mlp()；末尾 run_output_head()', 'T_attention / T_mlp / T_head：待测'),
        ('writer 等待', 'wait_prefill(previous)：当前层提交后等上一层；末尾 wait_prefill(last)：等最后一层', 'W_join：只计到达 join 后仍未完成的部分'),
        ('每个 APPEND', 'Guest ioctl 阻塞至 CQ；成功只表示 tail 接收。pending 后 sleep(100 μs 请求值)，再重试', 'W_append：待测；100 μs 不是固定总延迟'),
        ('生成末尾 drain', '全部 Decode 结束后 ioctl(BUFFER_DRAIN)；不足组补零写出，pending 重试至全部完成', 'W_drain：待测；不属于最后一个 Execute'),
    ]
    for i, (a, b, c) in enumerate(rows):
        y = 935 + 78 * i
        s.rect(65, y, 2470, 72, BG if i % 2 == 0 else '#ffffff')
        s.text(82, y + 30, a, 23, bold=True)
        s.text(315, y + 29, b, 21)
        s.text(315, y + 58, c, 20, MUTED)
    s.save('01-prefill-overview')


def decode():
    s = SVG(1640, '02  单层 Decode：调用、传输与等待放在同一条横向时间轴上',
            '结构时间轴，宽度不按时长比例。τ0…τ10 为逻辑边界，不是已有时间戳；各段 Guest 耗时 Δ1…Δ10 均待测。')
    left, step = 290, 224
    x = lambda i: left + i * step
    s.text(65, 143, '本层入口还需 weight_prefetcher.wait() 取得权重（待测）；下一层权重可提前加载。下方从 QKV 调用开始。', 20, MUTED)
    headings = ['GPU 投影', 'KV 导出', '打包 / KV 追加', 'Q / mean 准备', 'slot / 请求准备', '输入上传', '执行 Attention', '结果回读', '输出回 GPU', '本层后半段']
    for i, label in enumerate(headings):
        s.text(x(i) + step / 2, 181, f'{i+1:02}  {label}', 22, bold=True, anchor='middle')
        s.line(x(i), 203, x(i), 648, dash=True)
        s.text(x(i), 228, f'τ{i}', 21, MUTED, anchor='middle')
        s.text(x(i) + step / 2, 264, f'Δ{i+1}：待测', 20, MUTED, anchor='middle')
    s.text(x(10), 228, 'τ10', 21, MUTED, anchor='middle')
    s.line(left, 235, x(10), 235, arrow=True, color=MUTED)
    for y, label in [(315, 'Guest 主线程'), (412, 'Guest GPU'), (509, 'slot workers'), (606, 'Host 设备侧')]:
        s.text(65, y + 22, label, 23, bold=True)
        s.line(left, y, x(10), y)
    def b(i, n, y, label, color=BLUE, size=21):
        s.box(x(i) + 4, y - 25, step * n - 8, 74, label, color, size)
    b(0, 1, 315, 'prepare_decode\n_attention()')
    b(1, 1, 315, '_kv() 导出 K/V\nGPU → Guest CPU', GREEN)
    b(2, 1, 315, '逐包打包 + ioctl\n至全部 APPEND 成功', AMBER, 20)
    b(3, 1, 315, '_update_mean()\n_decode_args()', GREEN)
    b(4, 4, 315, 'run_decode()：获取 slot，随后 future.result() 阻塞；直到输出与 trace 收齐', AMBER, 23)
    b(8, 1, 315, 'finish_decode\n_attention()')
    b(9, 1, 315, 'Wo / 残差 / LN\nrun_mlp()')
    b(0, 1, 412, 'LN / QKV')
    b(1, 1, 412, 'K/V → Guest CPU', GREEN, 20)
    b(3, 1, 412, 'Q → Guest CPU', GREEN, 20)
    b(8, 1, 412, 'Attention 输出 H2D', GREEN, 20)
    b(9, 1, 412, 'GPU 实际计算')
    b(4, 1, 509, '_prepare_decode\n预算 / 布局')
    b(5, 1, 509, 'pwrite：MR0 / MR4\n然后 metadata', GREEN, 20)
    b(6, 1, 509, 'Execute ioctl\n等待父命令 CQ', AMBER)
    b(7, 1, 509, 'pread：MR9 → MR10\n完成 future', GREEN, 20)
    b(2, 1, 606, 'PRP → wire → tail\n接收成功 → CQ', GREEN, 20)
    b(5, 1, 606, 'memory_write()\nPRP → FDM', GREEN)
    b(6, 1, 606, 'SparF engine\n详细时间见图 03', PURPLE)
    b(7, 1, 606, 'memory_read()\nFDM → Guest', GREEN)
    s.bracket(x(4), x(8), 681, 'W_attention：包括 slot / worker 调度、上传、Execute 和回读')
    s.bracket(x(6), x(7), 744, 'W_execute', '#8255ae')
    s.text(65, 789, 'Host buffer_worker：满组写 NAND 可横跨以上阶段；APPEND 不等待 NAND 全部写完。背景区间不能再与主线相加。', 23, MUTED)
    s.text(65, 846, '边界处调用什么 / 等到什么才继续', 27, bold=True)
    rows = [
        ('01–02  产生与导出 KV', 'prepare_decode_attention() → append_decode() → _kv()；随后进入 NativeKvStore._append_heads()', '新 K/V 从 GPU 到 Guest CPU；第 03 段逐包打包、逐包 ioctl；同包只含同层、同 batch 的多个 head。'),
        ('03  APPEND syscall', 'ioctl(/dev/ng0n1, NVME_IOCTL_IO_CMD, BUFFER_APPEND_BATCH=0xD1)', 'NVMe SQ → native_execute() → PRP copy → tail ring → CQ；pending 时请求 sleep 100 μs 后重试。'),
        ('04–06  请求与输入', '_decode_args() → submit_decode() → _prepare_decode() → write_tensor() → pwrite(FDM fd)', 'Q 写 MR0、mean 写 MR4；fdmfs → NVMe ns2 → memory_write()。随后组织 Execute metadata。'),
        ('07  父 Execute', 'execute_autonomous() → CemuClient.execute() → ioctl(/dev/ng0n3)', 'program_execute() → sparf_submit_autonomous() → engine → to_poller → CQ → ioctl 返回；输出仍在 FDM。'),
        ('08  回读并唤醒', '_collect_decode() → read_range() → pread(MR9)，再 pread(MR10)', '必要时 Host CSD GPU → Host FDM RAM，再 PRP → Guest CPU；future 完成后模型主线程继续。'),
        ('09–10  完成本层', 'finish_decode_attention() → CPU→GPU → Wo / 残差 / LN → run_mlp()', '下一层重复 01–10；层间还需取得权重。全部层结束后 run_output_head()，进入下一 token。'),
    ]
    for i, (a, b_, c) in enumerate(rows):
        y = 867 + 96 * i
        s.rect(65, y, 2470, 90, BG if i % 2 == 0 else 'white')
        s.text(82, y + 32, a, 22, bold=True)
        s.text(405, y + 32, b_, 21)
        s.text(405, y + 67, c, 21, MUTED)
    s.rect(65, 1468, 2470, 107, '#fff5df')
    s.text(85, 1504, '如何填写时长：Δi = 同一 Guest 时钟下相邻边界之差；W_execute 是 Δ7 的调用包络，不能再与 Δ7 内部子段相加。', 23)
    s.text(85, 1543, '示例 B=4 / H=40 / d=128 / FP16：4 个 20 KiB APPEND 包；Q、mean、输出各 40 KiB。每个 40 KiB FDM 命令的模型增量为 20.48 μs，非 syscall 实测值。', 21)
    s.save('02-decode-layer')


def record():
    snapshot = OUT / 'execute-request-1.json'
    if snapshot.exists():
        return json.loads(snapshot.read_text())
    with SOURCE.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get('event') == 'complete' and r.get('session') == 906725105082679 and r.get('request') == 1:
                assert r['success']
                snapshot.write_text(json.dumps(r, ensure_ascii=False, indent=2) + '\n')
                return r
    raise RuntimeError('Expected historical request not found')


def measured():
    r = record()
    h0, v0 = r['admitted']['host_ns'], r['admitted']['virtual_ns']
    H = lambda p: (p['host_ns'] - h0) / 1e6
    V = lambda p: (p['virtual_ns'] - v0) / 1e6
    vn = lambda n: (n - v0) / 1e6
    hd, vd = H(r['ready']), V(r['ready'])
    reads = {a['name']: a for a in r['reads']}
    computes = {a['name']: a for a in r['computes']}
    s = SVG(1920, '03  一个 Execute 的实测时间轴：Host 实际时间与设备虚拟时间分开看',
            '历史日志 native-batch-device-timeline.jsonl · 2026-09-26 · request=1 / layer=0 / T=1025 / success=true；不是新运行。')
    s.box(65, 147, 810, 101, f'Host：admitted → ready  {hd:.3f} ms\n实际线程执行、控制开销与等待的包络', BLUE, 26)
    s.box(895, 147, 810, 101, f'虚拟时间：admitted → ready  {vd:.3f} ms\nQEMU_CLOCK_VIRTUAL 的推进量', PURPLE, 26)
    s.box(1725, 147, 810, 101, '不含完整 Guest syscall 前后路径\n不能写成 execute_wait_wall 的实测值', AMBER, 24)
    left, width = 335, 2170
    def axis(y, ymax, end, ticks, caption):
        s.text(65, y - 57, caption, 26, bold=True)
        for t in range(0, end + 1, ticks):
            xx = left + width * t / end
            s.line(xx, y, xx, ymax, dash=True)
            s.text(xx, y - 12, str(t), 21, MUTED, anchor='middle')
        s.line(left, y, left + width, y, arrow=True, color=MUTED)
        s.text(left + width, ymax + 31, '相对 admitted 的 ms', 20, MUTED, anchor='end')
        return lambda t: left + width * t / end
    hx = axis(351, 685, 190, 10, 'A  Host 墙钟轴：真实经过时间，横向长度按 ms 比例')
    rows_csv = []
    names = [('channel_k', 'channel-K 读'), ('token_k', 'selected-K 读'), ('token_v', 'selected-V 读')]
    for i, (name, label) in enumerate(names):
        a = reads[name]
        start, obs, end = H(a['submitted']), H(a['observed']), H(a['ready'])
        y = 383 + i * 78
        s.text(65, y + 30, label, 24, bold=True)
        s.rect(hx(start), y, hx(obs) - hx(start), 45, GREEN, radius=3)
        s.rect(hx(obs), y, hx(end) - hx(obs), 45, BLUE, radius=3)
        s.text((hx(start) + hx(obs))/2, y + 30, f'{obs-start:.3f}', 20, anchor='middle')
        s.text((hx(obs) + hx(end))/2, y + 30, f'{end-obs:.3f}', 20, anchor='middle')
        s.text(hx(end) + 10, y + 30, f'就绪 @{end:.3f}', 20, MUTED)
        rows_csv += [[label, 'Host', start, obs, obs-start, 'submitted→observed：观测窗口，不等同纯 gather'], [label, 'Host', obs, end, end-obs, 'observed→ready：后续等待/收尾窗口']]
    s.text(65, 658, '实际功能计算', 24, bold=True)
    for key, label in [('approx', 'Approx'), ('qk', 'QK'), ('pv', 'PV')]:
        a = computes[key]
        start, end = H(a['functional_start']), H(a['functional_end'])
        s.rect(hx(start), 627, max(2, hx(end)-hx(start)), 45, PURPLE, radius=3)
        s.rect(hx(end), 627, max(2, hx(H(a['observed_done']))-hx(end)), 45, BLUE, radius=2)
        if key != 'pv':
            s.text((hx(start)+hx(end))/2, 657, f'{label} {end-start:.3f}', 20, anchor='middle')
        else:
            s.text(hx(start)-10, 657, f'PV {end-start:.3f} →', 20, anchor='end')
        rows_csv += [[label+' functional', 'Host', start, end, end-start, 'functional_start→functional_end'], [label+' completion observation', 'Host', end, H(a['observed_done']), H(a['observed_done'])-end, '功能结束→观察 CU 完成；含控制、等待及观察开销'], [label+' functional', 'Virtual', V(a['functional_start']), V(a['functional_end']), V(a['functional_end'])-V(a['functional_start']), '该功能区间虚拟时间不推进']]
    s.text(65, 752, '绿色：submitted → observed；蓝色：observed → ready。V 读绿色窗口包含 QK 功能计算及其模型等待，不能再相加。', 23)
    s.text(65, 791, '紫色三个功能区间的 V 增量均为 0。日志未细分所有 freeze enter/leave 与 top-r；空白不等于空闲，H−V 也不能直接当总冻结时长。', 22, MUTED)
    vx = axis(902, 1213, 90, 10, 'B  设备虚拟时间轴：读目标 deadline 与 CU 模拟运行，横向长度按 ms 比例')
    for i, (name, label) in enumerate(names):
        a = reads[name]
        start, end = V(a['submitted']), vn(a['deadline_ns'])
        y = 934 + i * 69
        s.text(65, y + 30, label + '目标', 23, bold=True)
        s.rect(vx(start), y, vx(end)-vx(start), 43, BLUE, radius=3)
        s.text((vx(start)+vx(end))/2, y + 29, f'{(a["deadline_ns"]-a["submitted"]["virtual_ns"])/1e6:.4f} ms', 22, anchor='middle')
        rows_csv += [[label+' target', 'Virtual', start, end, end-start, '模型起点→deadline，不是纯 Host sleep']]
    s.text(65, 1186, 'CU 模拟计算', 24, bold=True)
    for key, label in [('approx', 'Approx'), ('qk', 'QK'), ('pv', 'PV')]:
        a = computes[key]
        start, end = vn(a['submitted_ns']), vn(a['completed_ns'])
        s.rect(vx(start), 1156, max(2,vx(end)-vx(start)), 43, PURPLE, radius=2)
        tx = vx(end) + 10 if key != 'pv' else vx(start) - 12
        s.text(tx, 1186, f'{label} {end-start:.3f}', 21, anchor='start' if key != 'pv' else 'end')
        rows_csv += [[label+' scheduled', 'Virtual', start, end, end-start, f'submit→completed；effective={a["effective_ns"]/1e6:.6f} ms']]
    s.text(65, 1273, 'V 读目标区间与 QK 的 CU 区间重叠：QK 完成后只等 V 读剩余时间；随后执行 PV。上、下两条轴不能按 x 坐标互相对齐。', 23)
    s.text(65, 1330, '计算阶段：实际做功能，与模型等待是两个不同区间', 27, bold=True)
    headers = [('阶段 / 调用',65),('Host 功能起止（ms）',540),('功能 / 后续观察（H）',1075),('有效预算（V）',1480),('提交至完成（V）',1950)]
    for label, xx in headers:
        s.text(xx + 15, 1380, label, 23, bold=True)
    for i, (key, label) in enumerate([('approx','Approx / softmax / top-k'),('qk','run_sparf_phase(EXACT_QK)'),('pv','run_sparf_phase(PV)')]):
        a = computes[key]
        y = 1400 + 66*i
        s.rect(65,y,2470,62,BG)
        vals = [label, f'{H(a["functional_start"]):.3f} → {H(a["functional_end"]):.3f}', f'{H(a["functional_end"])-H(a["functional_start"]):.3f} / {H(a["observed_done"])-H(a["functional_end"]):.3f} ms',f'{a["effective_ns"]/1e6:.6f} ms',f'{(a["completed_ns"]-a["submitted_ns"])/1e6:.6f} ms']
        for value, (_, xx) in zip(vals,headers):
            s.text(xx + 15,y+40,value,23)
    s.rect(65,1620,2470,221,'#fff5df')
    s.text(85,1658,f'排队：admitted → engine started = {H(r["started"]):.6f} ms（H） / {V(r["started"]):.6f} ms（V）；上图没有把短排队放大成等宽块。',23)
    s.text(85,1702,'功能区：enter_compute_section() → 功能调用 → leave_compute_section()；随后 WAIT_TIME 或 sched_submit_internal() 等待。',22)
    s.text(85,1745,'后续观察 = functional_end → observed_done：包含功能收尾、解冻控制、CU 排队、模拟等待与完成观察；并非纯 sleep。',22)
    s.text(85,1788,'读模型起点使用 model_cursor；submitted 的 H 与 V 不是严格配对采样。更多端点与精确值见同目录 CSV / 原始请求 JSON。',22)
    s.save('03-execute-measured')
    with (OUT/'execute-timings.csv').open('w', newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['stage','clock','start_ms_from_admitted','end_ms_from_admitted','duration_ms','meaning'])
        writer.writerows(rows_csv)
    events = [(key, r[key]) for key in ('admitted', 'started', 'ready')]
    for a in r['reads']:
        events.extend((a['name'] + '.' + key, a[key]) for key in ('submitted', 'observed', 'ready'))
    for a in r['computes']:
        events.extend((a['name'] + '.' + key, a[key]) for key in ('functional_start', 'functional_end', 'observed_done'))
    with (OUT/'execute-events.csv').open('w', newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['event','host_ms_from_admitted','virtual_ms_from_admitted','note'])
        for name, stamp in sorted(events, key=lambda item: item[1]['host_ns']):
            writer.writerow([name, f'{H(stamp):.6f}', f'{V(stamp):.6f}',
                             'V 是模型起点，不是与 H 严格配对的采样' if name.endswith('.submitted') else ''])


if __name__ == '__main__':
    prefill()
    decode()
    measured()
