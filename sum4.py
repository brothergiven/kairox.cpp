"""정책 격자 덤프 요약.  usage: python3 sum4.py   (DUMPS=dumps3 LOGS=logs3 STEPS=1536)

주의: t/s 는 덤프 런(KAIROX_DUMP_ACTIVATION=1)에서 읽은 값이다. 계측 오버헤드가
      섞여 있고 반복도 없으므로 셀 간 방향 확인용이지 처리량 수치가 아니다.
"""
import csv, glob, os, re, unicodedata

STEPS = int(os.environ.get("STEPS", 3 * 512))   # BENCH_RUNS x N
DUMPS = os.environ.get("DUMPS", "dumps3")
LOGS  = os.environ.get("LOGS",  "logs3")
RE_TS = re.compile(r"decode mean:\s*([0-9.]+)")

d = {}
for p in sorted(glob.glob(f"{DUMPS}/*.csv")):
    stem = os.path.basename(p)[:-4]
    if "__" not in stem:
        continue
    name, cell = stem.split("__", 1)

    lay = {}
    act = hit = 0
    with open(p) as f:
        for r in csv.DictReader(f):
            v = lay.setdefault(r["layer"], [0, 0])
            v[0] += int(r["total_loads"])
            v[1] += int(r["wasted_loads"])
            act  += int(r["activation_count"])
            hit  += int(r["hit_count"])
    tot = sum(v[0] for v in lay.values())
    wst = sum(v[1] for v in lay.values())
    nl  = sum(1 for v in lay.values() if v[0] > 0)
    if not tot or not nl:
        continue

    ts = None
    try:
        with open(f"{LOGS}/{stem}.log", errors="ignore") as f:
            m = RE_TS.findall(f.read())
            if m:
                ts = float(m[-1])
    except OSError:
        pass

    d[(name, cell)] = dict(tot=tot, use=tot - wst, wst=wst, den=nl * STEPS, nl=nl,
                           ratio=wst / tot * 100, hit=hit / act * 100 if act else 0,
                           ts=ts)

models = sorted({k[0] for k in d})


def w(s):  # 한글은 터미널에서 두 칸을 차지한다
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def rj(s, n):
    return " " * max(0, n - w(s)) + s


def lj(s, n):
    return s + " " * max(0, n - w(s))


H = (lj("모델", 13) + lj("셀", 4) + rj("총전송", 12) + rj("유용", 12) + rj("낭비", 12)
     + rj("낭비율", 7) + rj("적중률", 7) + rj("총/Ls", 8) + rj("유용/Ls", 8)
     + rj("낭비/Ls", 8) + rj("t/s", 8))


def num(v):
    return f"{v:>8.2f}" if v is not None else rj("—", 8)


def dnum(v):
    return f"{v:>+8.2f}" if v is not None else rj("—", 8)


for pol, lo, hi in (("배포본  A", "A0", "A1"),
                    ("논문ANB B", "B0", "B1"),
                    ("되먹임X C", "C0", "C1"),
                    ("상한 iso D", "D0", "D1"),
                    ("상한 1/2 E", "E0", "E1"),
                    ("상한 1/4 F", "F0", "F1")):
    pairs = [(n, d.get((n, lo)), d.get((n, hi))) for n in models]
    if not any(a or b for _, a, b in pairs):
        continue
    print(f"\n=== {pol}   기본 g  ->  g=1 + 병합")
    print(H)
    for name, a, b in pairs:
        for cell, v in ((lo, a), (hi, b)):
            if not v:
                continue
            print(f"{name:<13}{cell:<4}{v['tot']:>12,}{v['use']:>12,}{v['wst']:>12,}"
                  f"{v['ratio']:>6.1f}%{v['hit']:>6.1f}%"
                  f"{v['tot']/v['den']:>8.1f}{v['use']/v['den']:>8.1f}{v['wst']/v['den']:>8.1f}"
                  f"{num(v['ts'])}")
        if a and b:
            dts = (b['ts'] - a['ts']) if (a['ts'] and b['ts']) else None
            rts = (b['ts'] / a['ts']) if (a['ts'] and b['ts']) else None
            print(lj('',13) + lj('Δ',4) + f"{b['tot']-a['tot']:>+12,}{b['use']-a['use']:>+12,}"
                  f"{b['wst']-a['wst']:>+12,}{b['ratio']-a['ratio']:>+6.1f}p"
                  f"{b['hit']-a['hit']:>+6.1f}p"
                  f"{(b['tot']-a['tot'])/a['den']:>+8.1f}{(b['use']-a['use'])/a['den']:>+8.1f}"
                  f"{(b['wst']-a['wst'])/a['den']:>+8.1f}{dnum(dts)}")
            rw = f"{b['wst']/a['wst']:.3f}x" if a['wst'] else "—"
            print(lj('',13) + lj('배',4) + f"{b['tot']/a['tot']:>11.3f}x{b['use']/a['use']:>11.3f}x"
                  f"{rw:>12}{'':>38}"
                  f"{rj(f'{rts:.3f}x' if rts else '—', 8)}")
            print()

print("\n=== 정책 교차 (총전송 배 / t/s 배)")
print(lj('모델',13)+rj('기본g B/A',16)+rj('기본g C/A',16)+rj('g=1 B/A',16)+rj('g=1 C/A',16))
for name in models:
    out = [f"{name:<13}"]
    for hi_cell, lo_cell in (("B0", "A0"), ("C0", "A0"), ("B1", "A1"), ("C1", "A1")):
        p, q = d.get((name, lo_cell)), d.get((name, hi_cell))
        if not (p and q):
            out.append(rj('—', 16))
            continue
        t = f"{q['tot']/p['tot']:.2f}"
        s = f"{q['ts']/p['ts']:.2f}" if (p['ts'] and q['ts']) else "—"
        out.append(rj(t + ' / ' + s, 16))
    print("".join(out))

print("\n=== 제안(g=1+병합+상한) 대 배포본 A0   셀마다  총전송배 / 낭비율 / 적중률 / t/s배")
print(lj("모델", 13) + rj("D1 (iso)", 27) + rj("E1 (1/2)", 27) + rj("F1 (1/4)", 27))
for name in models:
    a = d.get((name, "A0"))
    row = [lj(name, 13)]
    for c in ("D1", "E1", "F1"):
        q = d.get((name, c))
        if not (a and q):
            row.append(rj("—", 27))
            continue
        s_ = f"{q['ts']/a['ts']:.2f}" if (a['ts'] and q['ts']) else "—"
        row.append(rj(f"{q['tot']/a['tot']:.2f} / {q['ratio']:.1f}% / {q['hit']:.1f}% / {s_}", 27))
    print("".join(row))

print("\n=== 동일전송 확인 (총/Ls 가 비슷해야 같은 바이트 천장)")
print(lj("모델", 13) + rj("A0", 10) + rj("D0", 10) + rj("D1", 10)
      + rj("E0", 10) + rj("E1", 10) + rj("F0", 10) + rj("F1", 10))
for name in models:
    row = [lj(name, 13)]
    for c in ("A0", "D0", "D1", "E0", "E1", "F0", "F1"):
        q = d.get((name, c))
        row.append(rj(f"{q['tot']/q['den']:.1f}" if q else "—", 10))
    print("".join(row))
