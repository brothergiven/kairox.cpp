"""절제 실험 요약.  usage: python3 abl_sum.py

읽는 것
  prof/abl_<모델>_<셀>_cuda_gpu_kern_sum.csv    커널별 시간/호출수   (ablation.sh nsys)
  prof/abl_<모델>_<셀>_cuda_gpu_mem_*_sum.csv   H2D/D2H 시간과 바이트
  abl_logs/plan__<모델>__<셀>.log               스텝수, 실행량, 호스트 비용, |S|/K  (plan)
  abl_nsys_wall.txt                             nsys 런과 짝지어진 벽시계           (nsys)
  abl_clocks.txt                                긴 런 처리량과 클럭/온도            (ts)

주의
  - 커널 시간 합은 벽시계가 아니다. 차액이 GPU 유휴(겹침)다. 그래서 둘을 같이 낸다.
  - nsys 런(-n 128, 1회)과 ts 런(-n 512, 3회)은 수준이 다르다. 유휴 비율은
    abl_nsys_wall.txt(같은 인자)로만 계산하고, ts 값은 처리량 판정에만 쓴다.
"""
import csv, glob, os, re, statistics, unicodedata

PROF  = os.environ.get("PROF",  "prof")
LOGS  = os.environ.get("LOGS",  "abl_logs")
CELLS = ["Ga", "Na", "G", "N", "Niso", "Nhalf", "Nns", "Nnsc", "Nb"]

# 커널을 역할로 묶는다. 위에서부터 먼저 맞는 것.
ROLES = [
    ("정렬",   ("RadixSort", "cub::", "DeviceRadix", "init_indices", "init_offsets")),
    ("마스크", ("kairox_dfr_mask", "index_mask")),
    ("임계",   ("shifted_step",)),
    ("전송",   ("kairox_zerocopy", "kairox_scatter", "kairox_gather")),
]
ORDER = ["정렬", "마스크", "임계", "전송", "본연산", "기타"]
COMPUTE = ("axpy", "dequantize", "mul_mat", "_mmv", "_mmq", "gemm", "soft_max",
           "rms_norm", "rope", "silu", "relu", "cpy", "add", "mul_f32", "norm")


def role_of(name):
    for r, keys in ROLES:
        if any(k in name for k in keys):
            return r
    low = name.lower()
    if any(k in low for k in COMPUTE):
        return "본연산"
    return "기타"


def col(hdr, *cands):
    for c in cands:
        for h in hdr:
            if h.strip().lower().startswith(c):
                return h
    return None


def read_csv(path):
    if not os.path.exists(path):
        return None, []
    with open(path, newline="") as f:
        rd = csv.DictReader(f)
        return rd.fieldnames or [], list(rd)


def f(v):
    try:
        return float(str(v).replace(",", ""))
    except (TypeError, ValueError):
        return 0.0


RE_STEPS = re.compile(r"호출\s*:\s*[\d,]+\s*\(([\d,]+)\s*스텝")
RE_SCAN  = re.compile(r"스캔\s*:\s*([0-9.]+)\s*ms")
RE_APPLY = re.compile(r"적용\s*:\s*([0-9.]+)\s*ms")
RE_PAIRS = re.compile(r"실행\s+([0-9.]+)")
RE_SEL   = re.compile(r"\(([0-9.]+)\s*배\)")
RE_NG    = re.compile(r"n_groups=(\d+)")


def read_plan(model, cell):
    p = f"{LOGS}/plan__{model}__{cell}.log"
    if not os.path.exists(p):
        return {}
    t = open(p, errors="ignore").read()

    def one(rx, cast=float):
        m = rx.findall(t)
        return cast(m[-1].replace(",", "")) if m else None

    return dict(steps=one(RE_STEPS, lambda s: int(float(s))), scan=one(RE_SCAN),
                apply=one(RE_APPLY), pairs=one(RE_PAIRS), sel=one(RE_SEL),
                ngroups=one(RE_NG, lambda s: int(float(s))))


def read_pairs_file(path, keep_na=False):
    """모델 셀 값 ... 형태의 줄을 {(모델,셀): [값...]} 로 모은다."""
    out = {}
    if not os.path.exists(path):
        return out
    for line in open(path, errors="ignore"):
        p = line.split()
        if len(p) < 3:
            continue
        model, cell = p[0], p[1]
        vals = [x for x in p[2:] if re.fullmatch(r"[0-9.]+", x)]
        if not vals and not keep_na:
            continue
        if vals:
            out.setdefault((model, cell), []).append(float(vals[0]))
    return out


nsys_wall = read_pairs_file("abl_nsys_wall.txt")
ts_wall   = {}
for line in (open("abl_clocks.txt", errors="ignore") if os.path.exists("abl_clocks.txt") else []):
    p = line.split()
    if len(p) >= 4 and re.fullmatch(r"[0-9.]+", p[3]):
        ts_wall.setdefault((p[0], p[1]), []).append(float(p[3]))


def load(model, cell):
    tag = f"{PROF}/abl_{model}_{cell}"
    hdr, rows = read_csv(f"{tag}_cuda_gpu_kern_sum.csv")
    if not rows:
        return None
    c_t, c_n, c_i = col(hdr, "total time"), col(hdr, "name"), col(hdr, "instances", "count")
    if not (c_t and c_n):
        return None

    kern, roles = [], {}
    for r in rows:
        name, ns, inst = r[c_n], f(r[c_t]), int(f(r[c_i])) if c_i else 0
        kern.append((name, ns, inst))
        roles[role_of(name)] = roles.get(role_of(name), 0.0) + ns
    total = sum(k[1] for k in kern)

    mem = {}
    hdr2, rows2 = read_csv(f"{tag}_cuda_gpu_mem_time_sum.csv")
    if rows2:
        ct, co = col(hdr2, "total time"), col(hdr2, "operation")
        if ct and co:
            for r in rows2:
                mem[r[co]] = mem.get(r[co], 0.0) + f(r[ct])
    size = {}
    hdr3, rows3 = read_csv(f"{tag}_cuda_gpu_mem_size_sum.csv")
    if rows3:
        cs, co = col(hdr3, "total"), col(hdr3, "operation")
        if cs and co:
            for r in rows3:
                size[r[co]] = size.get(r[co], 0.0) + f(r[cs])

    return dict(kern=kern, roles=roles, total=total, mem=mem, size=size,
                plan=read_plan(model, cell))


def w(s):
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def rj(s, n):
    return " " * max(0, n - w(s)) + s


def lj(s, n):
    return s + " " * max(0, n - w(s))


models = sorted({os.path.basename(p).split("_")[1]
                 for p in glob.glob(f"{PROF}/abl_*_cuda_gpu_kern_sum.csv")}) or \
         sorted({os.path.basename(p).split("__")[1]
                 for p in glob.glob(f"{LOGS}/plan__*.log")})

for model in models:
    data = {c: load(model, c) for c in CELLS}
    data = {c: v for c, v in data.items() if v}
    plans = {c: read_plan(model, c) for c in CELLS}

    # ---- 1. 동일전송과 선택 크기 확인. 이게 안 맞으면 아래 표는 전부 무효다.
    print(f"\n{'='*96}\n{model}\n{'='*96}")
    print("\n[1] 전송량과 선택 크기   (plan 패밀리 — 이 표가 안 맞으면 아래는 무효)")
    print(lj("셀", 7) + rj("n_groups", 10) + rj("실행짝/Ls", 11) + rj("뉴런/Ls", 10)
          + rj("vs G", 8) + rj("|S|/K", 8) + rj("스캔ms", 8) + rj("적용ms", 8) + rj("호스트ms", 9))
    # g=1 split 의 n_groups 가 곧 n_ff 이므로, 가장 큰 n_groups 로 나누면 group_size 가 나온다.
    ngs = [p["ngroups"] for p in plans.values() if p.get("ngroups")]
    n_ff = max(ngs) if ngs else 0

    def neurons(p):
        ng = p.get("ngroups") or 0
        return p["pairs"] * (n_ff / ng) if (ng and n_ff) else None

    gref = neurons(plans["G"]) if (plans.get("G") or {}).get("pairs") else None
    for c in CELLS:
        p = plans.get(c) or {}
        if not p.get("pairs"):
            continue
        ng = p.get("ngroups") or 0
        neu = neurons(p) or 0.0
        host = (p.get("scan") or 0) + (p.get("apply") or 0)
        print(lj(c, 7) + rj(f"{ng:,}", 10) + rj(f"{p['pairs']:.1f}", 11) + rj(f"{neu:.1f}", 10)
              + rj(f"{neu/gref:.3f}" if gref else "—", 8)
              + rj(f"{p['sel']:.3f}" if p.get("sel") else "—", 8)
              + rj(f"{p.get('scan') or 0:.3f}", 8) + rj(f"{p.get('apply') or 0:.3f}", 8)
              + rj(f"{host:.3f}", 9))

    if not data:
        print("\n(nsys 데이터 없음 — bash ablation.sh nsys)")
        continue

    # ---- 2. 역할별 커널 시간. ms/토큰.
    def steps(c):
        return (plans.get(c) or {}).get("steps") or (data[c]["plan"].get("steps")) or 0

    print("\n[2] 역할별 커널 시간  ms/토큰   (스텝수는 plan 로그에서 읽는다)")
    print(lj("셀", 7) + rj("스텝", 7) + "".join(rj(r, 9) for r in ORDER)
          + rj("Σ커널", 9) + rj("H2D", 8) + rj("H2D MB", 9))
    for c in CELLS:
        v = data.get(c)
        if not v:
            continue
        s = steps(c)
        if not s:
            print(lj(c, 7) + rj("—", 7) + "  (스텝수 모름 — plan 패밀리 먼저)")
            continue
        row = lj(c, 7) + rj(f"{s:,}", 7)
        for r in ORDER:
            row += rj(f"{v['roles'].get(r, 0.0)/1e6/s:.3f}", 9)
        h2d = sum(t for k, t in v["mem"].items() if "HtoD" in k or "Host-to" in k)
        mb  = sum(t for k, t in v["size"].items() if "HtoD" in k or "Host-to" in k)
        row += rj(f"{v['total']/1e6/s:.3f}", 9) + rj(f"{h2d/1e6/s:.3f}", 8) + rj(f"{mb/s:.3f}", 9)
        print(row)

    # ---- 3. 커널 시간 합 대 벽시계. 차액이 겹침이다.
    print("\n[3] 커널 합 대 벽시계   (짝 벽시계 = nsys 와 같은 인자의 맨런)")
    print(lj("셀", 7) + rj("Σ커널ms", 10) + rj("짝t/s", 9) + rj("짝ms/tok", 10)
          + rj("GPU유휴", 9) + rj("긴t/s", 9) + rj("긴/Ga", 8))
    ga_long = statistics.median(ts_wall[(model, "Ga")]) if (model, "Ga") in ts_wall else None
    for c in CELLS:
        v = data.get(c)
        if not v:
            continue
        s = steps(c)
        ker = v["total"] / 1e6 / s if s else None
        pw  = statistics.median(nsys_wall[(model, c)]) if (model, c) in nsys_wall else None
        pms = 1000.0 / pw if pw else None
        idle = (1 - ker / pms) * 100 if (ker and pms) else None
        lw  = statistics.median(ts_wall[(model, c)]) if (model, c) in ts_wall else None
        print(lj(c, 7) + rj(f"{ker:.3f}" if ker else "—", 10)
              + rj(f"{pw:.2f}" if pw else "—", 9)
              + rj(f"{pms:.3f}" if pms else "—", 10)
              + rj(f"{idle:.1f}%" if idle is not None else "—", 9)
              + rj(f"{lw:.2f}" if lw else "—", 9)
              + rj(f"{lw/ga_long:.3f}" if (lw and ga_long) else "—", 8))

    # ---- 4. 한 변수씩. 각 쌍이 무엇을 분리하는지 이름을 달아 둔다.
    print("\n[4] 한 변수 비교   Δms/토큰 (커널) 과 Δ벽시계")
    PAIRS = [("Nns",  "Niso", "정렬 제거"),
             ("Nnsc", "Nns",  "호스트 스캔 제거"),
             ("Nb",   "Niso", "전송 경로 batch"),
             ("Niso", "G",    "입도+전송 (동일전송)"),
             ("G",    "Ga",   "되먹임 제거 (그룹)"),
             ("N",    "Niso", "상한 해제"),
             ("Na",   "Ga",   "입도 (되먹임 켠 채)"),
             ("Nhalf","Niso", "상한 절반")]
    print(lj("비교", 16) + lj("무엇이 바뀌나", 24)
          + "".join(rj(r, 8) for r in ORDER) + rj("ΣΔ", 8) + rj("Δ벽시계", 10) + rj("배", 7))
    for hi, lo, what in PAIRS:
        a, b = data.get(lo), data.get(hi)
        if not (a and b and steps(lo) and steps(hi)):
            continue
        sa, sb = steps(lo), steps(hi)
        row = lj(f"{hi} - {lo}", 16) + lj(what, 24)
        for r in ORDER:
            row += rj(f"{b['roles'].get(r,0.0)/1e6/sb - a['roles'].get(r,0.0)/1e6/sa:+.3f}", 8)
        row += rj(f"{b['total']/1e6/sb - a['total']/1e6/sa:+.3f}", 8)
        wa = statistics.median(nsys_wall[(model, lo)]) if (model, lo) in nsys_wall else None
        wb = statistics.median(nsys_wall[(model, hi)]) if (model, hi) in nsys_wall else None
        if wa and wb:
            row += rj(f"{1000/wb - 1000/wa:+.3f}", 10) + rj(f"{wb/wa:.3f}", 7)
        else:
            row += rj("—", 10) + rj("—", 7)
        print(row)

    # ---- 5. 선택 기구를 구성하는 커널 전부. 묶음이 뭘 숨기는지 보이게.
    print("\n[5] 선택 기구 커널 상세   ms/토큰 (정렬+마스크+임계만)")
    for c in CELLS:
        v = data.get(c)
        if not v or not steps(c):
            continue
        s = steps(c)
        sel = [(n, t, i) for n, t, i in v["kern"] if role_of(n) in ("정렬", "마스크", "임계")]
        if not sel:
            continue
        print(f"  {c}")
        for n, t, i in sorted(sel, key=lambda x: -x[1]):
            print(f"      {t/1e6/s:8.3f}  {i:>9,}회  {n[:68]}")
