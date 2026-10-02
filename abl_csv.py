"""절제 실험 비교표를 CSV 로 뽑는다.  usage: python3 abl_csv.py [출력디렉터리]

네 비교에 대해 모듈별 시간(ms/토큰)과 처리량을 한 장씩 낸다.
  abl_cmp_Ga_Niso.csv     배포본(Ga)  vs 제안(Niso)
  abl_cmp_G_Niso.csv      구현본(G)   vs 제안(Niso)
  abl_cmp_Ga_Nhalf.csv    배포본(Ga)  vs 제안(Nhalf)
  abl_cmp_G_Nhalf.csv     구현본(G)   vs 제안(Nhalf)

헤더와 정책 라벨은 영어, 인코딩은 utf-8-sig 다 — Excel 이 BOM 없는 UTF-8 을
현지 코드페이지로 읽어 한글이 깨진다.

읽는 것
  prof/abl_<모델>_<셀>_cuda_gpu_kern_sum.csv    커널별 시간      (ablation.sh nsys)
  prof/abl_<모델>_<셀>_cuda_gpu_mem_time_sum.csv H2D 시간
  abl_logs/plan__<모델>__<셀>.log               호스트 비용, 전송량 (plan)
  abl_clocks.txt                                처리량           (ts)
  abl_dumps/<모델>__<셀>.csv                    활성/낭비        (dump, 없으면 빈 열)

주의
  - 전송 과 H2D 는 같은 일의 두 장부다. naive 는 복사 엔진(H2D), zerocopy 는 커널(전송).
    따로 읽으면 안 되고 GPU합 = Σ커널 + H2D 로 봐야 비교가 된다.
  - 커널은 nsys 런(-n 128), 호스트와 전송량은 plan 런(-n 512), t/s 는 ts 런(-n 512 x 3회,
    3반복 중위)에서 왔다. 전부 토큰당으로 환산했지만 런 길이가 다르다.
"""
import csv, glob, os, re, statistics, sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
MS = ["Bamboo", "SparseQwen2", "opt-30b", "opt-6.7b", "prosparse-7b"]

ROLES = [("정렬",   ("RadixSort", "cub::", "DeviceRadix", "init_indices", "init_offsets")),
         ("마스크", ("kairox_dfr_mask", "index_mask")),
         ("임계",   ("shifted_step",)),
         ("압축",   ("kairox_compact",)),
         ("전송",   ("kairox_zerocopy", "kairox_scatter", "kairox_gather"))]
COMPUTE = ("axpy", "dequantize", "mul_mat", "_mmv", "_mmq", "gemm", "soft_max",
           "rms_norm", "rope", "silu", "relu", "cpy", "add", "mul_f32", "norm")


def role(n):
    for r, keys in ROLES:
        if any(k in n for k in keys):
            return r
    return "본연산" if any(k in n.lower() for k in COMPUTE) else "기타"


def num(v):
    try:
        return float(str(v).replace(",", ""))
    except (TypeError, ValueError):
        return 0.0


def pick(hdr, *pre):
    for p in pre:
        for h in hdr:
            if h.strip().lower().startswith(p):
                return h
    return None


# ---- 처리량 (ts)
ts = {}
if os.path.exists("abl_clocks.txt"):
    for line in open("abl_clocks.txt", errors="ignore"):
        p = line.split()
        if len(p) >= 4 and re.fullmatch(r"[0-9.]+", p[3]):
            ts.setdefault((p[0], p[1]), []).append(float(p[3]))

# ---- 호스트 / 전송량 / 선택크기 (plan)
PL = {}
for p in glob.glob("abl_logs/plan__*.log"):
    stem = os.path.basename(p)[6:-4]
    if "__" not in stem:
        continue
    m, c = stem.split("__", 1)
    t = open(p, errors="ignore").read()
    one = lambda rx: (re.findall(rx, t) or [None])[-1]
    PL[(m, c)] = dict(scan=one(r"스캔\s*:\s*([0-9.]+)"), apply=one(r"적용\s*:\s*([0-9.]+)"),
                      pairs=one(r"실행\s+([0-9.]+)"), ng=one(r"n_groups=(\d+)"),
                      nl=one(r"스텝 x (\d+) 레이어"), sel=one(r"\(([0-9.]+)\s*배\)"))

# ---- 커널 / H2D (nsys)
K = {}
for p in glob.glob("prof/abl_*_cuda_gpu_kern_sum.csv"):
    base = os.path.basename(p)[4:-22]
    mod = cell = None
    for m in MS:
        if base.startswith(m + "_"):
            mod, cell = m, base[len(m) + 1:]
            break
    if not mod:
        continue
    rows = list(csv.DictReader(open(p, newline="")))
    if not rows:
        continue
    hdr = rows[0].keys()
    ct, cn = pick(hdr, "total time"), pick(hdr, "name")
    ci = pick(hdr, "instances", "count")
    d, inst = {}, 0
    for r in rows:
        d[role(r[cn])] = d.get(role(r[cn]), 0.0) + num(r[ct])
        if "shifted_step" in r[cn]:
            inst = int(num(r[ci]))
    nl = int((PL.get((mod, cell)) or {}).get("nl") or 32)
    steps = inst / nl if inst else 129          # nsys 런의 실제 스텝수를 역산한다

    h2d = 0.0
    q = p.replace("kern_sum", "mem_time_sum")
    if os.path.exists(q):
        rr = list(csv.DictReader(open(q, newline="")))
        if rr:
            hh = rr[0].keys()
            cc, co = pick(hh, "total"), pick(hh, "operation")
            if cc and co:
                h2d = sum(num(r[cc]) for r in rr if "Host-to" in r[co]) / 1e6 / steps
    K[(mod, cell)] = dict(h2d=h2d, kern=sum(d.values()) / 1e6 / steps,
                          **{k: v / 1e6 / steps for k, v in d.items()})

# ---- 활성 / 낭비 (dump, 있으면)
DP = {}
for p in glob.glob("abl_dumps/*.csv"):
    stem = os.path.basename(p)[:-4]
    if "__" not in stem:
        continue
    m, c = stem.split("__", 1)
    tot = wst = act = hit = 0
    layers = set()
    for r in csv.DictReader(open(p)):
        tot += int(r["total_loads"]); wst += int(r["wasted_loads"])
        act += int(r["activation_count"]); hit += int(r["hit_count"])
        if int(r["total_loads"]) or int(r["activation_count"]):
            layers.add(r["layer"])
    if tot:
        DP[(m, c)] = dict(tot=tot, wst=wst, act=act, hit=hit, nl=len(layers))

# 역할 이름은 커널 분류에 쓰는 내부 키라 한글로 두고, CSV 헤더만 영어로 낸다.
# (Excel 이 BOM 없는 UTF-8 을 못 읽어 한글 헤더가 깨졌다 — 아래 utf-8-sig 도 같은 이유)
#
# 열을 묶는 규칙
#   selection = 정렬 + 마스크 + 임계   선택 기구. 세 커널이 늘 같이 움직인다
#   transfer  = 전송커널 + H2D         같은 바이트의 두 장부. naive 는 복사 엔진,
#                                      zerocopy 는 커널 — 따로 읽으면 비교가 안 된다
#   other     = 압축 + 기타            이 네 비교에는 압축 셀이 없어 늘 0 이다
# 네 열(selection/transfer/compute/other)의 합이 정확히 gpu_total 이다 — 검산용.
# 버린 열: sel_over_K (NOSORT 셀이 없어 전 행 1.000), kernel_total/host_total (파생).
GROUPS = [("selection", ["정렬", "마스크", "임계"]),
          ("transfer",  ["전송"]),                      # h2d 는 아래에서 더한다
          ("compute",   ["본연산"]),
          ("other",     ["압축", "기타"])]
HDR = (["model", "policy", "cell", "selection", "transfer", "compute", "other",
        "gpu_total", "host_scan", "host_apply",
        "rebalanced", "actived", "wasted", "wasted_rate_pct", "tok_per_s"])


def nff_of(m):
    ngs = [int(v["ng"]) for (mm, _), v in PL.items() if mm == m and v.get("ng")]
    return max(ngs) if ngs else 0


def row(m, label, cell):
    k = K.get((m, cell))
    p = PL.get((m, cell)) or {}
    v = ts.get((m, cell))
    d = DP.get((m, cell))
    nff = nff_of(m)
    reb = (float(p["pairs"]) * (nff / int(p["ng"]))) if (p.get("pairs") and p.get("ng") and nff) else None
    scan = float(p["scan"]) if p.get("scan") else None
    app = float(p["apply"]) if p.get("apply") else None
    f = lambda x, n=2: ("" if x is None else f"{x:.{n}f}")
    out = [m, label, cell]
    for name, keys in GROUPS:
        g = sum(k.get(x, 0.0) for x in keys) if k else None
        if name == "transfer" and k:
            g += k["h2d"]        # 같은 바이트의 두 장부를 합친다
        out.append(f(g))
    out += [f(k["kern"] + k["h2d"]) if k else "", f(scan, 3), f(app, 3), f(reb, 1)]
    if d:
        den = d["nl"] * 1  # Actived/Wasted 는 전송량과 같은 단위로 맞춘다 (로드당 비율로 환산)
        use = d["tot"] - d["wst"]
        out += [f(reb * use / d["tot"], 1) if reb else "",
                f(reb * d["wst"] / d["tot"], 1) if reb else "",
                f(d["wst"] / d["tot"] * 100, 1)]
    else:
        out += ["", "", ""]
    out += [f(statistics.median(v)) if v else ""]
    return out


os.makedirs(OUT, exist_ok=True)
made = []
for base, blab, prop in (("Ga", "배포본", "Niso"), ("G", "구현본", "Niso"),
                         ("Ga", "배포본", "Nhalf"), ("G", "구현본", "Nhalf")):
    path = os.path.join(OUT, f"abl_cmp_{base}_{prop}.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.writer(fh)
        wr.writerow(HDR)
        for m in MS:
            a, b = row(m, "baseline", base), row(m, "proposed", prop)
            wr.writerow(a)
            wr.writerow(b)
            # Δ 행: 수치 열만 뺀다
            dl = [m, "delta", f"{prop}-{base}"]
            for i in range(3, len(HDR)):
                try:
                    dl.append(f"{float(b[i]) - float(a[i]):+.2f}")
                except ValueError:
                    dl.append("")
            wr.writerow(dl)
    made.append(path)

print("쓴 파일:")
for p in made:
    print("  " + p)
print("\n덤프 열(Actived/Wasted/WastedRate):",
      "채워짐" if DP else "비어 있음 — bash ablation.sh dump 필요")
