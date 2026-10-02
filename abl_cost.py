"""미스당 CPU 비용.  usage: python3 abl_cost.py

이 한 숫자가 "뉴런 단위가 CPU 를 아낀다" 의 생사를 가른다.

    CPU 시간 = 미스 수 x 미스당 비용

미스 수가 줄어든 건 측정됐다(적중률 +2.1~13.6%p). 그런데 CPU 시간은 늘었다
(work +30~69%). 둘 다 참이려면 미스당 비용이 그보다 더 올랐다는 뜻인데,
그 비율 R 을 아직 안 쟀다.

    R x (미스 비율) < 1   ->  이점이 실재하고 지금은 다른 팔에 가려져 있을 뿐.
                              CPU 가 max 팔이 되는 시나리오를 만들면 드러난다
    R x (미스 비율) > 1   ->  구조적으로 거짓. 어떤 시나리오에서도 안 된다.
                              선택 단위와 메모리 지역성이 같은 변수에 묶여 있는 한계다

R 은 메모리 시스템의 성질이라 어느 팔이 묶이는지와 무관하다. 지금 작동점에서
그대로 잴 수 있다.

읽는 것
  abl_logs/cpu__<모델>__<셀>.log   work / join / evsync   (ablation.sh cpu)
  abl_dumps/<모델>__<셀>.csv       활성 / 적중            (ablation.sh dump)
"""
import csv, glob, os, re, sys

LOGS = os.environ.get("LOGS", "abl_logs")
DUMP = os.environ.get("DUMPS", "abl_dumps")
BASE = os.environ.get("BASE", "Ga")          # 비교 기준 (그룹 입도)
MS   = ["opt-6.7b", "prosparse-7b", "SparseQwen2", "Bamboo", "opt-30b"]
CE   = ["Gorig", "Ga", "Gnsc", "Gansc", "N", "Niso", "Nhalf", "Nns", "Nnsc"]


def w(s):
    import unicodedata
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def rj(s, n):
    return " " * max(0, n - w(s)) + s


def lj(s, n):
    return s + " " * max(0, n - w(s))


# ---- CPU 팔 (cpu 패밀리)
CPU = {}
for p in glob.glob(f"{LOGS}/cpu__*.log"):
    stem = os.path.basename(p)[5:-4]
    if "__" not in stem:
        continue
    m, c = stem.split("__", 1)
    t = open(p, errors="ignore").read()
    one = lambda rx: (re.findall(rx, t) or [None])[-1]
    steps = one(r"프로파일 \(([\d.]+) 스텝\)")
    work  = one(r"work\s*:\s*([0-9.]+)")
    join  = one(r"join\s*:\s*([0-9.]+)")
    evs   = one(r"evsync\s*:\s*([0-9.]+)")
    if work:
        CPU[(m, c)] = dict(work=float(work), join=float(join or 0),
                           evsync=float(evs or 0), steps=float(steps or 0))

# ---- 미스 수 (dump 패밀리)
DP = {}
for p in glob.glob(f"{DUMP}/*.csv"):
    stem = os.path.basename(p)[:-4]
    if "__" not in stem:
        continue
    m, c = stem.split("__", 1)
    act = hit = 0
    for r in csv.DictReader(open(p)):
        act += int(r["activation_count"])
        hit += int(r["hit_count"])
    if act:
        DP[(m, c)] = dict(act=act, hit=hit, miss=act - hit)

if not CPU or not DP:
    print("필요한 데이터가 없다.")
    print(f"  cpu  로그 {len(CPU)} 개 — bash ablation.sh cpu")
    print(f"  dump 파일 {len(DP)} 개 — bash ablation.sh dump")
    sys.exit(0)

print("미스당 CPU 비용   (work 는 cpu 런, 미스는 dump 런 — 따로 재서 합친다)\n")
print(lj("모델", 14) + lj("셀", 7) + rj("work ms/tok", 12) + rj("적중률", 8)
      + rj("미스/토큰", 11) + rj("us/미스", 9) + rj("join", 8) + rj("evsync", 8)
      + rj("max 팔", 9))

ROWS = {}
for m in MS:
    shown = False
    for c in CE:
        k, d = CPU.get((m, c)), DP.get((m, c))
        if not (k and d):
            continue
        steps = k["steps"] or 513
        miss = d["miss"] / steps                       # 토큰당 미스 뉴런 수
        per = k["work"] * 1000.0 / miss if miss else 0  # us/미스
        ROWS[(m, c)] = dict(work=k["work"], miss=miss, per=per)
        print(lj(m if not shown else "", 14) + lj(c, 7)
              + rj(f"{k['work']:.3f}", 12)
              + rj(f"{d['hit']/d['act']*100:.1f}%", 8)
              + rj(f"{miss:,.0f}", 11)
              + rj(f"{per:.3f}", 9)
              + rj(f"{k['join']:.2f}", 8) + rj(f"{k['evsync']:.2f}", 8)
              + rj("CPU" if k["join"] > k["evsync"] else "GPU", 9))
        shown = True
    if shown:
        print()

print(f"\n판정   기준 {BASE} 대비.  미스비 x R < 1 이면 이점이 실재한다\n")
print(lj("모델", 14) + lj("셀", 7) + rj("미스비", 9) + rj("R", 9)
      + rj("곱", 9) + rj("판정", 24))
for m in MS:
    b = ROWS.get((m, BASE))
    if not b:
        continue
    for c in CE:
        v = ROWS.get((m, c))
        if not v or c == BASE:
            continue
        mr = v["miss"] / b["miss"]
        R  = v["per"] / b["per"]
        prod = mr * R
        verdict = ("이점 실재 — 가려져 있을 뿐" if prod < 0.98 else
                   "동률" if prod < 1.02 else "지역성 손실이 더 크다")
        print(lj(m, 14) + lj(c, 7) + rj(f"{mr:.3f}", 9) + rj(f"{R:.3f}", 9)
              + rj(f"{prod:.3f}", 9) + rj(verdict, 24))
