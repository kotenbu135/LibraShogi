#!/usr/bin/env python3
"""頭打ちの引き金（docs/release.md §0）がどれだけ効くかを数える。

実データの点のあとに節目を足していく計算を条件ごとに何度も回し、
「本当に伸びが落ちたとき鳴るか」「落ちていないのに鳴らないか」を数える。
引き金や節目の置き方、自己評価の局数を見直すときはこれを回してから決める
（2026-09-20 のユーザーの依頼「どこで伸びが鈍った、と判断したらいいか検討」）。

    python3 tools/plateau_trigger_power.py

REAL は 2026-09-20（280 万局）時点の `libra rating --curve` の点。
新しい節目が出たら足す（progress ブランチの progress/ls.json の rating.curve.points）。
"""
import math, json, random

REAL = [(4738,0.0),(18720,163.5),(120171,374.8),(200361,471.0),(300500,623.1),(400021,736.4),
        (800281,908.7),(1200126,1014.4),(1402864,1059.6),(1600085,1090.2),(2000241,1148.4),
        (2399947,1185.8),(2799909,1210.9)]
CI = 35.0          # 1 点の 95% 区間の半幅（実測 ±35.5）
SD = CI/1.96

def fit(pts):
    xs=[math.log2(g) for g,_ in pts]; ys=[e for _,e in pts]
    n=len(xs); mx=sum(xs)/n; my=sum(ys)/n
    den=sum((x-mx)**2 for x in xs)
    if den==0: return 0.0,my,0.0
    b=sum((x-mx)*(y-my) for x,y in zip(xs,ys))/den
    a=my-b*mx
    rms=math.sqrt(sum((y-(a+b*x))**2 for x,y in zip(xs,ys))/n)
    return b,a,rms

def fit_recent(pts, doublings=4, minpts=4):
    cut = pts[-1][0]/2**doublings
    tail=[p for p in pts if p[0]>=cut]
    return fit(tail if len(tail)>=minpts else pts)

def fires(hist):
    """release.md §0 の引き金: 続けて 2 つ以上、目安の線より下で、95% 区間が線をまたがない"""
    below=0
    for i in range(len(hist)-1, max(len(hist)-3,0)-1, -1):
        b,a,_ = fit_recent(hist[:i])          # その点より前の点だけで線を引く
        g,e = hist[i]
        pred = a + b*math.log2(g)
        if e + CI < pred: below += 1
        else: break
    return below >= 2

def run(true_slope_after, brk, n_more=30, seed=0):
    """brk 局以降の真の傾きを true_slope_after にして、40 万局ごとの節目を足していく"""
    rnd = random.Random(seed)
    b,a,_ = fit(REAL[3:])                     # 20 万局以降の実データの傾き
    hist = list(REAL)
    true_at_brk = a + b*math.log2(brk)
    for k in range(1, n_more+1):
        g = brk + 400000*k
        true = true_at_brk + true_slope_after*math.log2(g/brk)
        hist.append((g, true + rnd.gauss(0, SD)))
        if fires(hist):
            return k, g
    return None, None

print("=== 実データで今の引き金が鳴っているか ===")
for i in range(6, len(REAL)+1):
    h = REAL[:i]
    b,a,_ = fit_recent(h[:-1])
    g,e = h[-1]
    pred = a+b*math.log2(g)
    print(f"  {g:>9,} 実測 {e:7.1f}  線 {pred:7.1f}  ずれ {e-pred:+6.1f}  "
          f"{'下に外れた' if e+CI<pred else ''}")
print("  → 鳴っている?", fires(REAL))

print("\n=== 280 万局で真の伸びが落ちたら、何回目の節目で鳴るか（40 万局ごと、20 回 = 各条件） ===")
print("  落ちた後の傾き | 鳴るまでの節目 | そのときの総局数 | 鳴らなかった回")
for s in (0.0, 47.0, 95.0, 142.0):
    ks=[]; miss=0
    for seed in range(20):
        k,g = run(s, 2799909, seed=seed)
        if k is None: miss+=1
        else: ks.append((k,g))
    if ks:
        ks.sort()
        med = ks[len(ks)//2]
        print(f"  +{s:5.0f} / 2倍     | 中央 {med[0]:>2} 回      | {med[1]:>10,}      | {miss}/20")
    else:
        print(f"  +{s:5.0f} / 2倍     | 30 回でも鳴らず |            | {miss}/20")

print("\n=== 自己評価の局数を増やすと、鈍りをどれだけ捕まえられるか ===")
print("  （1 点の 95% 区間は局数の平方根で縮む。1,000 局で ±35 Elo の実測から換算）")


def _power(ci, slope, trials=400, n=20, start=2799909, step=400000):
    global CI, SD
    keep_ci, keep_sd = CI, SD
    CI, SD = ci, ci / 1.96
    try:
        b, a, _ = fit(REAL[3:])
        base = a + b * math.log2(start)
        ok = []
        for seed in range(trials):
            rnd = random.Random(seed)
            hist = list(REAL)
            hit = None
            for i in range(1, n + 1):
                g = start + step * i
                hist.append((g, base + slope * math.log2(g / start) + rnd.gauss(0, SD)))
                if hit is None and fires(hist):
                    hit = (i, g)
            ok.append(hit)
        got = sorted(h for h in ok if h)
        return len(got) / trials, (got[len(got) // 2] if got else None)
    finally:
        CI, SD = keep_ci, keep_sd


print("  自己評価の局数 | 点の幅 | 誤報 | 25% の鈍りを捕まえる | 半減を捕まえる")
for games, ci in ((1000, 35.0), (2000, 25.0), (4000, 18.0)):
    fa, _ = _power(ci, 191.4)
    md, mm = _power(ci, 142.0)
    hf, hh = _power(ci, 95.0)
    sm = f"{mm[1] / 1e6:.2f}M" if mm else "—"
    sh = f"{hh[1] / 1e6:.2f}M" if hh else "—"
    print(f"  {games:>6,} 局     | ±{ci:4.0f}  | {fa * 100:4.1f}% | {md * 100:5.1f}% ({sm})      | {hf * 100:5.1f}% ({sh})")
