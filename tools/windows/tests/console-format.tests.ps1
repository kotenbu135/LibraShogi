# SPDX-License-Identifier: Apache-2.0
# 管理コンソール（libra-console.ps1）の Elo の文言を確かめる。
# コンソールは 1 枚のスクリプトで、読み込むと WinForms の画面を作ってしまうので、AST から純関数だけを取り出して試す。
# 走らせ方: pwsh -NoProfile -File tools/windows/tests/console-format.tests.ps1
$ErrorActionPreference = "Stop"
$src = Join-Path $PSScriptRoot "..\libra-console.ps1"

$tokens = $null; $errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile((Resolve-Path $src), [ref]$tokens, [ref]$errors)
if ($errors.Count -gt 0) {
    $errors | ForEach-Object { Write-Host ("構文エラー {0}:{1} {2}" -f $_.Extent.StartLineNumber, $_.Extent.StartColumnNumber, $_.Message) }
    exit 1
}
$want = @("Format-Int", "Format-Ago", "From-Unix", "Ci-Val", "Format-Ci", "Format-Pct", "Format-Saturated", "New-Series", "Add-Pt", "Pt-XVal", "Get-XAxis", "Format-Elo-Anchor", "Format-Elo-Best", "Format-Elo-References", "Format-Elo-Rating", "Format-Elo-Note", "Build-Rating-Series")
$found = @{}
foreach ($f in $ast.FindAll({ $args[0] -is [System.Management.Automation.Language.FunctionDefinitionAst] }, $false)) {
    if ($want -contains $f.Name) { $found[$f.Name] = $true; . ([scriptblock]::Create($f.Extent.Text)) }
}
foreach ($n in $want) { if (-not $found.ContainsKey($n)) { Write-Host "関数が見つからない: $n"; exit 1 } }

$fails = 0
function Check([string]$name, $got, $want) {
    if ([string]$got -ne [string]$want) {
        $script:fails++
        Write-Host "NG $name`n  期待: $want`n  実際: $got"
    } else { Write-Host "OK $name" }
}
# 「3 時間前」の部分は今の時刻で変わるので、そこだけ伏せて比べる
function Strip([string]$s) { return ($s -replace "[0-9,\.]+ (秒|分|時間|日)前", "<いつ>") }

$now = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()

# --- 基準比: 2026-09-18 の ls の実データ（progress/ls.json の anchor の末尾） ---
$anchor = [pscustomobject]@{ t = $now; step = 62426; n = 1000; score_new = 0.832; anchor_step = 28908; offset = 658.6;
                             elo_vs_anchor = 277.9; elo = 936.5; ci95 = @(909.2, 967.1) }
Check "基準比" (Strip (Format-Elo-Anchor $anchor)) `
    "step 62,426 で +936.5 [+909, +967]（基準 step 28,908 に 83%、<いつ>）"

# --- 最強比: 打った相手（そのときの最強）と、測った step の両方を出す ---
$best = [pscustomobject]@{ t = $now; step = 62426; best_step = 28908; n = 1000; score_new = 0.832;
                           elo_vs_best = 277.9; ci95 = @(250.6, 308.5); improved = $true; stall = 0 }
Check "最強比（更新）" (Strip (Format-Elo-Best $best)) `
    "最強比 +277.9 [+251, +309]（step 62,426 が step 28,908 に 83%で最強を更新、<いつ>）"

$stalled = [pscustomobject]@{ t = $now; step = 70000; best_step = 62426; n = 1000; score_new = 0.49;
                              elo_vs_best = -7.0; ci95 = @(-34.0, 20.0); improved = $false; stall = 3 }
Check "最強比（足踏み）" (Strip (Format-Elo-Best $stalled)) `
    "最強比 -7.0 [-34, +20]（step 70,000 が step 62,426 に 49%で最強を抜けず、足踏み 3 回、<いつ>）"

# --- 天井: 得点が 15%〜85% の外なら Elo が当てにならない旨を付ける ---
Check "割合（地域によらず空白なし）" (Format-Pct 0.832) "83%"
Check "割合（値なし）" (Format-Pct $null) "-"
Check "天井（勝ちすぎ）" (Format-Saturated 0.9025) "・天井"
Check "天井（負けすぎ）" (Format-Saturated 0.14) "・天井"
Check "天井（範囲内）" (Format-Saturated 0.545) ""
Check "天井（値なし）" (Format-Saturated $null) ""

# --- 固定の参照: 参照ごとに最新の 1 行だけ、名前の順で並べる ---
$refs = @(
    [pscustomobject]@{ t = $now; step = 28908; ref = "win1m.pt"; n = 200; score_new = 0.545; elo = 31.4; ci95 = @(-16.7, 80.6) },
    [pscustomobject]@{ t = $now; step = 62426; ref = "ckpt_000646699.pt"; n = 200; score_new = 0.9025; elo = 386.6; ci95 = @(317.4, 489.4) },
    [pscustomobject]@{ t = $now; step = 62426; ref = "win1m.pt"; n = 200; score_new = 0.8; elo = 240.8; ci95 = @(185.8, 308.9) }
)
Check "固定の参照" (Strip (Format-Elo-References $refs)) (
    "対 ckpt_000646699.pt +386.6 [+317, +489]（step 62,426、90%・天井、<いつ>）`r`n" +
    "対 win1m.pt +240.8 [+186, +309]（step 62,426、80%、<いつ>）")
Check "固定の参照（まだ無い）" (Format-Elo-References @()) ""
Check "固定の参照（項目なし）" (Format-Elo-References $null) ""

# --- 強さの目盛り（Bradley-Terry）: 2026-09-19 の ls の実データ（progress/ls.json の rating.curve.points の末尾） ---
$rating = [pscustomobject]@{ anchor = "step 402"; points = @(
    [pscustomobject]@{ step = 117436; games = 1402864; t = ($now - 300000); elo = 1091.9; ci95 = @(1058.9, 1124.9); n = 3400 },
    [pscustomobject]@{ step = 135740; games = 1600085; t = ($now - 200000); elo = 1106.0; ci95 = @(1069.9, 1142.2); n = 2440 },
    [pscustomobject]@{ step = 172923; games = 2000241; t = ($now - 100000); elo = 1151.8; ci95 = @(1112.2, 1191.5); n = 1400 }
) }
Check "強さの目盛り" (Format-Elo-Rating $rating) `
    "目盛り +1151.8 [+1112, +1192]（step 172,923、2,000,241 局の時点）、前の点（1,600,085 局）から +45.8"
$one = [pscustomobject]@{ points = @([pscustomobject]@{ step = 402; games = 4738; t = $now; elo = 0.0; ci95 = @(0.0, 0.0); n = 3400 }) }
Check "強さの目盛り（1 点だけ）" (Format-Elo-Rating $one) "目盛り 0 [0, 0]（step 402、4,738 局の時点）"
Check "強さの目盛り（まだ無い）" (Format-Elo-Rating ([pscustomobject]@{ points = @() })) $null
Check "強さの目盛り（項目なし）" (Format-Elo-Rating $null) $null
# rating が出せなかったとき（status が error だけを返す）でも落ちない
Check "強さの目盛り（出せなかった）" (Format-Elo-Rating ([pscustomobject]@{ error = "ValueError: x" })) $null

# --- 目盛りの系列: t か局数が欠けた点は描かない ---
$rs = Build-Rating-Series "ls 強さの目盛り" 0 $rating
Check "目盛りの系列の点の数" $rs.pts.Count 3
Check "目盛りの系列の名前" $rs.name "ls 強さの目盛り"
Check "目盛りの系列の末尾" ("{0} {1} {2} {3}" -f $rs.pts[2].y, $rs.pts[2].lo, $rs.pts[2].hi, $rs.pts[2].g) "1151.8 1112.2 1191.5 2000241"
Check "目盛りの系列の札" $rs.pts[2].label "目盛り step 172,923"
$holes = [pscustomobject]@{ points = @(
    [pscustomobject]@{ step = 1; games = 100; t = $null; elo = 5.0; ci95 = $null; n = 200 },
    [pscustomobject]@{ step = 2; games = $null; t = $now; elo = 6.0; ci95 = $null; n = 200 },
    [pscustomobject]@{ step = 3; games = 300; t = $now; elo = $null; ci95 = $null; n = 200 },
    [pscustomobject]@{ step = 4; games = 400; t = $now; elo = 8.0; ci95 = $null; n = 200 }
) }
Check "欠けた点は描かない" (Build-Rating-Series "s" 0 $holes).pts.Count 1
Check "目盛りの系列（項目なし）" (Build-Rating-Series "s" 0 $null).pts.Count 0

# --- グラフの注記: 既定は目盛りだけ、内訳を出したとき、目盛りがまだ無いとき ---
Check "注記（既定）" (Format-Elo-Note $false $true $true) (
    "「強さの目盛り」＝ 全部の対局をまとめて 1 本にした Elo。0 はいちばん古い重み。相手ごとの線は「内訳を出す」で足せる。全期間を表示")
Check "注記（内訳）" (Format-Elo-Note $true $true $true) (
    "太い線が「強さの目盛り」（全部の対局をまとめて 1 本にした Elo）。相手ごとの線は 1 点 200 局で幅が広く、練習相手の入れ替えと相性で上下するので、目盛りのほうで伸びを見る。全期間を表示")
Check "注記（目盛りがまだ無い）" (Format-Elo-Note $false $false $true) (
    "目盛りがまだ出せないので相手ごとの線を出している。「基準比」の 0 は系列の最初の重み、「対 …」の 0 はその参照と互角で、0 の意味が違う。全期間を表示")
Check "注記（横軸が時間）" (Format-Elo-Note $false $true $false) (
    "「強さの目盛り」＝ 全部の対局をまとめて 1 本にした Elo。0 はいちばん古い重み。相手ごとの線は「内訳を出す」で足せる。全期間を表示。横軸が時間だと止めた間も伸びが寝て見えるので、ふだんは「総局数」で見る")

# --- 区間が無い（古い行）でも落ちない ---
Check "区間なし" (Format-Ci $null) ""
Check "区間が片側だけ" (Format-Ci @(1.0, $null)) ""
$noci = [pscustomobject]@{ t = $now; step = 100; n = 100; score_new = 0.6; anchor_step = 50; offset = 0.0; elo_vs_anchor = 70.0; elo = 70.0; ci95 = $null }
Check "基準比（区間なし）" (Strip (Format-Elo-Anchor $noci)) "step 100 で +70.0（基準 step 50 に 60%、<いつ>）"

# --- グラフの横軸（Elo・対外対局は総局数、ほかは時間） ---
$now = [datetime]::new(2026, 9, 19, 12, 0, 0)
function Mk($pts) {
    # $pts は @(@{t=<分前>; y=<値>; g=<総局数 or $null>}, ...)
    $s = New-Series "s" 0
    foreach ($q in $pts) { Add-Pt $s $now.AddMinutes(-[double]$q.t) ([double]$q.y) $null $null "" $q.g }
    return @($s)
}
$min = [datetime]::MinValue
# 総局数の軸: 最初と最後の点の局数がそのまま端になる（右に「今」までの空白を作らない）
$a = Get-XAxis (Mk @(@{t=600; y=1; g=120171}, @{t=60; y=2; g=800281})) $min $true $now
Check "総局数の軸" ("{0} {1} {2} {3}" -f $a.games, $a.min, $a.max, $a.n) "True 120171 800281 2"
# 時間の軸: 右端は「今」なので、10 時間前の最後の点の右に空白ができる
$b = Get-XAxis (Mk @(@{t=600; y=1; g=120171}, @{t=60; y=2; g=800281})) $min $false $now
$bNow = [double]([System.DateTimeOffset]::new($now).ToUnixTimeMilliseconds() / 1000.0)
Check "時間の軸は今まで伸びる" ("{0} {1}" -f $b.games, ($b.max -eq $bNow)) "False True"
Check "時間の軸の幅" ([Math]::Round(($b.max - $b.min) / 60)) 600
# 局数の分からない点しか無ければ時間の軸に落ちる
$c = Get-XAxis (Mk @(@{t=600; y=1; g=$null}, @{t=60; y=2; g=$null})) $min $true $now
Check "局数が無ければ時間の軸" $c.games $false
# 一部だけ局数がある: 総局数の軸にし、無い点は描かない
$d = Get-XAxis (Mk @(@{t=600; y=1; g=$null}, @{t=60; y=2; g=500000})) $min $true $now
# 1 点しか残らないので、幅 0 で割らないように左へ 1 だけ広げる
Check "局数のある点だけ描く" ("{0} {1} {2} {3}" -f $d.games, $d.min, $d.max, $d.n) "True 499999 500000 1"
# 点が無い
$e2 = Get-XAxis (Mk @()) $min $true $now
Check "点が無い" $e2.n 0
# 局数が 1 点だけでも幅 0 で割らない
$f = Get-XAxis (Mk @(@{t=60; y=2; g=800281})) $min $true $now
Check "1 点でも幅がある" ($f.span -ge 1) $true

if ($fails -gt 0) { Write-Host "`n$fails 件 失敗"; exit 1 }
Write-Host "`nすべて通過"
