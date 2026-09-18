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
$want = @("Format-Int", "Format-Ago", "From-Unix", "Ci-Val", "Format-Ci", "Format-Pct", "Format-Saturated", "Format-Elo-Anchor", "Format-Elo-Best", "Format-Elo-References")
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

# --- 区間が無い（古い行）でも落ちない ---
Check "区間なし" (Format-Ci $null) ""
Check "区間が片側だけ" (Format-Ci @(1.0, $null)) ""
$noci = [pscustomobject]@{ t = $now; step = 100; n = 100; score_new = 0.6; anchor_step = 50; offset = 0.0; elo_vs_anchor = 70.0; elo = 70.0; ci95 = $null }
Check "基準比（区間なし）" (Strip (Format-Elo-Anchor $noci)) "step 100 で +70.0（基準 step 50 に 60%、<いつ>）"

if ($fails -gt 0) { Write-Host "`n$fails 件 失敗"; exit 1 }
Write-Host "`nすべて通過"
