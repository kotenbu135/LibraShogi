# SPDX-License-Identifier: Apache-2.0
# Libra 管理コンソール（Windows 用 GUI）。
# WSL 内の bin/libra を wsl.exe 経由で呼び、本体 ls と搾取者 lx の進捗・速度を表示し、
# 一時停止 / 再開 / 停止 / 起動 / 同時局数の絞り込みを行う。デスクトップの bat と同じ操作を 1 画面にまとめたもの。
# 起動: libra-console.bat（powershell -ExecutionPolicy Bypass -File libra-console.ps1）
# 自動テスト: -Screenshot C:\path\shot.png で 1 回更新して画面を PNG に保存し終了する（要約を stdout に出す）。
#             -Do "lx:pause" のようにボタンと同じ操作だけを GUI なしで実行して結果を出す。
param(
    [string]$Distro = "Ubuntu-24.04",
    [string]$Libra = "/home/sakis/LibraShogi/bin/libra",
    [string[]]$Runs = @("ls", "lx"),
    [int]$IntervalSec = 15,
    [int]$LogLines = 8,
    [string]$Screenshot = "",
    [string]$Do = "",
    [int]$ChartHours = 24
)
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
[System.Windows.Forms.Application]::EnableVisualStyles()

$script:RunTitles = @{ ls = "ls  本体 L-S"; lx = "lx  搾取者" }
$script:TaskNames = @{ ls = "LibraShogi run"; lx = "LibraShogi run lx" }
$script:Colors = @{ ls = [System.Drawing.Color]::FromArgb(31, 119, 180); lx = [System.Drawing.Color]::FromArgb(255, 127, 14) }
$script:HistDir = Join-Path $env:LOCALAPPDATA "LibraShogi"
$script:HistFile = Join-Path $script:HistDir "console-history.csv"
$script:Hist = @{}      # run -> ArrayList of [pscustomobject]@{t; games; step; gpd}
$script:Pending = @{}   # run -> @{proc; out; err; started}
$script:Last = @{}      # run -> parsed status object
$script:Ui = @{}        # run -> @{vals=@{}; log=TextBox; buttons=@()}
$script:NextFetch = [datetime]::MinValue
$script:ShotDone = $false
$script:Errors = @{}

function Format-Int($v) {
    if ($null -eq $v) { return "-" }
    return ("{0:N0}" -f [double]$v)
}
function Format-Ago([datetime]$t) {
    $d = [datetime]::Now - $t
    if ($d.TotalMinutes -lt 1) { return ("{0:N0} 秒前" -f $d.TotalSeconds) }
    if ($d.TotalHours -lt 1) { return ("{0:N0} 分前" -f $d.TotalMinutes) }
    return ("{0:N1} 時間前" -f $d.TotalHours)
}

# ---- WSL 呼び出し ----
function Get-LibraArgs([string]$run, [string[]]$cmd) {
    return (@("-d", $Distro, "--", $Libra, "--run", $run) + $cmd) -join " "
}
function Invoke-Libra([string]$run, [string[]]$cmd) {
    # 同期呼び出し（pause/resume/stop/throttle は 0.3 秒程度）。stdout+stderr を返す。
    $p = New-Object System.Diagnostics.Process
    $p.StartInfo.FileName = "wsl.exe"
    $p.StartInfo.Arguments = Get-LibraArgs $run $cmd
    $p.StartInfo.UseShellExecute = $false
    $p.StartInfo.RedirectStandardOutput = $true
    $p.StartInfo.RedirectStandardError = $true
    $p.StartInfo.StandardOutputEncoding = [System.Text.Encoding]::UTF8
    $p.StartInfo.StandardErrorEncoding = [System.Text.Encoding]::UTF8
    $p.StartInfo.CreateNoWindow = $true
    [void]$p.Start()
    $out = $p.StandardOutput.ReadToEndAsync()
    $err = $p.StandardError.ReadToEndAsync()
    $p.WaitForExit()
    return (($out.Result + $err.Result).Trim())
}
function Start-Fetch([string]$run) {
    if ($script:Pending.ContainsKey($run)) { return }
    $p = New-Object System.Diagnostics.Process
    $p.StartInfo.FileName = "wsl.exe"
    $p.StartInfo.Arguments = Get-LibraArgs $run @("status", "--json", "--tail", "$LogLines")
    $p.StartInfo.UseShellExecute = $false
    $p.StartInfo.RedirectStandardOutput = $true
    $p.StartInfo.RedirectStandardError = $true
    $p.StartInfo.StandardOutputEncoding = [System.Text.Encoding]::UTF8
    $p.StartInfo.StandardErrorEncoding = [System.Text.Encoding]::UTF8
    $p.StartInfo.CreateNoWindow = $true
    [void]$p.Start()
    $script:Pending[$run] = @{ proc = $p; out = $p.StandardOutput.ReadToEndAsync(); err = $p.StandardError.ReadToEndAsync(); started = [datetime]::Now }
}
function Complete-Fetches {
    foreach ($run in @($script:Pending.Keys)) {
        $f = $script:Pending[$run]
        if (-not $f.proc.HasExited -or -not $f.out.IsCompleted) {
            if (([datetime]::Now - $f.started).TotalSeconds -gt 60) {
                try { $f.proc.Kill() } catch {}
                $script:Errors[$run] = "status がタイムアウト（60 秒）"
                $script:Pending.Remove($run)
            }
            continue
        }
        $script:Pending.Remove($run)
        $text = $f.out.Result.Trim()
        try {
            if (-not $text) { throw "出力なし: " + $f.err.Result.Trim() }
            $obj = $text | ConvertFrom-Json
            $script:Last[$run] = $obj
            $script:Errors.Remove($run)
            Add-Sample $run $obj
            Update-Panel $run $obj
        } catch {
            $script:Errors[$run] = ("status の読み取りに失敗: " + $_.Exception.Message)
            Update-Panel $run $null
        }
    }
}

# ---- 履歴（速度の折れ線用。%LOCALAPPDATA%\LibraShogi\console-history.csv に追記） ----
function Load-History {
    foreach ($r in $Runs) { $script:Hist[$r] = New-Object System.Collections.ArrayList }
    if (-not (Test-Path $script:HistFile)) { return }
    $cut = [datetime]::Now.AddDays(-7)
    foreach ($line in Get-Content $script:HistFile) {
        $c = $line.Split(",")
        if ($c.Length -lt 5 -or -not $script:Hist.ContainsKey($c[0])) { continue }
        $t = [datetime]::ParseExact($c[1], "yyyy-MM-dd HH:mm:ss", $null)
        if ($t -lt $cut) { continue }
        [void]$script:Hist[$c[0]].Add([pscustomobject]@{ t = $t; games = [double]$c[2]; step = [double]$c[3]; gpd = [double]$c[4] })
    }
}
function Add-Sample([string]$run, $obj) {
    if ($null -eq $obj.status) { return }
    $st = $obj.status
    $h = $script:Hist[$run]
    $now = [datetime]::Now
    if ($h.Count -gt 0 -and ($now - $h[$h.Count - 1].t).TotalSeconds -lt 30) { return }
    $s = [pscustomobject]@{ t = $now; games = [double]$st.games_total; step = [double]$st.step; gpd = [double]$st.games_per_day_1h }
    [void]$h.Add($s)
    if (-not (Test-Path $script:HistDir)) { [void](New-Item -ItemType Directory -Path $script:HistDir) }
    Add-Content -Path $script:HistFile -Value ("{0},{1},{2},{3},{4}" -f $run, $now.ToString("yyyy-MM-dd HH:mm:ss"), $s.games, $s.step, $s.gpd)
}
function Get-MeasuredRate([string]$run) {
    # 直近 10 分以上前のサンプルからの実測 局/日（プロセスの再起動で games_total は減らない）
    $h = $script:Hist[$run]
    if ($h.Count -lt 2) { return $null }
    $last = $h[$h.Count - 1]
    $base = $null
    for ($i = $h.Count - 2; $i -ge 0; $i--) {
        $base = $h[$i]
        if (($last.t - $base.t).TotalMinutes -ge 10) { break }
    }
    $dt = ($last.t - $base.t).TotalDays
    if ($dt -le 0 -or $last.games -lt $base.games) { return $null }
    return [pscustomobject]@{ rate = ($last.games - $base.games) / $dt; minutes = ($last.t - $base.t).TotalMinutes }
}

# ---- 画面 ----
$form = New-Object System.Windows.Forms.Form
$form.Text = "Libra 管理コンソール"
$form.Font = New-Object System.Drawing.Font("Yu Gothic UI", 9)
$form.StartPosition = "CenterScreen"
$form.Size = New-Object System.Drawing.Size(1000, 940)
$form.MinimumSize = New-Object System.Drawing.Size(820, 760)

$root = New-Object System.Windows.Forms.TableLayoutPanel
$root.Dock = "Fill"
$root.ColumnCount = 1
$root.RowCount = 4
[void]$root.RowStyles.Add((New-Object System.Windows.Forms.RowStyle("AutoSize")))
[void]$root.RowStyles.Add((New-Object System.Windows.Forms.RowStyle("Percent", 100)))
[void]$root.RowStyles.Add((New-Object System.Windows.Forms.RowStyle("Absolute", 170)))
[void]$root.RowStyles.Add((New-Object System.Windows.Forms.RowStyle("AutoSize")))
$form.Controls.Add($root)

# ツールバー
$bar = New-Object System.Windows.Forms.FlowLayoutPanel
$bar.Dock = "Fill"
$bar.AutoSize = $true
$bar.WrapContents = $false
$bar.Padding = New-Object System.Windows.Forms.Padding(4)
function New-Button([string]$text, [scriptblock]$onClick, [int]$w = 96) {
    $b = New-Object System.Windows.Forms.Button
    $b.Text = $text
    $b.Width = $w
    $b.Height = 28
    $b.Add_Click($onClick)
    return $b
}
$bar.Controls.Add((New-Button "今すぐ更新" { $script:NextFetch = [datetime]::MinValue }))
$lblIv = New-Object System.Windows.Forms.Label
$lblIv.Text = "更新間隔(秒)"
$lblIv.AutoSize = $true
$lblIv.Margin = New-Object System.Windows.Forms.Padding(12, 8, 2, 0)
$bar.Controls.Add($lblIv)
$numIv = New-Object System.Windows.Forms.NumericUpDown
$numIv.Minimum = 5; $numIv.Maximum = 600; $numIv.Value = [Math]::Max(5, $IntervalSec); $numIv.Width = 60
$numIv.Margin = New-Object System.Windows.Forms.Padding(0, 4, 12, 0)
$bar.Controls.Add($numIv)
$bar.Controls.Add((New-Button "全部 一時停止" { Invoke-All @("pause") } 110))
$bar.Controls.Add((New-Button "全部 再開" { Invoke-All @("resume") }))
$bar.Controls.Add((New-Button "全部 停止" { if (Confirm-Action "両方の run に STOP を送ります（チェックポイントを書いて終了。再起動前の手順）。よろしいですか？") { Invoke-All @("stop") } }))
$root.Controls.Add($bar, 0, 0)

# run ごとのパネル
$runsPanel = New-Object System.Windows.Forms.TableLayoutPanel
$runsPanel.Dock = "Fill"
$runsPanel.ColumnCount = $Runs.Length
$runsPanel.RowCount = 1
$pct = [single](100.0 / $Runs.Length)
foreach ($r in $Runs) { [void]$runsPanel.ColumnStyles.Add((New-Object System.Windows.Forms.ColumnStyle("Percent", $pct))) }
$root.Controls.Add($runsPanel, 0, 1)

$script:Keys = @(
    @("process", "状態"), @("updated", "status 更新"), @("step", "step / 世代"), @("games_total", "総局数"),
    @("gpd", "局/日（1 時間平均）"), @("measured", "局/日（実測）"), @("active", "同時局数"), @("elapsed", "稼働 / セッション局数"),
    @("results", "先手 / 引分 / 後手"), @("plies", "平均手数 / sims/手"), @("loss", "loss / policy / value"),
    @("lr", "lr / 学習 1 回"), @("gpu", "GPU メモリ"), @("ckpt", "最終チェックポイント"), @("exploiter", "対本体 勝率"), @("restarts", "再起動")
)
function New-RunPanel([string]$run) {
    $g = New-Object System.Windows.Forms.GroupBox
    $title = if ($script:RunTitles.ContainsKey($run)) { $script:RunTitles[$run] } else { $run }
    $g.Text = $title
    $g.Dock = "Fill"
    $g.Padding = New-Object System.Windows.Forms.Padding(6)
    $inner = New-Object System.Windows.Forms.TableLayoutPanel
    $inner.Dock = "Fill"
    $inner.ColumnCount = 1
    $inner.RowCount = 3
    [void]$inner.RowStyles.Add((New-Object System.Windows.Forms.RowStyle("AutoSize")))
    [void]$inner.RowStyles.Add((New-Object System.Windows.Forms.RowStyle("AutoSize")))
    [void]$inner.RowStyles.Add((New-Object System.Windows.Forms.RowStyle("Percent", 100)))
    $g.Controls.Add($inner)

    $grid = New-Object System.Windows.Forms.TableLayoutPanel
    $grid.Dock = "Fill"
    $grid.AutoSize = $true
    $grid.ColumnCount = 2
    [void]$grid.ColumnStyles.Add((New-Object System.Windows.Forms.ColumnStyle("Absolute", 150)))
    [void]$grid.ColumnStyles.Add((New-Object System.Windows.Forms.ColumnStyle("Percent", 100)))
    $vals = @{}
    $row = 0
    foreach ($k in $script:Keys) {
        $lk = New-Object System.Windows.Forms.Label
        $lk.Text = $k[1]; $lk.AutoSize = $true; $lk.ForeColor = [System.Drawing.Color]::DimGray
        $lk.Margin = New-Object System.Windows.Forms.Padding(2, 3, 2, 3)
        $lv = New-Object System.Windows.Forms.Label
        $lv.Text = "-"; $lv.AutoSize = $true
        $lv.Margin = New-Object System.Windows.Forms.Padding(2, 3, 2, 3)
        $grid.Controls.Add($lk, 0, $row); $grid.Controls.Add($lv, 1, $row)
        $vals[$k[0]] = $lv
        $row++
    }
    $inner.Controls.Add($grid, 0, 0)

    $btns = New-Object System.Windows.Forms.FlowLayoutPanel
    $btns.Dock = "Fill"; $btns.AutoSize = $true; $btns.WrapContents = $true
    $btns.Margin = New-Object System.Windows.Forms.Padding(0, 6, 0, 6)
    $bl = @()
    $bl += New-Button "一時停止" { Invoke-Run $run @("pause") }.GetNewClosure() 80
    $bl += New-Button "再開" { Invoke-Run $run @("resume") }.GetNewClosure() 60
    $bl += New-Button "停止" { if (Confirm-Action "$run に STOP を送ります（チェックポイントを書いて終了）。よろしいですか？") { Invoke-Run $run @("stop") } }.GetNewClosure() 60
    $bl += New-Button "起動" { Start-Run $run }.GetNewClosure() 60
    $lt = New-Object System.Windows.Forms.Label
    $lt.Text = "絞る(局)"; $lt.AutoSize = $true; $lt.Margin = New-Object System.Windows.Forms.Padding(10, 8, 2, 0)
    $nt = New-Object System.Windows.Forms.NumericUpDown
    $nt.Minimum = 0; $nt.Maximum = 4096; $nt.Value = 0; $nt.Width = 64
    $nt.Margin = New-Object System.Windows.Forms.Padding(0, 4, 2, 0)
    $bt = New-Button "適用" { Invoke-Run $run @("throttle", "--games", [string][int]$nt.Value) }.GetNewClosure() 56
    foreach ($b in $bl) { $btns.Controls.Add($b) }
    $btns.Controls.Add($lt); $btns.Controls.Add($nt); $btns.Controls.Add($bt)
    $inner.Controls.Add($btns, 0, 1)

    $log = New-Object System.Windows.Forms.TextBox
    $log.Multiline = $true; $log.ReadOnly = $true; $log.ScrollBars = "Vertical"; $log.WordWrap = $false
    $log.Dock = "Fill"
    $log.Font = New-Object System.Drawing.Font("Consolas", 8.5)
    $log.BackColor = [System.Drawing.Color]::White
    $inner.Controls.Add($log, 0, 2)

    $script:Ui[$run] = @{ vals = $vals; log = $log; buttons = ($bl + @($bt)); throttle = $nt }
    return $g
}
$col = 0
foreach ($r in $Runs) { $runsPanel.Controls.Add((New-RunPanel $r), $col, 0); $col++ }

# 速度の折れ線
$chart = New-Object System.Windows.Forms.Panel
$chart.Dock = "Fill"
$chart.BackColor = [System.Drawing.Color]::White
$chart.BorderStyle = "FixedSingle"
$chart.Add_Paint({
    param($s, $e)
    $g = $e.Graphics
    $g.SmoothingMode = "AntiAlias"
    $w = $s.ClientSize.Width; $h = $s.ClientSize.Height
    $left = 70; $right = 10; $top = 30; $bottom = 22
    $font = New-Object System.Drawing.Font("Yu Gothic UI", 8)
    $gray = [System.Drawing.Brushes]::Gray
    $g.DrawString("局/日（1 時間平均）の推移  直近 $ChartHours 時間", $font, [System.Drawing.Brushes]::Black, 6, 3)
    $now = [datetime]::Now
    $t0 = $now.AddHours(-$ChartHours)
    $maxY = 1.0
    foreach ($r in $Runs) { foreach ($p in $script:Hist[$r]) { if ($p.t -ge $t0 -and $p.gpd -gt $maxY) { $maxY = $p.gpd } } }
    $maxY = $maxY * 1.1
    $pen = New-Object System.Drawing.Pen([System.Drawing.Color]::LightGray)
    for ($i = 0; $i -le 4; $i++) {
        $y = $top + ($h - $top - $bottom) * (1 - $i / 4)
        $g.DrawLine($pen, $left, $y, $w - $right, $y)
        $g.DrawString(("{0:N0}" -f ($maxY * $i / 4)), $font, $gray, 4, $y - 7)
    }
    $g.DrawString($t0.ToString("MM/dd HH:mm"), $font, $gray, $left, $h - $bottom + 4)
    $g.DrawString($now.ToString("MM/dd HH:mm"), $font, $gray, $w - $right - 70, $h - $bottom + 4)
    $lx = $left + 6
    foreach ($r in $Runs) {
        $color = if ($script:Colors.ContainsKey($r)) { $script:Colors[$r] } else { [System.Drawing.Color]::Green }
        $rp = New-Object System.Drawing.Pen($color, 2)
        $pts = New-Object System.Collections.ArrayList
        foreach ($p in $script:Hist[$r]) {
            if ($p.t -lt $t0) { continue }
            $x = $left + ($w - $left - $right) * (($p.t - $t0).TotalSeconds / ($now - $t0).TotalSeconds)
            $y = $top + ($h - $top - $bottom) * (1 - $p.gpd / $maxY)
            [void]$pts.Add((New-Object System.Drawing.PointF([single]$x, [single]$y)))
        }
        if ($pts.Count -ge 2) { $g.DrawLines($rp, [System.Drawing.PointF[]]$pts.ToArray()) }
        elseif ($pts.Count -eq 1) { $g.FillEllipse((New-Object System.Drawing.SolidBrush($color)), $pts[0].X - 3, $pts[0].Y - 3, 6, 6) }
        $g.FillRectangle((New-Object System.Drawing.SolidBrush($color)), $w - 160 + $lx - $left - 6, 6, 10, 10)
        $g.DrawString($r, $font, [System.Drawing.Brushes]::Black, $w - 146 + $lx - $left - 6, 3)
        $lx += 40
    }
})
$root.Controls.Add($chart, 0, 2)

# ステータス行
$status = New-Object System.Windows.Forms.Label
$status.Dock = "Fill"; $status.AutoSize = $true
$status.Padding = New-Object System.Windows.Forms.Padding(6, 4, 6, 4)
$status.Text = "起動中…"
$root.Controls.Add($status, 0, 3)

# ---- 表示の更新 ----
function Update-Panel([string]$run, $obj) {
    $u = $script:Ui[$run]
    $v = $u.vals
    if ($null -eq $obj) {
        $v.process.Text = "取得失敗"; $v.process.ForeColor = [System.Drawing.Color]::Firebrick
        return
    }
    $running = ($obj.process -eq "running")
    $flags = @($obj.flags)
    $ptxt = if ($running) { "稼働中" } else { "停止" }
    if ($flags -contains "PAUSE") { $ptxt += "（一時停止中）" }
    if ($flags -contains "STOP") { $ptxt += "（停止処理中）" }
    if ($null -ne $obj.throttle) { $ptxt += "（絞り $($obj.throttle) 局）" }
    if (-not $obj.exists) { $ptxt = "run なし（$($obj.root)）" }
    $v.process.Text = $ptxt
    $v.process.ForeColor = if (-not $running) { [System.Drawing.Color]::Firebrick } elseif ($flags.Count -gt 0) { [System.Drawing.Color]::DarkOrange } else { [System.Drawing.Color]::ForestGreen }
    $v.process.Font = New-Object System.Drawing.Font($form.Font, [System.Drawing.FontStyle]::Bold)
    foreach ($b in $u.buttons) { $b.Enabled = $true }
    $st = $obj.status
    if ($null -eq $st) {
        foreach ($k in $script:Keys) { if ($k[0] -ne "process") { $v[$k[0]].Text = "-" } }
        $u.log.Text = ""
        return
    }
    $stTime = [datetime]::ParseExact($st.time, "yyyy-MM-dd HH:mm:ss", $null)
    $v.updated.Text = "{0}（{1}）" -f $st.time, (Format-Ago $stTime)
    $v.updated.ForeColor = if ($running -and ([datetime]::Now - $stTime).TotalMinutes -gt 5 -and -not ($flags -contains "PAUSE")) { [System.Drawing.Color]::Firebrick } else { [System.Drawing.Color]::Black }
    $v.step.Text = "{0} / {1}" -f (Format-Int $st.step), $st.generation
    $v.games_total.Text = Format-Int $st.games_total
    $v.gpd.Text = Format-Int $st.games_per_day_1h
    $m = Get-MeasuredRate $run
    $v.measured.Text = if ($null -eq $m) { "（サンプル待ち）" } else { "{0}（直近 {1:N0} 分）" -f (Format-Int $m.rate), $m.minutes }
    $v.active.Text = "{0}（窓 {1}）" -f $st.active_games, (Format-Int $st.window_games)
    $v.elapsed.Text = "{0:N1} h / {1}" -f [double]$st.elapsed_h, (Format-Int $st.games_session)
    $e = $st.engine
    if ($null -ne $e) {
        $tot = [Math]::Max(1, [double]$e.games)
        $v.results.Text = "{0} / {1} / {2}（先手 {3:P1}）" -f (Format-Int $e.sente_wins), (Format-Int $e.draws), (Format-Int $e.gote_wins), ([double]$e.sente_wins / $tot)
        $v.plies.Text = "{0:N1} / {1:N1}" -f ([double]$e.plies_sum / $tot), ([double]$e.sims / [Math]::Max(1, [double]$e.moves))
    }
    $t = $st.train
    if ($null -ne $t) {
        $v.loss.Text = "{0:N3} / {1:N3} / {2:N3}" -f [double]$t.loss, [double]$t.policy, [double]$t.value
        $v.lr.Text = "{0} / {1:N1} 秒（{2} steps）" -f $t.lr, [double]$t.sec, $t.steps
    }
    if ($null -ne $st.gpu) { $v.gpu.Text = "{0} MB 確保 / {1} MB 予約" -f $st.gpu.mem_alloc_mb, $st.gpu.mem_reserved_mb }
    if ($null -ne $obj.state -and $null -ne $obj.state.last_checkpoint) {
        $ck = [DateTimeOffset]::FromUnixTimeSeconds([long][double]$obj.state.last_checkpoint).LocalDateTime
        $v.ckpt.Text = "{0}（{1}、step {2}）" -f $ck.ToString("MM/dd HH:mm:ss"), (Format-Ago $ck), (Format-Int $obj.state.step)
    }
    $v.exploiter.Text = if ($null -ne $st.exploiter) { "{0:P1}（{1} 局）" -f [double]$st.exploiter.winrate, (Format-Int $st.exploiter.games) } else { "-" }
    $rs = @($st.restarts)
    $v.restarts.Text = if ($rs.Count -gt 0) { "{0} 回（最終 {1}）" -f $rs.Count, $rs[$rs.Count - 1] } else { "0 回" }
    if ($null -ne $obj.log_tail) {
        $u.log.Text = (@($obj.log_tail) -join "`r`n")
        $u.log.SelectionStart = $u.log.Text.Length
        $u.log.ScrollToCaret()
    }
    $chart.Invalidate()
}
function Update-StatusBar {
    $parts = @()
    foreach ($r in $Runs) {
        if ($script:Errors.ContainsKey($r)) { $parts += "$r : " + $script:Errors[$r] }
    }
    $wait = [Math]::Max(0, ($script:NextFetch - [datetime]::Now).TotalSeconds)
    $pend = if ($script:Pending.Count -gt 0) { "  取得中…" } else { "" }
    $status.Text = ("次の更新まで {0:N0} 秒{1}   {2}" -f $wait, $pend, ($parts -join "   "))
    $status.ForeColor = if ($parts.Count -gt 0) { [System.Drawing.Color]::Firebrick } else { [System.Drawing.Color]::DimGray }
}

# ---- 操作 ----
function Confirm-Action([string]$msg) {
    $r = [System.Windows.Forms.MessageBox]::Show($form, $msg, "確認", "YesNo", "Question")
    return ($r -eq "Yes")
}
function Invoke-Run([string]$run, [string[]]$cmd) {
    try {
        $out = Invoke-Libra $run $cmd
        $status.Text = "$run : " + ($cmd -join " ") + " → " + $out
        $status.ForeColor = [System.Drawing.Color]::DimGray
    } catch {
        $status.Text = "$run : " + ($cmd -join " ") + " に失敗: " + $_.Exception.Message
        $status.ForeColor = [System.Drawing.Color]::Firebrick
    }
    $script:NextFetch = [datetime]::Now.AddSeconds(1)
}
function Invoke-All([string[]]$cmd) {
    foreach ($r in $Runs) { Invoke-Run $r $cmd }
}
function Start-Run([string]$run) {
    # タスク スケジューラの「LibraShogi run [lx]」を起動する（失敗時の自動再起動を含めて bat と同じ経路）。
    # タスクが無ければ wsl.exe を非表示で直接起動する。
    $obj = $script:Last[$run]
    if ($null -ne $obj -and $obj.process -eq "running") {
        $status.Text = "$run は既に稼働中です"
        return
    }
    $task = if ($script:TaskNames.ContainsKey($run)) { $script:TaskNames[$run] } else { "LibraShogi run $run" }
    $out = & schtasks.exe /Run /TN $task 2>&1
    if ($LASTEXITCODE -eq 0) {
        $status.Text = "$run : タスク「$task」を起動しました"
    } else {
        Start-Process -FilePath "wsl.exe" -ArgumentList (Get-LibraArgs $run @("run")) -WindowStyle Hidden
        $status.Text = "$run : タスク「$task」が無いので wsl.exe を直接起動しました（$out）"
    }
    $status.ForeColor = [System.Drawing.Color]::DimGray
    $script:NextFetch = [datetime]::Now.AddSeconds(5)
}

# ---- タイマー ----
$timer = New-Object System.Windows.Forms.Timer
$timer.Interval = 500
$timer.Add_Tick({
    try {
        Complete-Fetches
        if ($script:Pending.Count -eq 0 -and [datetime]::Now -ge $script:NextFetch) {
            $script:NextFetch = [datetime]::Now.AddSeconds([int]$numIv.Value)
            foreach ($r in $Runs) { Start-Fetch $r }
        }
        Update-StatusBar
        if ($Screenshot -and -not $script:ShotDone -and $script:Pending.Count -eq 0 -and $script:Last.Count -eq $Runs.Count) {
            $script:ShotDone = $true
            $bmp = New-Object System.Drawing.Bitmap($form.Width, $form.Height)
            $form.DrawToBitmap($bmp, (New-Object System.Drawing.Rectangle(0, 0, $form.Width, $form.Height)))
            $bmp.Save($Screenshot, [System.Drawing.Imaging.ImageFormat]::Png)
            $bmp.Dispose()
            [Console]::WriteLine("screenshot: " + $Screenshot)
            foreach ($r in $Runs) { [Console]::WriteLine(("{0}: {1} flags={2} step={3} games={4} gpd={5}" -f $r, $script:Last[$r].process, (@($script:Last[$r].flags) -join "+"), $script:Last[$r].status.step, $script:Last[$r].status.games_total, $script:Last[$r].status.games_per_day_1h)) }
            $form.Close()
        }
    } catch {
        $status.Text = "内部エラー: " + $_.Exception.Message
        $status.ForeColor = [System.Drawing.Color]::Firebrick
    }
})
$form.Add_Shown({ $timer.Start() })
$form.Add_FormClosing({ $timer.Stop(); foreach ($f in $script:Pending.Values) { try { $f.proc.Kill() } catch {} } })

if ($Do) {
    # GUI なしでボタンと同じ呼び出しを実行する（例: -Do "lx:pause"、-Do "ls:throttle --games 256"）
    $run, $rest = $Do.Split(":", 2)
    [Console]::WriteLine((Invoke-Libra $run ($rest.Trim().Split(" "))))
    exit 0
}
Load-History
[void]$form.ShowDialog()
