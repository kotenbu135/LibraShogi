# SPDX-License-Identifier: Apache-2.0
# Libra 管理コンソール（Windows 用 GUI）。
# WSL 内の bin/libra を wsl.exe 経由で呼び、本体 ls と搾取者 lx の進捗・速度・強さの推移を表示し、
# 一時停止 / 再開 / 停止 / 起動 / 自己評価・対外対局の前倒しを行う。
# 起動: libra-console.bat（powershell -ExecutionPolicy Bypass -File libra-console.ps1）
# 自動テスト: -Screenshot C:\path\shot.png で 1 回更新して画面を PNG に保存し終了する（要約を stdout に出す）。
#             -Tab <タブ名> で保存時に表示するグラフを選ぶ。
#             -Do "lx:pause" のようにボタンと同じ操作だけを GUI なしで実行して結果を出す。
#             -UpdateDesktopModel で desktop（天秤将棋GUI）に登録した libra.exe のモデルを最新の latest.onnx に置き換えて終了する。
param(
    [string]$Distro = "Ubuntu-24.04",
    [string]$Libra = "/home/sakis/LibraShogi/bin/libra",
    [string[]]$Runs = @("ls", "lx"),
    [int]$IntervalSec = 15,
    [int]$HistorySec = 300,
    [int]$LogLines = 8,
    [string]$Screenshot = "",
    [string]$Tab = "",
    [string]$Do = "",
    [string]$RunRoot = "/home/sakis/libra-run",
    [string]$ModelRun = "ls",
    [string]$DesktopExe = "C:\Users\sakis\AppData\Local\天秤将棋GUI\tenbin-shogi-gui.exe",
    [string]$DesktopEngineDir = (Join-Path $env:APPDATA "com.fusekishogi.tenbin\engines\libra\engine"),
    [switch]$UpdateDesktopModel
)
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
[System.Windows.Forms.Application]::EnableVisualStyles()
# 非表示の PowerShell で起動されるので、途中の例外はメッセージボックスで見せる
trap {
    [System.Windows.Forms.MessageBox]::Show(($_.ToString() + "`r`n" + $_.ScriptStackTrace), "Libra 管理コンソール エラー", "OK", "Error") | Out-Null
    exit 1
}

$script:RunTitles = @{ ls = "ls  本体 L-S"; lx = "lx  搾取者" }
$script:TaskNames = @{ ls = "LibraShogi run"; lx = "LibraShogi run lx" }
$script:Colors = @{ ls = [System.Drawing.Color]::FromArgb(31, 119, 180); lx = [System.Drawing.Color]::FromArgb(255, 127, 14) }
$script:Palette = @([System.Drawing.Color]::FromArgb(31, 119, 180), [System.Drawing.Color]::FromArgb(255, 127, 14), [System.Drawing.Color]::FromArgb(44, 160, 44), [System.Drawing.Color]::FromArgb(148, 103, 189), [System.Drawing.Color]::FromArgb(214, 39, 40))
$script:HistDir = Join-Path $env:LOCALAPPDATA "LibraShogi"
$script:HistFile = Join-Path $script:HistDir "console-history.csv"
$script:Hist = @{}      # run -> ArrayList（コンソール自身の観測: t, games, step, gpd。実測 局/日 に使う）
$script:Data = @{}      # run -> status --history の結果（metrics, evals, matches, archives, auto, auto_cfg）
$script:Pending = @{}   # run -> @{proc; out; err; started}
$script:Last = @{}      # run -> 直近の status オブジェクト
$script:Ui = @{}        # run -> @{vals; log; buttons; autoButtons; autoTips}
$script:NextFetch = [datetime]::MinValue
$script:NextHistory = [datetime]::MinValue
$script:ShotDone = $false
$script:Errors = @{}
$script:HistNote = ""
$script:Tip = New-Object System.Windows.Forms.ToolTip
$script:Tip.AutoPopDelay = 12000

function Format-Int($v) {
    if ($null -eq $v) { return "-" }
    return ("{0:N0}" -f [double]$v)
}
function Format-Ago([datetime]$t) {
    $d = [datetime]::Now - $t
    if ($d.Ticks -lt 0) { $d = [timespan]::Zero }
    if ($d.TotalMinutes -lt 1) { return ("{0:N0} 秒前" -f $d.TotalSeconds) }
    if ($d.TotalHours -lt 1) { return ("{0:N0} 分前" -f $d.TotalMinutes) }
    if ($d.TotalDays -lt 2) { return ("{0:N1} 時間前" -f $d.TotalHours) }
    return ("{0:N1} 日前" -f $d.TotalDays)
}
function From-Unix($sec) {
    return [DateTimeOffset]::FromUnixTimeSeconds([long][double]$sec).LocalDateTime
}
function Ci-Val($ci, [int]$i) {
    # ci95 は [下限, 上限]。行が古い / 要素が null のことがあるので、取れないときは $null を返す
    if ($null -eq $ci) { return $null }
    $a = @($ci)
    if ($a.Count -le $i -or $null -eq $a[$i]) { return $null }
    return [double]$a[$i]
}

# ---- WSL 呼び出し ----
function Get-LibraArgs([string]$run, [string[]]$cmd) {
    return (@("-d", $Distro, "--", $Libra, "--run", $run) + $cmd) -join " "
}
function New-LibraProcess([string]$run, [string[]]$cmd) {
    $p = New-Object System.Diagnostics.Process
    $p.StartInfo.FileName = "wsl.exe"
    $p.StartInfo.Arguments = Get-LibraArgs $run $cmd
    $p.StartInfo.UseShellExecute = $false
    $p.StartInfo.RedirectStandardOutput = $true
    $p.StartInfo.RedirectStandardError = $true
    $p.StartInfo.StandardOutputEncoding = [System.Text.Encoding]::UTF8
    $p.StartInfo.StandardErrorEncoding = [System.Text.Encoding]::UTF8
    $p.StartInfo.CreateNoWindow = $true
    return $p
}
function Invoke-Libra([string]$run, [string[]]$cmd, [int]$TimeoutMs = 20000) {
    # 同期呼び出し（pause/resume/stop/eval-now/match-now は 0.3 秒程度）。stdout+stderr を返す。
    # UI スレッドから呼ぶので必ず上限を付ける。WSL が起動していないと wsl.exe は長時間返らない。
    $p = New-LibraProcess $run $cmd
    [void]$p.Start()
    $out = $p.StandardOutput.ReadToEndAsync()
    $err = $p.StandardError.ReadToEndAsync()
    $sec = [int]($TimeoutMs / 1000)
    if (-not $p.WaitForExit($TimeoutMs)) {
        try { $p.Kill() } catch {}
        throw "wsl.exe が $sec 秒で返りません（WSL が動いているか確認してください）: libra --run $run $($cmd -join ' ')"
    }
    if (-not [System.Threading.Tasks.Task]::WaitAll(@($out, $err), 5000)) {
        try { $p.Kill() } catch {}
        throw "wsl.exe の出力を読み終えられません: libra --run $run $($cmd -join ' ')"
    }
    return (($out.Result + $err.Result).Trim())
}
function Start-Fetch([string]$run, [bool]$withHistory) {
    if ($script:Pending.ContainsKey($run)) { return }
    $cmd = @("status", "--json", "--tail", "$LogLines")
    if ($withHistory) { $cmd += @("--history", "600") }
    $p = New-LibraProcess $run $cmd
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
            if ($null -ne $obj.PSObject.Properties["metrics"]) { $script:Data[$run] = $obj }
            $script:Errors.Remove($run)
            Add-Sample $run $obj
            Update-Panel $run $obj
        } catch {
            $script:Errors[$run] = ("status の読み取りに失敗: " + $_.Exception.Message)
            Update-Panel $run $null
        }
    }
}

# ---- コンソール自身の観測履歴（実測 局/日 用。%LOCALAPPDATA%\LibraShogi\console-history.csv） ----
function Load-History {
    # 壊れた 1 行でコンソールが起動しなくなるのを避けるため、行ごとに握りつぶす。
    # （追記が改行なしで途切れると次の追記と連結し、5 列あるのに数値が壊れた行ができる）
    foreach ($r in $Runs) { $script:Hist[$r] = New-Object System.Collections.ArrayList }
    if (-not (Test-Path $script:HistFile)) { return }
    $cut = [datetime]::Now.AddDays(-7)
    $total = 0; $bad = 0
    foreach ($line in Get-Content $script:HistFile) {
        $total++
        try {
            $c = $line.Split(",")
            if ($c.Length -lt 5 -or -not $script:Hist.ContainsKey($c[0])) { continue }
            $t = [datetime]::ParseExact($c[1], "yyyy-MM-dd HH:mm:ss", $null)
            if ($t -lt $cut) { continue }
            [void]$script:Hist[$c[0]].Add([pscustomobject]@{ t = $t; games = [double]$c[2]; step = [double]$c[3]; gpd = [double]$c[4] })
        } catch { $bad++ }
    }
    $kept = 0
    foreach ($r in $Runs) { $kept += $script:Hist[$r].Count }
    # 放っておくと 30 秒ごとに増え続ける（2 run で約 5,760 行/日）。壊れた行があるか大きくなったら書き戻す
    if ($bad -gt 0 -or $total -gt $kept + 20000) { Save-History $bad $total }
}
function Save-History([int]$bad, [int]$total) {
    # 読み込めた 7 日ぶんだけを書き戻す（run ごとにまとまるが、読み直しで run ごとに分けるので問題ない）
    try {
        $lines = New-Object System.Collections.ArrayList
        foreach ($r in $Runs) {
            foreach ($p in $script:Hist[$r]) {
                [void]$lines.Add(("{0},{1},{2},{3},{4}" -f $r, $p.t.ToString("yyyy-MM-dd HH:mm:ss"), $p.games, $p.step, $p.gpd))
            }
        }
        $tmp = $script:HistFile + ".tmp"
        Set-Content -Path $tmp -Value $lines -Encoding ASCII   # Add-Sample の Add-Content と同じ符号化
        Move-Item -Path $tmp -Destination $script:HistFile -Force
        $script:HistNote = "履歴 CSV を整理しました（{0} 行 → {1} 行{2}）" -f $total, $lines.Count, $(if ($bad -gt 0) { "、壊れた $bad 行を除去" } else { "" })
    } catch {
        $script:HistNote = "履歴 CSV の書き戻しに失敗: " + $_.Exception.Message
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
    $h = $script:Hist[$run]
    if ($h.Count -lt 2) { return $null }
    $last = $h[$h.Count - 1]
    $base = $null
    for ($i = $h.Count - 2; $i -ge 0; $i--) {
        if (($last.t - $h[$i].t).TotalMinutes -ge 10) { $base = $h[$i]; break }
    }
    # 10 分に満たない窓で割ると桁違いの値が出る（measurements.md に転記する数字なので出さない）
    if ($null -eq $base) { return $null }
    $dt = ($last.t - $base.t).TotalDays
    if ($dt -le 0 -or $last.games -lt $base.games) { return $null }
    return [pscustomobject]@{ rate = ($last.games - $base.games) / $dt; minutes = ($last.t - $base.t).TotalMinutes }
}

# ---- 画面 ----
$form = New-Object System.Windows.Forms.Form
$form.Text = "Libra 管理コンソール"
$form.Font = New-Object System.Drawing.Font("Yu Gothic UI", 9)
$form.StartPosition = "CenterScreen"
$form.Size = New-Object System.Drawing.Size(1040, 1000)
$form.MinimumSize = New-Object System.Drawing.Size(860, 800)

$root = New-Object System.Windows.Forms.TableLayoutPanel
$root.Dock = "Fill"
$root.ColumnCount = 1
$root.RowCount = 4
[void]$root.RowStyles.Add((New-Object System.Windows.Forms.RowStyle("AutoSize")))
[void]$root.RowStyles.Add((New-Object System.Windows.Forms.RowStyle("Percent", 100)))
[void]$root.RowStyles.Add((New-Object System.Windows.Forms.RowStyle("Absolute", 240)))
[void]$root.RowStyles.Add((New-Object System.Windows.Forms.RowStyle("AutoSize")))
$form.Controls.Add($root)

function New-Button([string]$text, [scriptblock]$onClick, [int]$w = 96) {
    $b = New-Object System.Windows.Forms.Button
    $b.Text = $text
    $b.Width = $w
    $b.Height = 28
    $b.Add_Click($onClick)
    return $b
}
function New-Label([string]$text, [int]$left = 12) {
    $l = New-Object System.Windows.Forms.Label
    $l.Text = $text; $l.AutoSize = $true
    $l.Margin = New-Object System.Windows.Forms.Padding($left, 8, 2, 0)
    return $l
}

# ツールバー
$bar = New-Object System.Windows.Forms.FlowLayoutPanel
$bar.Dock = "Fill"; $bar.AutoSize = $true; $bar.WrapContents = $true
$bar.Padding = New-Object System.Windows.Forms.Padding(4)
$bar.Controls.Add((New-Button "今すぐ更新" { $script:NextFetch = [datetime]::MinValue; $script:NextHistory = [datetime]::MinValue }))
$bar.Controls.Add((New-Label "更新間隔(秒)"))
$numIv = New-Object System.Windows.Forms.NumericUpDown
$numIv.Minimum = 5; $numIv.Maximum = 600; $numIv.Value = [Math]::Max(5, $IntervalSec); $numIv.Width = 60
$numIv.Margin = New-Object System.Windows.Forms.Padding(0, 4, 12, 0)
$bar.Controls.Add($numIv)
$bar.Controls.Add((New-Button "全部 一時停止" { Invoke-All @("pause") } 110))
$bar.Controls.Add((New-Button "全部 再開" { Invoke-All @("resume") }))
$bar.Controls.Add((New-Button "全部 停止" { if (Confirm-Action "両方の run に STOP を送ります（チェックポイントを書いて終了。再起動前の手順）。よろしいですか？") { Invoke-All @("stop") } }))
$bar.Controls.Add((New-Label "グラフの期間" 24))
$cmbRange = New-Object System.Windows.Forms.ComboBox
$cmbRange.DropDownStyle = "DropDownList"; $cmbRange.Width = 90
[void]$cmbRange.Items.AddRange(@("6 時間", "24 時間", "7 日", "全部"))
$cmbRange.SelectedIndex = 1
$cmbRange.Margin = New-Object System.Windows.Forms.Padding(0, 4, 8, 0)
$cmbRange.Add_SelectedIndexChanged({ $tabs.Invalidate($true) })
$bar.Controls.Add($cmbRange)
$bar.Controls.Add((New-Label "学習・統計の run" 8))
$cmbRun = New-Object System.Windows.Forms.ComboBox
$cmbRun.DropDownStyle = "DropDownList"; $cmbRun.Width = 60
[void]$cmbRun.Items.AddRange($Runs)
$cmbRun.SelectedIndex = 0
$cmbRun.Margin = New-Object System.Windows.Forms.Padding(0, 4, 8, 0)
$cmbRun.Add_SelectedIndexChanged({ $tabs.Invalidate($true) })
$bar.Controls.Add($cmbRun)
$bar.Controls.Add((New-Button "desktop で対局" { Play-Desktop } 110))
$lblModel = New-Label "" 4
$lblModel.ForeColor = [System.Drawing.Color]::DimGray
$bar.Controls.Add($lblModel)
$root.Controls.Add($bar, 0, 0)

# ---- desktop（天秤将棋GUI）で最新ネットと対局する ----
function Get-DesktopModelInfo {
    $j = Join-Path $DesktopEngineDir "libra.onnx.json"
    if (Test-Path $j) { try { return (Get-Content $j -Raw -Encoding UTF8 | ConvertFrom-Json) } catch {} }
    return $null
}
function Model-StepText($info) {
    if ($null -ne $info.step) { return "step " + (Format-Int $info.step) }
    if ($info.stale) { return "step 不明（latest.onnx がチェックポイントより古い＝書き出しに失敗している）" }
    return "step 不明"
}
function Update-DesktopModelLabel {
    $i = Get-DesktopModelInfo
    if ($null -eq $i) { $lblModel.Text = "desktop のモデル: 未更新（登録時のまま）"; return }
    $ot = if ($i.PSObject.Properties["onnx_time"] -and $i.onnx_time) { "、onnx " + $i.onnx_time } else { "" }
    $lblModel.Text = "desktop のモデル: {0}（更新 {1}{2}）" -f (Model-StepText $i), $i.time, $ot
}
function Update-DesktopModel {
    # WSL 側の latest.onnx を desktop に登録した libra.exe の横（libra.onnx）へ写す。手順は一時名 → 置き換え。
    if (-not (Test-Path $DesktopEngineDir)) { throw "desktop のエンジン フォルダがありません: $DesktopEngineDir（desktop で libra.exe を登録してください。runbook §8）" }
    $src = "\\wsl.localhost\$Distro\" + ($RunRoot.TrimStart("/") -replace "/", "\") + "\$ModelRun\checkpoints\latest.onnx"
    if (-not (Test-Path $src)) { throw "latest.onnx が見つかりません: $src（WSL が起動していて run が動いているか確認）" }
    $srcItem = Get-Item $src
    $dst = Join-Path $DesktopEngineDir "libra.onnx"
    $tmp = $dst + ".tmp"
    Copy-Item -Path $src -Destination $tmp -Force
    Move-Item -Path $tmp -Destination $dst -Force
    # state.step はチェックポイント時に書かれるので latest.onnx の step と一致する。
    # ただし export は失敗してもランを止めない設計（runner.export_onnx）なので、
    # onnx がチェックポイントより古ければ書き出しに失敗していて step は当てにならない。
    $step = $null; $stale = $false
    try {
        $obj = Invoke-Libra $ModelRun @("status", "--json") | ConvertFrom-Json
        if ($null -ne $obj.state) {
            $step = $obj.state.step
            if ($null -ne $obj.state.last_checkpoint -and $srcItem.LastWriteTime -lt (From-Unix $obj.state.last_checkpoint).AddMinutes(-1)) {
                $stale = $true
                $step = $null
            }
        }
    } catch {}
    $info = @{ step = $step; stale = $stale; onnx_time = $srcItem.LastWriteTime.ToString("yyyy-MM-dd HH:mm");
               time = [datetime]::Now.ToString("yyyy-MM-dd HH:mm"); source = $src; size = (Get-Item $dst).Length }
    ($info | ConvertTo-Json -Compress) | Set-Content -Path (Join-Path $DesktopEngineDir "libra.onnx.json") -Encoding UTF8
    return $info
}
function Play-Desktop {
    try {
        $info = Update-DesktopModel
        Update-DesktopModelLabel
        $running = Get-Process -Name "tenbin-shogi-gui" -ErrorAction SilentlyContinue
        if ($running) {
            $status.Text = "desktop のモデルを {0} に更新しました。desktop は起動中です。エンジンを立て直す（desktop を開き直すか、対局設定でエンジンを選び直す）と新しいネットで指します。" -f (Model-StepText $info)
        } else {
            if (-not (Test-Path $DesktopExe)) { throw "desktop が見つかりません: $DesktopExe" }
            Start-Process -FilePath $DesktopExe -WorkingDirectory (Split-Path $DesktopExe)
            $status.Text = "desktop のモデルを {0} に更新して起動しました。「対局」でエンジンに「LibraShogi」を選んでください。" -f (Model-StepText $info)
        }
        $status.ForeColor = [System.Drawing.Color]::DimGray
    } catch {
        $status.Text = "desktop: " + $_.Exception.Message
        $status.ForeColor = [System.Drawing.Color]::Firebrick
    }
}

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
    @("lr", "lr / 学習 1 回"), @("gpu", "GPU メモリ"), @("ckpt", "最終チェックポイント"), @("exploiter", "対本体 勝率"), @("restarts", "再起動"),
    @("elo", "強さ（基準比 Elo）"), @("match", "対外対局 勝率"), @("auto", "自動計測")
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
        $lk.Margin = New-Object System.Windows.Forms.Padding(2, 1, 2, 1)
        $lv = New-Object System.Windows.Forms.Label
        $lv.Text = "-"; $lv.AutoSize = $true
        $lv.Margin = New-Object System.Windows.Forms.Padding(2, 1, 2, 1)
        $grid.Controls.Add($lk, 0, $row); $grid.Controls.Add($lv, 1, $row)
        $vals[$k[0]] = $lv
        $row++
    }
    $inner.Controls.Add($grid, 0, 0)

    $btns = New-Object System.Windows.Forms.FlowLayoutPanel
    $btns.Dock = "Fill"; $btns.AutoSize = $true; $btns.WrapContents = $true
    $btns.Margin = New-Object System.Windows.Forms.Padding(0, 4, 0, 4)
    $bl = @()
    $bl += New-Button "一時停止" { Invoke-Run $run @("pause") }.GetNewClosure() 76
    $bl += New-Button "再開" { Invoke-Run $run @("resume") }.GetNewClosure() 56
    $bl += New-Button "停止" { if (Confirm-Action "$run に STOP を送ります（チェックポイントを書いて終了）。よろしいですか？") { Invoke-Run $run @("stop") } }.GetNewClosure() 56
    $bl += New-Button "起動" { Start-Run $run }.GetNewClosure() 56
    $be = New-Button "今すぐ自己評価" { if (Confirm-Action "$run : 次のチェックポイント（10 分以内）で archive を作り、直前の archive と自己評価します（GPU を共有、約 10 分）。よろしいですか？") { Invoke-Run $run @("eval-now") } }.GetNewClosure() 110
    $bm = New-Button "今すぐ対外対局" { if (Confirm-Action "$run : 次のチェックポイント（10 分以内）で外部エンジンとの計測対局を積みます（GPU と CPU を共有、10 局で 15 分程度）。よろしいですか？") { Invoke-Run $run @("match-now") } }.GetNewClosure() 110
    foreach ($b in $bl) { $btns.Controls.Add($b) }
    $btns.Controls.Add($be); $btns.Controls.Add($bm)
    $inner.Controls.Add($btns, 0, 1)

    $log = New-Object System.Windows.Forms.TextBox
    $log.Multiline = $true; $log.ReadOnly = $true; $log.ScrollBars = "Vertical"; $log.WordWrap = $false
    $log.Dock = "Fill"
    $log.Font = New-Object System.Drawing.Font("Consolas", 8.5)
    $log.BackColor = [System.Drawing.Color]::White
    $inner.Controls.Add($log, 0, 2)

    $tipEval = "次のチェックポイントで archive を作り、基準ネットと 100 局対局する"
    $tipMatch = "次のチェックポイントで外部エンジンとの計測対局を積む"
    $script:Tip.SetToolTip($be, $tipEval)
    $script:Tip.SetToolTip($bm, $tipMatch)
    $script:Ui[$run] = @{ vals = $vals; log = $log; buttons = $bl; autoButtons = @($be, $bm); autoTips = @($tipEval, $tipMatch) }
    return $g
}
$col = 0
foreach ($r in $Runs) { $runsPanel.Controls.Add((New-RunPanel $r), $col, 0); $col++ }

# ---- グラフ ----
function Get-Range {
    switch ($cmbRange.SelectedIndex) {
        0 { return [datetime]::Now.AddHours(-6) }
        1 { return [datetime]::Now.AddHours(-24) }
        2 { return [datetime]::Now.AddDays(-7) }
        default { return [datetime]::MinValue }
    }
}
function New-Series([string]$name, $color, [bool]$marker = $false, [bool]$dash = $false) {
    return @{ name = $name; color = $color; pts = (New-Object System.Collections.ArrayList); marker = $marker; dash = $dash; gap = $false }
}
function Add-Pt($series, [datetime]$t, [double]$y, $lo = $null, $hi = $null, [string]$label = "") {
    [void]$series.pts.Add(@{ t = $t; y = $y; lo = $lo; hi = $hi; label = $label })
}
function Run-Color([string]$run, [int]$i = 0) {
    if ($script:Colors.ContainsKey($run)) { return $script:Colors[$run] }
    return $script:Palette[$i % $script:Palette.Length]
}
function Draw-Chart($g, [int]$w, [int]$h, [string]$title, $series, [string]$yfmt, [bool]$zeroBase, [string]$note) {
    $g.SmoothingMode = "AntiAlias"
    $g.Clear([System.Drawing.Color]::White)
    $font = New-Object System.Drawing.Font("Yu Gothic UI", 8)
    $gray = [System.Drawing.Brushes]::Gray
    $black = [System.Drawing.Brushes]::Black
    $left = 72; $right = 14; $top = 30; $bottom = 24
    $g.DrawString($title, $font, $black, 6, 3)
    $t0 = Get-Range
    $now = [datetime]::Now
    $tmin = $null; $tmax = $now
    $ymin = [double]::MaxValue; $ymax = [double]::MinValue
    $n = 0
    foreach ($s in $series) {
        foreach ($p in $s.pts) {
            if ($p.t -lt $t0) { continue }
            $n++
            if ($null -eq $tmin -or $p.t -lt $tmin) { $tmin = $p.t }
            $lo = if ($null -ne $p.lo) { [double]$p.lo } else { $p.y }
            $hi = if ($null -ne $p.hi) { [double]$p.hi } else { $p.y }
            if ($lo -lt $ymin) { $ymin = $lo }
            if ($hi -gt $ymax) { $ymax = $hi }
        }
    }
    if ($n -eq 0) {
        $g.DrawString("（データなし）", $font, $gray, $left, $top + 10)
        if ($note) { $g.DrawString($note, $font, $gray, $left, $top + 28) }
        return
    }
    if ($t0 -gt [datetime]::MinValue) { $tmin = $t0 }
    if (($tmax - $tmin).TotalSeconds -lt 600) { $tmin = $tmax.AddMinutes(-10) }
    if ($zeroBase) { $ymin = [Math]::Min(0.0, $ymin); if ($ymax -le $ymin) { $ymax = $ymin + 1 } }
    $pad = ($ymax - $ymin) * 0.1
    if ($pad -le 0) { $pad = [Math]::Max(1.0, [Math]::Abs($ymax) * 0.1) }
    $ymax += $pad
    if (-not $zeroBase -or $ymin -lt 0) { $ymin -= $pad }
    if ($yfmt -eq "{0:P0}") { $ymax = [Math]::Min(1.0, $ymax); $ymin = [Math]::Max(0.0, $ymin) }
    $pen = New-Object System.Drawing.Pen([System.Drawing.Color]::LightGray)
    $ph = $h - $top - $bottom; $pw = $w - $left - $right
    for ($i = 0; $i -le 4; $i++) {
        $y = $top + $ph * (1 - $i / 4.0)
        $g.DrawLine($pen, $left, $y, $w - $right, $y)
        $g.DrawString(($yfmt -f ($ymin + ($ymax - $ymin) * $i / 4.0)), $font, $gray, 4, $y - 7)
    }
    if ($ymin -lt 0 -and $ymax -gt 0) {
        $y0 = $top + $ph * (1 - (0 - $ymin) / ($ymax - $ymin))
        $g.DrawLine((New-Object System.Drawing.Pen([System.Drawing.Color]::DarkGray)), $left, $y0, $w - $right, $y0)
    }
    $span = ($tmax - $tmin).TotalSeconds
    $fmt = if ($span -gt 3 * 86400) { "MM/dd" } else { "MM/dd HH:mm" }
    $g.DrawString($tmin.ToString($fmt), $font, $gray, $left, $h - $bottom + 4)
    $mid = $tmin.AddSeconds($span / 2)
    $g.DrawString($mid.ToString($fmt), $font, $gray, $left + $pw / 2 - 30, $h - $bottom + 4)
    $g.DrawString($tmax.ToString($fmt), $font, $gray, $w - $right - 70, $h - $bottom + 4)
    $shown = @($series | Where-Object { @($_.pts | Where-Object { $_.t -ge $tmin }).Count -gt 0 })
    $legendW = 0
    foreach ($s in $shown) { $legendW += 22 + $g.MeasureString($s.name, $font).Width + 8 }
    $lx = $w - $right - $legendW
    foreach ($s in $shown) {
        $brush = New-Object System.Drawing.SolidBrush($s.color)
        $rp = New-Object System.Drawing.Pen($s.color, 2)
        if ($s.dash) { $rp.DashStyle = [System.Drawing.Drawing2D.DashStyle]::Dash }
        $pts = New-Object System.Collections.ArrayList
        $last = $null
        # 定期観測の系列（gap）は、点の間隔の中央値の 3 倍（最低 15 分）より空いたところで線を切る。
        # 停止・一時停止の区間を直線でつなぐと、その間も同じ値で動いていたように見えるため
        $gapSec = [double]::MaxValue
        if ($s.gap -and $s.pts.Count -ge 3) {
            $d = @(for ($k = 1; $k -lt $s.pts.Count; $k++) { ($s.pts[$k].t - $s.pts[$k - 1].t).TotalSeconds }) | Sort-Object
            $gapSec = [Math]::Max(900.0, 3 * $d[[int][Math]::Floor($d.Count / 2)])
        }
        $prevT = $null
        foreach ($p in $s.pts) {
            if ($p.t -lt $tmin) { continue }
            if ($null -ne $prevT -and ($p.t - $prevT).TotalSeconds -gt $gapSec) {
                if ($pts.Count -ge 2) { $g.DrawLines($rp, [System.Drawing.PointF[]]$pts.ToArray()) }
                elseif ($pts.Count -eq 1) { $g.FillEllipse($brush, $pts[0].X - 2, $pts[0].Y - 2, 4, 4) }
                $pts.Clear()
            }
            $prevT = $p.t
            $x = $left + $pw * (($p.t - $tmin).TotalSeconds / $span)
            $y = $top + $ph * (1 - ($p.y - $ymin) / ($ymax - $ymin))
            [void]$pts.Add((New-Object System.Drawing.PointF([single]$x, [single]$y)))
            if ($null -ne $p.lo -and $null -ne $p.hi) {
                $yl = $top + $ph * (1 - ([double]$p.lo - $ymin) / ($ymax - $ymin))
                $yh = $top + $ph * (1 - ([double]$p.hi - $ymin) / ($ymax - $ymin))
                $g.DrawLine((New-Object System.Drawing.Pen($s.color, 1)), [single]$x, [single]$yl, [single]$x, [single]$yh)
            }
            if ($s.marker) { $g.FillEllipse($brush, [single]($x - 3), [single]($y - 3), 6, 6) }
            $last = @{ x = $x; y = $y; p = $p }
        }
        if ($pts.Count -ge 2) { $g.DrawLines($rp, [System.Drawing.PointF[]]$pts.ToArray()) }
        elseif ($pts.Count -eq 1) { $g.FillEllipse($brush, $pts[0].X - 3, $pts[0].Y - 3, 6, 6) }
        if ($null -ne $last) {
            $txt = $yfmt -f $last.p.y
            if ($last.p.label) { $txt += " " + $last.p.label }
            $sz = $g.MeasureString($txt, $font)
            $tx = [Math]::Min($last.x + 4, $w - $right - $sz.Width)
            $g.DrawString($txt, $font, $brush, [single]$tx, [single]($last.y - 16))
        }
        $g.FillRectangle($brush, [single]$lx, 6, 10, 10)
        $g.DrawString($s.name, $font, $black, [single]($lx + 12), 3)
        $lx += 22 + $g.MeasureString($s.name, $font).Width + 8
    }
    if ($note) { $g.DrawString($note, $font, $gray, [single](6 + $g.MeasureString($title, $font).Width + 12), 3) }
}

function Get-Metrics([string]$run) {
    if ($script:Data.ContainsKey($run) -and $null -ne $script:Data[$run].metrics) { return @($script:Data[$run].metrics) }
    return @()
}
function Build-Series([string]$tab) {
    $series = @()
    $note = ""
    $yfmt = "{0:N0}"; $zero = $true; $title = $tab
    $sel = [string]$cmbRun.SelectedItem
    switch ($tab) {
        "局/日" {
            $title = "局/日（5 分平均）の推移"
            $i = 0
            foreach ($r in $Runs) {
                $s = New-Series $r (Run-Color $r $i)
                # gpd_5m は libra status --history が metrics.jsonl の隣り合う行の局数差から出す（間が 15 分を超えたら null）
                foreach ($m in (Get-Metrics $r)) { if ($null -ne $m.gpd_5m) { Add-Pt $s (From-Unix $m.t) ([double]$m.gpd_5m) } }
                if ($s.pts.Count -eq 0) {
                    # metrics.jsonl が無いときはコンソール自身の 30 秒ごとの観測から、5 分以上 15 分以内の窓で出す
                    $h = $script:Hist[$r]; $b = 0
                    for ($k = 1; $k -lt $h.Count; $k++) {
                        while ($b + 1 -lt $k -and ($h[$k].t - $h[$b + 1].t).TotalMinutes -ge 5) { $b++ }
                        $dt = ($h[$k].t - $h[$b].t).TotalMinutes
                        if ($dt -lt 5 -or $dt -gt 15 -or $h[$k].games -lt $h[$b].games) { continue }
                        Add-Pt $s $h[$k].t (($h[$k].games - $h[$b].games) / ($dt / 1440))
                    }
                }
                $series += $s; $i++
            }
        }
        "Elo" {
            $title = "強さの推移（Elo。0 は乱数初期化のネット。実線は固定の基準との差、点線は前の世代との差を足した鎖）"
            $i = 0
            foreach ($r in $Runs) {
                $s = New-Series ($r + " 基準比") (Run-Color $r $i) $true
                if ($script:Data.ContainsKey($r)) {
                    foreach ($a in @($script:Data[$r].anchor)) {
                        $lo = Ci-Val $a.ci95 0
                        $hi = Ci-Val $a.ci95 1
                        Add-Pt $s (From-Unix $a.t) ([double]$a.elo) $lo $hi ("step " + (Format-Int $a.step))
                    }
                }
                $series += $s
                $c = New-Series ($r + " 鎖") (Run-Color $r $i) $false $true
                if ($script:Data.ContainsKey($r)) {
                    foreach ($e in @($script:Data[$r].evals)) {
                        if ($null -eq $e.cumulative) { continue }
                        Add-Pt $c (From-Unix $e.time) ([double]$e.cumulative)
                    }
                }
                $series += $c
                $i++
            }
            $note = "基準比は 1 日 1 回 100 局。基準に 85% 勝つと基準を置き換えて差を足す"
        }
        "対外対局" {
            $title = "外部エンジン（fuseki_usi_server.py = 方策ネット＋やねうら王/水匠5）との勝率"
            $yfmt = "{0:P0}"
            $i = 0
            foreach ($r in $Runs) {
                $s = New-Series $r (Run-Color $r $i) $true
                if ($script:Data.ContainsKey($r)) {
                    foreach ($m in @($script:Data[$r].matches)) { if ($null -ne $m.winrate) { Add-Pt $s (From-Unix $m.time) ([double]$m.winrate) $null $null ("{0} 局 {1}" -f $m.n, $m.go) } }
                }
                $series += $s; $i++
            }
            $note = "計測のみ（相手専用の対策はしない）。1 日 10 局・movetime 1000"
        }
        "学習" {
            $title = "学習の損失（$sel）"
            $yfmt = "{0:N2}"; $zero = $false
            $names = @("loss", "policy", "value", "v41"); $i = 0
            foreach ($k in $names) {
                $s = New-Series $k $script:Palette[$i]
                foreach ($m in (Get-Metrics $sel)) { if ($null -ne $m.train -and $null -ne $m.train.$k) { Add-Pt $s (From-Unix $m.t) ([double]$m.train.$k) } }
                $series += $s; $i++
            }
            $s = New-Series "policy_acc" $script:Palette[4]
            foreach ($m in (Get-Metrics $sel)) { if ($null -ne $m.train -and $null -ne $m.train.policy_acc) { Add-Pt $s (From-Unix $m.t) ([double]$m.train.policy_acc) } }
            $series += $s
        }
        "終局内訳" {
            $title = "終局の内訳と先手勝率（$sel、5 分ごとの新規対局の割合）"
            $yfmt = "{0:P0}"
            $ru = New-Series "41 手目裁定" $script:Palette[0]; $ma = New-Series "詰み" $script:Palette[1]; $se = New-Series "先手勝ち" $script:Palette[2]; $dr = New-Series "引き分け" $script:Palette[3]
            $prev = $null
            foreach ($m in (Get-Metrics $sel)) {
                if ($null -ne $prev -and $null -ne $m.engine -and $null -ne $prev.engine -and $m.engine.games -gt $prev.engine.games) {
                    $dg = [double]($m.engine.games - $prev.engine.games)
                    $t = From-Unix $m.t
                    Add-Pt $ru $t (($m.engine.ruling41 - $prev.engine.ruling41) / $dg)
                    Add-Pt $ma $t (($m.engine.no_legal_move - $prev.engine.no_legal_move) / $dg)
                    Add-Pt $se $t (($m.engine.sente_wins - $prev.engine.sente_wins) / $dg)
                    Add-Pt $dr $t (($m.engine.draws - $prev.engine.draws) / $dg)
                }
                $prev = $m
            }
            $series = @($ru, $ma, $se, $dr)
        }
        "手数" {
            $title = "平均手数と sims/手（$sel、5 分ごと）"
            $yfmt = "{0:N1}"
            $pl = New-Series "平均手数" $script:Palette[0]; $sm = New-Series "sims/手" $script:Palette[1]
            $prev = $null
            foreach ($m in (Get-Metrics $sel)) {
                if ($null -ne $prev -and $null -ne $m.engine -and $null -ne $prev.engine -and $m.engine.games -gt $prev.engine.games -and $m.engine.moves -gt $prev.engine.moves) {
                    $t = From-Unix $m.t
                    Add-Pt $pl $t (($m.engine.plies_sum - $prev.engine.plies_sum) / ($m.engine.games - $prev.engine.games))
                    Add-Pt $sm $t (($m.engine.sims - $prev.engine.sims) / ($m.engine.moves - $prev.engine.moves))
                }
                $prev = $m
            }
            $series = @($pl, $sm)
        }
    }
    # 5 分ごとの metrics（とコンソールの 30 秒観測）から作る系列は、観測の途切れで線を切る
    if (@("局/日", "学習", "終局内訳", "手数") -contains $tab) { foreach ($s in $series) { $s.gap = $true } }
    return @{ title = $title; series = $series; yfmt = $yfmt; zero = $zero; note = $note }
}

$tabs = New-Object System.Windows.Forms.TabControl
$tabs.Dock = "Fill"
$script:TabNames = @("局/日", "Elo", "対外対局", "学習", "終局内訳", "手数")
foreach ($name in $script:TabNames) {
    $page = New-Object System.Windows.Forms.TabPage
    $page.Text = $name
    $panel = New-Object System.Windows.Forms.Panel
    $panel.Dock = "Fill"
    $panel.BackColor = [System.Drawing.Color]::White
    $panel.Tag = $name
    $panel.Add_Paint({
        param($s, $e)
        $b = Build-Series ([string]$s.Tag)
        Draw-Chart $e.Graphics $s.ClientSize.Width $s.ClientSize.Height $b.title $b.series $b.yfmt $b.zero $b.note
    })
    $panel.Add_Resize({ param($s, $e) $s.Invalidate() })
    $page.Controls.Add($panel)
    [void]$tabs.TabPages.Add($page)
}
$tabs.Add_SelectedIndexChanged({ $tabs.SelectedTab.Controls[0].Invalidate() })
if ($Tab) { $idx = [array]::IndexOf($script:TabNames, $Tab); if ($idx -ge 0) { $tabs.SelectedIndex = $idx } }
$root.Controls.Add($tabs, 0, 2)

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
    if ($flags -contains "EVAL_NOW") { $ptxt += "（自己評価 予約）" }
    if ($flags -contains "MATCH_NOW") { $ptxt += "（対外対局 予約）" }
    if (-not $obj.exists) { $ptxt = "run なし（$($obj.root)）" }
    $v.process.Text = $ptxt
    $v.process.ForeColor = if (-not $running) { [System.Drawing.Color]::Firebrick } elseif ($flags -contains "PAUSE" -or $flags -contains "STOP") { [System.Drawing.Color]::DarkOrange } else { [System.Drawing.Color]::ForestGreen }
    $v.process.Font = New-Object System.Drawing.Font($form.Font, [System.Drawing.FontStyle]::Bold)
    foreach ($b in $u.buttons) { $b.Enabled = $true }
    # 自動計測が無効な run では前倒しのボタンは効かない（フラグを消費するものが無い）ので押せなくする
    $ac = $obj.auto_cfg
    $autoOn = ($null -ne $ac -and $ac.enabled)
    for ($i = 0; $i -lt $u.autoButtons.Count; $i++) {
        $u.autoButtons[$i].Enabled = $autoOn
        $script:Tip.SetToolTip($u.autoButtons[$i], $(if ($autoOn) { $u.autoTips[$i] } else { "$run は自動計測が無効です（$($obj.root)/config.toml の [auto] enabled = true で使えます）" }))
    }
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
        $ck = From-Unix $obj.state.last_checkpoint
        $v.ckpt.Text = "{0}（{1}、step {2}）" -f $ck.ToString("MM/dd HH:mm:ss"), (Format-Ago $ck), (Format-Int $obj.state.step)
    }
    if ($null -ne $st.exploiter) {
        $ex = $st.exploiter
        $v.exploiter.Text = "{0:P1}（{1} 局、相手 step {2}{3}）" -f [double]$ex.winrate, (Format-Int $ex.games), (Format-Int $ex.main_step),
            $(if ($null -ne $ex.refreshed_at) { "、作り直し " + (Format-Ago (From-Unix $ex.refreshed_at)) } else { "" })
    } else { $v.exploiter.Text = "-" }
    $rs = @($st.restarts)
    $v.restarts.Text = if ($rs.Count -gt 0) { "{0} 回（最終 {1}）" -f $rs.Count, $rs[$rs.Count - 1] } else { "0 回" }
    if ($null -ne $obj.log_tail) {
        $u.log.Text = (@($obj.log_tail) -join "`r`n")
        $u.log.SelectionStart = $u.log.Text.Length
        $u.log.ScrollToCaret()
    }
    if ($script:Data.ContainsKey($run)) {
        $d = $script:Data[$run]
        $anc = @($d.anchor)
        $chain = @(@($d.evals) | Where-Object { $null -ne $_.cumulative })
        if ($anc.Count -gt 0) {
            $la = $anc[$anc.Count - 1]
            $lo = Ci-Val $la.ci95 0; $hi = Ci-Val $la.ci95 1
            $ci = if ($null -ne $lo -and $null -ne $hi) { " [{0:+0;-0;0}, {1:+0;-0;0}]" -f $lo, $hi } else { "" }
            $v.elo.Text = "{0:+0.0;-0.0;0}{1}（基準 step {2} に {3:P0}、{4}）" -f [double]$la.elo, $ci, (Format-Int $la.anchor_step), [double]$la.score_new, (Format-Ago (From-Unix $la.t))
        } elseif ($chain.Count -gt 0) {
            $le = $chain[$chain.Count - 1]
            # eval の ci95 は a−b の区間。鎖は b−a を足すので、符号を反転して上下を入れ替える
            $aLo = Ci-Val $le.ci95 0; $aHi = Ci-Val $le.ci95 1
            $ci = if ($null -ne $aLo -and $null -ne $aHi) { " [{0:+0;-0;0}, {1:+0;-0;0}]" -f (-$aHi), (-$aLo) } else { "" }
            $v.elo.Text = "鎖 {0:+0.0;-0.0;0}（前回 {1:+0.0;-0.0;0}{2}、step {3}→{4}、{5}）" -f [double]$le.cumulative, (-[double]$le.elo), $ci, (Format-Int $le.step_a), (Format-Int $le.step_b), (Format-Ago (From-Unix $le.time))
        } else { $v.elo.Text = "（まだ無い。archive {0} 個）" -f @($d.archives).Count }
        $ms = @($d.matches)
        if ($ms.Count -gt 0) {
            $lm = $ms[$ms.Count - 1]
            $v.match.Text = "{0} / {1} 局（{2:P0}、{3}、{4}）" -f $lm.a_points, $lm.n, [double]$lm.winrate, $lm.go, (Format-Ago (From-Unix $lm.time))
        } else { $v.match.Text = "（まだ無い）" }
        $au = $d.auto; $ac = $d.auto_cfg
        if ($null -eq $ac -or -not $ac.enabled) { $v.auto.Text = "無効（config.toml の [auto]）" }
        elseif ($null -ne $au -and $null -ne $au.running) { $v.auto.Text = "{0} 実行中（{1}）" -f $au.running.kind, (Format-Ago (From-Unix $au.running.started)) }
        else {
            $q = if ($null -ne $au) { @($au.queue).Count } else { 0 }
            $next = if ($null -ne $au -and $null -ne $au.last_archive) { (From-Unix $au.last_archive).AddHours([double]$ac.every_hours).ToString("MM/dd HH:mm") } else { "次のチェックポイント" }
            $v.auto.Text = "待機（次の自己評価 {0}、待ち {1} 件、{2} 時間ごと）" -f $next, $q, $ac.every_hours
        }
    }
    $tabs.SelectedTab.Controls[0].Invalidate()
}
function Update-StatusBar {
    $parts = @()
    foreach ($r in $Runs) {
        if ($script:Errors.ContainsKey($r)) { $parts += "$r : " + $script:Errors[$r] }
    }
    if ($script:HistNote) { $parts += $script:HistNote }
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
    try {
        # $ErrorActionPreference は関数の中だけ Continue にする（外側の Stop はそのまま）。
        # Stop のままだと schtasks が stderr に 1 行でも書いた時点で終端エラーになり、
        # 下の $LASTEXITCODE の分岐にも wsl.exe のフォールバックにも到達しない。
        # しかも WinForms のクリック ハンドラでは例外が黙って捨てられ、押しても無反応になる。
        $ErrorActionPreference = "Continue"
        $out = (& schtasks.exe /Run /TN $task 2>&1 | Out-String).Trim()
        if ($LASTEXITCODE -eq 0) {
            $status.Text = "$run : タスク「$task」を起動しました"
        } else {
            Start-Process -FilePath "wsl.exe" -ArgumentList (Get-LibraArgs $run @("run")) -WindowStyle Hidden
            $status.Text = "$run : タスク「$task」を起動できないので wsl.exe を直接起動しました（$out）"
        }
        $status.ForeColor = [System.Drawing.Color]::DimGray
    } catch {
        $status.Text = "$run : 起動に失敗: " + $_.Exception.Message
        $status.ForeColor = [System.Drawing.Color]::Firebrick
    }
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
            $withHistory = ([datetime]::Now -ge $script:NextHistory)
            if ($withHistory) { $script:NextHistory = [datetime]::Now.AddSeconds($HistorySec) }
            foreach ($r in $Runs) { Start-Fetch $r $withHistory }
        }
        Update-StatusBar
        if ($Screenshot -and -not $script:ShotDone -and $script:Pending.Count -eq 0 -and $script:Data.Count -eq $Runs.Count) {
            $script:ShotDone = $true
            $tabs.SelectedTab.Controls[0].Refresh()
            $bmp = New-Object System.Drawing.Bitmap($form.Width, $form.Height)
            $form.DrawToBitmap($bmp, (New-Object System.Drawing.Rectangle(0, 0, $form.Width, $form.Height)))
            $bmp.Save($Screenshot, [System.Drawing.Imaging.ImageFormat]::Png)
            $bmp.Dispose()
            [Console]::WriteLine("screenshot: " + $Screenshot + " tab=" + $tabs.SelectedTab.Text)
            foreach ($r in $Runs) {
                $d = $script:Data[$r]
                [Console]::WriteLine(("{0}: {1} flags={2} step={3} games={4} gpd={5} metrics={6} evals={7} matches={8} archives={9}" -f $r, $d.process, (@($d.flags) -join "+"), $d.status.step, $d.status.games_total, $d.status.games_per_day_1h, @($d.metrics).Count, @($d.evals).Count, @($d.matches).Count, @($d.archives).Count))
            }
            $form.Close()
        }
    } catch {
        $status.Text = "内部エラー: " + $_.Exception.Message
        $status.ForeColor = [System.Drawing.Color]::Firebrick
        if ($Screenshot) { [Console]::WriteLine("error: " + $_.Exception.Message + " " + $_.ScriptStackTrace); $form.Close() }
    }
})
$form.Add_Shown({ $timer.Start() })
$form.Add_FormClosing({ $timer.Stop(); foreach ($f in $script:Pending.Values) { try { $f.proc.Kill() } catch {} } })

if ($UpdateDesktopModel) {
    $info = Update-DesktopModel
    [Console]::WriteLine(("desktop model updated: step={0} stale={1} onnx={2} size={3} dir={4}" -f $info.step, $info.stale, $info.onnx_time, $info.size, $DesktopEngineDir))
    exit 0
}
if ($Do) {
    # GUI なしでボタンと同じ呼び出しを実行する（例: -Do "lx:pause"、-Do "ls:eval-now"）
    $run, $rest = $Do.Split(":", 2)
    if ([string]::IsNullOrWhiteSpace($rest)) {
        [Console]::WriteLine("-Do は <run>:<コマンド> の形で指定してください（例: ls:status、lx:pause）")
        exit 1
    }
    [Console]::WriteLine((Invoke-Libra $run (@($rest.Trim().Split(" ")) | Where-Object { $_ })))
    exit 0
}
Load-History
Update-DesktopModelLabel
[void]$form.ShowDialog()
