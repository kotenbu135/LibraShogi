# SPDX-License-Identifier: Apache-2.0
# Libra 管理コンソール（Windows 用 GUI）。
# WSL 内の bin/libra を wsl.exe 経由で呼び、本体 ls と搾取者 lx の進捗・速度・強さの推移を表示し、
# 起動 / 停止 / 自己評価・対外対局の前倒しを行う（一時停止・再開は 2026-09-14 に廃止。docs/decisions.md）。
# 縦長のウィンドウが前提（2026-09-15）。上のタブで ls / lx / クラウド / クラウド履歴 を切り替え、タブの見出しに稼働状態を出す。
# クラウド タブ: bin/libra-vast で vast.ai の自己対局ワーカーを起動 / 停止し、段階・回収局数・費用・残高を表示する。
# クラウド履歴 タブ: bin/libra-vast history で過去のセッションごとの費用・有効局・100 万局あたりの費用と今月の合計を表示する。
# 起動: libra-console.bat（powershell -ExecutionPolicy Bypass -File libra-console.ps1）
# 置き場所（ディストロ名・WSL 内の bin/libra・run の置き場）は同じフォルダの libra-paths.json から読む。
# これは tools/windows/install.sh が書く。コマンド行の -Distro / -Libra などを渡せばそちらが優先される。
# 自動テスト: -Screenshot C:\path\shot.png で 1 回更新して画面を PNG に保存し終了する（要約を stdout に出す）。
#             -Tab <名前>[,<名前>] で保存時に表示するタブを選ぶ（上のタブ: ls, lx, クラウド, クラウド履歴。グラフ: 局/日, Elo, 対外対局, 学習, 終局内訳, 手数, ログ）。
#             -Size 600x1200 でウィンドウの大きさを指定する（-Screenshot のときは前回の位置と大きさを復元しない）。
#             -Do "lx:stop" のようにボタンと同じ操作だけを GUI なしで実行して結果を出す（起動は含まない）。
#             -Do "vast:status" で bin/libra-vast を呼ぶ（例: vast:offers --gpu RTX_5070_Ti）。
#             -UpdateDesktopModel で desktop（天秤将棋GUI）に登録した libra.exe のモデルを最新の latest.onnx に置き換えて終了する。
param(
    [string]$Distro = "",        # 空なら wsl.exe の既定のディストロ（libra-paths.json で埋まる）
    [string]$Libra = "",         # WSL 内の bin/libra（libra-paths.json で埋まる）
    [string[]]$Runs = @("ls", "lx"),
    [int]$IntervalSec = 15,
    [int]$HistorySec = 300,
    [int]$LogLines = 40,          # グラフの「ログ」タブに出す log.txt の末尾の行数
    [string]$Screenshot = "",
    [string]$Tab = "",
    [string]$Size = "",
    [string]$Do = "",
    [string]$RunRoot = "",       # WSL 内の run の置き場（libra-paths.json で埋まる）
    [string]$ModelRun = "ls",
    [string]$LibraVast = "",     # WSL 内の bin/libra-vast（libra-paths.json で埋まる）
    [int]$VastAccountSec = 300,
    [double]$UsdJpy = 150,        # クラウド履歴の月の費用を円に直す目安
    [int]$BudgetJpy = 10000,      # 追加の計算費用の月の上限（docs/decisions.md 2026-09-13 のユーザーの決定）
    [string]$DesktopExe = "",    # 天秤将棋GUI の exe（libra-paths.json か %LOCALAPPDATA% から）
    [string]$DesktopEngineDir = (Join-Path $env:APPDATA "com.fusekishogi.tenbin\engines\libra\engine"),
    [switch]$UpdateDesktopModel
)
$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
[System.Windows.Forms.Application]::EnableVisualStyles()
# 非表示の PowerShell で起動されるので、途中の例外はメッセージボックスで見せる。
# ただし無人実行（-Screenshot / -Do / -UpdateDesktopModel）では誰も押せないので stderr に出して終わる
$script:Headless = [bool]($Screenshot -or $Do -or $UpdateDesktopModel)
trap {
    if ($script:Headless) {
        [Console]::Error.WriteLine($_.ToString())
        [Console]::Error.WriteLine($_.ScriptStackTrace)
        exit 1
    }
    [System.Windows.Forms.MessageBox]::Show(($_.ToString() + "`r`n" + $_.ScriptStackTrace), "Libra 管理コンソール エラー", "OK", "Error") | Out-Null
    exit 1
}

# ---- 置き場所の解決（コマンド行 > libra-paths.json > その場の既定） ----
$script:PathsFile = Join-Path $PSScriptRoot "libra-paths.json"
if (Test-Path $script:PathsFile) {
    $cfg = Get-Content $script:PathsFile -Raw -Encoding UTF8 | ConvertFrom-Json
    foreach ($k in @("Distro", "Libra", "RunRoot", "LibraVast", "DesktopExe")) {
        if (-not $PSBoundParameters.ContainsKey($k) -and $cfg.PSObject.Properties.Name -contains $k -and $cfg.$k) {
            Set-Variable -Name $k -Value ([string]$cfg.$k) -Scope Script
        }
    }
}
if (-not $DesktopExe) { $DesktopExe = Join-Path $env:LOCALAPPDATA "天秤将棋GUI\tenbin-shogi-gui.exe" }
if (-not $Libra) {
    throw "bin/libra の場所が分かりません。WSL から tools/windows/install.sh を実行して libra-paths.json を作るか、-Libra <WSL 内のパス> を渡してください。"
}
if (-not $RunRoot) { $RunRoot = (Split-Path (Split-Path $Libra -Parent) -Parent) -replace "\\", "/" }
if (-not $LibraVast) { $LibraVast = ($Libra + "-vast") }
# wsl.exe に渡すディストロの指定。$Distro が空なら既定のディストロを使う
function Get-WslDistroArgs { if ($Distro) { return @("-d", $Distro) } else { return @() } }
# \\wsl.localhost\<ディストロ> のためにディストロ名が要る場面では、空のときだけ既定の名前を引く
function Get-DistroName {
    if ($Distro) { return $Distro }
    if (-not $script:DefaultDistro) {
        $out = & wsl.exe -l -q 2>$null
        $script:DefaultDistro = ($out | Where-Object { $_.Trim() } | Select-Object -First 1).Trim()
    }
    return $script:DefaultDistro
}

$script:TabTitles = @{ ls = "ls 本体"; lx = "lx 搾取者"; cloud = "クラウド"; history = "クラウド履歴" }
$script:TabState = @{}  # 上のタブの Tag -> @{text; color}（見出しの右に出す稼働状態）
$script:TaskNames = @{ ls = "LibraShogi run"; lx = "LibraShogi run lx" }
$script:Colors = @{ ls = [System.Drawing.Color]::FromArgb(31, 119, 180); lx = [System.Drawing.Color]::FromArgb(255, 127, 14) }
$script:Palette = @([System.Drawing.Color]::FromArgb(31, 119, 180), [System.Drawing.Color]::FromArgb(255, 127, 14), [System.Drawing.Color]::FromArgb(44, 160, 44), [System.Drawing.Color]::FromArgb(148, 103, 189), [System.Drawing.Color]::FromArgb(214, 39, 40))
# 固定の参照の系列の色。Run-Color は run に色が決まっていれば $i を見ないので、参照を run の色で描くと基準比と重なって見分けが付かない
$script:RefPalette = @([System.Drawing.Color]::FromArgb(44, 160, 44), [System.Drawing.Color]::FromArgb(148, 103, 189), [System.Drawing.Color]::FromArgb(214, 39, 40), [System.Drawing.Color]::FromArgb(140, 86, 75), [System.Drawing.Color]::FromArgb(227, 119, 194), [System.Drawing.Color]::FromArgb(23, 190, 207))
$script:HistDir = Join-Path $env:LOCALAPPDATA "LibraShogi"
$script:HistFile = Join-Path $script:HistDir "console-history.csv"
$script:Hist = @{}      # run -> ArrayList（コンソール自身の観測: t, games, step, gpd。実測 局/日 に使う）
$script:Data = @{}      # run -> status --history の結果（metrics, evals, matches, archives, auto, auto_cfg）
$script:Pending = @{}   # run -> @{proc; out; err; started}
$script:Last = @{}      # run -> 直近の status オブジェクト
$script:Notes = @{}     # run -> @{text; error; until}（操作の結果。ステータスバーに until まで出す）
$script:StartCheck = @{} # run -> 起動を押した後、稼働を確かめる期限
$script:Ui = @{}        # run -> @{vals; log; buttons; autoButtons; autoTips}
$script:chkEloDetail = $null  # Elo のグラフに相手ごとの線も出すか（画面を作る前は $null ＝ 出さない）
$script:NextFetch = [datetime]::MinValue
$script:NextHistory = [datetime]::MinValue
$script:ShotDone = $false
$script:Errors = @{}
$script:HistNote = ""
$script:Vast = $null          # 直近の libra-vast status --json
$script:VastPending = $null   # @{proc; out; err; started}
$script:NextVast = [datetime]::MinValue
$script:NextVastAccount = [datetime]::MinValue
$script:VastError = ""
$script:VastLeak = $false     # セッションが動いていないのにインスタンスが残っている
$script:VastSessionKey = ""   # 直近のセッションの dir と alive。変わったら履歴を取り直す
$script:VastHist = $null      # 直近の libra-vast history --json
$script:VastHistPending = $null
$script:NextVastHist = [datetime]::MinValue
$script:VastHistAt = [datetime]::MinValue
$script:VastHistError = ""
$script:LayoutFile = Join-Path $script:HistDir "console-layout.json"
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

# ---- Elo の文言（純関数。tools/windows/tests/console-format.tests.ps1 が AST で取り出して試す） ----
function Format-Ci($ci) {
    $lo = Ci-Val $ci 0; $hi = Ci-Val $ci 1
    if ($null -eq $lo -or $null -eq $hi) { return "" }
    return " [{0:+0;-0;0}, {1:+0;-0;0}]" -f $lo, $hi
}
function Format-Pct($score) {
    # {0:P0} は地域によって「83 %」と空白が入る。日本語の文なので空白を入れない形に固定する
    if ($null -eq $score) { return "-" }
    return ("{0}%" -f [Math]::Round([double]$score * 100))
}
function Format-Saturated($score) {
    # 得点が 15%〜85% の外だと差が開きすぎて、Elo の値も区間も当てにならない（docs/scaling-2026-09-18.md の「天井」）
    if ($null -eq $score) { return "" }
    $s = [double]$score
    if ($s -ge 0.85 -or $s -le 0.15) { return "・天井" }
    return ""
}
function Format-Elo-Anchor($row) {
    # 基準比。どの step を測った値かを頭に出す（今の step とは違うことが多い）
    if ($null -eq $row) { return $null }
    return "step {0} で {1:+0.0;-0.0;0}{2}（基準 step {3} に {4}{5}、{6}）" -f (Format-Int $row.step), [double]$row.elo, (Format-Ci $row.ci95),
        (Format-Int $row.anchor_step), (Format-Pct $row.score_new), (Format-Saturated $row.score_new), (Format-Ago (From-Unix $row.t))
}
function Format-Elo-Best($row) {
    # 最強比。best_step は「打った相手（そのときの最強）」で、更新していれば今の最強は row.step のほう
    if ($null -eq $row) { return $null }
    $tail = if ($row.improved) { "最強を更新" } else { "最強を抜けず、足踏み {0} 回" -f $row.stall }
    return "最強比 {0:+0.0;-0.0;0}{1}（step {2} が step {3} に {4}{5}で{6}、{7}）" -f [double]$row.elo_vs_best, (Format-Ci $row.ci95),
        (Format-Int $row.step), (Format-Int $row.best_step), (Format-Pct $row.score_new), (Format-Saturated $row.score_new), $tail, (Format-Ago (From-Unix $row.t))
}
function Format-Elo-Rating($rating) {
    # 強さの目盛り（全部の対局をまとめて 1 本にした Bradley-Terry の Elo）。いちばん新しい点と、その前の点との差。
    # 相手ごとの値（基準比・最強比・対 …）と違って、練習相手を入れ替えても 0 の意味が動かない
    if ($null -eq $rating) { return $null }
    $pts = @(@($rating.points) | Where-Object { $null -ne $_ -and $null -ne $_.elo -and $null -ne $_.games })
    if ($pts.Count -eq 0) { return $null }
    $last = $pts[$pts.Count - 1]
    $out = "目盛り {0:+0.0;-0.0;0}{1}（step {2}、{3} 局の時点）" -f [double]$last.elo, (Format-Ci $last.ci95),
        (Format-Int $last.step), (Format-Int $last.games)
    if ($pts.Count -gt 1) {
        $prev = $pts[$pts.Count - 2]
        $out += "、前の点（{0} 局）から {1:+0.0;-0.0;0}" -f (Format-Int $prev.games), ([double]$last.elo - [double]$prev.elo)
    }
    return $out
}
function Format-Elo-References($rows) {
    # 固定の参照は run をまたいで同じ相手なので絶対の物差しになる（docs/restart-plan.md §3 M4）。参照ごとに最新の 1 行だけ出す
    $last = [ordered]@{}
    foreach ($e in @($rows)) {
        if ($null -eq $e -or $null -eq $e.elo) { continue }
        $last[[string]$e.ref] = $e
    }
    $out = @()
    foreach ($k in (@($last.Keys) | Sort-Object)) {
        $e = $last[$k]
        $out += "対 {0} {1:+0.0;-0.0;0}{2}（step {3}、{4}{5}、{6}）" -f $k, [double]$e.elo, (Format-Ci $e.ci95),
            (Format-Int $e.step), (Format-Pct $e.score_new), (Format-Saturated $e.score_new), (Format-Ago (From-Unix $e.t))
    }
    if ($out.Count -eq 0) { return $null }
    return ($out -join "`r`n")
}

# ---- WSL 呼び出し ----
function Get-LibraArgs([string]$run, [string[]]$cmd) {
    return ((Get-WslDistroArgs) + @("--", $Libra, "--run", $run) + $cmd) -join " "
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
    # 同期呼び出し（stop/eval-now/match-now は 0.3 秒程度）。stdout+stderr を返す。
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
# ---- bin/libra-vast（クラウド タブ） ----
function New-VastProcess([string[]]$cmd) {
    # 引数は wsl.exe のコマンド行になるので空白を含めない（GPU 名は RTX_5070_Ti のように _ でつなぐ）
    $p = New-LibraProcess "" @()
    $p.StartInfo.Arguments = ((Get-WslDistroArgs) + @("--", $LibraVast) + $cmd) -join " "
    return $p
}
function Invoke-Vast([string[]]$cmd, [int]$TimeoutMs = 60000) {
    # 同期呼び出し（start・stop は 1 秒程度、offers・cleanup は vast.ai の API で数秒）。終了コードと stdout+stderr を返す
    $p = New-VastProcess $cmd
    [void]$p.Start()
    $out = $p.StandardOutput.ReadToEndAsync()
    $err = $p.StandardError.ReadToEndAsync()
    if (-not $p.WaitForExit($TimeoutMs)) {
        try { $p.Kill() } catch {}
        throw ("wsl.exe が {0} 秒で返りません: libra-vast {1}" -f [int]($TimeoutMs / 1000), ($cmd -join " "))
    }
    [void][System.Threading.Tasks.Task]::WaitAll(@($out, $err), 5000)
    return @{ code = $p.ExitCode; text = (($out.Result + $err.Result).Trim()) }
}
function Start-VastFetch {
    if ($null -ne $script:VastPending) { return }
    $cmd = @("status", "--json", "--tail", "12")
    if ([datetime]::Now -ge $script:NextVastAccount) {
        $cmd += "--account"
        $script:NextVastAccount = [datetime]::Now.AddSeconds($VastAccountSec)
    }
    $p = New-VastProcess $cmd
    [void]$p.Start()
    $script:VastPending = @{ proc = $p; out = $p.StandardOutput.ReadToEndAsync(); err = $p.StandardError.ReadToEndAsync(); started = [datetime]::Now }
}
function Complete-VastFetch {
    $f = $script:VastPending
    if ($null -eq $f) { return }
    if (-not $f.proc.HasExited -or -not $f.out.IsCompleted) {
        if (([datetime]::Now - $f.started).TotalSeconds -gt 90) {
            try { $f.proc.Kill() } catch {}
            $script:VastError = "libra-vast status がタイムアウト（90 秒）"
            $script:VastPending = $null
        }
        return
    }
    $script:VastPending = $null
    $text = $f.out.Result.Trim()
    try {
        if (-not $text) { throw ("出力なし: " + $f.err.Result.Trim()) }
        $obj = $text | ConvertFrom-Json
        # 残高とインスタンスは VastAccountSec ごとにしか取らないので、取らなかった回は前の値を引き継ぐ
        if ($null -eq $obj.PSObject.Properties["account"] -and $null -ne $script:Vast -and $null -ne $script:Vast.PSObject.Properties["account"]) {
            $obj | Add-Member -NotePropertyName account -NotePropertyValue $script:Vast.account
        }
        $script:Vast = $obj
        $script:VastError = ""
        Update-VastPanel
    } catch {
        $script:VastError = "libra-vast status の読み取りに失敗: " + $_.Exception.Message
    }
}
function Start-VastHistFetch {
    if ($null -ne $script:VastHistPending) { return }
    $p = New-VastProcess @("history", "--json")
    [void]$p.Start()
    $script:VastHistPending = @{ proc = $p; out = $p.StandardOutput.ReadToEndAsync(); err = $p.StandardError.ReadToEndAsync(); started = [datetime]::Now }
}
function Complete-VastHistFetch {
    $f = $script:VastHistPending
    if ($null -eq $f) { return }
    if (-not $f.proc.HasExited -or -not $f.out.IsCompleted) {
        if (([datetime]::Now - $f.started).TotalSeconds -gt 60) {
            try { $f.proc.Kill() } catch {}
            $script:VastHistError = "libra-vast history がタイムアウト（60 秒）"
            $script:VastHistPending = $null
        }
        return
    }
    $script:VastHistPending = $null
    $text = $f.out.Result.Trim()
    try {
        if (-not $text) { throw ("出力なし: " + $f.err.Result.Trim()) }
        $script:VastHist = $text | ConvertFrom-Json
        $script:VastHistAt = [datetime]::Now
        $script:VastHistError = ""
        Update-HistPanel
    } catch {
        $script:VastHistError = "libra-vast history の読み取りに失敗: " + $_.Exception.Message
    }
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
# 縦長が前提（画面の作業領域より高くしない）。前回の位置と大きさは Restore-Layout が戻す
$wa = [System.Windows.Forms.Screen]::PrimaryScreen.WorkingArea
$form.Size = New-Object System.Drawing.Size(640, [Math]::Min(1280, [Math]::Max(760, $wa.Height)))
$form.MinimumSize = New-Object System.Drawing.Size(540, 900)   # これより低いと run のタブでグラフの場所が無くなる
if ($Size -match '^(\d+)x(\d+)$') { $form.Size = New-Object System.Drawing.Size([int]$Matches[1], [int]$Matches[2]) }

$root = New-Object System.Windows.Forms.TableLayoutPanel
$root.Dock = "Fill"
$root.ColumnCount = 1
$root.RowCount = 3
[void]$root.RowStyles.Add((New-Object System.Windows.Forms.RowStyle("AutoSize")))
[void]$root.RowStyles.Add((New-Object System.Windows.Forms.RowStyle("Percent", 100)))
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
$bar.Controls.Add((New-Button "今すぐ更新" { $script:NextFetch = [datetime]::MinValue; $script:NextHistory = [datetime]::MinValue; $script:NextVast = [datetime]::MinValue; $script:NextVastHist = [datetime]::MinValue }))
$bar.Controls.Add((New-Label "更新間隔(秒)"))
$numIv = New-Object System.Windows.Forms.NumericUpDown
$numIv.Minimum = 5; $numIv.Maximum = 600; $numIv.Value = [Math]::Max(5, $IntervalSec); $numIv.Width = 60
$numIv.Margin = New-Object System.Windows.Forms.Padding(0, 4, 12, 0)
$bar.Controls.Add($numIv)
$bar.Controls.Add((New-Button "全部 起動" { foreach ($r in $Runs) { Start-Run $r } }))
$bStopAll = New-Button "全部 停止" { if (Confirm-Action "両方の run に STOP を送ります（チェックポイントを書いて終了。再起動前の手順）。よろしいですか？") { Invoke-All @("stop") } }
$bar.Controls.Add($bStopAll)
$bar.SetFlowBreak($bStopAll, $true)
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
    $src = "\\wsl.localhost\" + (Get-DistroName) + "\" + ($RunRoot.TrimStart("/") -replace "/", "\") + "\$ModelRun\checkpoints\latest.onnx"
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

# 上のタブ（ls / lx / クラウド / クラウド履歴）。見出しの右に稼働状態を色付きで描く（タブの裏の run が止まっても見えるように）
$runTabs = New-Object System.Windows.Forms.TabControl
$runTabs.Dock = "Fill"
$runTabs.Multiline = $true
$runTabs.DrawMode = "OwnerDrawFixed"
$runTabs.Padding = New-Object System.Drawing.Point(10, 5)
$runTabs.Add_DrawItem({
    param($s, $e)
    $page = $s.TabPages[$e.Index]
    $key = [string]$page.Tag
    $bg = if ($e.Index -eq $s.SelectedIndex) { [System.Drawing.Color]::White } else { [System.Drawing.SystemColors]::Control }
    $e.Graphics.FillRectangle((New-Object System.Drawing.SolidBrush($bg)), $e.Bounds)
    $flags = [System.Windows.Forms.TextFormatFlags]::NoPadding -bor [System.Windows.Forms.TextFormatFlags]::VerticalCenter -bor [System.Windows.Forms.TextFormatFlags]::SingleLine
    $title = $script:TabTitles[$key]
    $x = $e.Bounds.X + 6
    $tw = [System.Windows.Forms.TextRenderer]::MeasureText($e.Graphics, $title, $s.Font, $e.Bounds.Size, $flags).Width
    [System.Windows.Forms.TextRenderer]::DrawText($e.Graphics, $title, $s.Font, (New-Object System.Drawing.Rectangle($x, $e.Bounds.Y, ($tw + 2), $e.Bounds.Height)), [System.Drawing.Color]::Black, $flags)
    $st = $script:TabState[$key]
    if ($null -ne $st -and $st.text) {
        $x2 = $x + $tw + 6
        [System.Windows.Forms.TextRenderer]::DrawText($e.Graphics, $st.text, $s.Font, (New-Object System.Drawing.Rectangle($x2, $e.Bounds.Y, [Math]::Max(1, $e.Bounds.Right - $x2), $e.Bounds.Height)), $st.color, $flags)
    }
})
function Set-TabState([string]$key, [string]$text, $color) {
    $old = $script:TabState[$key]
    if ($null -ne $old -and $old.text -eq $text -and $old.color -eq $color) { return }
    $script:TabState[$key] = @{ text = $text; color = $color }
    foreach ($pg in $runTabs.TabPages) {
        if ([string]$pg.Tag -ne $key) { continue }
        # 見出しの幅は Text から決まるので、描く文字列（題名＋状態）と同じ長さの Text にする
        $t = $script:TabTitles[$key] + $(if ($text) { "  " + $text } else { "" })
        if ($pg.Text -ne $t) { $pg.Text = $t }
    }
    $runTabs.Invalidate()
}
function New-TopPage([string]$key) {
    if (-not $script:TabTitles.ContainsKey($key)) { $script:TabTitles[$key] = $key }
    $p = New-Object System.Windows.Forms.TabPage
    $p.Text = $script:TabTitles[$key]; $p.Tag = $key
    $p.BackColor = [System.Drawing.Color]::White
    [void]$runTabs.TabPages.Add($p)
    return $p
}
function Selected-Run {
    # 学習・終局内訳・手数のグラフの run。クラウドのタブを見ている間は直前に選んでいた run
    $k = if ($null -ne $runTabs.SelectedTab) { [string]$runTabs.SelectedTab.Tag } else { "" }
    if ($Runs -contains $k) { $script:LastRun = $k }
    if (-not $script:LastRun) { $script:LastRun = $Runs[0] }
    return $script:LastRun
}
function Fit-Labels([int]$width, [int]$keyW, $labels) {
    # 値の欄の長い文字列は折り返す（縦長のウィンドウで右にはみ出さないように）
    $w = [Math]::Max(120, $width - $keyW - 20)
    foreach ($l in $labels) { if ($l.MaximumSize.Width -ne $w) { $l.MaximumSize = New-Object System.Drawing.Size($w, 0) } }
}
function Set-RowVisible($u, [string]$key, [bool]$on) {
    foreach ($c in $u.rows[$key]) { $c.Visible = $on }
}
$root.Controls.Add($runTabs, 0, 1)

$script:Keys = @(
    @("process", "状態"), @("updated", "status 更新"), @("step", "step / 世代"), @("games_total", "総局数"),
    @("gpd", "局/日（1 時間平均）"), @("measured", "局/日（実測）"), @("active", "同時局数"), @("elapsed", "稼働 / セッション局数"),
    @("results", "先手 / 引分 / 後手"), @("plies", "平均手数 / sims/手"), @("loss", "loss / policy / value"),
    @("lr", "lr / 学習 1 回"), @("gpu", "GPU メモリ"), @("ckpt", "最終チェックポイント"), @("exploiter", "対本体 勝率"), @("restarts", "再起動"),
    @("elo", "強さ（Elo）"), @("reference", "強さ（固定の参照 Elo）"), @("match", "対外対局 勝率"), @("auto", "自動計測")
)
function New-RunPanel([string]$run) {
    # 上から 状態の表 / ボタン / グラフ（Move-Charts が選んでいる run の chartSlot に付け替える。log.txt の末尾はグラフの「ログ」タブ）
    $g = New-Object System.Windows.Forms.Panel
    $g.Dock = "Fill"
    $g.Padding = New-Object System.Windows.Forms.Padding(6, 4, 6, 2)
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
    $rows = @{}
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
        $rows[$k[0]] = @($lk, $lv)
        $row++
    }
    $inner.Controls.Add($grid, 0, 0)

    $btns = New-Object System.Windows.Forms.FlowLayoutPanel
    $btns.Dock = "Fill"; $btns.AutoSize = $true; $btns.WrapContents = $true
    $btns.Margin = New-Object System.Windows.Forms.Padding(0, 4, 0, 4)
    $bl = @()
    $bStart = New-Button "起動" { Start-Run $run }.GetNewClosure() 56
    $bStop = New-Button "停止" { if (Confirm-Action "$run を停止します（チェックポイントを書いて終了）。よろしいですか？") { Invoke-Run $run @("stop") } }.GetNewClosure() 56
    $bl += $bStart
    $bl += $bStop
    $be = New-Button "今すぐ自己評価" { if (Confirm-Action "$run : 次のチェックポイント（10 分以内）で archive を作り、直前の archive と自己評価します（GPU を共有、約 10 分）。よろしいですか？") { Invoke-Run $run @("eval-now") } }.GetNewClosure() 110
    $bm = New-Button "今すぐ対外対局" { if (Confirm-Action "$run : 次のチェックポイント（10 分以内）で外部エンジンとの計測対局を積みます（GPU と CPU を共有、10 局で 15 分程度）。よろしいですか？") { Invoke-Run $run @("match-now") } }.GetNewClosure() 110
    foreach ($b in $bl) { $btns.Controls.Add($b) }
    $btns.Controls.Add($be); $btns.Controls.Add($bm)
    $inner.Controls.Add($btns, 0, 1)

    $slot = New-Object System.Windows.Forms.Panel
    $slot.Dock = "Fill"
    $slot.Margin = New-Object System.Windows.Forms.Padding(0, 4, 0, 0)
    $inner.Controls.Add($slot, 0, 2)
    $inner.Add_Resize({ Fit-Labels $inner.ClientSize.Width 150 $vals.Values }.GetNewClosure())

    $tipEval = "次のチェックポイントで archive を作り、基準ネットと 100 局対局する"
    $tipMatch = "次のチェックポイントで外部エンジンとの計測対局を積む"
    $script:Tip.SetToolTip($be, $tipEval)
    $script:Tip.SetToolTip($bm, $tipMatch)
    $script:Ui[$run] = @{ vals = $vals; rows = $rows; logText = ""; startButton = $bStart; stopButton = $bStop; autoButtons = @($be, $bm); autoTips = @($tipEval, $tipMatch); chartSlot = $slot }
    return $g
}
foreach ($r in $Runs) { $pg = New-TopPage $r; $pg.Controls.Add((New-RunPanel $r)) }

# ---- グラフ ----
function Get-Range {
    switch ($cmbRange.SelectedIndex) {
        0 { return [datetime]::Now.AddHours(-6) }
        1 { return [datetime]::Now.AddHours(-24) }
        2 { return [datetime]::Now.AddDays(-7) }
        default { return [datetime]::MinValue }
    }
}
function Get-XAxis($series, [datetime]$t0, [bool]$xGames, [datetime]$now, [bool]$xLog = $false) {
    # 横軸の範囲。総局数の軸は局数（games_at）の付いた点が 1 つでもあるときだけで、無ければ時間の軸に落ちる。
    # 返すのは @{ games; log; min; max; span; n }（min・max は描く座標＝対数の軸なら log2。n は描ける点の数）
    $useGames = $false
    if ($xGames) {
        foreach ($s in $series) { foreach ($p in $s.pts) { if ($null -ne $p.g) { $useGames = $true; break } }; if ($useGames) { break } }
    }
    # 対数にできるのは総局数の軸だけ（時間の Unix 秒を対数にしても読めない）
    $useLog = ($xLog -and $useGames)
    $min = $null; $max = $null; $n = 0
    foreach ($s in $series) {
        foreach ($p in $s.pts) {
            if ($p.t -lt $t0) { continue }
            $xv = X-Scale (Pt-XVal $p $useGames) $useLog
            if ($null -eq $xv) { continue }
            $n++
            if ($null -eq $min -or $xv -lt $min) { $min = $xv }
            if ($null -eq $max -or $xv -gt $max) { $max = $xv }
        }
    }
    if ($n -eq 0) { return @{ games = $useGames; log = $useLog; min = 0.0; max = 1.0; span = 1.0; n = 0 } }
    if ($useGames) {
        # 測った点のところまで（右に「今」までの空白を作らない）
        $floor = if ($useLog) { 0.5 } else { 1.0 }   # 対数の軸は幅が「何倍か」なので 1 局ぶんでは足りない
        if ($max - $min -lt $floor) { $min = [Math]::Max(0.0, $max - $floor) }
    } else {
        # 時間の軸は右端を「今」にして、いつの値かが分かるようにする
        $max = [double]([System.DateTimeOffset]::new($now).ToUnixTimeMilliseconds() / 1000.0)
        if ($t0 -gt [datetime]::MinValue) { $min = [double]([System.DateTimeOffset]::new($t0).ToUnixTimeMilliseconds() / 1000.0) }
        if ($max - $min -lt 600) { $min = $max - 600 }
    }
    $minSpan = if ($useLog) { 0.5 } else { 1.0 }
    return @{ games = $useGames; log = $useLog; min = $min; max = $max; span = [Math]::Max($minSpan, $max - $min); n = $n }
}
function Show-Elo-Detail {
    # Elo のグラフに相手ごとの線（基準比・鏡・対 …）も出すか。既定は出さない
    if ($null -eq $script:chkEloDetail) { return $false }
    return [bool]$script:chkEloDetail.Checked
}
function Format-Elo-Note([bool]$detail, [bool]$haveRating, [bool]$useGames, $trendFit = $null, [bool]$useLog = $false) {
    # グラフの下の注記（純関数。tools/windows/tests/console-format.tests.ps1 が試す）
    $n = if (-not $haveRating) {
        "目盛りがまだ出せないので相手ごとの線を出している。「基準比」の 0 は系列の最初の重み、「対 …」の 0 はその参照と互角で、0 の意味が違う"
    } elseif ($detail) {
        "太い線が「強さの目盛り」（全部の対局をまとめて 1 本にした Elo）。相手ごとの線は 1 点 200 局で幅が広く、練習相手の入れ替えと相性で上下するので、目盛りのほうで伸びを見る"
    } else {
        "「強さの目盛り」＝ 全部の対局をまとめて 1 本にした Elo。0 はいちばん古い重み。相手ごとの線は「内訳を出す」で足せる"
    }
    # 目安の線の読み方。これが「頭打ちか」の答えになる
    if ($null -ne $trendFit -and $null -ne $trendFit.elo_per_doubling) {
        $n += "。点線は目安で、最近の伸び（局数 2 倍あたり {0:+0;-0;0} Elo）をそのまま延ばしたもの。**点が点線に乗っているかぎり頭打ちではない**（下に離れていけば頭打ち）" -f [double]$trendFit.elo_per_doubling
        if (-not $useLog) { $n += "。横軸を「総局数（対数）」にすると点線が直線になり、ずれが見やすい" }
    }
    $n += "。全期間を表示"
    if (-not $useGames) { $n += "。横軸が時間だと止めた間も伸びが寝て見えるので、ふだんは「総局数」で見る" }
    elseif ($useLog) { $n += "。横軸は総局数の対数（右へ 1 目盛りで局数が 2 倍）。伸びは局数の対数にほぼ比例するので、ふつうの横軸では一定の伸びでも右で寝て見える" }
    return $n
}
function Format-Auto-Status($ac, $au, $games, [int]$queued) {
    # 「自動計測」の行（純関数。tools/windows/tests/console-format.tests.ps1 が試す）。
    # 節目は総局数の倍数で決まる（[auto] every_games）。時間区切りは 2026-09-19 に廃止したので、
    # 「あと何局で次の計測か」を出す（時刻は局/日で変わるので出さない）
    if ($null -eq $ac -or -not $ac.enabled) { return "無効（config.toml の [auto]）" }
    if ($null -ne $au -and $null -ne $au.running) { return "{0} 実行中（{1}）" -f $au.running.kind, (Format-Ago (From-Unix $au.running.started)) }
    $every = if ($null -ne $ac.every_games) { [double]$ac.every_games } else { 0 }
    if ($every -le 0) { return "節目なし（[auto] every_games が 0。eval-now / match-now のときだけ測る、待ち {0} 件）" -f $queued }
    # 40 万局は「40 万局ごと」と読みやすく出す（万で割り切れないときはそのまま局で）
    $unit = if ($every -ge 10000 -and ($every % 10000) -eq 0) { (Format-Int ($every / 10000)) + " 万局ごと" } else { (Format-Int $every) + " 局ごと" }
    if ($null -eq $games) { return "待機（{0}、待ち {1} 件）" -f $unit, $queued }
    $g = [double]$games
    $next = ([math]::Floor($g / $every) + 1) * $every
    return "待機（次の自己評価は総局数 {0}、あと {1} 局、{2}、待ち {3} 件）" -f (Format-Int $next), (Format-Int ($next - $g)), $unit, $queued
}
function Use-GamesAxis {
    # Elo と 対外対局 の横軸。学習を進めるのは時間ではなく局数（止めている間・GPU を分け合う間は
    # 同じ時間でも進みが違う）ので、既定は総局数。$cmbAxis がまだ無い起動直後も総局数。
    # 並びは 0 総局数 / 1 時間 / 2 総局数（対数）。**2 を末尾に足したのは**、
    # console-layout.json に番号で残っているため（0 と 1 の意味を動かさない）
    if ($null -eq $script:cmbAxis) { return $true }
    return ($script:cmbAxis.SelectedIndex -ne 1)
}
function Use-LogAxis {
    # 横軸を総局数の対数にするか。一定の伸びが直線になるので、頭打ちかどうかが形で読める
    if ($null -eq $script:cmbAxis) { return $false }
    return ($script:cmbAxis.SelectedIndex -eq 2)
}
function New-Series([string]$name, $color, [bool]$marker = $false, [bool]$dash = $false, [single]$width = 2) {
    return @{ name = $name; color = $color; pts = (New-Object System.Collections.ArrayList); marker = $marker; dash = $dash; gap = $false; width = $width }
}
function Add-Pt($series, [datetime]$t, [double]$y, $lo = $null, $hi = $null, [string]$label = "", $g = $null) {
    # g は「その重みを保存した時点の総局数」（status --history の games_at）。横軸を総局数にするときに使う
    [void]$series.pts.Add(@{ t = $t; y = $y; lo = $lo; hi = $hi; label = $label; g = $g })
}
function X-Scale($v, [bool]$log) {
    # 横軸の値を描く座標に直す。対数なら log2（局数は正）。伸びは局数の対数にほぼ比例する
    # （AlphaZero 系。docs/scaling-2026-09-18.md）ので、対数の軸なら**一定の伸びが直線**になり、
    # 頭打ちかどうかが形で読める。ふつうの軸では同じ伸びでも必ず右で寝て見える
    if ($null -eq $v) { return $null }
    if (-not $log) { return [double]$v }
    return [Math]::Log([Math]::Max(1.0, [double]$v), 2)
}
function X-Unscale([double]$v, [bool]$log) {
    # 目盛りの文字にするとき、描く座標から元の値に戻す
    if (-not $log) { return $v }
    return [Math]::Pow(2, $v)
}
function Pt-XVal($p, [bool]$useGames) {
    # 横軸の値。総局数の軸なら局数、時間の軸なら Unix 秒。総局数が分からない点は総局数の軸では描かない
    if ($useGames) {
        if ($null -eq $p.g) { return $null }
        return [double]$p.g
    }
    return [double]([System.DateTimeOffset]::new($p.t).ToUnixTimeMilliseconds() / 1000.0)
}
function Build-Rating-Series([string]$name, $color, $rating) {
    # 「強さの目盛り」の系列。status --history の rating.points（全部の対局をまとめて 1 本にした
    # Bradley-Terry の Elo）から作る。相手ごとの線は 1 点 200 局で幅が広く、参照の入れ替えと
    # じゃんけん（非推移性）で上下するので、下がっていないのに下がって見える。目盛りは全部の
    # 対局を一度に当てはめるので、参照を入れ替えても 0 の意味が動かない（2026-09-19 のユーザーの
    # 「Elo さがってませんか」から）
    $s = New-Series $name $color $true $false 3.5
    if ($null -eq $rating) { return $s }
    foreach ($p in @($rating.points)) {
        # t が無い点は時間の横軸で置く場所が決まらないので描かない（step から引き直せた点だけ出す）
        if ($null -eq $p -or $null -eq $p.elo -or $null -eq $p.t -or $null -eq $p.games) { continue }
        Add-Pt $s (From-Unix $p.t) ([double]$p.elo) (Ci-Val $p.ci95 0) (Ci-Val $p.ci95 1) `
            ("目盛り step " + (Format-Int $p.step)) $p.games
    }
    return $s
}
function Build-Trend-Series([string]$name, $color, $fit, $points) {
    # 「目安の線」。`rating.curve_fit`（log2(総局数) に対する Elo の直線当てはめ）を描く。
    # **点がこの線に乗っているかぎり、伸びは初めからの傾きのまま**で、頭打ちではない。
    # 伸びは局数の対数にほぼ比例するので、ふつうの横軸では一定の伸びでも必ず右で寝て見え、
    # グラフの形だけでは頭打ちか区別が付かない（2026-09-19 のユーザーの「初期に比べると伸びが
    # 緩やかなので頭打ちなのかグラフからわかりにくい」）。対数の横軸ならこの線が直線になる
    $s = New-Series $name $color $false $true 1.5
    if ($null -eq $fit -or $null -eq $fit.elo_per_doubling -or $null -eq $fit.intercept) { return $s }
    $from = $fit.games_from; $to = $fit.games_to
    if ($null -eq $from -or $null -eq $to -or [double]$from -lt 1 -or [double]$to -le [double]$from) { return $s }
    # 線を引く時刻は実際の点のものを使う（時間の横軸で置く場所が要る。無ければ描かない）
    $pts = @(@($points) | Where-Object { $null -ne $_ -and $null -ne $_.t })
    if ($pts.Count -eq 0) { return $s }
    $t0 = From-Unix $pts[0].t
    $b = [double]$fit.elo_per_doubling; $a = [double]$fit.intercept
    $lo = [Math]::Log([double]$from, 2); $hi = [Math]::Log([double]$to, 2)
    for ($k = 0; $k -le 40; $k++) {
        $x = $lo + ($hi - $lo) * $k / 40.0
        Add-Pt $s $t0 ($a + $b * $x) $null $null "" ([Math]::Round([Math]::Pow(2, $x)))
    }
    return $s
}
function Run-Color([string]$run, [int]$i = 0) {
    if ($script:Colors.ContainsKey($run)) { return $script:Colors[$run] }
    return $script:Palette[$i % $script:Palette.Length]
}
function Draw-Chart($g, [int]$w, [int]$h, [string]$title, $series, [string]$yfmt, [bool]$zeroBase, [string]$note, [bool]$allTime = $false, [bool]$xGames = $false, [bool]$xLog = $false) {
    $g.SmoothingMode = "AntiAlias"
    $g.Clear([System.Drawing.Color]::White)
    $font = New-Object System.Drawing.Font("Yu Gothic UI", 8)
    $gray = [System.Drawing.Brushes]::Gray
    $black = [System.Drawing.Brushes]::Black
    $left = 72; $right = 14; $top = 30; $bottom = 24; $lineH = 15
    $g.DrawString($title, $font, $black, 6, 3)
    # 1 日 1 回の計測（Elo・対外対局）は、期間の選択が短いと前回の点が外れて推移が見えないので常に全期間を描く
    $t0 = if ($allTime) { [datetime]::MinValue } else { Get-Range }
    $ax = Get-XAxis $series $t0 $xGames ([datetime]::Now) $xLog
    $useGames = [bool]$ax.games
    $useLog = [bool]$ax.log
    $xmin = [double]$ax.min; $xmax = [double]$ax.max; $xspan = [double]$ax.span
    if ([int]$ax.n -eq 0) {
        $g.DrawString("（データなし）", $font, $gray, $left, $top + 10)
        if ($note) { $g.DrawString($note, $font, $gray, $left, $top + 28) }
        return
    }
    $ymin = [double]::MaxValue; $ymax = [double]::MinValue
    foreach ($s in $series) {
        foreach ($p in $s.pts) {
            if ($p.t -lt $t0 -or $null -eq (Pt-XVal $p $useGames)) { continue }
            $lo = if ($null -ne $p.lo) { [double]$p.lo } else { $p.y }
            $hi = if ($null -ne $p.hi) { [double]$p.hi } else { $p.y }
            if ($lo -lt $ymin) { $ymin = $lo }
            if ($hi -gt $ymax) { $ymax = $hi }
        }
    }
    # 見出し: タイトルの右に注記、その右に凡例を置き、入らなければ次の行に回して描画域を下げる（重なって読めなくなるため）
    $shown = @($series | Where-Object { @($_.pts | Where-Object { $_.t -ge $t0 -and $null -ne (Pt-XVal $_ $useGames) }).Count -gt 0 })
    $legendW = 0
    foreach ($s in $shown) { $legendW += 22 + $g.MeasureString($s.name, $font).Width + 8 }
    $rowEnd = 6 + $g.MeasureString($title, $font).Width
    $noteRow = 0; $noteX = 0
    if ($note) {
        $nw = $g.MeasureString($note, $font).Width
        if ($rowEnd + 12 + $nw -le $w - $right) { $noteX = $rowEnd + 12 } else { $noteRow = 1; $noteX = 6 }
        $rowEnd = $noteX + $nw
    }
    $legendRow = $noteRow
    if ($rowEnd + 12 + $legendW -gt $w - $right) { $legendRow++ }
    $top += $lineH * $legendRow
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
    $fmt = if ($xspan -gt 3 * 86400) { "MM/dd" } else { "MM/dd HH:mm" }
    $xlab = {
        param([double]$v)
        if ($useGames) { return (Format-Int (X-Unscale $v $useLog)) }
        return (From-Unix $v).ToString($fmt)
    }
    $g.DrawString((& $xlab $xmin), $font, $gray, $left, $h - $bottom + 4)
    $g.DrawString((& $xlab ($xmin + $xspan / 2)), $font, $gray, $left + $pw / 2 - 30, $h - $bottom + 4)
    $lastLab = & $xlab $xmax
    $g.DrawString($lastLab, $font, $gray, $w - $right - $g.MeasureString($lastLab, $font).Width, $h - $bottom + 4)
    $lx = $w - $right - $legendW
    $ly = 3 + $lineH * $legendRow
    foreach ($s in $shown) {
        $brush = New-Object System.Drawing.SolidBrush($s.color)
        # 線の太さ。強さの目盛りだけ太くして主役にする（古い系列に width が無くても既定の 2 で描く）
        $lw = if ($null -ne $s.width) { [single]$s.width } else { [single]2 }
        $rp = New-Object System.Drawing.Pen($s.color, $lw)
        if ($s.dash) { $rp.DashStyle = [System.Drawing.Drawing2D.DashStyle]::Dash }
        $pts = New-Object System.Collections.ArrayList
        $last = $null
        # 定期観測の系列（gap）は、点の間隔の中央値の 3 倍（最低 15 分）より空いたところで線を切る。
        # 停止の区間を直線でつなぐと、その間も同じ値で動いていたように見えるため
        $gapSec = [double]::MaxValue
        if ($s.gap -and $s.pts.Count -ge 3) {
            $d = @(for ($k = 1; $k -lt $s.pts.Count; $k++) { ($s.pts[$k].t - $s.pts[$k - 1].t).TotalSeconds }) | Sort-Object
            $gapSec = [Math]::Max(900.0, 3 * $d[[int][Math]::Floor($d.Count / 2)])
        }
        $prevT = $null
        foreach ($p in $s.pts) {
            if ($p.t -lt $t0) { continue }
            $xv = X-Scale (Pt-XVal $p $useGames) $useLog
            if ($null -eq $xv) { continue }
            if ($null -ne $prevT -and ($p.t - $prevT).TotalSeconds -gt $gapSec) {
                if ($pts.Count -ge 2) { $g.DrawLines($rp, [System.Drawing.PointF[]]$pts.ToArray()) }
                elseif ($pts.Count -eq 1) { $g.FillEllipse($brush, $pts[0].X - 2, $pts[0].Y - 2, 4, 4) }
                $pts.Clear()
            }
            $prevT = $p.t
            $x = $left + $pw * (($xv - $xmin) / $xspan)
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
        $g.FillRectangle($brush, [single]$lx, [single]($ly + 3), 10, 10)
        $g.DrawString($s.name, $font, $black, [single]($lx + 12), [single]$ly)
        $lx += 22 + $g.MeasureString($s.name, $font).Width + 8
    }
    if ($note) { $g.DrawString($note, $font, $gray, [single]$noteX, [single](3 + $lineH * $noteRow)) }
}

function Get-Metrics([string]$run) {
    if ($script:Data.ContainsKey($run) -and $null -ne $script:Data[$run].metrics) { return @($script:Data[$run].metrics) }
    return @()
}
function Build-Series([string]$tab) {
    $series = @()
    $note = ""; $all = $false; $xg = $false
    $yfmt = "{0:N0}"; $zero = $true; $title = $tab
    $sel = Selected-Run
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
            # 既定は「強さの目盛り」1 本だけ。相手ごとの線は 1 点 200 局で幅が広く、参照の入れ替えと
            # じゃんけんで上下するので、並べると下がっていないのに下がって見える（2026-09-19 のユーザーの
            # 「Elo さがってませんか？線がいっぱいあってよくわからない」）。内訳は「内訳を出す」で足す
            $detail = Show-Elo-Detail
            $title = if ($detail) { "強さの推移（Elo。縦線は 95% 区間）" } else { "強さの推移（Elo の目盛り。縦線は 95% 区間）" }
            $all = $true; $xg = $true
            $i = 0
            $haveRating = $false
            $trendFit = $null
            foreach ($r in $Runs) {
                if ($script:Data.ContainsKey($r)) {
                    $rt = $script:Data[$r].rating
                    $rs = Build-Rating-Series ($r + " 強さの目盛り") (Run-Color $r $i) $rt
                    if ($rs.pts.Count -gt 0) {
                        $series += $rs; $haveRating = $true
                        # 最近の伸びをそのまま延ばした目安の線。点がこれに乗っていれば頭打ちではない。
                        # 全部の点に当てはめた curve_fit は、学習の初めの伸び方が違うぶん形が合わない
                        # （実データで残差 88、最新の点が +62 上に出て「加速している」と誤読させる）ので使わない
                        $tf = $rt.curve_fit_recent
                        if ($null -eq $tf) { $tf = $rt.curve_fit }
                        $ts = Build-Trend-Series ($r + " 目安（2 倍あたり）") ([System.Drawing.Color]::FromArgb(130, (Run-Color $r $i))) $tf $rt.points
                        if ($ts.pts.Count -gt 0) { $series += $ts; if ($null -eq $trendFit) { $trendFit = $tf } }
                    }
                }
                $i++
            }
            if ($detail -or -not $haveRating) {
                $i = 0
                foreach ($r in $Runs) {
                    $s = New-Series ($r + " 基準比") ([System.Drawing.Color]::FromArgb(150, (Run-Color $r $i))) $true
                    if ($script:Data.ContainsKey($r)) {
                        foreach ($a in @($script:Data[$r].anchor)) {
                            $lo = Ci-Val $a.ci95 0
                            $hi = Ci-Val $a.ci95 1
                            Add-Pt $s (From-Unix $a.t) ([double]$a.elo) $lo $hi ("基準比 step " + (Format-Int $a.step)) $a.games_at
                        }
                    }
                    $series += $s
                    # 鎖は同じ重みを基準比と別の方法で測った補助の値。同じ色で並ぶと基準比が下がったように見えるので薄くして名前を付ける
                    $c = New-Series ($r + " 鎖") ([System.Drawing.Color]::FromArgb(110, (Run-Color $r $i))) $false $true
                    if ($script:Data.ContainsKey($r)) {
                        foreach ($e in @($script:Data[$r].evals)) {
                            if ($null -eq $e.cumulative) { continue }
                            Add-Pt $c (From-Unix $e.time) ([double]$e.cumulative) $null $null "鎖" $e.games_at
                        }
                    }
                    $series += $c
                    # 固定の参照（[auto] reference_ckpts）との差。run をまたいで同じ相手なので絶対の物差しになる（docs/restart-plan.md §3 M4）
                    if ($script:Data.ContainsKey($r)) {
                        # 参照ごとに RefPalette の色を割り当てる（run の色で描くと基準比と同じ青になって見分けが付かない）。run が 2 つ目以降なら破線
                        $byRef = @{}
                        foreach ($e in @($script:Data[$r].reference)) {
                            if ($null -eq $e.elo) { continue }
                            if (-not $byRef.ContainsKey([string]$e.ref)) { $byRef[[string]$e.ref] = (New-Object System.Collections.ArrayList) }
                            [void]$byRef[[string]$e.ref].Add($e)
                        }
                        $j = 0
                        foreach ($k in ($byRef.Keys | Sort-Object)) {
                            $rfs = New-Series ($r + " 対 " + $k) $script:RefPalette[$j % $script:RefPalette.Length] $true ($i -gt 0)
                            foreach ($e in $byRef[$k]) { Add-Pt $rfs (From-Unix $e.t) ([double]$e.elo) (Ci-Val $e.ci95 0) (Ci-Val $e.ci95 1) ("対 " + $k + " step " + (Format-Int $e.step)) $e.games_at }
                            $series += $rfs
                            $j++
                        }
                    }
                    $i++
                }
            }
            $note = Format-Elo-Note $detail $haveRating (Use-GamesAxis) $trendFit (Use-LogAxis)
        }
        "対外対局" {
            $title = "外部エンジン（fuseki_usi_server.py = 方策ネット＋やねうら王/水匠5）との勝率"
            $yfmt = "{0:P0}"; $all = $true; $xg = $true
            $i = 0
            foreach ($r in $Runs) {
                $s = New-Series $r (Run-Color $r $i) $true
                if ($script:Data.ContainsKey($r)) {
                    foreach ($m in @($script:Data[$r].matches)) { if ($null -ne $m.winrate) { Add-Pt $s (From-Unix $m.time) ([double]$m.winrate) $null $null ("{0} 局 {1}" -f $m.n, $m.go) $m.games_at } }
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
        "学習目標" {
            # 学習目標・v41・探索値と実際の結果の差（得点の尺度、学習 1 回ぶんの平均。docs/method-evidence.md §4.4 (a)）
            $title = "学習目標と実際の結果の差（$sel、0 に近いほど偏りなし）"
            $yfmt = "{0:N3}"
            $defs = [ordered]@{
                "target_minus_z"       = "布石の目標 − 結果（先手から）"
                "v41_minus_z"          = "v41 − 結果（先手から）"
                "rootq_minus_z_fuseki" = "探索値 − 結果（布石、手番側）"
                "rootq_minus_z_normal" = "探索値 − 結果（本将棋、手番側）"
            }
            $i = 0
            foreach ($k in $defs.Keys) {
                $s = New-Series $defs[$k] $script:Palette[$i]
                foreach ($m in (Get-Metrics $sel)) {
                    if ($null -ne $m.train -and $null -ne $m.train.target -and $null -ne $m.train.target.$k) { Add-Pt $s (From-Unix $m.t) ([double]$m.train.target.$k) }
                }
                $series += $s; $i++
            }
            $s = New-Series "引き分け: 目標 − 実際（布石）" $script:Palette[4]
            foreach ($m in (Get-Metrics $sel)) {
                if ($null -eq $m.train -or $null -eq $m.train.target) { continue }
                $g = $m.train.target
                if ($null -ne $g.draw_target -and $null -ne $g.draw_actual) { Add-Pt $s (From-Unix $m.t) ([double]$g.draw_target - [double]$g.draw_actual) }
            }
            $series += $s
            $note = "0 から離れた線は、学習目標が実際の結果からずれている。2026-09-15 の集計では布石の目標 約 −0.02、引き分け 約 +0.32"
        }
        "較正" {
            # 同じネットの自己対局の較正（calib.jsonl、1 時間ごと。docs/decisions.md 2026-09-17）。ECE は予測と実際の差を区間の局面数で平均した値
            $title = "探索値の較正 ECE（$sel、0 に近いほど予測の勝率が実際に合う）"
            $yfmt = "{0:N3}"
            $defs = [ordered]@{
                "start"        = "3 手目（両玉の直後）"
                "fuseki_sente" = "布石・先手の番"
                "fuseki_gote"  = "布石・後手の番"
                "normal_sente" = "本将棋・先手の番"
                "normal_gote"  = "本将棋・後手の番"
            }
            $rows = @()
            if ($script:Data.ContainsKey($sel) -and $null -ne $script:Data[$sel].PSObject.Properties["calib"]) { $rows = @($script:Data[$sel].calib) }
            $i = 0
            foreach ($k in $defs.Keys) {
                $s = New-Series $defs[$k] $script:Palette[$i]
                foreach ($r in $rows) {
                    $g = $r.groups.$k
                    if ($null -ne $g -and $g.n -gt 0) { Add-Pt $s (From-Unix $r.t) ([double]$g.ece) }
                }
                $series += $s; $i++
            }
            $note = "曲線と偏りは bin/libra calib（lx は --run lx）"
            if ($rows.Count -gt 0) {
                $a = $rows[-1].groups.all
                if ($null -ne $a -and $a.n -gt 0) { $note = "最新 step {0}: 全体 ECE {1:N3}・偏り（予測 − 実際）{2:+0.000;-0.000}。{3}" -f (Format-Int $rows[-1].step), [double]$a.ece, [double]$a.bias, $note }
            }
        }
        "処理時間" {
            # 学習器のループの実時間の内訳（metrics.jsonl の timing、5 分の窓。libra-league/libra_league/looptime.py）
            $title = "処理時間の内訳（$sel、5 分ごとの実時間に占める割合）"
            $yfmt = "{0:P0}"
            $defs = [ordered]@{
                "評価 GPU"     = @("sp_eval")                                                             # 自己対局のネットの評価の待ち
                "探索 CPU"     = @("sp_collect", "sp_proof", "sp_apply")                                  # 自己対局の収集・証明・反映
                "リーグ"       = @("lg_collect", "lg_eval", "lg_proof", "lg_apply", "lg_other")           # 対 lx
                "学習"         = @("train_sample", "train_step", "train_publish")                         # バッチ・ステップ・重みの反映
                "保存ほか"     = @("replay", "ingest", "housekeeping", "status", "checkpoint", "other")
            }
            $i = 0; $last = $null
            foreach ($name in $defs.Keys) {
                $s = New-Series $name $script:Palette[$i]
                foreach ($m in (Get-Metrics $sel)) {
                    if ($null -eq $m.PSObject.Properties["timing"] -or $null -eq $m.timing -or [double]$m.timing.window_s -le 0) { continue }
                    $v = 0.0
                    foreach ($k in $defs[$name]) { if ($null -ne $m.timing.sec.$k) { $v += [double]$m.timing.sec.$k } }
                    Add-Pt $s (From-Unix $m.t) ($v / [double]$m.timing.window_s)
                    $last = $m.timing
                }
                $series += $s; $i++
            }
            $note = "run の起動し直しの後から記録。内訳は bin/libra status の timing 行"
            if ($null -ne $last) {
                $r = [double]$last.rounds; $sec = $last.sec
                $parts = ""
                if ($r -gt 0) { $parts = "1 ラウンド 収集 {0:N1}・評価 {1:N1}・証明 {2:N1}・反映 {3:N1} ms " -f ($sec.sp_collect / $r * 1000), ($sec.sp_eval / $r * 1000), ($sec.sp_proof / $r * 1000), ($sec.sp_apply / $r * 1000) }
                if ([double]$last.train_steps -gt 0) { $parts += "学習 {0:N0} ms/step" -f ($sec.train_step / [double]$last.train_steps * 1000) }
                $note = "最新 " + $parts + "（評価は同じ GPU の別 run の待ちを含む）"
            }
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
    if (@("局/日", "学習", "学習目標", "較正", "処理時間", "終局内訳", "手数") -contains $tab) { foreach ($s in $series) { $s.gap = $true } }
    return @{ title = $title; series = $series; yfmt = $yfmt; zero = $zero; note = $note; all = $all;
              xgames = ($xg -and (Use-GamesAxis)); xlog = ($xg -and (Use-GamesAxis) -and (Use-LogAxis)) }
}

# グラフは 1 組だけ作り、選んでいる run のタブの下半分（chartSlot）に Move-Charts が付け替える
$chartHost = New-Object System.Windows.Forms.TableLayoutPanel
$chartHost.Dock = "Fill"; $chartHost.ColumnCount = 1; $chartHost.RowCount = 2
$chartHost.Margin = New-Object System.Windows.Forms.Padding(0)
[void]$chartHost.RowStyles.Add((New-Object System.Windows.Forms.RowStyle("AutoSize")))
[void]$chartHost.RowStyles.Add((New-Object System.Windows.Forms.RowStyle("Percent", 100)))
$cbar = New-Object System.Windows.Forms.FlowLayoutPanel
$cbar.Dock = "Fill"; $cbar.AutoSize = $true; $cbar.WrapContents = $true
$cbar.Controls.Add((New-Label "グラフの期間" 2))
$cmbRange = New-Object System.Windows.Forms.ComboBox
$cmbRange.DropDownStyle = "DropDownList"; $cmbRange.Width = 90
[void]$cmbRange.Items.AddRange(@("6 時間", "24 時間", "7 日", "全部"))
$cmbRange.SelectedIndex = 1
$cmbRange.Margin = New-Object System.Windows.Forms.Padding(0, 4, 8, 0)
$cmbRange.Add_SelectedIndexChanged({ $tabs.Invalidate($true) })
$cbar.Controls.Add($cmbRange)
# Elo と 対外対局 の横軸。既定は総局数（伸びを決めるのは時間ではなく局数。時間だと止めた間も寝て見える）
$cbar.Controls.Add((New-Label "Elo の横軸" 2))
$script:cmbAxis = New-Object System.Windows.Forms.ComboBox
$script:cmbAxis.DropDownStyle = "DropDownList"; $script:cmbAxis.Width = 118
[void]$script:cmbAxis.Items.AddRange(@("総局数", "時間", "総局数（対数）"))
$script:cmbAxis.SelectedIndex = 0
$script:cmbAxis.Margin = New-Object System.Windows.Forms.Padding(0, 4, 8, 0)
$script:cmbAxis.Add_SelectedIndexChanged({ $tabs.Invalidate($true) })
$cbar.Controls.Add($script:cmbAxis)
# Elo のグラフは既定で「強さの目盛り」1 本。相手ごとの線（基準比・鏡・対 …）はここで足す
$script:chkEloDetail = New-Object System.Windows.Forms.CheckBox
$script:chkEloDetail.Text = "Elo の内訳を出す"; $script:chkEloDetail.AutoSize = $true
$script:chkEloDetail.Margin = New-Object System.Windows.Forms.Padding(2, 6, 8, 0)
$script:chkEloDetail.Add_CheckedChanged({ $tabs.Invalidate($true) })
$cbar.Controls.Add($script:chkEloDetail)
$lblChartNote = New-Label "学習・学習目標・較正・処理時間・終局内訳・手数はこのタブの run、ほかは両方" 4
$lblChartNote.ForeColor = [System.Drawing.Color]::DimGray
$cbar.Controls.Add($lblChartNote)
$chartHost.Controls.Add($cbar, 0, 0)
$tabs = New-Object System.Windows.Forms.TabControl
$tabs.Dock = "Fill"
$script:TabNames = @("局/日", "Elo", "対外対局", "学習", "学習目標", "較正", "処理時間", "終局内訳", "手数")
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
        Draw-Chart $e.Graphics $s.ClientSize.Width $s.ClientSize.Height $b.title $b.series $b.yfmt $b.zero $b.note ([bool]$b.all) ([bool]$b.xgames) ([bool]$b.xlog)
    })
    $panel.Add_Resize({ param($s, $e) $s.Invalidate() })
    $page.Controls.Add($panel)
    [void]$tabs.TabPages.Add($page)
}
# 「ログ」タブ: 選んでいる run の log.txt の末尾（縦長でグラフの場所を取らないように、表の下からここへ移した）
$logPage = New-Object System.Windows.Forms.TabPage
$logPage.Text = "ログ"
$chartLog = New-Object System.Windows.Forms.TextBox
$chartLog.Multiline = $true; $chartLog.ReadOnly = $true; $chartLog.ScrollBars = "Both"; $chartLog.WordWrap = $false
$chartLog.Dock = "Fill"; $chartLog.Font = New-Object System.Drawing.Font("Consolas", 8.5); $chartLog.BackColor = [System.Drawing.Color]::White
$logPage.Controls.Add($chartLog)
[void]$tabs.TabPages.Add($logPage)
function Show-RunLog([string]$run) {
    if ($run -ne (Selected-Run) -or -not $script:Ui.ContainsKey($run)) { return }
    $t = [string]$script:Ui[$run].logText
    if ($chartLog.Text -eq $t) { return }
    $chartLog.Text = $t
    $chartLog.SelectionStart = $chartLog.Text.Length
    $chartLog.ScrollToCaret()
}
# ---- クラウド タブ（vast.ai の自己対局ワーカー。bin/libra-vast。docs/runbook.md） ----
function New-Num([decimal]$min, [decimal]$max, [decimal]$val, [decimal]$inc, [int]$dec, [int]$w = 64) {
    $n = New-Object System.Windows.Forms.NumericUpDown
    $n.Minimum = $min; $n.Maximum = $max; $n.DecimalPlaces = $dec; $n.Increment = $inc; $n.Value = $val; $n.Width = $w
    $n.Margin = New-Object System.Windows.Forms.Padding(0, 4, 8, 0)
    return $n
}
# 縦に 操作 / 状態の表 / launcher.log の末尾 を並べる
$vroot = New-Object System.Windows.Forms.TableLayoutPanel
$vroot.Dock = "Fill"; $vroot.ColumnCount = 1; $vroot.RowCount = 3
$vroot.BackColor = [System.Drawing.Color]::White
$vroot.Padding = New-Object System.Windows.Forms.Padding(6, 4, 6, 2)
[void]$vroot.RowStyles.Add((New-Object System.Windows.Forms.RowStyle("AutoSize")))
[void]$vroot.RowStyles.Add((New-Object System.Windows.Forms.RowStyle("AutoSize")))
[void]$vroot.RowStyles.Add((New-Object System.Windows.Forms.RowStyle("Percent", 100)))
$vbar = New-Object System.Windows.Forms.FlowLayoutPanel
$vbar.Dock = "Fill"; $vbar.AutoSize = $true; $vbar.WrapContents = $true
$vbar.Controls.Add((New-Label "GPU" 4))
$cmbGpu = New-Object System.Windows.Forms.ComboBox
$cmbGpu.DropDownStyle = "DropDownList"; $cmbGpu.Width = 100
[void]$cmbGpu.Items.AddRange(@("RTX 5070 Ti", "RTX 5080", "RTX 4090", "RTX 5090", "RTX 3090"))
$cmbGpu.SelectedIndex = 0
$cmbGpu.Margin = New-Object System.Windows.Forms.Padding(0, 4, 8, 0)
$vbar.Controls.Add($cmbGpu)
$vbar.Controls.Add((New-Label "借り方" 4))
$cmbRent = New-Object System.Windows.Forms.ComboBox
$cmbRent.DropDownStyle = "DropDownList"; $cmbRent.Width = 90
[void]$cmbRent.Items.AddRange(@("入札", "on-demand"))
$cmbRent.SelectedIndex = 0
$cmbRent.Margin = New-Object System.Windows.Forms.Padding(0, 4, 8, 0)
$vbar.Controls.Add($cmbRent)
$vbar.Controls.Add((New-Label '上限 $/h' 4)); $numDph = New-Num 0.05 2.00 0.28 0.01 2; $vbar.Controls.Add($numDph)
$vbar.Controls.Add((New-Label "時間" 4)); $numHours = New-Num 0.5 24 3 0.5 1; $vbar.Controls.Add($numHours)
$vbar.Controls.Add((New-Label "信頼度の下限" 4)); $numRel = New-Num 0.90 0.99 0.94 0.01 2; $vbar.Controls.Add($numRel)
$vbar.Controls.Add((New-Label "CPU GHz の下限" 4)); $numGhz = New-Num 0 9 4.4 0.1 1; $vbar.Controls.Add($numGhz)
$vbar.Controls.Add((New-Label "コア数の下限" 4)); $numCores = New-Num 1 128 16 1 0 52; $vbar.Controls.Add($numCores)
$vbar.Controls.Add((New-Label '転送料の上限 $/GB' 4)); $numInet = New-Num 0 0.2 0.02 0.005 3; $vbar.Controls.Add($numInet)
# 「候補を見る」の「1 つ緩めれば借りられる」の key（libra_cloud.bench.offer_rejects）と入力欄の対応
$script:VastNums = @{ max_dph = $numDph; min_rel = $numRel; min_cpu_ghz = $numGhz; min_cores = $numCores; max_inet_cost = $numInet }
$bVastStart = New-Button "起動" { Start-Vast } 56
$bVastStop = New-Button "停止" { Stop-Vast } 56
$bVastOffers = New-Button "候補を見る" { Show-VastOffers } 90
$bVastAccount = New-Button "残高を更新" { $script:NextVastAccount = [datetime]::MinValue; $script:NextVast = [datetime]::MinValue } 90
$bVastCleanup = New-Button "後始末" { Cleanup-Vast } 70
foreach ($b in @($bVastStart, $bVastStop, $bVastOffers, $bVastAccount, $bVastCleanup)) { $vbar.Controls.Add($b) }
$script:Tip.SetToolTip($bVastStart, "GPU を借りて自己対局ワーカーを起動し、ls に局を足す（準備に 5〜15 分。時間が来たら残りの局を取ってインスタンスを消す）")
$script:Tip.SetToolTip($bVastStop, "ワーカーを止めて残りの局を取り、インスタンスを消す")
$script:Tip.SetToolTip($bVastOffers, "検索したオファーを安い順に、落ちた条件と「1 つ緩めれば借りられる」値を付けて出す（借りない）")
$script:Tip.SetToolTip($numCores, "ホストの実効コア数の下限（自己対局のスレッドは 12 まで。少ないと局/日が落ちる）")
$script:Tip.SetToolTip($numInet, "転送料（上り・下りの高い方）の上限。重みと局の転送で 1 日に数 GB〜数十 GB 流れる")
$script:Tip.SetToolTip($bVastCleanup, "libra- で始まるラベルのインスタンスをすべて消す（起動の途中で落ちて残ったとき）")
$script:Tip.SetToolTip($numGhz, "CPU が遅いホストでは探索が律速して GPU が遊ぶ（5070 Ti で Xeon 2.4 GHz 35 万局/日、Ryzen 4.5 GHz 48 万局/日）")
$vroot.Controls.Add($vbar, 0, 0)
$script:VastKeys = @(
    @("phase", "状態"), @("session", "セッション"), @("gpu", "GPU / ホスト"), @("time", "借りた時間 / 残り"), @("cost", "費用（見積もり）"),
    @("bridge", "回収（ブリッジ）"), @("learner", "取り込み（ls）"), @("verify", "検査"), @("waste", "打ち切り（これまで）"),
    @("credit", "残高"), @("instances", "借りているインスタンス")
)
$vgrid = New-Object System.Windows.Forms.TableLayoutPanel
$vgrid.Dock = "Fill"; $vgrid.ColumnCount = 2; $vgrid.AutoSize = $true
$vgrid.Margin = New-Object System.Windows.Forms.Padding(0, 6, 0, 6)
[void]$vgrid.ColumnStyles.Add((New-Object System.Windows.Forms.ColumnStyle("Absolute", 150)))
[void]$vgrid.ColumnStyles.Add((New-Object System.Windows.Forms.ColumnStyle("Percent", 100)))
$script:VastVals = @{}
$vrow = 0
foreach ($k in $script:VastKeys) {
    $lk = New-Object System.Windows.Forms.Label
    $lk.Text = $k[1]; $lk.AutoSize = $true; $lk.ForeColor = [System.Drawing.Color]::DimGray
    $lk.Margin = New-Object System.Windows.Forms.Padding(2, 1, 2, 1)
    $lv = New-Object System.Windows.Forms.Label
    $lv.Text = "-"; $lv.AutoSize = $true
    $lv.Margin = New-Object System.Windows.Forms.Padding(2, 1, 2, 1)
    $vgrid.Controls.Add($lk, 0, $vrow); $vgrid.Controls.Add($lv, 1, $vrow)
    $script:VastVals[$k[0]] = $lv
    $vrow++
}
$vroot.Controls.Add($vgrid, 0, 1)
$vlog = New-Object System.Windows.Forms.TextBox
$vlog.Multiline = $true; $vlog.ReadOnly = $true; $vlog.ScrollBars = "Both"; $vlog.WordWrap = $false
$vlog.Dock = "Fill"; $vlog.Font = New-Object System.Drawing.Font("Consolas", 8.5); $vlog.BackColor = [System.Drawing.Color]::White
$vroot.Controls.Add($vlog, 0, 2)
$vroot.Add_Resize({ Fit-Labels $vroot.ClientSize.Width 150 $script:VastVals.Values })
$vpage = New-TopPage "cloud"
$vpage.Controls.Add($vroot)
$chartHost.Controls.Add($tabs, 0, 1)
$tabs.Add_SelectedIndexChanged({ if ($null -ne $tabs.SelectedTab) { $tabs.SelectedTab.Controls[0].Invalidate() } })

# ---- クラウド履歴 タブ（bin/libra-vast history。セッションごとの費用対効果と今月の合計） ----
function Fmt-Num($v, [string]$f = "N0", [string]$prefix = "") {
    if ($null -eq $v) { return "-" }
    return $prefix + ([double]$v).ToString($f)
}
$hroot = New-Object System.Windows.Forms.TableLayoutPanel
$hroot.Dock = "Fill"; $hroot.ColumnCount = 1; $hroot.RowCount = 5
$hroot.BackColor = [System.Drawing.Color]::White
$hroot.Padding = New-Object System.Windows.Forms.Padding(6, 4, 6, 2)
[void]$hroot.RowStyles.Add((New-Object System.Windows.Forms.RowStyle("AutoSize")))
[void]$hroot.RowStyles.Add((New-Object System.Windows.Forms.RowStyle("AutoSize")))
[void]$hroot.RowStyles.Add((New-Object System.Windows.Forms.RowStyle("Percent", 35)))
[void]$hroot.RowStyles.Add((New-Object System.Windows.Forms.RowStyle("Percent", 65)))
[void]$hroot.RowStyles.Add((New-Object System.Windows.Forms.RowStyle("Absolute", 150)))
$hbar = New-Object System.Windows.Forms.FlowLayoutPanel
$hbar.Dock = "Fill"; $hbar.AutoSize = $true; $hbar.WrapContents = $true
$hbar.Controls.Add((New-Button "更新" { $script:NextVastHist = [datetime]::MinValue } 56))
$lblHistAt = New-Label "" 8
$lblHistAt.ForeColor = [System.Drawing.Color]::DimGray
$hbar.Controls.Add($lblHistAt)
$hroot.Controls.Add($hbar, 0, 0)
$lblHistSum = New-Object System.Windows.Forms.Label
$lblHistSum.AutoSize = $true; $lblHistSum.Text = "（読み込み中）"
$lblHistSum.Margin = New-Object System.Windows.Forms.Padding(2, 4, 2, 6)
$hroot.Controls.Add($lblHistSum, 0, 1)
$hroot.Add_Resize({ Fit-Labels $hroot.ClientSize.Width 0 @($lblHistSum) })
$lvHist = New-Object System.Windows.Forms.ListView
$lvHist.View = "Details"; $lvHist.FullRowSelect = $true; $lvHist.GridLines = $true; $lvHist.HideSelection = $false; $lvHist.MultiSelect = $false
$lvHist.Dock = "Fill"
foreach ($c in @(@("開始", 76, "Left"), @("GPU", 76, "Left"), @('$/100万局', 72, "Right"), @("有効局", 66, "Right"), @("費用", 50, "Right"),
                 @("局/日", 68, "Right"), @("借りた h", 58, "Right"), @('$/h', 50, "Right"), @("捨てた", 52, "Right"), @("打ち切りの損", 78, "Right"), @("結果", 170, "Left"), @("CPU・場所", 280, "Left"))) {
    $ch = New-Object System.Windows.Forms.ColumnHeader
    $ch.Text = $c[0]; $ch.Width = $c[1]; $ch.TextAlign = $c[2]
    [void]$lvHist.Columns.Add($ch)
}
$hroot.Controls.Add($lvHist, 0, 2)
$histChart = New-Object System.Windows.Forms.Panel
$histChart.Dock = "Fill"; $histChart.BackColor = [System.Drawing.Color]::White
$histChart.Add_Paint({ param($s, $e) Draw-HistChart $e.Graphics $s.ClientSize.Width $s.ClientSize.Height })
$histChart.Add_Resize({ param($s, $e) $s.Invalidate() })
$hroot.Controls.Add($histChart, 0, 3)
$txtHist = New-Object System.Windows.Forms.TextBox
$txtHist.Multiline = $true; $txtHist.ReadOnly = $true; $txtHist.ScrollBars = "Vertical"; $txtHist.WordWrap = $true
$txtHist.Dock = "Fill"; $txtHist.BackColor = [System.Drawing.Color]::White
$hroot.Controls.Add($txtHist, 0, 4)
$hpage = New-TopPage "history"
$hpage.Controls.Add($hroot)

$script:HistBars = @()   # 棒ごとの @(上端 y, 下端 y, セッション名)。クリックで一覧の行を選ぶ
function Draw-HistChart($g, [int]$w, [int]$h) {
    $g.SmoothingMode = "AntiAlias"
    $g.Clear([System.Drawing.Color]::White)
    $font = New-Object System.Drawing.Font("Yu Gothic UI", 8)
    $black = [System.Drawing.Brushes]::Black; $gray = [System.Drawing.Brushes]::Gray
    $g.DrawString("各回の 100 万局あたりの費用（有効局で割った値。濃い部分は回収局で割った値、薄い部分は捨てた局のぶん。点線は合計の平均）", $font, $black, 4, 3)
    $script:HistBars = @()
    if ($null -eq $script:VastHist) { $g.DrawString("（まだ読めていません）", $font, $gray, 6, 26); return }
    $rows = @(@($script:VastHist.sessions) | Where-Object { $null -ne $_.usd_per_1m })
    [array]::Reverse($rows)
    if ($rows.Count -eq 0) { $g.DrawString("（まだ局を回収した回がありません）", $font, $gray, 6, 26); return }
    $left = 118; $right = 60; $top = 26; $bottom = 22
    $avg = $script:VastHist.totals.usd_per_1m
    $maxv = 0.0
    foreach ($r in $rows) { $maxv = [Math]::Max($maxv, [double]$r.usd_per_1m) }
    if ($null -ne $avg) { $maxv = [Math]::Max($maxv, [double]$avg) }
    $maxv *= 1.05
    $pw = [Math]::Max(10, $w - $left - $right)
    $bh = [Math]::Min(28.0, ($h - $top - $bottom) / $rows.Count)
    $gpus = @($rows | ForEach-Object { [string]$_.gpu } | Sort-Object -Unique)
    $selName = if ($lvHist.SelectedItems.Count -gt 0) { $lvHist.SelectedItems[0].Tag.name } else { "" }
    for ($i = 0; $i -lt $rows.Count; $i++) {
        $r = $rows[$i]
        $y = $top + $i * $bh
        $ty0 = [single]($y + ($bh - 13) / 2)
        $col = $script:Palette[[Array]::IndexOf($gpus, [string]$r.gpu) % $script:Palette.Length]
        $label = "{0} {1}" -f $(if ($null -ne $r.started) { (From-Unix $r.started).ToString("MM/dd HH:mm") } else { "-" }), (([string]$r.gpu) -replace '^RTX ', '')
        $g.DrawString($label, $font, $black, 2, $ty0)
        $th = [single]([Math]::Max(3.0, $bh * 0.62)); $ty = [single]($y + ($bh - $th) / 2)
        $wNet = [single]($pw * [double]$r.usd_per_1m / $maxv)
        $wGross = if ($null -ne $r.usd_per_1m_gross) { [single]($pw * [double]$r.usd_per_1m_gross / $maxv) } else { $wNet }
        $g.FillRectangle((New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(80, $col))), [single]$left, $ty, $wNet, $th)
        $g.FillRectangle((New-Object System.Drawing.SolidBrush($col)), [single]$left, $ty, $wGross, $th)
        if ($r.name -eq $selName) { $g.DrawRectangle((New-Object System.Drawing.Pen([System.Drawing.Color]::Black, 1.5)), [single]($left - 2), [single]($ty - 2), [single]($wNet + 4), [single]($th + 4)) }
        $g.DrawString(('${0:N2}' -f [double]$r.usd_per_1m), $font, $black, [single]($left + $wNet + 4), $ty0)
        $script:HistBars += , @([double]$y, [double]($y + $bh), [string]$r.name)
    }
    if ($null -ne $avg) {
        $ax = [single]($left + $pw * [double]$avg / $maxv)
        $yEnd = [single]($top + $rows.Count * $bh)
        $pen = New-Object System.Drawing.Pen([System.Drawing.Color]::DimGray, 1)
        $pen.DashStyle = [System.Drawing.Drawing2D.DashStyle]::Dash
        $g.DrawLine($pen, $ax, [single]($top - 2), $ax, $yEnd)
        $g.DrawString(('平均 ${0:N2}' -f [double]$avg), $font, $gray, [single]([Math]::Max($left, $ax - 30)), [single]($yEnd + 3))
    }
}
$histChart.Add_MouseClick({
    param($s, $e)
    foreach ($b in $script:HistBars) {
        if ($e.Y -lt $b[0] -or $e.Y -ge $b[1]) { continue }
        foreach ($it in $lvHist.Items) { $it.Selected = ($it.Tag.name -eq $b[2]) }
        if ($lvHist.SelectedItems.Count -gt 0) { $lvHist.SelectedItems[0].EnsureVisible() }
    }
})
function Format-Vast-Session-Waste($r) {
    # 選んだ回が打ち切られた（または打ち切りの借り直しだった）ときだけ、その回の損を 1 行で出す
    $out = @()
    if ($r.lost) {
        $out += "打ち切り: 最後の回収から次が打ち始めるまで {0:N2} 時間{1}、この回で打てなかったのは約 {2:N0} 局" -f
                [double]$r.dark_h, $(if ([double]$r.unused_h -gt 0) { "、借り直せず予定の残り {0:N2} 時間も捨てた" -f [double]$r.unused_h } else { "" }), [double]$r.lost_games
    }
    if ($r.continues) {
        $out += "借り直し（{0} の打ち切りから）: 準備の {1:N2} 時間 {2} は打ち切りが無ければ払わずに済んだぶん" -f
                $r.continues, [double]$r.setup_h, (Fmt-Num $r.extra_setup_usd "N2" '$')
    }
    return ($out -join "`r`n")
}
function Update-HistDetail {
    if ($lvHist.SelectedItems.Count -eq 0) { $txtHist.Text = ""; $histChart.Invalidate(); return }
    $r = $lvHist.SelectedItems[0].Tag
    $lines = @()
    $lines += "{0}（run {1}）  {2}{3}" -f $r.name, $r.run, $r.phase, $(if ($r.stopped_by_user) { "（ユーザーが停止）" } else { "" })
    $lines += "予定 {0} 時間・上限 {1}/h" -f (Fmt-Num $r.hours "0.#"), (Fmt-Num $r.max_dph "N2" '$')
    if ($null -ne $r.instance) {
        $lines += "インスタンス #{0}: {1}、{2}（{3}、信頼度 {4}）、{5}/h" -f $r.instance, $r.gpu, $r.cpu, $r.where, (Fmt-Num $r.reliability "N3"), (Fmt-Num $r.dph "N3" '$')
        $lines += "借りた {0} 時間（ssh まで {1} 秒）、ブリッジ {2} 時間、費用 {3}（借りた時間 {4} ＋ 転送料 {5}{6}）" -f (Fmt-Num $r.rented_h "N2"), (Fmt-Num $r.t_ready_s), (Fmt-Num $r.bridge_h "N2"),
                  (Fmt-Num $r.total_usd "N2" '$'), (Fmt-Num $r.est_cost_usd "N2" '$'), (Fmt-Num $r.transfer_usd "N3" '$'), $(if ($r.transfer_estimated) { "、見積もり" } else { "" })
        $lines += "回収 {0} 局（{1} ファイル、弾いた {2}、エラー {3}、検査 {4} ms/局）、捨てた {5}、有効 {6}" -f (Fmt-Num $r.games), (Fmt-Num $r.files), (Fmt-Num $r.rejected_files), (Fmt-Num $r.errors), (Fmt-Num $r.verify_ms_per_game "N2"), (Fmt-Num $r.stale_games), (Fmt-Num $r.net_games)
        $lines += "局/日（ブリッジの時間で換算）{0}、100 万局あたり {1}（回収局で割ると {2}）" -f (Fmt-Num $r.games_per_day), (Fmt-Num $r.usd_per_1m "N2" '$'), (Fmt-Num $r.usd_per_1m_gross "N2" '$')
        $lines += Format-Vast-Session-Waste $r
    } else { $lines += "インスタンスを借りていません（費用なし）" }
    $lines = @($lines | Where-Object { $_ })
    $txtHist.Text = $lines -join "`r`n"
    $histChart.Invalidate()
}
$lvHist.Add_SelectedIndexChanged({ Update-HistDetail })
# 打ち切り（借りたホストを入札で止められる・落ちること）の損。libra-vast history --json の interrupts（libra_cloud/interrupts.py）。
# 打ち切られると (1) 借り直しの準備の 5〜15 分は課金されるのに局が出ず、(2) 止まってから次が打ち始めるまで局が 1 つも増えない。
# ここはその 2 つを「100 万局あたりいくら余計に払ったか」に直して出す（クラウド費用を効率化する判断の材料）
function Format-Vast-Waste($w) {
    if ($null -eq $w) { return "" }
    if ([int]$w.interruptions -eq 0) {
        return ("打ち切り 0 回（打った {0:N1} 時間）。入札で止められた回はまだありません" -f [double]$w.bridge_h)
    }
    $l1 = "打ち切り {0} 回（打った {1:N1} 時間、平均 {2:N1} 時間に 1 回。借り直し {3} 回、借り直せず {4} 回）: 余分な準備代 {5} ＋ 打てなかった {6:N1} 時間 = 約 {7:N0} 局" -f
          [int]$w.interruptions, [double]$w.bridge_h, [double]$w.h_per_loss, [int]$w.relaunches, [int]$w.not_relaunched,
          (Fmt-Num $w.extra_setup_usd "N2" '$'), [double]$w.lost_h, [double]$w.lost_games
    $l2 = if ($null -ne $w.usd_per_1m -and $null -ne $w.usd_per_1m_ideal) {
        "  100 万局あたり {0}（打ち切りが無ければ {1}、+{2:N1}%）" -f (Fmt-Num $w.usd_per_1m "N2" '$'), (Fmt-Num $w.usd_per_1m_ideal "N2" '$'), [double]$w.waste_pct
    } else { "" }
    $parts = @()
    foreach ($g in @($w.by_rent)) {
        $parts += "{0} {1} 回・打ち切り {2} 回{3}{4}" -f $g.name, [int]$g.sessions, [int]$g.lost,
                  $(if ($null -ne $g.h_per_loss) { "（{0:N1} h に 1 回）" -f [double]$g.h_per_loss } else { "" }),
                  $(if ($null -ne $g.usd_per_1m) { "・100 万局あたり " + (Fmt-Num $g.usd_per_1m "N2" '$') } else { "" })
    }
    if ($parts.Count -gt 0) { $l2 = ($l2 + "　借り方: " + ($parts -join "、")).TrimStart() }
    return (@($l1, $l2) | Where-Object { $_ }) -join "`r`n"
}
# 入札と on-demand のどちらが実際に安く済んでいるかの助言（どちらも 3 回以上あって 1 割以上違うときだけ出す）
function Format-Vast-Waste-Short($w) {
    # クラウド タブの 1 行。借りる前に「どれくらいの頻度で止められ、そのぶん 100 万局あたりいくら余計に払っているか」が見えるように
    if ($null -eq $w) { return "-" }
    if ([int]$w.interruptions -eq 0) { return ("0 回 / 打った {0:N1} 時間" -f [double]$w.bridge_h) }
    return ("{0} 回 / 打った {1:N1} 時間（平均 {2:N1} 時間に 1 回）。100 万局あたり {3} → {4}（+{5:N1}%）" -f
            [int]$w.interruptions, [double]$w.bridge_h, [double]$w.h_per_loss,
            (Fmt-Num $w.usd_per_1m_ideal "N2" '$'), (Fmt-Num $w.usd_per_1m "N2" '$'), [double]$w.waste_pct)
}
function Format-Vast-Rent-Advice($w) {
    if ($null -eq $w) { return "" }
    $bid = @($w.by_rent) | Where-Object { $_.name -eq "bid" -or $_.name -eq "入札" } | Select-Object -First 1
    $od = @($w.by_rent) | Where-Object { $_.name -eq "on-demand" } | Select-Object -First 1
    if ($null -eq $bid -or $null -eq $od) { return "" }
    if ([int]$bid.sessions -lt 3 -or [int]$od.sessions -lt 3) { return "" }
    if ($null -eq $bid.usd_per_1m -or $null -eq $od.usd_per_1m) { return "" }
    $b = [double]$bid.usd_per_1m; $o = [double]$od.usd_per_1m
    if ($b -gt $o * 1.1) { return ("入札は打ち切りのぶんを入れると on-demand より {0:N0}% 高くついています（借り方を on-demand にするか、入札の上乗せを増やす）" -f (($b / $o - 1) * 100)) }
    if ($o -gt $b * 1.1) { return ("入札のほうが on-demand より {0:N0}% 安く済んでいます（打ち切りを入れても得）" -f (($o / $b - 1) * 100)) }
    return "入札と on-demand で 100 万局あたりの費用は 1 割以内の差です"
}
function Update-HistPanel {
    $h = $script:VastHist
    if ($null -eq $h) { return }
    $t = $h.totals
    $mon = [datetime]::Now.ToString("yyyy-MM")
    $m = @($h.months) | Where-Object { $_.month -eq $mon } | Select-Object -First 1
    $mc = if ($null -ne $m) { [double]$(if ($null -ne $m.total_usd) { $m.total_usd } else { $m.est_cost_usd }) } else { 0.0 }
    $lblHistAt.Text = "読み込み " + $script:VastHistAt.ToString("HH:mm:ss") + "（5 分ごと、セッションの開始・終了時）"
    $adv = Format-Vast-Rent-Advice $h.interrupts
    $advice = if ($adv) { $adv + "`r`n" } else { "" }
    $lblHistSum.Text = ("合計 {0} 回（借りた {1} 回・{2} 時間）費用 {3}、有効 {4} 局（捨てた {5} 局）、100 万局あたり {6}" -f $t.sessions, $t.rented, (Fmt-Num $t.rented_h "N2"),
                        (Fmt-Num $(if ($null -ne $t.total_usd) { $t.total_usd } else { $t.est_cost_usd }) "N2" '$'), (Fmt-Num $t.net_games), (Fmt-Num $t.stale_games), (Fmt-Num $t.usd_per_1m "N2" '$')) + "`r`n" +
                       ("今月（{0}）: {1} ≈ {2:N0} 円 / 上限 {3:N0} 円（{4:P1}。1 ドル {5} 円で換算）" -f $mon, (Fmt-Num $mc "N2" '$'), ($mc * $UsdJpy), $BudgetJpy, ($mc * $UsdJpy / [Math]::Max(1, $BudgetJpy)), $UsdJpy) + "`r`n" +
                       (Format-Vast-Waste $h.interrupts) + "`r`n" +
                       $advice +
                       '有効局 = 回収局 − 学習側が古すぎて捨てた局。費用は借りた時間 × $/h ＋ 転送料（記録の無い古い回は見積もり）。局/日はブリッジの時間で換算'
    Set-TabState "history" ("今月 " + (Fmt-Num $mc "N2" '$')) ([System.Drawing.Color]::DimGray)
    $selName = if ($lvHist.SelectedItems.Count -gt 0) { $lvHist.SelectedItems[0].Tag.name } else { "" }
    $rows = @($h.sessions)
    [array]::Reverse($rows)
    $lvHist.BeginUpdate()
    $lvHist.Items.Clear()
    foreach ($r in $rows) {
        $it = New-Object System.Windows.Forms.ListViewItem($(if ($null -ne $r.started) { (From-Unix $r.started).ToString("MM/dd HH:mm") } else { "-" }))
        $cpu = if ($r.cpu) { "{0}（{1}）" -f ($r.cpu -replace '\s+\d+-Core Processor$', ''), $r.where } else { "-" }
        foreach ($txt in @((([string]$r.gpu) -replace '^RTX ', ''), (Fmt-Num $r.usd_per_1m "N2" '$'), (Fmt-Num $r.net_games), (Fmt-Num $(if ($null -ne $r.total_usd) { $r.total_usd } else { $r.est_cost_usd }) "N2" '$'),
                           (Fmt-Num $r.games_per_day), (Fmt-Num $r.rented_h "N2"), (Fmt-Num $r.dph "N3" '$'), (Fmt-Num $r.stale_games),
                           $(if ([double]$r.lost_games -gt 0) { "{0:N0} 局" -f [double]$r.lost_games } elseif ([double]$r.extra_setup_usd -gt 0) { Fmt-Num $r.extra_setup_usd "N2" '$' } else { "-" }),
                           ([string]$r.phase + $(if ($r.stopped_by_user) { "（停止）" } else { "" })), $cpu)) {
            [void]$it.SubItems.Add([string]$txt)
        }
        $it.Tag = $r
        if ($r.alive) { $it.ForeColor = [System.Drawing.Color]::ForestGreen }
        elseif ([string]$r.phase -like "異常終了*") { $it.ForeColor = [System.Drawing.Color]::Firebrick }
        elseif ($null -eq $r.est_cost_usd) { $it.ForeColor = [System.Drawing.Color]::Gray }
        [void]$lvHist.Items.Add($it)
        if ($r.name -eq $selName) { $it.Selected = $true }
    }
    $lvHist.EndUpdate()
    if ($lvHist.SelectedItems.Count -eq 0 -and $lvHist.Items.Count -gt 0) { $lvHist.Items[0].Selected = $true }
    Update-HistDetail
}
function Get-VastPast([string]$gpu, [double]$hours) {
    # 起動の確認に出す見込み（同じ GPU の過去の回の、借りた時間あたりの有効局と 100 万局あたりの費用）
    if ($null -eq $script:VastHist) { return "過去の実績: 履歴をまだ読めていません。" }
    $rs = @(@($script:VastHist.sessions) | Where-Object { $_.gpu -eq $gpu -and $null -ne $_.est_cost_usd -and [double]$_.rented_h -gt 0 -and [double]$_.net_games -gt 0 })
    if ($rs.Count -eq 0) { return "過去の $gpu の実績はありません。" }
    $g = 0.0; $hh = 0.0; $c = 0.0
    foreach ($r in $rs) { $g += [double]$r.net_games; $hh += [double]$r.rented_h; $c += [double]$(if ($null -ne $r.total_usd) { $r.total_usd } else { $r.est_cost_usd }) }
    return ('過去の {0} の実績（{1} 回）: 借りた 1 時間あたり 約 {2:N0} 局、100 万局あたり ${3:N2}。{4} 時間なら 約 {5:N0} 局の見込み。' -f $gpu, $rs.Count, ($g / $hh), ($c / $g * 1e6), $hours, ($g / $hh * $hours))
}

function Move-Charts {
    # 表示の前（ハンドルが無い間）は SelectedTab が null なので最初の run に置く
    $k = if ($null -ne $runTabs.SelectedTab) { [string]$runTabs.SelectedTab.Tag } else { $Runs[0] }
    if (-not $script:Ui.ContainsKey($k)) { return }
    $slot = $script:Ui[$k].chartSlot
    if (-not [object]::ReferenceEquals($chartHost.Parent, $slot)) { $slot.Controls.Add($chartHost) }
    Show-RunLog $k
    if ($null -ne $tabs.SelectedTab) { $tabs.SelectedTab.Controls[0].Invalidate() }
}
$runTabs.Add_SelectedIndexChanged({
    Move-Charts
    if ([string]$runTabs.SelectedTab.Tag -eq "history" -and ([datetime]::Now - $script:VastHistAt).TotalSeconds -gt 60) { $script:NextVastHist = [datetime]::MinValue }
})
function Save-Layout {
    # 次に開いたときに位置・大きさ・選んでいたタブを戻す（%LOCALAPPDATA%\LibraShogi\console-layout.json）
    try {
        $b = if ($form.WindowState -eq "Normal") { $form.Bounds } else { $form.RestoreBounds }
        if (-not (Test-Path $script:HistDir)) { [void](New-Item -ItemType Directory -Path $script:HistDir) }
        $o = @{ x = $b.X; y = $b.Y; w = $b.Width; h = $b.Height; top = $runTabs.SelectedIndex; chart = $tabs.SelectedIndex; range = $cmbRange.SelectedIndex; axis = $script:cmbAxis.SelectedIndex }
        ($o | ConvertTo-Json -Compress) | Set-Content -Path $script:LayoutFile -Encoding ASCII
    } catch {}
}
function Restore-Layout {
    try {
        if (-not (Test-Path $script:LayoutFile)) { return }
        $o = Get-Content $script:LayoutFile -Raw | ConvertFrom-Json
        $rect = New-Object System.Drawing.Rectangle([int]$o.x, [int]$o.y, [int]$o.w, [int]$o.h)
        $visible = @([System.Windows.Forms.Screen]::AllScreens | Where-Object { $_.WorkingArea.IntersectsWith($rect) }).Count -gt 0
        if ($visible -and $rect.Width -ge $form.MinimumSize.Width -and $rect.Height -ge $form.MinimumSize.Height) {
            $form.StartPosition = "Manual"
            $form.Bounds = $rect
        }
        if ($null -ne $o.top -and [int]$o.top -ge 0 -and [int]$o.top -lt $runTabs.TabCount) { $runTabs.SelectedIndex = [int]$o.top }
        if ($null -ne $o.chart -and [int]$o.chart -ge 0 -and [int]$o.chart -lt $tabs.TabCount) { $tabs.SelectedIndex = [int]$o.chart }
        if ($null -ne $o.range -and [int]$o.range -ge 0 -and [int]$o.range -lt $cmbRange.Items.Count) { $cmbRange.SelectedIndex = [int]$o.range }
        if ($null -ne $o.axis -and [int]$o.axis -ge 0 -and [int]$o.axis -lt $script:cmbAxis.Items.Count) { $script:cmbAxis.SelectedIndex = [int]$o.axis }
    } catch {}
}
if ($Tab) {
    foreach ($name in @($Tab.Split(",") | ForEach-Object { $_.Trim() })) {
        foreach ($pg in $runTabs.TabPages) { if ([string]$pg.Tag -eq $name -or $script:TabTitles[[string]$pg.Tag] -eq $name) { $runTabs.SelectedTab = $pg } }
        foreach ($pg in $tabs.TabPages) { if ($pg.Text -eq $name) { $tabs.SelectedTab = $pg } }
    }
}
Move-Charts

# ステータス行
$status = New-Object System.Windows.Forms.Label
$status.Dock = "Fill"; $status.AutoSize = $true
$status.Padding = New-Object System.Windows.Forms.Padding(6, 4, 6, 4)
$status.Text = "起動中…"
$root.Controls.Add($status, 0, 2)
$form.Add_Resize({ $status.MaximumSize = New-Object System.Drawing.Size([Math]::Max(200, $form.ClientSize.Width - 12), 0) })

# ---- 表示の更新 ----
function Update-Panel([string]$run, $obj) {
    $u = $script:Ui[$run]
    $v = $u.vals
    if ($null -eq $obj) {
        $v.process.Text = "取得失敗"; $v.process.ForeColor = [System.Drawing.Color]::Firebrick
        Set-TabState $run "● 取得失敗" ([System.Drawing.Color]::Firebrick)
        return
    }
    $running = ($obj.process -eq "running")
    $flags = @($obj.flags)
    # 操作は起動と停止だけ。停止処理中（STOP があり、まだ動いている）に起動を押すと、止まるのを待ってから起動する
    $stopping = ($running -and ($flags -contains "STOP"))
    $ptxt = if ($stopping) { "停止処理中" } elseif ($running) { "稼働中" } else { "停止" }
    if ($flags -contains "EVAL_NOW") { $ptxt += "（自己評価 予約）" }
    if ($flags -contains "MATCH_NOW") { $ptxt += "（対外対局 予約）" }
    if (-not $obj.exists) { $ptxt = "run なし（$($obj.root)）" }
    $v.process.Text = $ptxt
    $v.process.ForeColor = if (-not $running) { [System.Drawing.Color]::Firebrick } elseif ($stopping) { [System.Drawing.Color]::DarkOrange } else { [System.Drawing.Color]::ForestGreen }
    $v.process.Font = New-Object System.Drawing.Font($form.Font, [System.Drawing.FontStyle]::Bold)
    $tabText = if (-not $obj.exists) { "● run なし" } elseif ($stopping) { "● 停止処理中" } elseif ($running) { "● 稼働中" } else { "● 停止" }
    Set-TabState $run $tabText $v.process.ForeColor
    $u.startButton.Enabled = (-not $running -or $stopping)
    $u.stopButton.Enabled = ($running -and -not $stopping)
    if ($script:StartCheck.ContainsKey($run)) {
        if ($running -and -not $stopping) {
            $script:StartCheck.Remove($run)
            Set-Note $run "起動しました" $false 60
        } elseif ([datetime]::Now -gt $script:StartCheck[$run]) {
            $script:StartCheck.Remove($run)
            Set-Note $run "起動を確認できません。下のログ欄（start: の行）と stdout.log を見てください" $true 900
        }
    }
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
        $u.logText = ""
        Show-RunLog $run
        return
    }
    $stTime = [datetime]::ParseExact($st.time, "yyyy-MM-dd HH:mm:ss", $null)
    $v.updated.Text = "{0}（{1}）" -f $st.time, (Format-Ago $stTime)
    $v.updated.ForeColor = if ($running -and ([datetime]::Now - $stTime).TotalMinutes -gt 5) { [System.Drawing.Color]::Firebrick } else { [System.Drawing.Color]::Black }
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
        $v.exploiter.Text = "{0:P1}（{1} 局、相手 step {2}{3}{4}）" -f [double]$ex.winrate, (Format-Int $ex.games), (Format-Int $ex.main_step),
            $(if ($null -ne $ex.source_step -and $null -ne $ex.main_step) { "（本体より " + (Format-Int ([long]$ex.source_step - [long]$ex.main_step)) + " 古い）" } else { "" }),
            $(if ($null -ne $ex.refreshed_at) { "、作り直し " + (Format-Ago (From-Unix $ex.refreshed_at)) } else { "" })
    } else { $v.exploiter.Text = "-" }
    Set-RowVisible $u "exploiter" ($null -ne $st.exploiter)   # 本体には無い行なので隠して縦を詰める
    $rs = @($st.restarts)
    $v.restarts.Text = if ($rs.Count -gt 0) { "{0} 回（最終 {1}）" -f $rs.Count, $rs[$rs.Count - 1] } else { "0 回" }
    if ($null -ne $obj.log_tail) {
        $u.logText = (@($obj.log_tail) -join "`r`n")
        Show-RunLog $run
    }
    if ($script:Data.ContainsKey($run)) {
        $d = $script:Data[$run]
        $anc = @($d.anchor)
        $chain = @(@($d.evals) | Where-Object { $null -ne $_.cumulative })
        # 目盛り（Bradley-Terry）を先頭に出す。基準比・最強比は相手が動くので、伸びはまずこの行で見る
        $ratingText = Format-Elo-Rating $d.rating
        if ($anc.Count -gt 0) {
            $v.elo.Text = Format-Elo-Anchor $anc[$anc.Count - 1]
        } elseif ($chain.Count -gt 0) {
            $le = $chain[$chain.Count - 1]
            # eval の ci95 は a−b の区間。鎖は b−a を足すので、符号を反転して上下を入れ替える
            $aLo = Ci-Val $le.ci95 0; $aHi = Ci-Val $le.ci95 1
            $ci = if ($null -ne $aLo -and $null -ne $aHi) { " [{0:+0;-0;0}, {1:+0;-0;0}]" -f (-$aHi), (-$aLo) } else { "" }
            $v.elo.Text = "鎖 {0:+0.0;-0.0;0}（前回 {1:+0.0;-0.0;0}{2}、step {3}→{4}、{5}）" -f [double]$le.cumulative, (-[double]$le.elo), $ci, (Format-Int $le.step_a), (Format-Int $le.step_b), (Format-Ago (From-Unix $le.time))
        } else { $v.elo.Text = "（まだ無い。archive {0} 個）" -f @($d.archives).Count }
        # 最強比（[auto] best_games）: どの step が、そのときの最強だった step に勝ったか（docs/restart-plan.md §3 M2）
        $bs = @($d.best)
        if ($bs.Count -gt 0) { $v.elo.Text += "`r`n" + (Format-Elo-Best $bs[$bs.Count - 1]) }
        if ($null -ne $ratingText) { $v.elo.Text = $ratingText + "`r`n" + $v.elo.Text }
        # 固定の参照（[auto] reference_ckpts）: 基準比と違って相手が動かないので、世代をまたいで比べられる（同 §3 M4）
        $refText = Format-Elo-References $d.reference
        $v.reference.Text = if ($null -ne $refText) { $refText } else { "（まだ無い）" }
        $ms = @($d.matches)
        if ($ms.Count -gt 0) {
            $lm = $ms[$ms.Count - 1]
            $v.match.Text = "{0} / {1} 局（{2:P0}、{3}、{4}）" -f $lm.a_points, $lm.n, [double]$lm.winrate, $lm.go, (Format-Ago (From-Unix $lm.time))
        } else { $v.match.Text = "（まだ無い）" }
        # 自動計測が無効で結果も無い run（搾取者）では強さ・対外対局の行を隠す
        $autoCfgOn = ($null -ne $d.auto_cfg -and $d.auto_cfg.enabled)
        Set-RowVisible $u "elo" ($autoCfgOn -or $anc.Count -gt 0 -or $chain.Count -gt 0 -or $null -ne $ratingText)
        Set-RowVisible $u "reference" ($null -ne $refText)
        Set-RowVisible $u "match" ($autoCfgOn -or $ms.Count -gt 0)
        $au = $d.auto; $ac = $d.auto_cfg
        $q = if ($null -ne $au) { @($au.queue).Count } else { 0 }
        $v.auto.Text = Format-Auto-Status $ac $au $st.games_total $q
    }
    # クラウドのタブを見ている間はグラフにハンドルが無く SelectedTab が null
    if ($null -ne $tabs.SelectedTab) { $tabs.SelectedTab.Controls[0].Invalidate() }
}
function Update-StatusBar {
    # 0.5 秒ごとに書き直すので、操作の結果は $script:Notes に持って期限まで出し続ける
    $parts = @()
    $bad = $false
    foreach ($r in (@($Runs) + @("vast"))) {
        if ($script:Notes.ContainsKey($r)) {
            $n = $script:Notes[$r]
            if ([datetime]::Now -lt $n.until) { $parts += "$r : " + $n.text; if ($n.error) { $bad = $true } } else { $script:Notes.Remove($r) }
        }
        if ($script:Errors.ContainsKey($r)) { $parts += "$r : " + $script:Errors[$r]; $bad = $true }
    }
    if ($script:HistNote) { $parts += $script:HistNote; $bad = $true }
    if ($script:VastError) { $parts += "vast : " + $script:VastError; $bad = $true }
    if ($script:VastHistError) { $parts += "vast : " + $script:VastHistError; $bad = $true }
    if ($script:VastLeak) { $parts += "vast : セッションが動いていないのにインスタンスが残っています（課金中）。クラウド タブの「後始末」を押してください"; $bad = $true }
    $wait = [Math]::Max(0, ($script:NextFetch - [datetime]::Now).TotalSeconds)
    $pend = if ($script:Pending.Count -gt 0) { "  取得中…" } else { "" }
    $status.Text = ("次の更新まで {0:N0} 秒{1}   {2}" -f $wait, $pend, ($parts -join "   "))
    $status.ForeColor = if ($bad) { [System.Drawing.Color]::Firebrick } else { [System.Drawing.Color]::DimGray }
}

# ---- 操作 ----
function Set-Note([string]$run, [string]$text, [bool]$isError = $false, [int]$sec = 60) {
    $script:Notes[$run] = @{ text = $text; error = $isError; until = [datetime]::Now.AddSeconds($sec) }
}
function Confirm-Action([string]$msg) {
    $r = [System.Windows.Forms.MessageBox]::Show($form, $msg, "確認", "YesNo", "Question")
    return ($r -eq "Yes")
}
function Invoke-Run([string]$run, [string[]]$cmd) {
    try {
        $out = "$(Invoke-Libra $run $cmd)".Trim()
        $msg = if ($cmd[0] -eq "stop") { "停止を送りました（チェックポイントを書いて数十秒で止まります）" } else { ($cmd -join " ") + " → " + $out }
        Set-Note $run $msg $false 90
    } catch {
        Set-Note $run (($cmd -join " ") + " に失敗: " + $_.Exception.Message) $true 900
    }
    $script:NextFetch = [datetime]::Now.AddSeconds(1)
}
function Invoke-All([string[]]$cmd) {
    foreach ($r in $Runs) { Invoke-Run $r $cmd }
}
function Start-Run([string]$run) {
    # タスク スケジューラの「LibraShogi run [lx]」を起動する（失敗時の自動再起動を含めて bat と同じ経路）。
    # タスクが無ければ wsl.exe を非表示で直接起動する。
    # 停止処理中なら起動してよい（libra run が止まるのを待ってから起動する。残った STOP も消す）
    $obj = $script:Last[$run]
    if ($null -ne $obj -and $obj.process -eq "running" -and -not (@($obj.flags) -contains "STOP")) {
        Set-Note $run "既に稼働中です" $false 30
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
            Set-Note $run "起動中…（タスク「$task」。立ち上がりに 1 分ほど）" $false 300
        } else {
            Start-Process -FilePath "wsl.exe" -ArgumentList (Get-LibraArgs $run @("run")) -WindowStyle Hidden
            Set-Note $run "起動中…（タスク「$task」を起動できないので wsl.exe を直接起動: $out）" $false 300
        }
        # 停止処理の待ち（最大 180 秒）＋立ち上がりを見込んで 5 分以内に稼働を確かめる（Update-Panel）
        $script:StartCheck[$run] = [datetime]::Now.AddMinutes(5)
    } catch {
        Set-Note $run ("起動に失敗: " + $_.Exception.Message) $true 900
    }
    $script:NextFetch = [datetime]::Now.AddSeconds(5)
}

# ---- クラウドの操作と表示 ----
function Get-VastArgs {
    return @("--gpu", ([string]$cmbGpu.SelectedItem -replace " ", "_"), "--rent", $(if ([string]$cmbRent.SelectedItem -eq "入札") { "bid" } else { "on-demand" }),
             "--max-dph", ("{0:0.00}" -f [double]$numDph.Value),
             "--min-rel", ("{0:0.00}" -f [double]$numRel.Value), "--min-cpu-ghz", ("{0:0.0}" -f [double]$numGhz.Value),
             "--min-cores", ("{0:0}" -f [double]$numCores.Value), "--max-inet-cost", ("{0:0.000}" -f [double]$numInet.Value))
}
function Start-Vast {
    $o = $script:Vast
    if ($null -ne $o -and $null -ne $o.session -and $o.session.alive) { Set-Note "vast" "既に動いています" $false 30; return }
    $gpu = [string]$cmbGpu.SelectedItem
    $dph = [double]$numDph.Value; $hours = [double]$numHours.Value
    $credit = if ($null -ne $o -and $null -ne $o.account -and $null -ne $o.account.credit) { '${0:N2}' -f [double]$o.account.credit } else { "不明" }
    $rentNote = if ([string]$cmbRent.SelectedItem -eq "入札") { "入札（割り込みあり。止められたら残りの時間で自動で借り直す）" } else { "on-demand" }
    $msg = ('{0} を{3}で最大 ${1:N2}/h（実効単価）、{2} 時間借りて、ls に自己対局の局を足します。' -f $gpu, $dph, $hours, $rentNote) + "`r`n" +
           ('費用は最大 ${0:N2} 程度（準備の 5〜15 分を含む）。残高 {1}。' -f ($dph * ($hours + 0.25)), $credit) + "`r`n" +
           (Get-VastPast $gpu $hours) + "`r`n" +
           "時間が来たら残りの局を取ってインスタンスを消します。途中で止めるときは「停止」。よろしいですか？"
    if (-not (Confirm-Action $msg)) { return }
    try {
        $r = Invoke-Vast (@("start", "--run", "ls", "--hours", ("{0:0.0}" -f $hours)) + (Get-VastArgs))
        Set-Note "vast" $r.text ($r.code -ne 0) $(if ($r.code -eq 0) { 300 } else { 900 })
    } catch {
        Set-Note "vast" ("起動に失敗: " + $_.Exception.Message) $true 900
    }
    $script:NextVast = [datetime]::Now.AddSeconds(3)
}
function Stop-Vast {
    if (-not (Confirm-Action "クラウドのワーカーを止めます（残りの局を取ってからインスタンスを消します。数分）。よろしいですか？")) { return }
    try {
        $r = Invoke-Vast @("stop")
        Set-Note "vast" $r.text ($r.code -ne 0) 300
    } catch {
        Set-Note "vast" ("停止に失敗: " + $_.Exception.Message) $true 900
    }
    $script:NextVast = [datetime]::Now.AddSeconds(3)
}
function Show-VastOffers {
    # 検索した全件と落ちた理由を等幅の表で出す。「1 つ緩めれば借りられる」のボタンで入力欄の値を変えて検索し直し、
    # 「この条件で起動」で起動の確認へ進む（判定は libra-vast offers --json。起動も同じ Get-VastArgs を使うので結果が食い違わない）
    $action = "search"
    while ($action -eq "search") {
        $action = ""
        $form.Cursor = [System.Windows.Forms.Cursors]::WaitCursor
        try {
            $r = Invoke-Vast (@("offers", "--json") + (Get-VastArgs))
            $line = @($r.text -split "`n" | Where-Object { $_.TrimStart().StartsWith("{") }) | Select-Object -Last 1
            if ($r.code -ne 0 -or -not $line) { throw $r.text }
            $o = $line | ConvertFrom-Json
        } catch {
            Set-Note "vast" ("候補の取得に失敗: " + $_.Exception.Message) $true 300
            return
        } finally {
            $form.Cursor = [System.Windows.Forms.Cursors]::Default
        }
        $dlg = New-Object System.Windows.Forms.Form
        $dlg.Text = "候補（{0}）  検索 {1} 件、条件に合う {2} 件" -f [string]$cmbGpu.SelectedItem, [int]$o.offers, [int]$o.usable
        $dlg.StartPosition = "CenterParent"; $dlg.Width = 1180; $dlg.Height = 560; $dlg.MinimizeBox = $false
        $txt = New-Object System.Windows.Forms.TextBox
        $txt.Multiline = $true; $txt.ReadOnly = $true; $txt.ScrollBars = "Both"; $txt.WordWrap = $false; $txt.Dock = "Fill"
        $txt.Font = New-Object System.Drawing.Font("Consolas", 9.5); $txt.BackColor = [System.Drawing.Color]::White
        $txt.Text = ([string]$o.text) -replace "`r?`n", "`r`n"
        $btns = New-Object System.Windows.Forms.FlowLayoutPanel
        $btns.Dock = "Bottom"; $btns.AutoSize = $true; $btns.WrapContents = $true; $btns.Padding = New-Object System.Windows.Forms.Padding(4)
        foreach ($h in @($o.hints)) {
            $num = $script:VastNums[[string]$h.key]
            if ($null -eq $num) { continue }
            $b = New-Button ("{0} を {1} に" -f $h.label, $h.need_text) { $this.Tag.num.Value = $this.Tag.value; $this.FindForm().Tag = "search"; $this.FindForm().Close() }
            $b.AutoSize = $true
            $b.Tag = @{ num = $num; value = [decimal][double]$h.need }
            if ([decimal][double]$h.need -lt $num.Minimum -or [decimal][double]$h.need -gt $num.Maximum) { $b.Enabled = $false }
            $script:Tip.SetToolTip($b, ('入力欄の値を変えて検索し直す（${0:N3}/h {1}）{2}' -f [double]$h.offer.dph, $h.offer.cpu, $(if ($h.note) { "。" + $h.note } else { "" })))
            $btns.Controls.Add($b)
        }
        $bAgain = New-Button "検索し直す" { $this.FindForm().Tag = "search"; $this.FindForm().Close() } 90
        $bGo = New-Button "この条件で起動…" { $this.FindForm().Tag = "start"; $this.FindForm().Close() } 120
        $bGo.Enabled = ([int]$o.usable -gt 0 -and -not ($null -ne $script:Vast -and $null -ne $script:Vast.session -and $script:Vast.session.alive))
        $bClose = New-Button "閉じる" { $this.FindForm().Close() } 70
        foreach ($b in @($bAgain, $bGo, $bClose)) { $btns.Controls.Add($b) }
        $dlg.Controls.Add($txt)
        $dlg.Controls.Add($btns)
        $dlg.CancelButton = $bClose
        [void]$dlg.ShowDialog($form)
        $action = [string]$dlg.Tag
        $dlg.Dispose()
    }
    if ($action -eq "start") { Start-Vast }
}
function Cleanup-Vast {
    $o = $script:Vast
    $n = if ($null -ne $o -and $null -ne $o.account -and $null -ne $o.account.instances) { @($o.account.instances).Count } else { "?" }
    if (-not (Confirm-Action ("libra- で始まるラベルの vast.ai インスタンスをすべて消します（表示中 {0} 台。動いているセッションがあるときは消しません）。よろしいですか？" -f $n))) { return }
    try {
        $r = Invoke-Vast @("cleanup", "--yes")
        Set-Note "vast" $r.text ($r.code -ne 0) 300
    } catch {
        Set-Note "vast" ("後始末に失敗: " + $_.Exception.Message) $true 900
    }
    $script:NextVastAccount = [datetime]::MinValue
    $script:NextVast = [datetime]::MinValue
}
function Update-VastPanel {
    $v = $script:VastVals
    $o = $script:Vast
    if ($null -eq $o) { return }
    $black = [System.Drawing.Color]::Black
    $s = $o.session
    $alive = ($null -ne $s -and [bool]$s.alive)
    $bVastStart.Enabled = -not $alive
    $bVastStop.Enabled = ($alive -and -not $s.stop_requested)
    $bVastCleanup.Enabled = -not $alive
    if ($null -eq $s) {
        $v.phase.Text = "セッションはまだありません"; $v.phase.ForeColor = [System.Drawing.Color]::DimGray
        foreach ($k in @("session", "gpu", "time", "cost", "bridge", "verify")) { $v[$k].Text = "-" }
    } else {
        $v.phase.Text = [string]$s.phase
        $v.phase.ForeColor = if ($s.phase -like "異常終了*") { [System.Drawing.Color]::Firebrick } elseif ($alive) { [System.Drawing.Color]::ForestGreen } else { [System.Drawing.Color]::DimGray }
        $v.phase.Font = New-Object System.Drawing.Font($form.Font, [System.Drawing.FontStyle]::Bold)
        $started = if ($null -ne $s.started) { (From-Unix $s.started).ToString("MM/dd HH:mm") } else { "-" }
        $v.session.Text = ('{0}（開始 {1}、{2} を最大 ${3:N2}/h で {4} 時間）' -f ([string]$s.dir -split "/")[-1], $started, $s.gpu, [double]$s.max_dph, $s.hours) +
                          $(if ($s.continues) { "　※ {0} が打ち切られたあとの借り直し" -f $s.continues } else { "" })
        $inst = $o.instance
        if ($null -ne $inst -and $null -ne $inst.offer) {
            $off = $inst.offer
            $v.gpu.Text = '{0}、{1}（{2}）#{3}、${4:N3}/h' -f $off.gpu_name, ([string]$off.cpu_name).Trim(), $off.geolocation, $inst.instance, [double]$off.dph_total
        } else { $v.gpu.Text = "-" }
        $v.time.Text = if ($null -ne $o.rented_h) { '{0:N2} 時間{1}' -f [double]$o.rented_h, $(if ($null -ne $o.remaining_h) { ' / 残り {0:N2} 時間' -f [double]$o.remaining_h } else { "" }) } else { "-" }
        $v.cost.Text = if ($null -ne $o.est_cost_usd) { '${0:N2}' -f [double]$o.est_cost_usd } else { "-" }
        $b = $o.bridge
        if ($null -ne $b) {
            $last = if ($null -ne $b.last_pull) { "、最終 " + (Format-Ago (From-Unix $b.last_pull)) } else { "" }
            $v.bridge.Text = '{0} 局（{1} ファイル）、弾いた {2}、エラー {3}{4}' -f (Format-Int $b.games), $b.files, $b.rejected_files, $b.errors, $last
            $v.bridge.ForeColor = if ([int]$b.rejected_files -gt 0 -or [int]$b.errors -gt 0) { [System.Drawing.Color]::DarkOrange } else { $black }
            $v.verify.Text = if ($null -ne $b.verify_ms_per_game) { '{0} ms/局' -f $b.verify_ms_per_game } else { "-" }
        } else { $v.bridge.Text = "-"; $v.verify.Text = "-" }
    }
    $v.waste.Text = Format-Vast-Waste-Short $(if ($null -ne $script:VastHist) { $script:VastHist.interrupts } else { $null })
    $v.waste.ForeColor = if ($null -ne $script:VastHist -and $null -ne $script:VastHist.interrupts -and [double]$script:VastHist.interrupts.waste_pct -ge 20) { [System.Drawing.Color]::DarkOrange } else { $black }
    $wk = $null
    if ($script:Last.ContainsKey("ls") -and $null -ne $script:Last["ls"].status) { $wk = $script:Last["ls"].status.workers }
    $v.learner.Text = if ($null -ne $wk) { '{0} 局（古くて捨てた {1}、不正 {2}）' -f (Format-Int $wk.games), (Format-Int $wk.stale_games), $wk.rejected_files } else { "（ls の [workers] が無効か、まだ取り込みなし）" }
    $ac = $o.account
    if ($null -ne $ac) {
        if ($null -ne $ac.error) {
            $v.credit.Text = "取得失敗: " + $ac.error; $v.credit.ForeColor = [System.Drawing.Color]::Firebrick
        } else {
            $v.credit.Text = '${0:N2}（{1}）' -f [double]$ac.credit, (Format-Ago (From-Unix $ac.time)); $v.credit.ForeColor = $black
            $ins = @($ac.instances)
            $v.instances.Text = if ($ins.Count -eq 0) { "なし" } else { ($ins | ForEach-Object { '#{0} {1} {2} {3} ${4:N3}/h' -f $_.id, $_.label, $_.status, $_.gpu, [double]$_.dph }) -join "、" }
            $script:VastLeak = ($ins.Count -gt 0 -and -not $alive)
            $v.instances.ForeColor = if ($script:VastLeak) { [System.Drawing.Color]::Firebrick } else { $black }
        }
    }
    # セッションが始まった・終わったら履歴を取り直す
    $skey = if ($null -ne $s) { "{0}|{1}" -f $s.dir, $alive } else { "" }
    if ($skey -ne $script:VastSessionKey) { $script:VastSessionKey = $skey; $script:NextVastHist = [datetime]::MinValue }
    if ($script:VastLeak) { Set-TabState "cloud" "● インスタンスが残っている" ([System.Drawing.Color]::Firebrick) }
    elseif ($alive) { Set-TabState "cloud" ("● " + [string]$s.phase) ([System.Drawing.Color]::ForestGreen) }
    elseif ($null -ne $s -and [string]$s.phase -like "異常終了*") { Set-TabState "cloud" "● 異常終了" ([System.Drawing.Color]::Firebrick) }
    else { Set-TabState "cloud" "" ([System.Drawing.Color]::DimGray) }
    if ($null -ne $o.log_tail) {
        $vlog.Text = (@($o.log_tail) -join "`r`n")
        $vlog.SelectionStart = $vlog.Text.Length
        $vlog.ScrollToCaret()
    }
}

# ---- タイマー ----
$timer = New-Object System.Windows.Forms.Timer
$timer.Interval = 500
$timer.Add_Tick({
    try {
        Complete-Fetches
        Complete-VastFetch
        Complete-VastHistFetch
        if ($null -eq $script:VastHistPending -and [datetime]::Now -ge $script:NextVastHist) {
            # 動いているセッションの行（回収局・費用）は 1 分ごと、ほかは VastAccountSec ごと
            $script:NextVastHist = [datetime]::Now.AddSeconds($(if ($script:VastSessionKey -like "*|True") { 60 } else { $VastAccountSec }))
            Start-VastHistFetch
        }
        if ($null -eq $script:VastPending -and [datetime]::Now -ge $script:NextVast) {
            $script:NextVast = [datetime]::Now.AddSeconds([int]$numIv.Value)
            Start-VastFetch
        }
        if ($script:Pending.Count -eq 0 -and [datetime]::Now -ge $script:NextFetch) {
            $script:NextFetch = [datetime]::Now.AddSeconds([int]$numIv.Value)
            $withHistory = ([datetime]::Now -ge $script:NextHistory)
            if ($withHistory) { $script:NextHistory = [datetime]::Now.AddSeconds($HistorySec) }
            foreach ($r in $Runs) { Start-Fetch $r $withHistory }
        }
        Update-StatusBar
        if ($Screenshot -and -not $script:ShotDone -and $script:Pending.Count -eq 0 -and $script:Data.Count -eq $Runs.Count -and $null -eq $script:VastPending -and ($null -ne $script:Vast -or $script:VastError) -and $null -eq $script:VastHistPending -and ($null -ne $script:VastHist -or $script:VastHistError)) {
            $script:ShotDone = $true
            if ($null -ne $tabs.SelectedTab) { $tabs.SelectedTab.Controls[0].Refresh() }
            $histChart.Refresh()
            $bmp = New-Object System.Drawing.Bitmap($form.Width, $form.Height)
            $form.DrawToBitmap($bmp, (New-Object System.Drawing.Rectangle(0, 0, $form.Width, $form.Height)))
            $bmp.Save($Screenshot, [System.Drawing.Imaging.ImageFormat]::Png)
            $bmp.Dispose()
            [Console]::WriteLine("screenshot: " + $Screenshot + " tab=" + $tabs.SelectedTab.Text)
            foreach ($r in $Runs) {
                $d = $script:Data[$r]
                [Console]::WriteLine(("{0}: {1} flags={2} step={3} games={4} gpd={5} metrics={6} evals={7} matches={8} archives={9}" -f $r, $d.process, (@($d.flags) -join "+"), $d.status.step, $d.status.games_total, $d.status.games_per_day_1h, @($d.metrics).Count, @($d.evals).Count, @($d.matches).Count, @($d.archives).Count))
            }
            $vphase = if ($null -ne $script:Vast -and $null -ne $script:Vast.session) { $script:Vast.session.phase } else { "-" }
            [Console]::WriteLine(("vast: phase={0} credit={1} error={2}" -f $vphase, $(if ($null -ne $script:Vast -and $null -ne $script:Vast.account) { $script:Vast.account.credit } else { "-" }), $script:VastError))
            $ht = if ($null -ne $script:VastHist) { $script:VastHist.totals } else { $null }
            [Console]::WriteLine(("vast history: sessions={0} cost={1} net_games={2} usd_per_1m={3} error={4} top={5} chart={6} size={7}x{8}" -f $ht.sessions, $(if ($null -ne $ht.total_usd) { $ht.total_usd } else { $ht.est_cost_usd }), $ht.net_games, $ht.usd_per_1m, $script:VastHistError, [string]$runTabs.SelectedTab.Tag, $tabs.SelectedTab.Text, $form.Width, $form.Height))
            $form.Close()
        }
    } catch {
        $status.Text = "内部エラー: " + $_.Exception.Message
        $status.ForeColor = [System.Drawing.Color]::Firebrick
        if ($Screenshot) { [Console]::WriteLine("error: " + $_.Exception.Message + " " + $_.ScriptStackTrace); $form.Close() }
    }
})
$form.Add_Shown({ Move-Charts; $timer.Start() })
$form.Add_FormClosing({
    $timer.Stop()
    foreach ($f in $script:Pending.Values) { try { $f.proc.Kill() } catch {} }
    foreach ($f in @($script:VastPending, $script:VastHistPending)) { if ($null -ne $f) { try { $f.proc.Kill() } catch {} } }
    if (-not $Screenshot) { Save-Layout }
})

if ($UpdateDesktopModel) {
    $info = Update-DesktopModel
    [Console]::WriteLine(("desktop model updated: step={0} stale={1} onnx={2} size={3} dir={4}" -f $info.step, $info.stale, $info.onnx_time, $info.size, $DesktopEngineDir))
    exit 0
}
if ($Do) {
    # GUI なしでボタンと同じ呼び出しを実行する（例: -Do "lx:stop"、-Do "ls:eval-now"。起動は含まない）
    if ($Do.StartsWith("vast:")) {
        $r = Invoke-Vast (@($Do.Substring(5).Trim().Split(" ")) | Where-Object { $_ })
        [Console]::WriteLine($r.text)
        exit $r.code
    }
    $run, $rest = $Do.Split(":", 2)
    if ([string]::IsNullOrWhiteSpace($rest)) {
        [Console]::WriteLine("-Do は <run>:<コマンド> の形で指定してください（例: ls:status、lx:stop）")
        exit 1
    }
    [Console]::WriteLine((Invoke-Libra $run (@($rest.Trim().Split(" ")) | Where-Object { $_ })))
    exit 0
}
Load-History
Update-DesktopModelLabel
if (-not $Screenshot -and -not $Size) { Restore-Layout }
[void]$form.ShowDialog()
