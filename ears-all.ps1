param(
    [Parameter(Mandatory=$true)][string]$InFile,
    [string]$OutFile = "",
    [string]$FwModel = "small"
)
$ErrorActionPreference = "Continue"
$PSNativeCommandUseErrorActionPreference = $false
$stage = "C:\Users\robse\pcloud-stage"
$senseBin = "$stage\sense\sherpa-onnx-v1.13.8-win-x64-shared-MD-Release\bin"
$whisperExe = "$stage\whisper\Release\whisper-cli.exe"
$whisperModel = "$stage\whisper\ggml-base.en.bin"
$svModel = "$stage\sense\model\model.int8.onnx"
$svTokens = "$stage\sense\model\tokens.txt"
$tagModel = "$stage\sense\tagger\sherpa-onnx-zipformer-audio-tagging-2024-04-09\model.onnx"
$tagLabels = "$stage\sense\tagger\sherpa-onnx-zipformer-audio-tagging-2024-04-09\class_labels_indices.csv"
$fwPy = "$stage\fwbench\venv\Scripts\python.exe"
$fwScript = "$stage\fwbench\fw_transcribe.py"
if ($OutFile -eq "") { $OutFile = [System.IO.Path]::ChangeExtension($InFile, ".ears.txt") }
$work = Join-Path $env:TEMP ("ears-" + [System.IO.Path]::GetFileNameWithoutExtension($InFile))
New-Item -ItemType Directory -Force $work | Out-Null
$wav = Join-Path $work "audio16.wav"
& ffmpeg -y -v error -i $InFile -ar 16000 -ac 1 $wav
if (!(Test-Path $wav)) { throw "ffmpeg produced no wav" }
$dur = & ffprobe -v error -show_entries format=duration -of csv=p=0 $wav

# EAR 1 - Whisper.cpp (words)
$wBase = Join-Path $work "whisper"
& $whisperExe -m $whisperModel -f $wav --no-timestamps -otxt -of $wBase 2>$null | Out-Null
if (!(Test-Path "$wBase.txt")) { throw "whisper produced no transcript" }
$wText = (Get-Content "$wBase.txt" -Raw).Trim()

# EAR 2 - SenseVoice (words + emotion + event + token times)
$svRaw = & "$senseBin\sherpa-onnx-offline.exe" --sense-voice-model=$svModel --tokens=$svTokens --sense-voice-use-itn=1 $wav 2>$null
$svLine = $svRaw | Where-Object { $_ -like "{*" } | Select-Object -First 1
if ($null -eq $svLine) { throw "SenseVoice produced no result" }
$sv = $svLine | ConvertFrom-Json
$svText = $sv.text.Trim()
$emotion = $sv.emotion -replace "<\|", "(" -replace "\|>", ")"
$svEvent = $sv.event -replace "<\|", "(" -replace "\|>", ")"

# EAR 3 - faster-whisper (words; model selectable, default small, CPU int8)
# WAV input is decoded by fw_transcribe.py via stdlib+numpy (no PyAV dependency).
$fwText = ""
$fwInfo = ""
if ((Test-Path $fwPy) -and (Test-Path $fwScript)) {
    $fwOut = & $fwPy $fwScript $wav $FwModel int8 2>$null
    if (Test-Path "$wav.fw.txt") {
        $fwText = (Get-Content "$wav.fw.txt" -Raw).Trim()
        $fwInfo = ($fwOut | Select-Object -First 1)
    }
}

# EAR 4 - Audio tagger (sound events, full vocabulary)
$tagRaw = & "$senseBin\sherpa-onnx-offline-audio-tagging.exe" --zipformer-model=$tagModel --labels=$tagLabels --top-k=8 $wav 2>$null
$events = @()
foreach ($line in $tagRaw) {
    if ($line -match 'name="([^"]+)", index=\d+, prob=([0-9.]+)') {
        $p = [double]$Matches[2]
        if ($p -ge 0.05) { $events += ("({0} {1}%)" -f $Matches[1], [math]::Round($p * 100)) }
    }
}

# CONSENSUS - pairwise token overlap between the word-ears
function Tokens($s) { return @($s.ToLower() -replace "[^a-z0-9' ]", " " -split "\s+" | Where-Object { $_ -ne "" }) }
function Agree($aText, $bText) {
    $at = Tokens $aText
    $bt = Tokens $bText
    $bag = @{}
    foreach ($t in $bt) { $bag[$t] = ($bag[$t] + 1) }
    $hit = 0
    foreach ($t in $at) { if ($bag[$t] -gt 0) { $hit++; $bag[$t]-- } }
    $denom = [math]::Max($at.Count, $bt.Count)
    if ($denom -gt 0) { return [math]::Round(100.0 * $hit / $denom) } else { return 100 }
}
$aWS = Agree $wText $svText
$scores = @($aWS)
$aWF = $null
$aSF = $null
if ($fwText -ne "") {
    $aWF = Agree $wText $fwText
    $aSF = Agree $svText $fwText
    $scores += $aWF
    $scores += $aSF
}
$agree = [math]::Round(($scores | Measure-Object -Average).Average)
$verdict = if ($agree -ge 85) { "EARS AGREE" } else { "EARS DISAGREE - FLAG FOR A READER" }

$out = @()
$out += "EARS REPORT - all layers, one file"
$out += ("Source: " + $InFile)
$out += ("Duration: " + $dur + " s")
$out += ""
$out += "WORDS (Whisper.cpp base.en):"
$out += $wText
$out += ""
$out += "WORDS (SenseVoice):"
$out += $svText
$out += ""
$out += "WORDS (faster-whisper " + $FwModel + "):"
if ($fwText -ne "") { $out += $fwText; $out += ("[" + $fwInfo + "]") } else { $out += "(unavailable - fwbench not installed)" }
$out += ""
$out += ("EMOTION (SenseVoice): " + $emotion)
$out += ("EVENT (SenseVoice): " + $svEvent)
$out += ("SOUND EVENTS (tagger): " + ($events -join " "))
$out += ""
$out += ("CONSENSUS pairwise: Whisper/SenseVoice " + $aWS + "%")
if ($null -ne $aWF) {
    $out += ("CONSENSUS pairwise: Whisper/FasterWhisper " + $aWF + "%")
    $out += ("CONSENSUS pairwise: SenseVoice/FasterWhisper " + $aSF + "%")
}
$out += ("CONSENSUS average: " + $agree + "% - " + $verdict)
$out -join "`r`n" | Set-Content -Path $OutFile -Encoding UTF8
Write-Output ("Wrote " + $OutFile)
Write-Output ("Consensus " + $agree + "% - " + $verdict)
