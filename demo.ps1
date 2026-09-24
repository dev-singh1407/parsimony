# Parsimony demo launcher.
#
# Removes the two ways a live demo goes wrong before it starts: being in the
# wrong folder, and using the wrong Python. Both happened in rehearsal.
#
#   .\demo.ps1                  list the acts
#   .\demo.ps1 warmup           load the model (run this BEFORE the guide arrives)
#   .\demo.ps1 doc              a document, live, with and without the layers
#   .\demo.ps1 sentences        which sentences of the documents survived
#   .\demo.ps1 web              the visualiser in a browser: heatmap, A/B, counters
#   .\demo.ps1 proof            the proof PDF: every removal struck through and labelled
#   .\demo.ps1 followups        which history strategy keeps the needed fact
#   .\demo.ps1 1                Act 1 - what the system does
#   .\demo.ps1 2                Act 2 - the gate blocking a bad edit
#   .\demo.ps1 3                Act 3 - cache safety
#   .\demo.ps1 4                Act 4 - regenerate every number
#   .\demo.ps1 5                Act 5 - self-improving, measured
#   .\demo.ps1 check            verify everything is ready
#
# Anything else is passed straight through to the CLI:
#   .\demo.ps1 chat "your question" --provider ollama

$ErrorActionPreference = "Stop"

$ProjectDir = $PSScriptRoot
$VenvPython = "C:\Users\dev singh\.venvs\parsimony\Scripts\python.exe"
$Ollama     = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "Cannot find the project's Python at:" -ForegroundColor Red
    Write-Host "  $VenvPython" -ForegroundColor Red
    exit 1
}
Set-Location $ProjectDir
$env:PYTHONIOENCODING = "utf-8"

function Run-Cli { & $VenvPython -m parsimony.surfaces.cli.main @args }

function Show-Banner($n, $title, $say) {
    Write-Host ""
    Write-Host ("=" * 78) -ForegroundColor DarkGray
    Write-Host "  ACT $n  -  $title" -ForegroundColor Cyan
    Write-Host ("=" * 78) -ForegroundColor DarkGray
    Write-Host "  Say: $say" -ForegroundColor DarkYellow
    Write-Host ""
}

$act = if ($args.Count -gt 0) { $args[0] } else { "" }
$rest = if ($args.Count -gt 1) { $args[1..($args.Count - 1)] } else { @() }

switch ($act) {

    "" {
        Write-Host ""
        Write-Host "  Parsimony demo" -ForegroundColor Cyan
        Write-Host "  --------------" -ForegroundColor DarkGray
        Write-Host "  .\demo.ps1 check     is everything ready?"
        Write-Host "  .\demo.ps1 warmup    load the model  <-- run this first, 15 min before"
        Write-Host ""
        Write-Host "  .\demo.ps1 doc       a document, live, with and without   ~30 s" -ForegroundColor Cyan
        Write-Host "  .\demo.ps1 sentences which sentences survived, and why    instant" -ForegroundColor Cyan
        Write-Host "  .\demo.ps1 web       the visualiser in a browser           live" -ForegroundColor Cyan
        Write-Host "  .\demo.ps1 proof     the proof PDF, every removal labelled  ~10 s" -ForegroundColor Cyan
        Write-Host "  .\demo.ps1 tour      every module, one at a time, before/after" -ForegroundColor Cyan
        Write-Host "  .\demo.ps1 ask       type questions freely, no quotes" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "  .\demo.ps1 1         what the system does          ~40 s"
        Write-Host "  .\demo.ps1 2         the gate blocking a bad edit  ~5 s"
        Write-Host "  .\demo.ps1 3         cache safety                  ~5 s"
        Write-Host "  .\demo.ps1 4         regenerate every number       ~100 s"
        Write-Host "  .\demo.ps1 5         self-improving, measured      ~5 s"
        Write-Host ""
    }

    "check" {
        # Every line reports OK or a fix, and the verdict at the end is
        # computed rather than left to the reader. An earlier version printed
        # two lines in green and three in white under the caption "if all four
        # lines are green" — three ways wrong at once, on the one screen whose
        # entire job is to say whether you are ready.
        Write-Host ""
        $problems = @()

        Write-Host ("  {0,-9} {1}" -f "folder", $ProjectDir)
        Write-Host ("  {0,-9} {1}" -f "python", $VenvPython)

        if (Test-Path $Ollama) {
            Write-Host ("  {0,-9} {1}" -f "ollama", "OK  installed") -ForegroundColor Green
            try {
                $null = Invoke-WebRequest "http://localhost:11434/api/tags" -TimeoutSec 5 -UseBasicParsing
                Write-Host ("  {0,-9} {1}" -f "server", "OK  responding") -ForegroundColor Green
            } catch {
                Write-Host ("  {0,-9} {1}" -f "server", "NOT RESPONDING") -ForegroundColor Red
                $problems += "Start it: & '$Ollama' serve"
            }
        } else {
            Write-Host ("  {0,-9} {1}" -f "ollama", "NOT FOUND") -ForegroundColor Red
            $problems += "Install Ollama, then: ollama pull qwen2.5:1.5b-instruct"
        }

        & $VenvPython -c "import parsimony" 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host ("  {0,-9} {1}" -f "package", "OK  importable") -ForegroundColor Green
        } else {
            Write-Host ("  {0,-9} {1}" -f "package", "NOT IMPORTABLE") -ForegroundColor Red
            $problems += "Reinstall: & '$VenvPython' -m pip install -e ."
        }

        Write-Host ""
        if ($problems.Count -eq 0) {
            Write-Host "  READY.  Next: .\demo.ps1 warmup" -ForegroundColor Green
        } else {
            Write-Host "  NOT READY:" -ForegroundColor Red
            $problems | ForEach-Object { Write-Host "    $_" -ForegroundColor Yellow }
        }
        Write-Host ""
    }

    "warmup" {
        Write-Host ""
        Write-Host "Loading the model into memory (~10 s the first time)..." -ForegroundColor DarkYellow
        Run-Cli chat "hello" --provider ollama --no-trace
        Write-Host ""
        Write-Host "Warm. Every query from now on takes about two seconds." -ForegroundColor Green
        Write-Host ""
    }

    "tour" {
        # --pause so each module can be talked through before the next appears.
        Run-Cli tour --pause @rest
    }

    "ask" {
        # The loop lives in Python now: it keeps the conversation, so history
        # accumulates and M3/M4 actually have something to work on. A
        # PowerShell loop calling a one-shot command could not.
        Run-Cli ask @rest
    }

    "doc" {
        Show-Banner "D" "A document, live" `
            "Watch each layer run on an attached handbook, then the same question with every layer off. Both timings are the runtime's own."
        Run-Cli chat "What is the annual travel budget for the Tallinn office?" `
            --file examples/staff-handbook.md --provider ollama --compare
    }

    "web" {
        # Opens a browser on a page served by the standard library. Nothing to
        # install, nothing leaves the laptop, and it works with the wifi off --
        # which is the same claim the middleware makes for itself.
        Show-Banner "W" "The visualiser" `
            "Heatmap: every sentence shaded by its score, hover for the decision. A/B: the same question with the layers off and on, against the real model. Demo: the counters."
        Run-Cli web @rest
    }

    "proof" {
        # The artefact to hand someone who was not in the room.
        Show-Banner "P" "The proof document" `
            "The compressed prompt with every removal struck through and labelled with the branch that removed it."
        Run-Cli export --proof kestrel_q2 --out proof-kestrel.pdf
    }

    "sentences" {
        Show-Banner "S" "Which sentences survived" `
            "Six documents, every sentence marked kept or removed, the answer sentences in bold. No model needed."
        Run-Cli longctx --show kestrel_q2
    }

    "followups" {
        # Instant without a model: it reports which history strategy keeps the
        # fact the final question needs. With --provider ollama it also asks.
        Run-Cli followups @rest
    }

    "recall" {
        # No model: seconds, deterministic, identical every time. The best act
        # to run when the room is impatient or Ollama is busy.
        Show-Banner "R" "Did the answer survive?" `
            "Evidence recall separates our failure from the model's. We keep the answer 1.9x as often as truncation on the same budget - and score the same, because the model cannot use it either way."
        Run-Cli longbench --recall @rest
    }

    "longbench" {
        # Needs LongBench's data.zip extracted somewhere and passed with --data.
        # Long prompts on a CPU take minutes each; the run is resumable.
        Show-Banner "L" "A benchmark we did not write" `
            "Same compressor, same settings, on the set the literature reports. Say that the absolute F1 is low because this is a 1.5B model on a CPU - the full-context arm is the baseline that matters."
        Run-Cli longbench @rest
    }

    "longctx" {
        # Without --provider ollama this is instant and needs no model: it
        # reports which methods keep the answer sentences. With it, every arm
        # is asked of the real model, which takes about an hour.
        Run-Cli longctx @rest
    }

    "1" {
        Show-Banner 1 "What the system does" `
            "Same question, same model, same machine. Only the middleware differs. Point at the PREFILL row."
        Run-Cli compare "How do I reverse a string in Python?" --turns 8
    }

    "2" {
        Show-Banner 2 "The gate refusing a saving" `
            "The compressor tried to delete a sentence carrying a date. The gate blocked it. A wrong answer being prevented."
        Run-Cli chat "Explain the deadline. The deadline is 15 March. The deadline is 16 March." --text
    }

    "3" {
        Show-Banner 3 "The published thresholds are unsafe" `
            "The trick pair sits at 0.924 - higher than every real paraphrase. The literature's safe range would serve the opposite answer."
        Run-Cli calibrate
    }

    "4" {
        Show-Banner 4 "Every number regenerates" `
            "This rebuilds every table in the report from raw logs. Takes 100 s - talk about the architecture while it runs."
        & $VenvPython reproduce.py --out figures
    }

    "5" {
        Show-Banner 5 "Self-improving, measured" `
            "Zero benefit when traffic never repeats, +17.8 points when it repeats often. The value is a property of the traffic."
        Run-Cli learning --rates 0,0.4,0.8 --conversations 90
    }

    default { Run-Cli @args }
}
