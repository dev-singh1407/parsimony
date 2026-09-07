# Parsimony demo launcher.
#
# Removes the two ways a live demo goes wrong before it starts: being in the
# wrong folder, and using the wrong Python. Both happened in rehearsal.
#
#   .\demo.ps1                  list the acts
#   .\demo.ps1 warmup           load the model (run this BEFORE the guide arrives)
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
        Write-Host "  .\demo.ps1 1         what the system does          ~40 s"
        Write-Host "  .\demo.ps1 2         the gate blocking a bad edit  ~5 s"
        Write-Host "  .\demo.ps1 3         cache safety                  ~5 s"
        Write-Host "  .\demo.ps1 4         regenerate every number       ~100 s"
        Write-Host "  .\demo.ps1 5         self-improving, measured      ~5 s"
        Write-Host ""
    }

    "check" {
        Write-Host ""
        Write-Host "folder   $ProjectDir"
        Write-Host "python   $VenvPython"
        if (Test-Path $Ollama) {
            Write-Host "ollama   installed" -ForegroundColor Green
            try {
                $null = Invoke-WebRequest "http://localhost:11434/api/tags" -TimeoutSec 5 -UseBasicParsing
                Write-Host "server   responding" -ForegroundColor Green
            } catch {
                Write-Host "server   NOT responding - run: $Ollama serve" -ForegroundColor Red
            }
        } else {
            Write-Host "ollama   not found" -ForegroundColor Red
        }
        & $VenvPython -c "import parsimony; print('package  importable')"
        Write-Host ""
        Write-Host "If all four lines are green, you are ready." -ForegroundColor Cyan
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
