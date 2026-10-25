# Federated Learning Lab - Multi-Terminal Launcher
# Reads configs/server.yaml and configs/client.yaml and spawns terminal windows.

param(
    [int]$Clients,
    [int]$Port,
    [int]$Rounds,
    [switch]$DryRun
)

$argsList = @("launch_terminals.py")
if ($Clients) { $argsList += @("--clients", $Clients) }
if ($Port) { $argsList += @("--port", $Port) }
if ($Rounds) { $argsList += @("--rounds", $Rounds) }
if ($DryRun) { $argsList += "--dry-run" }

python @argsList
