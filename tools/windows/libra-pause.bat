@echo off
rem applies to both runs: ls (main) and lx (exploiter)
for %%R in (ls lx) do (
  echo == %%R ==
  wsl.exe -d Ubuntu-24.04 -- /home/sakis/LibraShogi/bin/libra --run %%R pause
)
pause
