@echo off
set /p N=games for ls (0 to clear): 
wsl.exe -d Ubuntu-24.04 -- /home/sakis/LibraShogi/bin/libra --run ls throttle --games %N%
set /p M=games for lx (0 to clear): 
wsl.exe -d Ubuntu-24.04 -- /home/sakis/LibraShogi/bin/libra --run lx throttle --games %M%
pause
