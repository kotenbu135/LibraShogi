Set sh = CreateObject("WScript.Shell")
rc = sh.Run("wsl.exe -d Ubuntu-24.04 -- /home/sakis/LibraShogi/bin/libra --run lx run", 0, True)
WScript.Quit rc
