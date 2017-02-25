title Installing software
cls
@echo off
echo 1) Python 3.5.0 installing...
echo.
start /wait %~dp0\python-3.5.0.exe
echo 2) Pip updating...
echo.
@echo off
start /wait %~dp0\get-pip.py
echo 3) NumPy installing...
echo.
@echo off
start /wait %~dp0\install_numpy.bat
echo 4) PyQt 5 installing...
echo.
exit