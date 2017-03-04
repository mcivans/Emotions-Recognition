title Installing software
cls
@echo off
echo 1) Python 3.5.0 installing...
echo.
start /wait %~dp0\python-3.5.0.exe
@echo off
echo 2) Pip updating...
echo.
start /wait %~dp0\get-pip.py
@echo off
echo 3) NumPy installing...
echo.
start /wait %~dp0\install_numpy.bat
@echo off
echo 4) PIL installing...
echo.
start /wait %~dp0\install_PIL.bat
@echo off
echo 5) LMDB installing...
echo.
start /wait %~dp0\install_LMDB.bat
echo.
exit