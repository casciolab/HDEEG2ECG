@echo off
REM run from this script's own folder, wherever it is launched from
cd /d "%~dp0"
REM =====================================================================
REM  Build the standalone Windows .exe of the HDEEG2ECG toolbox (Stages 2-3).
REM
REM  BEFORE YOU RUN THIS:
REM    1. CLOSE the HDEEG2ECG_Toolbox app if it is open, and check Task
REM       Manager -> end any HDEEG2ECG_Toolbox.exe process. A running .exe
REM       is locked; the build then fails with "Access is denied" when it
REM       tries to overwrite it.
REM    2. Build on a LOCAL drive, not a network share. Antivirus on network
REM       drives scans and briefly locks the freshly-built .exe, which also
REM       causes "Access is denied".
REM
REM  Build environment (run once in Anaconda Prompt, then run this .bat
REM  from that same activated prompt):
REM    conda create -n rating_build python=3.11 -y
REM    conda activate rating_build
REM    pip install numpy scipy pandas mne matplotlib eeglabio pyinstaller
REM  Do NOT install torch here. eeglabio IS required -- mne uses it to
REM  WRITE .set files; without it "Save & next" silently fails.
REM  The released HDEEG2ECG_Toolbox.exe (GitHub release asset) was built with Python 3.11.15,
REM  PyInstaller 6.21.0, numpy 2.4.6, scipy 1.17.1, pandas 3.0.3, mne 1.12.1,
REM  matplotlib 3.11.0 and eeglabio 0.1.3.
REM =====================================================================

REM --- clear any stale build so nothing half-built lingers ---
if exist dist  rmdir /s /q dist
if exist build rmdir /s /q build

REM  The last line is the toolbox script.
pyinstaller --noconfirm --onefile --windowed ^
  --name HDEEG2ECG_Toolbox ^
  --collect-all mne ^
  --collect-all eeglabio ^
  --collect-submodules scipy ^
  --hidden-import matplotlib.backends.backend_tkagg ^
  --exclude-module torch ^
  --exclude-module hdeeg2ecg_model ^
  hdeeg2ecg_toolbox.py

echo.
echo ============================================================
echo  Look at the LAST lines above:
echo   - "Building EXE ... completed successfully" and NO traceback
echo        -^> success: dist\HDEEG2ECG_Toolbox.exe
echo   - "Access is denied"
echo        -^> the old .exe was still open, or AV-locked on a network
echo           drive. Close it, build on a LOCAL drive, run this again.
echo ============================================================
pause