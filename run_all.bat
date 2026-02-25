@echo off

SET "GAUSSIAN_EXE=C:\G09W\g09.exe"
SET "INPUT_DIR=C:\Users\Welcome\Downloads\dft\d_pbe_gjf"
SET "LOG_DIR=C:\Users\Welcome\Downloads\dft\d_pbe_gjf"

IF NOT EXIST "%LOG_DIR%" mkdir "%LOG_DIR%"

cd /d "%INPUT_DIR%"

FOR %%F IN (*.gjf) DO (
    echo ==================================
    echo Running Gaussian on %%F
    echo ==================================

    "%GAUSSIAN_EXE%" "%%F"

    IF EXIST "%%~nF.out" (
        move /Y "%%~nF.out" "%LOG_DIR%\%%~nF.out"
    )

    del /Q fort.* 2>nul
    del /Q "%%~nF.inp" 2>nul
)




