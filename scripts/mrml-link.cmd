@echo off
setlocal
for /f "usebackq delims=" %%I in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "MRML_VS=%%I"
if not defined MRML_VS (
  echo MRML could not find Visual Studio C++ Build Tools. Install the Desktop development with C++ workload. 1>&2
  exit /b 1
)
call "%MRML_VS%\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul
link.exe %*
