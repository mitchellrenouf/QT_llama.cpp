@echo off
setlocal
for /f "usebackq delims=" %%I in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "RUSTLLAMA_VS=%%I"
if not defined RUSTLLAMA_VS (
  echo RustLlama could not find Visual Studio C++ Build Tools. Install the Desktop development with C++ workload. 1>&2
  exit /b 1
)
call "%RUSTLLAMA_VS%\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul
link.exe %*
