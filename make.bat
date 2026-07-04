@ECHO OFF

pushd %~dp0

REM Command file for Sphinx documentation

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)
set SOURCEDIR=source
set BUILDDIR=build

%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
	echo.
	echo.The 'sphinx-build' command was not found. Make sure you have Sphinx
	echo.installed, then set the SPHINXBUILD environment variable to point
	echo.to the full path of the 'sphinx-build' executable. Alternatively you
	echo.may add the Sphinx directory to PATH.
	echo.
	echo.If you don't have Sphinx installed, grab it from
	echo.https://www.sphinx-doc.org/
	exit /b 1
)

if "%1" == "" goto help
if "%1" == "clean" goto clean
if "%1" == "autobuild" goto autobuild

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:clean
echo.Cleaning build directories...
rmdir /s /q %BUILDDIR% 2>NUL
rmdir /s /q %SOURCEDIR%\.doctrees 2>NUL
rmdir /s /q %SOURCEDIR%\__pycache__ 2>NUL
for /d /r %SOURCEDIR% %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"
del /s /q %SOURCEDIR%\*.pyc 2>NUL
echo.Cleaning complete.
goto end

:autobuild
sphinx-autobuild %SOURCEDIR% %BUILDDIR%\html
goto end

:end
popd
