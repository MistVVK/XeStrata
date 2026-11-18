@echo off
rem UPDATE.bat is kept so that the command people know still works, and so that an older clone's own UPDATE.bat can finish.
rem It does what START-HERE.bat --update does: it gets the newest code (git pull), then updates the engine and the settings.
rem Why this file must exist: cmd opens a running .bat file again after each program it calls. The pull that brought the 0.2.16
rem layout used to remove the old UPDATE.bat, and cmd then ended with "The batch file cannot be found." and exit code 1.
call "%~dp0START-HERE.bat" --update %*
exit /b %errorlevel%
