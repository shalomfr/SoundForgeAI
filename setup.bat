@echo off
chcp 65001 >nul
echo Copying SoundForge AI files...
xcopy "C:\Users\שלום\Downloads\סלומון\תוכנת פודקאסטים\*" "C:\Projects\SoundForgeAI\" /E /I /Y /Q
echo Done!
echo.
echo Now run these commands:
echo   cd C:\Projects\SoundForgeAI
echo   git init
echo   git add .
echo   git commit -m "Initial commit"
echo   git remote add origin https://github.com/shalomfr/SoundForgeAI.git
echo   git push -u origin main
pause

