@echo off
setlocal
cd /d "%~dp0"

python -c "import pygame, torch, numpy" >nul 2>&1
if errorlevel 1 (
    echo Missing a visualization dependency.
    echo Run: python -m pip install pygame
    pause
    exit /b 1
)

if not exist replays mkdir replays
python main.py play --agents dueling_ddqn_agent rule_based_agent rule_based_agent rule_based_agent --scenario classic --n-rounds 1 --seed 240824 --update-interval 0.15 --save-replay replays/visual_demo.pt --match-name visual-demo

if errorlevel 1 pause
endlocal
