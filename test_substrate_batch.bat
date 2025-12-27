@echo off
REM Batch file for running substrate blending tests
echo Running substrate blending tests...
echo.

echo Test 1: Germanic + Celtic (high influence)
python test_substrate.py --primary "Proto-Germanic" --substrate "Proto-Celtic" --influence 0.8 --seed 42
echo.
echo.

echo Test 2: Indo-Iranian + Dravidian (medium influence)
python test_substrate.py --primary "Proto-Indo-Iranian" --substrate "Proto-Dravidian" --influence 0.5 --seed 123
echo.
echo.

echo Test 3: Italic + Greek (low influence)
python test_substrate.py --primary "Proto-Italic" --substrate "Proto-Greek" --influence 0.3 --seed 456
echo.
echo.

echo All tests completed!
pause