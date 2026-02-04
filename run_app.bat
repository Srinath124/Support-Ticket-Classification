@echo off
echo ========================================
echo Support Ticket Classification Dashboard
echo ========================================
echo.
echo Starting the Streamlit application...
echo.
echo The dashboard will open in your default browser.
echo Press Ctrl+C to stop the server.
echo.
echo ========================================
echo.

streamlit run app.py --server.port 8501 --server.address localhost

pause
