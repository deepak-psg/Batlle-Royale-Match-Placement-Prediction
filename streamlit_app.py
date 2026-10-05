"""
Streamlit Root Runner.
Points directly to the enhanced tactical application in app/streamlit_app.py.
"""
from pathlib import Path
import runpy

if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "app" / "streamlit_app.py"
    runpy.run_path(str(target), run_name="__main__")
