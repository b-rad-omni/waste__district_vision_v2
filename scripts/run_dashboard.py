#!/usr/bin/env python3
"""Launch the detection highlights dashboard"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

if __name__ == "__main__":
    import streamlit.web.cli as stcli
    import sys
    
    # Point to the app module
    sys.argv = ["streamlit", "run", "src/dashboard/app.py"]
    stcli.main()