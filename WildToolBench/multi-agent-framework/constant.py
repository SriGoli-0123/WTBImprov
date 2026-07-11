from pathlib import Path

# NOTE: These paths are relative to the `wtb` directory where this script is located.
DOTENV_PATH = ".env"

# Construct the full path to use by other scripts
script_dir = Path(__file__).parent
DOTENV_PATH = (script_dir / DOTENV_PATH).resolve()
