from pathlib import Path

# NOTE: These paths are relative to the `wtb` directory where this script is located.
RESULT_PATH = "../result/"
SCORE_PATH = "../score/"
PROMPT_PATH = "../data/Wild-Tool-Bench.jsonl"
DOTENV_PATH = "../.env"
PROJECT_ROOT = "../"
TEST_IDS_TO_GENERATE_PATH = "../test_case_ids_to_generate.json"

# Construct the full path to use by other scripts
script_dir = Path(__file__).parent
RESULT_PATH = (script_dir / RESULT_PATH).resolve()
SCORE_PATH = (script_dir / SCORE_PATH).resolve()
PROMPT_PATH = (script_dir / PROMPT_PATH).resolve()
DOTENV_PATH = (script_dir / DOTENV_PATH).resolve()
PROJECT_ROOT = (script_dir / PROJECT_ROOT).resolve()
TEST_IDS_TO_GENERATE_PATH = (script_dir / TEST_IDS_TO_GENERATE_PATH).resolve()

RESULT_PATH.mkdir(parents=True, exist_ok=True)
SCORE_PATH.mkdir(parents=True, exist_ok=True)
