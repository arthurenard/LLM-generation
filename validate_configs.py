import sys
from src.utils.config_validator import check_config_files

if __name__ == "__main__":
    config_dir = sys.argv[1] if len(sys.argv) > 1 else "config"
    if not check_config_files(config_dir):
        sys.exit(1)
