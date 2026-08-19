import sys

from .cli import main

if __name__ == "__main__":
    try:
        main()
    except ModuleNotFoundError as e:   # public engine build: research and
        name = e.name or ""            # station modules are not included
        if name.startswith("gauntlet"):
            sys.exit(f"this command needs '{name}', which is not part of "
                     "the public engine build")
        raise
