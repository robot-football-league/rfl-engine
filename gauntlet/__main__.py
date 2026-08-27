import sys

from .cli import main

if __name__ == "__main__":
    try:
        main()
    except ModuleNotFoundError as e:
        name = e.name or ""
        if name.startswith("gauntlet"):
            # Two very different causes, and guessing wrong wastes an
            # outage: the public engine build genuinely omits the station
            # modules, but on the STATION the same error means a deploy
            # left a file behind (2026-08-21: schedule.py had never been
            # copied to the box at all). Name it and point at both.
            sys.exit(f"missing module '{name}' — either this is the public "
                     "engine build, which does not include the station "
                     "modules, or a deploy is incomplete: check with "
                     "scripts/deploy_box.sh")
        raise
