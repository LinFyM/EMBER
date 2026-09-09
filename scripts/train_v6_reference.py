#!/usr/bin/env python3
"""Train only the isolated historical v6 under the matched lawful protocol."""
from pathlib import Path
from ember.writer.training import main

if __name__ == "__main__":
    main(default_config=Path(__file__).resolve().parents[1] / "configs/pi05_v6_causal_reference.json",
         description=__doc__, modes=("profile", "exploratory"))
