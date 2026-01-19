#!/usr/bin/env python3
from __future__ import annotations

from .registry import show_registry_status

def main() -> None:
    print(show_registry_status())

if __name__ == "__main__":
    main()
