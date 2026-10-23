import argparse
import hashlib
import json
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(description="Measure neutral-pose mesh height from Webots Pedestrian assets")
    parser.add_argument("--assets", type=Path, default=Path.home() / ".cache/Cyberbotics/Webots/assets")
    args = parser.parse_args()
    parts = []
    for path in args.assets.rglob("*"):
        if not path.is_file():
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        name = re.search(r"PROTO (Pedestrian\w+)\s*\[", text)
        points = re.search(r"coord\s+Coordinate\s*\{\s*point\s*\[([^\]]+)\]", text)
        if name is None or points is None:
            continue
        values = [float(value) for value in re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", points[1])]
        if len(values) % 3:
            raise ValueError(f"Invalid mesh coordinates: {path}")
        zValues = values[2::3]
        parts.append({"part": name[1], "minimumZ": min(zValues), "maximumZ": max(zValues),
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    required = {"PedestrianHead", "PedestrianLeftFoot", "PedestrianRightFoot"}
    if not required.issubset({part["part"] for part in parts}):
        parser.error("Head/foot assets missing; run capture first so Webots caches its referenced PROTOs")
    minimum = min(part["minimumZ"] for part in parts)
    maximum = max(part["maximumZ"] for part in parts)
    print(json.dumps({"neutralPoseHeightMeters": maximum - minimum, "minimumZ": minimum,
                      "maximumZ": maximum, "parts": sorted(parts, key=lambda part: part["part"]),
                      "scope": "Mesh extents at zero joint angles; assumes the standard R2025a untranslated part geometries. Walking pose height varies."}, indent=2))


if __name__ == "__main__":
    main()
