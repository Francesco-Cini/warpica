"""Convert MATLAB's column-major JSON export to compressed NumPy fixtures."""

import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    directory = Path(__file__).resolve().parents[1] / "tests/reference/matlab"
    raw = directory / "raw.json"
    data = json.loads(raw.read_text())
    metadata = {
        key: data[key] for key in ("commit", "runtime", "generated_utc", "c", "G")
    }
    metadata["raw_json_sha256"] = hashlib.sha256(raw.read_bytes()).hexdigest()
    metadata["compatibility"] = (
        "CPU isgpuarray check; getEvenPointsOnSphere uses unchanged CPU formulas"
    )
    metadata["cases"] = {}
    arrays = {}
    for case, item in data.items():
        if not isinstance(item, dict) or "grid" not in item:
            continue
        metadata["cases"][case] = {
            key: item[key] for key in ("grid", "spacing", "center")
        }
        for name, packed in item.items():
            if isinstance(packed, dict) and "data" in packed:
                arrays[f"{case}__{name}"] = np.asarray(packed["data"]).reshape(
                    packed["shape"], order="F"
                )
    np.savez_compressed(directory / "warp_factory.npz", **arrays)
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Converted {len(arrays)} independent reference arrays")


if __name__ == "__main__":
    main()
