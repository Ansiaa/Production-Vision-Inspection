from __future__ import annotations

import argparse
import json
from pathlib import Path


def decide_status(drift_level: str, predicted_anomaly_ratio: float) -> str:
    if drift_level == "severe":
        return "retrain_recommended"

    if drift_level == "moderate":
        if predicted_anomaly_ratio >= 0.25:
            return "retrain_recommended"
        return "retrain_candidate"

    return "keep_monitoring"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drift-json", required=True)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    drift_path = Path(args.drift_json)
    payload = json.loads(drift_path.read_text(encoding="utf-8"))

    drift_level = payload.get("drift_level", "weak")
    current = payload.get("current", {})
    anomaly_ratio = float(current.get("predicted_anomaly_ratio", 0.0))

    status = decide_status(drift_level, anomaly_ratio)

    out = {
        "drift_json": str(drift_path),
        "drift_level": drift_level,
        "predicted_anomaly_ratio": anomaly_ratio,
        "status": status,
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[SAVE] {out_path}")


if __name__ == "__main__":
    main()
