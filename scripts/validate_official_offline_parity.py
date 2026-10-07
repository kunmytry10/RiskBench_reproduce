#!/usr/bin/env python
"""Compare offline official-rule outputs with available author JSON references."""

from __future__ import print_function

import argparse
import json
import os


CASES = [
    ("interactive", "10_i-1_1_c_f_f_1_rl_ClearSunset_low_"),
    ("collision", "10_i-1_1_c_r_l_0_HardRainNoon_low_"),
    ("obstacle", "10_i-1_0_r_sl_ClearSunset_low_"),
]
METHODS = ["Random", "Range", "Kalman filter"]


def positive_ids(frame):
    return {str(key) for key, value in frame.items() if value}


def compare(offline_root, reference_root):
    report = {
        "reference_root": os.path.abspath(reference_root),
        "offline_root": os.path.abspath(offline_root),
        "cases": [],
    }
    for data_type, scene_key in CASES:
        for method in METHODS:
            ours_path = os.path.join(offline_root, method, data_type + ".json")
            reference_path = os.path.join(reference_root, method, data_type + ".json")
            if not os.path.exists(ours_path) or not os.path.exists(reference_path):
                continue
            ours = json.load(open(ours_path)).get(scene_key, {})
            reference = json.load(open(reference_path)).get(scene_key, {})
            frames = sorted(set(ours) & set(reference), key=int)
            differences = []
            ours_positive = 0
            reference_positive = 0
            for frame in frames:
                ours_ids = positive_ids(ours[frame])
                reference_ids = positive_ids(reference[frame])
                ours_positive += len(ours_ids)
                reference_positive += len(reference_ids)
                if ours_ids != reference_ids:
                    differences.append(frame)
            report["cases"].append({
                "data_type": data_type,
                "scene_key": scene_key,
                "method": method,
                "common_frames": len(frames),
                "different_frames": len(differences),
                "offline_positive_count": ours_positive,
                "reference_positive_count": reference_positive,
                "interpretation": (
                    "stochastic method; compare candidate/output contract"
                    if method == "Random"
                    else "deterministic parity check"
                ),
            })
    return report


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline-root", required=True)
    parser.add_argument("--reference-root", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    report = compare(args.offline_root, args.reference_root)
    parent = os.path.dirname(args.output)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    with open(args.output, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2))
