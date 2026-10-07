#!/usr/bin/env python
"""Batch the lightweight RiskBench adapters over the official test-prefix view.

This runner produces one official ROI JSON and one continuous-score JSON per
method/data type. It is intentionally limited to Random, Range, and the local
constant-velocity Kalman adapter; learned methods use their own official
pipelines and are tracked separately in artifacts/method_status.md.
"""

from __future__ import print_function

import argparse
import json
import os
import time

from single_scene_baselines import generate


DATA_TYPES = ["interactive", "collision", "obstacle", "non-interactive"]
METHODS = ["Random", "Range", "Kalman filter"]
TEST_PREFIXES = ("10", "A6", "B3")


def data_type_root(data_root, data_type):
    nested = os.path.join(data_root, data_type, data_type)
    return nested if os.path.isdir(nested) else os.path.join(data_root, data_type)


def list_scenes(data_root, data_type):
    root = data_type_root(data_root, data_type)
    scenes = []
    for basic in sorted(os.listdir(root)):
        if not basic.startswith(TEST_PREFIXES):
            continue
        variant_root = os.path.join(root, basic, "variant_scenario")
        if not os.path.isdir(variant_root):
            continue
        for variant in sorted(os.listdir(variant_root)):
            if os.path.isdir(os.path.join(variant_root, variant)):
                scenes.append((basic, variant))
    return scenes


def write_json(path, value):
    parent = os.path.dirname(path)
    if not os.path.isdir(parent):
        os.makedirs(parent)
    with open(path, "w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)


def main(args):
    selected = []
    for data_type in DATA_TYPES:
        scenes = list_scenes(args.data_root, data_type)
        if args.limit_per_type:
            scenes = scenes[: args.limit_per_type]
        selected.extend((data_type, basic, variant) for basic, variant in scenes)

    manifest = {
        "data_root": os.path.abspath(args.data_root),
        "output_root": os.path.abspath(args.output_root),
        "test_prefixes": list(TEST_PREFIXES),
        "methods": METHODS,
        "selected_scene_count": len(selected),
        "selected_by_type": {
            data_type: sum(1 for item in selected if item[0] == data_type)
            for data_type in DATA_TYPES
        },
        "started_at_unix": time.time(),
        "records": [],
    }

    combined = {
        method: {data_type: {} for data_type in DATA_TYPES} for method in METHODS
    }
    for index, (data_type, basic, variant) in enumerate(selected, 1):
        for method in METHODS:
            run_args = argparse.Namespace(
                data_root=args.data_root,
                data_type=data_type,
                basic=basic,
                variant=variant,
                method=method,
                output_root=args.output_root,
                seed=args.seed,
                range_m=args.range_m,
                horizon=args.horizon,
            )
            started = time.time()
            try:
                scene_key, scores, roi = generate(run_args)
                combined[method][data_type][scene_key] = roi
                score_key = "%s_%s_scores" % (data_type, basic)
                score_path = os.path.join(args.output_root, method, score_key + ".json")
                # Keep a per-scene score file for auditability; the merged ROI
                # evaluator consumes only the boolean file below.
                existing_scores = {}
                if os.path.exists(score_path):
                    with open(score_path) as handle:
                        existing_scores = json.load(handle)
                existing_scores[scene_key] = scores
                write_json(score_path, existing_scores)
                manifest["records"].append(
                    {
                        "data_type": data_type,
                        "basic": basic,
                        "variant": variant,
                        "method": method,
                        "status": "ok",
                        "frames": len(scores),
                        "seconds": round(time.time() - started, 3),
                    }
                )
            except Exception as exc:
                manifest["records"].append(
                    {
                        "data_type": data_type,
                        "basic": basic,
                        "variant": variant,
                        "method": method,
                        "status": "error",
                        "error": repr(exc),
                        "seconds": round(time.time() - started, 3),
                    }
                )
        if index % 10 == 0 or index == len(selected):
            print("completed scenes %d/%d" % (index, len(selected)))

    for method in METHODS:
        for data_type in DATA_TYPES:
            write_json(
                os.path.join(args.output_root, method, data_type + ".json"),
                combined[method][data_type],
            )
    manifest["finished_at_unix"] = time.time()
    manifest["ok_records"] = sum(
        record["status"] == "ok" for record in manifest["records"]
    )
    manifest["error_records"] = sum(
        record["status"] == "error" for record in manifest["records"]
    )
    write_json(os.path.join(args.output_root, "batch_manifest.json"), manifest)
    print(json.dumps({key: manifest[key] for key in (
        "selected_scene_count", "selected_by_type", "ok_records", "error_records"
    )}, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--limit-per-type", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--range-m", type=float, default=10.0)
    parser.add_argument("--horizon", type=int, default=30)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
