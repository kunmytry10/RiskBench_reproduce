#!/usr/bin/env python
"""Export the shared test-scene universe from official baseline outputs."""

from __future__ import print_function

import argparse
import json
import os


DATA_TYPES = ["interactive", "collision", "obstacle", "non-interactive"]
METHODS = ["Random", "Range", "Kalman filter"]


def data_type_root(data_root, data_type):
    nested = os.path.join(data_root, data_type, data_type)
    return nested if os.path.isdir(nested) else os.path.join(data_root, data_type)


def available_scene_keys(data_root, data_type):
    root = data_type_root(data_root, data_type)
    result = set()
    for basic in os.listdir(root):
        variants = os.path.join(root, basic, "variant_scenario")
        if not os.path.isdir(variants):
            continue
        for variant in os.listdir(variants):
            if os.path.isdir(os.path.join(variants, variant)):
                result.add("%s_%s" % (basic, variant))
    return result


def export(args):
    manifest = {}
    summary = {"reference_root": os.path.abspath(args.reference_root), "by_type": {}}
    for data_type in DATA_TYPES:
        reference_sets = {}
        for method in METHODS:
            path = os.path.join(args.reference_root, method, data_type + ".json")
            with open(path) as handle:
                reference_sets[method] = set(json.load(handle))
        baseline = reference_sets[args.reference_method]
        inconsistent = {
            method: len(baseline.symmetric_difference(scene_keys))
            for method, scene_keys in reference_sets.items()
            if scene_keys != baseline
        }
        if inconsistent:
            raise ValueError("official baseline scene sets differ for %s: %s" %
                             (data_type, inconsistent))

        available = available_scene_keys(args.data_root, data_type)
        missing = sorted(baseline - available)
        if missing:
            raise ValueError("%d official %s test scenes are missing from data; examples: %s" %
                             (len(missing), data_type, missing[:5]))
        manifest[data_type] = sorted(baseline)
        summary["by_type"][data_type] = {
            "scene_count": len(baseline),
            "present_in_data": len(baseline & available),
            "missing_from_data": 0,
        }

    parent = os.path.dirname(args.output)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    with open(args.output, "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    summary["manifest"] = os.path.abspath(args.output)
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--reference-root", required=True)
    parser.add_argument("--reference-method", choices=METHODS, default="Random")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    export(parse_args())
