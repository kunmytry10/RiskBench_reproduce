#!/usr/bin/env python
"""Run lightweight RiskBench baselines on one stored scenario variant.

This adapter reads the currently available RiskBench_Dataset layout
(actors_data/ego_data/bbox.json). The upstream trajectory baseline scripts
expect a different intermediate layout containing trajectory_frame/*.csv;
we keep this conversion local to the reproduction repository and never
write to the read-only dataset.

The output has two files:
  * scores.json: continuous distance/risk scores where available;
  * roi.json: official ROI-tool compatible boolean decisions.

The Kalman implementation is deliberately marked exploratory: it uses a
constant-velocity forecast from the stored actor state and is intended to
validate the data/metric contract before porting the full upstream script.
"""

from __future__ import print_function

import argparse
import json
import math
import os
import random


def load_json(path):
    with open(path) as handle:
        return json.load(handle)


def norm_id(actor_id):
    return int(actor_id) % 65536


def xy(entry):
    loc = entry["location"]
    return float(loc["x"]), float(loc["y"])


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def frame_files(path):
    files = []
    for name in os.listdir(path):
        if name.endswith(".json") and name[:8].isdigit():
            files.append((int(name[:8]), os.path.join(path, name)))
    return sorted(files)


def predict_position(entry, steps):
    x, y = xy(entry)
    vel = entry.get("velocity", {})
    return (
        x + float(vel.get("x", 0.0)) * 0.05 * steps,
        y + float(vel.get("y", 0.0)) * 0.05 * steps,
    )


def generate(args):
    scenario_root = os.path.join(args.data_root, args.data_type, args.basic)
    nested_root = os.path.join(
        args.data_root, args.data_type, args.data_type, args.basic
    )
    if os.path.isdir(nested_root):
        scenario_root = nested_root
    variant_path = os.path.join(
        scenario_root, "variant_scenario", args.variant
    )
    actor_path = os.path.join(variant_path, "actors_data")
    ego_path = os.path.join(variant_path, "ego_data")
    bbox = load_json(os.path.join(variant_path, "bbox.json"))
    attrs = load_json(os.path.join(variant_path, "actor_attribute.json"))
    ego_instance_id = norm_id(attrs["ego_id"])

    actors_by_frame = {
        frame: load_json(path) for frame, path in frame_files(actor_path)
    }
    egos_by_frame = {
        frame: load_json(path) for frame, path in frame_files(ego_path)
    }
    frames = sorted(set(actors_by_frame) & set(egos_by_frame) & {
        int(key) for key in bbox
    })
    if not frames:
        raise RuntimeError("No common actor/ego/bbox frames found")

    rng = random.Random(args.seed)
    roi = {}
    scores = {}
    for frame in frames:
        key = "%08d" % frame
        actor_state = actors_by_frame[frame]
        ego_state = egos_by_frame[frame]
        ego_xy = xy(ego_state)
        visible = []
        for raw_id in bbox[key]:
            instance_id = norm_id(raw_id)
            if instance_id == ego_instance_id:
                continue
            state = None
            for full_id, candidate in actor_state.items():
                if not str(full_id).isdigit():
                    continue
                if norm_id(full_id) == instance_id:
                    state = candidate
                    break
            if state is not None:
                visible.append((str(instance_id), state))

        frame_scores = {}
        frame_roi = {}
        selected = None
        if args.method == "Random" and visible:
            selected = rng.choice(visible)[0]

        for instance_id, state in visible:
            current_distance = distance(xy(state), ego_xy)
            if args.method == "Random":
                score = 1.0 if instance_id == selected else 0.0
                risky = instance_id == selected
            elif args.method == "Range":
                score = max(0.0, 1.0 - current_distance / args.range_m)
                risky = current_distance <= args.range_m
            elif args.method == "Kalman filter":
                # Forecast relative motion for the next horizon. This is a
                # local contract smoke test, not yet the full upstream
                # rectangle-collision implementation.
                min_distance = current_distance
                for step in range(1, args.horizon + 1):
                    actor_future = predict_position(state, step)
                    ego_future = predict_position(ego_state, step)
                    min_distance = min(
                        min_distance, distance(actor_future, ego_future)
                    )
                score = max(0.0, 1.0 - min_distance / args.range_m)
                risky = min_distance <= args.range_m
            else:
                raise ValueError("Unsupported method: %s" % args.method)
            frame_scores[instance_id] = round(float(score), 6)
            frame_roi[instance_id] = bool(risky)

        roi[str(frame)] = frame_roi
        scores[str(frame)] = frame_scores

    scenario_key = "%s_%s" % (args.basic, args.variant)
    return scenario_key, scores, roi


def run(args):
    scenario_key, scores, roi = generate(args)
    output_dir = os.path.join(args.output_root, args.method)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, "%s_%s_scores.json" % (args.data_type, args.basic)), "w") as handle:
        json.dump({scenario_key: scores}, handle, indent=2, sort_keys=True)
    with open(os.path.join(output_dir, "%s.json" % args.data_type), "w") as handle:
        json.dump({scenario_key: roi}, handle, indent=2, sort_keys=True)
    print("method=%s scenario=%s frames=%d output=%s" %
          (args.method, scenario_key, len(frames), output_dir))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--data-type", default="interactive")
    parser.add_argument("--basic", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument(
        "--method", required=True,
        choices=["Random", "Range", "Kalman filter"],
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--range-m", type=float, default=10.0)
    parser.add_argument("--horizon", type=int, default=30)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
