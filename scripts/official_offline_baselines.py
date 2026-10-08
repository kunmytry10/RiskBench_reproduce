#!/usr/bin/env python
"""Offline port of the official RiskBench Random/Range/Kalman rules.

The released planning-aware implementation normally receives actor data from
CARLA. This adapter reconstructs that same per-frame information from the
read-only RiskBench_Dataset archive, then applies the official rules:

* Random: choose one actor from ``all_ids`` (actors within 35 m).
* Range: choose the nearest visible object/obstacle and report it only when
  its official recorded distance is <= 10 m.
* Kalman: call the upstream ``models.KalmanFilter.kf_inference`` unchanged,
  after rebuilding its 20-frame trajectory input.

This script is deliberately separate from the earlier smoke adapter. Its
outputs are the candidates for official-rule reproduction, not yet paper
results until the small-scene parity checks pass.
"""

from __future__ import print_function

import argparse
import json
import os
import random
import sys

import numpy as np
import pandas as pd
import cv2
from PIL import Image


DATA_TYPES = ["interactive", "collision", "obstacle", "non-interactive"]
METHODS = ["Random", "Range", "Kalman filter"]
OFFICIAL_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "codes", "RiskBench", "Planning_Aware_Metric",
)
if OFFICIAL_ROOT not in sys.path:
    sys.path.insert(0, OFFICIAL_ROOT)
from models.KalmanFilter import KalmanFilter, kf_inference  # noqa: E402


def reset_official_kalman_state():
    """Mirror a fresh official data_generator.py process per scenario.

    The upstream KalmanFilter stores ``cv2.KalmanFilter`` as a class variable,
    so a batch worker would otherwise carry its state from one scenario into
    the next, unlike the official shell runner (one Python process/scenario).
    """
    KalmanFilter.kf = cv2.KalmanFilter(4, 2)
    KalmanFilter.kf.measurementMatrix = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0]], np.float32
    )
    KalmanFilter.kf.transitionMatrix = np.array(
        [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
        np.float32,
    )


def load_json(path):
    with open(path) as handle:
        return json.load(handle)


def norm_id(value):
    return int(value) % 65536


def frame_files(path):
    return sorted(
        (int(name[:8]), os.path.join(path, name))
        for name in os.listdir(path)
        if name.endswith(".json") and name[:8].isdigit()
    )


def locate_variant(data_root, data_type, basic, variant):
    for root in (
        os.path.join(data_root, data_type, data_type, basic),
        os.path.join(data_root, data_type, basic),
    ):
        path = os.path.join(root, "variant_scenario", variant)
        if os.path.isdir(path):
            return path
    raise IOError("variant not found: %s/%s/%s" % (data_type, basic, variant))


def actor_id_lists(attrs):
    return {
        "vehicles": [int(value) for value in attrs.get("vehicle", {})],
        "pedestrians": [int(value) for value in attrs.get("pedestrian", {})],
        "obstacles": [int(value) for value in attrs.get("obstacle", {})],
    }


def obstacle_segmentation_ids(variant_path, frame):
    path = os.path.join(
        variant_path, "instance_segmentation", "front", "%08d.png" % frame
    )
    if not os.path.exists(path):
        return set()
    image = np.asarray(Image.open(path))
    if image.ndim != 3 or image.shape[2] < 3:
        return set()
    semantic = image[:, :, 0].astype(np.int64)
    instances = image[:, :, 1].astype(np.int64) + 256 * image[:, :, 2].astype(np.int64)
    obstacle_instances = instances[semantic == 21]
    ids, areas = np.unique(obstacle_instances, return_counts=True)
    return {
        norm_id(value)
        for value, area in zip(ids, areas)
        if area >= 400
    }


def visible_object_ids(bbox_frame, ego_id):
    """Equivalent of official get_ids() object IDs, excluding the ego."""
    result = set()
    for raw_id in bbox_frame:
        try:
            value = norm_id(raw_id)
        except (TypeError, ValueError):
            continue
        if value != norm_id(ego_id):
            result.add(value)
    return result


def actor_state_by_id(actor_state, instance_id):
    for raw_id, state in actor_state.items():
        try:
            if norm_id(raw_id) == norm_id(instance_id):
                return int(raw_id), state
        except (TypeError, ValueError):
            continue
    return None, None


def build_scene(args):
    variant_path = locate_variant(args.data_root, args.data_type, args.basic, args.variant)
    attrs = load_json(os.path.join(variant_path, "actor_attribute.json"))
    ego_id = int(attrs["ego_id"])
    ids = actor_id_lists(attrs)
    bbox = load_json(os.path.join(variant_path, "bbox.json"))
    actors = {
        frame: load_json(path)
        for frame, path in frame_files(os.path.join(variant_path, "actors_data"))
    }
    egos = {
        frame: load_json(path)
        for frame, path in frame_files(os.path.join(variant_path, "ego_data"))
    }
    frames = sorted(set(actors) & set(egos) & {int(key) for key in bbox})
    if not frames:
        raise RuntimeError("no common frames")

    # A trajectory table matching the columns constructed by data_generator.py.
    rows = []
    for frame in frames:
        for raw_id, state in actors[frame].items():
            if not isinstance(state, dict) or "location" not in state:
                continue
            location = state["location"]
            velocity = state.get("velocity", {})
            actor_type = state.get("type", "vehicle")
            if actor_type not in ("vehicle", "pedestrian", "obstacle"):
                # collect_actor_data() does not add traffic lights to
                # all_ids or the trajectory dataframe.
                continue
            if actor_type == "obstacle":
                object_type = attrs.get("obstacle", {}).get(str(raw_id), {}).get(
                    "type_id", "static.prop.trafficcone01"
                )
            elif actor_type == "pedestrian":
                object_type = "pedestrian"
            else:
                object_type = "vehicle"
            rows.append([
                frame,
                int(raw_id),
                object_type,
                float(location["x"]),
                float(location["y"]),
                float(velocity.get("x", 0.0)),
                float(velocity.get("y", 0.0)),
                float(state.get("rotation", {}).get("yaw", 0.0)),
            ])
        # Static props are not present as actor records in actors_data. The
        # official CARLA collector appends them to df_list on every frame.
        for raw_id, state in attrs.get("obstacle", {}).items():
            location = state.get("location", {})
            if "x" not in location or "y" not in location:
                continue
            rows.append([
                frame,
                int(raw_id),
                state.get("type_id", "static.prop.trafficcone01"),
                float(location["x"]),
                float(location["y"]),
                0.0,
                0.0,
                float(state.get("rotation", {}).get("yaw", 0.0)),
            ])
    trajectories = pd.DataFrame(
        rows,
        columns=["FRAME", "TRACK_ID", "OBJECT_TYPE", "X", "Y",
                 "VELOCITY_X", "VELOCITY_Y", "YAW"],
    )
    return variant_path, attrs, ids, bbox, actors, egos, frames, trajectories, ego_id


def static_obstacle_distance(obstacle, ego_state):
    location = obstacle.get("location", {})
    ego_location = ego_state.get("location", {})
    if not all(key in location and key in ego_location for key in ("x", "y", "z")):
        return None
    dx = float(location["x"]) - float(ego_location["x"])
    dy = float(location["y"]) - float(ego_location["y"])
    dz = float(location["z"]) - float(ego_location["z"])
    return float(np.sqrt(dx * dx + dy * dy + dz * dz))


def official_random(actor_state, attrs, ego_id, ego_state, rng):
    """Match collect_actor_data()'s all_ids construction."""
    candidates = []
    for raw_id, state in actor_state.items():
        if not isinstance(state, dict) or "distance" not in state:
            continue
        try:
            value = int(raw_id)
        except (TypeError, ValueError):
            continue
        if value == int(ego_id):
            continue
        if state.get("type") not in ("vehicle", "pedestrian", "obstacle"):
            continue
        if float(state["distance"]) < 35.0:
            candidates.append(value)
    obstacles = actor_state.get("obstacle_ids", [])
    for raw_id in obstacles:
        state = attrs.get("obstacle", {}).get(str(raw_id), {})
        distance = static_obstacle_distance(state, ego_state)
        if distance is not None and distance < 35.0:
            candidates.append(int(raw_id))
    return norm_id(rng.choice(candidates)) if candidates else None


def official_range(
    actor_state, attrs, ego_state, actor_ids, bbox_frame,
    obstacle_ids_frame, ego_id, range_m
):
    """Match the exact nearest-object rule in data_generator.py."""
    visible_ids = visible_object_ids(bbox_frame, ego_id) | set(obstacle_ids_frame)
    if not visible_ids:
        return None
    best_distance = 1000.0
    best_id = None

    for raw_id in actor_ids["obstacles"]:
        state = attrs.get("obstacle", {}).get(str(raw_id))
        if norm_id(raw_id) in visible_ids and state is not None:
            distance = static_obstacle_distance(state, ego_state)
            if distance is None:
                continue
            if distance < best_distance:
                best_distance, best_id = distance, int(raw_id)

    for group in ("vehicles", "pedestrians"):
        for raw_id in actor_ids[group]:
            if int(raw_id) == int(ego_id):
                continue
            raw_key, state = actor_state_by_id(actor_state, raw_id)
            if norm_id(raw_id) in visible_ids and state is not None:
                distance = float(state["distance"])
                if distance < best_distance:
                    best_distance, best_id = distance, raw_key

    return norm_id(best_id) if best_id is not None and best_distance <= range_m else None


def official_kalman(
    trajectories, frame, actor_ids, ego_id
):
    """Build the official 20-frame local trajectory list and call upstream code."""
    vehicle_list = []
    for track_id, group in trajectories.groupby("TRACK_ID"):
        remain = group[group.FRAME > (int(frame) - 20)].copy()
        remain = remain.reset_index(drop=True)
        now = remain[remain.FRAME == int(frame)]
        if now.empty:
            continue
        actor_pos_x = float(now["X"].values[0])
        actor_pos_y = float(now["Y"].values[0])
        ego = trajectories[trajectories.TRACK_ID == int(ego_id)]
        ego_now = ego[ego.FRAME == int(frame)]
        if ego_now.empty:
            continue
        ego_x = float(ego_now["X"].values[0])
        ego_y = float(ego_now["Y"].values[0])
        if abs(actor_pos_x - ego_x) <= 37.5 and abs(actor_pos_y - ego_y) <= 37.5:
            vehicle_list.append(remain)
    if not vehicle_list:
        return None
    # The upstream data_generator passes this list as integer CARLA IDs.
    # Preserve that exact type; KalmanFilter.py itself performs the string
    # comparison in its obstacle branch.
    obstacle_ids = list(actor_ids["obstacles"])
    result = kf_inference(
        vehicle_list,
        int(frame),
        int(ego_id),
        actor_ids["pedestrians"],
        actor_ids["vehicles"],
        obstacle_ids,
    )
    return norm_id(result[0]) if result else None


def generate(args):
    if args.method == "Kalman filter":
        reset_official_kalman_state()
    (
        variant_path, attrs, actor_ids, bbox, actors, egos, frames,
        trajectories, ego_id,
    ) = build_scene(args)
    rng = random.Random(args.seed)
    roi = {}
    scores = {}
    for frame in frames:
        obstacle_ids = (
            obstacle_segmentation_ids(variant_path, frame)
            if args.data_type == "obstacle" else set()
        )
        if args.method == "Random":
            selected = official_random(
                actors[frame], attrs, ego_id, egos[frame], rng
            )
        elif args.method == "Range":
            selected = official_range(
                actors[frame], attrs, egos[frame], actor_ids,
                bbox["%08d" % frame], obstacle_ids, ego_id, args.range_m,
            )
        elif args.method == "Kalman filter":
            selected = official_kalman(trajectories, frame, actor_ids, ego_id)
        else:
            raise ValueError(args.method)
        # The official planning code reports a single risky ID (or none).
        ids = set(visible_object_ids(bbox["%08d" % frame], ego_id))
        ids.update(obstacle_ids)
        roi[str(frame)] = {
            str(instance_id): bool(instance_id == selected)
            for instance_id in sorted(ids)
        }
        scores[str(frame)] = {
            str(instance_id): (1.0 if instance_id == selected else 0.0)
            for instance_id in sorted(ids)
        }
    scene_key = "%s_%s" % (args.basic, args.variant)
    return scene_key, scores, roi


def write_scene(args, scene_key, scores, roi):
    output = os.path.join(args.output_root, args.method)
    if not os.path.isdir(output):
        os.makedirs(output)
    with open(os.path.join(output, "%s_%s_scores.json" %
                           (args.data_type, args.basic)), "w") as handle:
        json.dump({scene_key: scores}, handle, indent=2, sort_keys=True)
    with open(os.path.join(output, "%s.json" % args.data_type), "w") as handle:
        json.dump({scene_key: roi}, handle, indent=2, sort_keys=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--data-type", required=True, choices=DATA_TYPES)
    parser.add_argument("--basic", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--range-m", type=float, default=10.0)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = generate(arguments)
    write_scene(arguments, *result)
    print("official-rule method=%s scene=%s frames=%d" %
          (arguments.method, result[0], len(result[1])))
