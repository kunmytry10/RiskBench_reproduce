#!/usr/bin/env python
"""Create compact, presentation-ready RiskBench report assets.

The script only reads the released archive and reproduction outputs. It writes
PNG/JSON/CSV files under the repository's artifacts directory.
"""

from __future__ import print_function

import argparse
import csv
import json
import os
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image


DATA_TYPES = ["interactive", "collision", "obstacle", "non-interactive"]
METHODS = ["Random", "Range", "Kalman filter"]
COLORS = {"Random": "#8c564b", "Range": "#1f77b4", "Kalman filter": "#2ca02c"}


def load_json(path):
    with open(path) as handle:
        return json.load(handle)


def norm_id(actor_id):
    return str(int(actor_id) % 65536)


def is_actor_id(value):
    try:
        int(value)
        return True
    except (TypeError, ValueError):
        return False


def frame_number(name):
    return int(os.path.splitext(name)[0])


def frame_files(directory):
    return sorted(
        (frame_number(name), os.path.join(directory, name))
        for name in os.listdir(directory)
        if name.endswith(".json") and name[:8].isdigit()
    )


def locate_variant(data_root, data_type, basic, variant):
    candidates = [
        os.path.join(data_root, data_type, data_type, basic, "variant_scenario", variant),
        os.path.join(data_root, data_type, basic, "variant_scenario", variant),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    raise IOError("Variant not found: %s/%s/%s" % (data_type, basic, variant))


def locate_metadata(metadata_root, name, data_type):
    path = os.path.join(metadata_root, name, data_type + ".json")
    return load_json(path)


def actor_xy(state):
    location = state["location"]
    return float(location["x"]), float(location["y"])


def distance(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def actor_states(variant_path):
    return {
        frame: load_json(path)
        for frame, path in frame_files(os.path.join(variant_path, "actors_data"))
    }


def ego_states(variant_path):
    return {
        frame: load_json(path)
        for frame, path in frame_files(os.path.join(variant_path, "ego_data"))
    }


def raw_state_by_normalized_id(states, instance_id):
    for raw_id, state in states.items():
        if norm_id(raw_id) == instance_id:
            return state
    return None


def obstacle_segmentation_boxes(variant_path, frame):
    """Mirror the official get_ids() obstacle branch.

    RiskBench raw instance segmentation stores the semantic class in channel
    0 and the instance ID in channels 1/2. Official code uses class 21 for
    static obstacles and computes a box from each instance mask.
    """
    path = os.path.join(
        variant_path, "instance_segmentation", "front", "%08d.png" % frame
    )
    if not os.path.exists(path):
        return {}
    image = np.asarray(Image.open(path))
    if image.ndim != 3 or image.shape[2] < 3:
        return {}
    semantic = image[:, :, 0].astype(np.int64)
    instance = image[:, :, 1].astype(np.int64) + 256 * image[:, :, 2].astype(np.int64)
    boxes = {}
    for raw_instance in np.unique(instance[semantic == 21]):
        ys, xs = np.where((semantic == 21) & (instance == raw_instance))
        if len(xs) == 0:
            continue
        boxes[str(int(raw_instance) % 65536)] = [
            int(xs.min()),
            int(ys.min()),
            int(xs.max()),
            int(ys.max()),
        ]
    return boxes


def obstacle_geometry(attrs):
    geometry = {}
    for raw_id, info in attrs.get("obstacle", {}).items():
        instance_id = norm_id(raw_id)
        corners = info.get("cord_bounding_box", {})
        try:
            # Same four ground-plane corners selected by the official BEV
            # visualizer: cord_0, cord_4, cord_6, cord_2.
            points = [
                (float(corners["cord_%d" % i][0]), float(corners["cord_%d" % i][1]))
                for i in (0, 4, 6, 2, 0)
            ]
        except (KeyError, TypeError, ValueError):
            continue
        geometry[instance_id] = {
            "points": points,
            "type_id": info.get("type_id", "obstacle"),
        }
    return geometry


def load_scores(score_root, method, data_type, scene_key):
    path = os.path.join(
        score_root,
        method,
        "%s_%s_scores.json" % (data_type, scene_key.rsplit("_", 3)[0]),
    )
    if not os.path.exists(path):
        candidates = [
            os.path.join(score_root, method, name)
            for name in os.listdir(os.path.join(score_root, method))
            if name.endswith("_scores.json")
        ]
        if not candidates:
            return {}
        path = candidates[0]
    data = load_json(path)
    return data.get(scene_key, {})


def load_roi(roi_root, method, data_type, scene_key):
    path = os.path.join(roi_root, method, data_type + ".json")
    if not os.path.exists(path):
        return {}
    return load_json(path).get(scene_key, {})


def plot_scene(args):
    variant_path = locate_variant(args.data_root, args.data_type, args.basic, args.variant)
    scene_key = "%s_%s" % (args.basic, args.variant)
    bbox = load_json(os.path.join(variant_path, "bbox.json"))
    attrs = load_json(os.path.join(variant_path, "actor_attribute.json"))
    ego_id = norm_id(attrs["ego_id"])
    obstacle_shapes = obstacle_geometry(attrs)
    gt_risk = locate_metadata(args.metadata_root, "GT_risk", args.data_type).get(scene_key, [])
    gt_ids = set(norm_id(item) for item in gt_risk)
    critical = locate_metadata(args.metadata_root, "GT_critical_point", args.data_type).get(scene_key)
    critical = int(critical) if critical is not None else None
    actors = actor_states(variant_path)
    egos = ego_states(variant_path)
    frames = sorted(set(actors) & set(egos) & set(int(k) for k in bbox))
    if not frames:
        raise RuntimeError("No common frames for %s" % scene_key)
    if critical not in frames:
        critical = frames[len(frames) // 2]

    behavior_interval = None
    behavior_path = os.path.join(args.metadata_root, "behavior", args.data_type + ".json")
    if os.path.exists(behavior_path):
        behavior = load_json(behavior_path)
        if args.basic in behavior and args.variant in behavior[args.basic]:
            behavior_interval = tuple(int(v) for v in behavior[args.basic][args.variant])
    if args.data_type == "collision":
        behavior_interval = (frames[0], frames[-1])

    score_data = {
        method: load_scores(args.score_root, method, args.data_type, scene_key)
        for method in METHODS
    }
    roi_data = {
        method: load_roi(args.roi_root, method, args.data_type, scene_key)
        for method in METHODS
    }

    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.08, 1.0], height_ratios=[1.0, 1.0])
    ax_front = fig.add_subplot(grid[0, 0])
    ax_bev = fig.add_subplot(grid[0, 1])
    ax_score = fig.add_subplot(grid[1, :])

    rgb_path = os.path.join(variant_path, "rgb", "front", "%08d.jpg" % critical)
    if not os.path.exists(rgb_path):
        rgb_path = os.path.join(variant_path, "rgb", "front", "%08d.png" % critical)
    if os.path.exists(rgb_path):
        ax_front.imshow(mpimg.imread(rgb_path))
    ax_front.set_title(
        "Front view | %s | frame %d | GT risk IDs: %s"
        % (scene_key, critical, ", ".join(sorted(gt_ids)) or "none"),
        fontsize=10,
    )
    frame_boxes = dict(bbox[str(critical).zfill(8)])
    if args.data_type == "obstacle":
        for instance_id, box in obstacle_segmentation_boxes(variant_path, critical).items():
            frame_boxes.setdefault(instance_id, box)
    frame_box_ids = {
        norm_id(raw_id) for raw_id in frame_boxes if is_actor_id(raw_id)
    }
    frame_roi = {
        method: roi_data[method].get(str(critical), {}) for method in METHODS
    }
    for raw_id, box in frame_boxes.items():
        if not is_actor_id(raw_id):
            continue
        instance_id = norm_id(raw_id)
        if instance_id == ego_id:
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        if instance_id in gt_ids:
            edge = "#d62728"
            linewidth = 3.0
            label = "GT"
        elif args.data_type == "obstacle" and instance_id in obstacle_shapes:
            edge = "#ff8c00"
            linewidth = 2.0
            label = "obstacle"
        else:
            edge = "#bdbdbd"
            linewidth = 1.0
            label = None
        ax_front.add_patch(
            Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor=edge,
                linewidth=linewidth,
            )
        )
        if label:
            ax_front.text(x1, max(0, y1 - 4), "%s %s" % (label, instance_id), color=edge, fontsize=8)
    missing_gt = sorted(gt_ids - frame_box_ids)
    if missing_gt:
        ax_front.text(
            0.01,
            0.03,
            "GT IDs without front bbox at this frame: %s"
            % ", ".join(missing_gt),
            transform=ax_front.transAxes,
            color="#b22222",
            fontsize=8,
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
        )
    ax_front.set_axis_off()

    visible_ids = set()
    for frame in frames:
        for raw_id in bbox[str(frame).zfill(8)]:
            if is_actor_id(raw_id):
                instance_id = norm_id(raw_id)
                if instance_id != ego_id:
                    visible_ids.add(instance_id)
    visible_ids.update(gt_ids)

    trajectories = {}
    for frame in frames:
        ego_xy = actor_xy(egos[frame])
        trajectories.setdefault(ego_id, []).append((frame, ego_xy[0], ego_xy[1]))
        for raw_id, state in actors[frame].items():
            if not is_actor_id(raw_id) or not isinstance(state, dict) or "location" not in state:
                continue
            instance_id = norm_id(raw_id)
            if instance_id == ego_id or instance_id not in visible_ids:
                continue
            if state.get("type") not in ("vehicle", "pedestrian"):
                continue
            trajectories.setdefault(instance_id, []).append(
                (frame, actor_xy(state)[0], actor_xy(state)[1])
            )
    for instance_id, points in trajectories.items():
        points.sort()
        xs = [item[1] for item in points]
        ys = [item[2] for item in points]
        if instance_id == ego_id:
            ax_bev.plot(xs, ys, color="black", linewidth=3, label="ego")
        elif instance_id in gt_ids:
            ax_bev.plot(xs, ys, color="#d62728", linewidth=2.5)
            ax_bev.scatter(xs[-1], ys[-1], color="#d62728", s=20)
        else:
            ax_bev.plot(xs, ys, color="#bdbdbd", linewidth=0.7, alpha=0.6)
    if args.data_type == "obstacle":
        critical_ego = actor_xy(egos[critical])
        active_obstacles = {
            norm_id(raw_id)
            for raw_id, state in actors[critical].items()
            if is_actor_id(raw_id)
            and isinstance(state, dict)
            and state.get("type") == "obstacle"
        }
        for instance_id, shape in obstacle_shapes.items():
            if instance_id not in active_obstacles and instance_id not in gt_ids:
                continue
            points = shape["points"]
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            center = (
                sum(point[0] for point in points[:-1]) / 4.0,
                sum(point[1] for point in points[:-1]) / 4.0,
            )
            near_ego = distance(center, critical_ego) <= 60.0
            if instance_id not in gt_ids and not near_ego:
                continue
            if instance_id in gt_ids:
                color, width, alpha = "#d62728", 2.5, 0.95
            else:
                color, width, alpha = "#ff8c00", 1.5, 0.75
            ax_bev.plot(xs, ys, color=color, linewidth=width, alpha=alpha)
            if instance_id in gt_ids:
                ax_bev.text(xs[0], ys[0], "GT %s" % instance_id, color=color, fontsize=8)
    ego_at_critical = actor_xy(egos[critical])
    ax_bev.scatter(
        [ego_at_critical[0]], [ego_at_critical[1]], marker="*", s=120, color="#111111", zorder=5
    )
    ax_bev.set_title("BEV trajectories (red = GT risky actor, star = ego at critical frame)", fontsize=10)
    ax_bev.set_xlabel("world x (m)")
    ax_bev.set_ylabel("world y (m)")
    ax_bev.grid(alpha=0.25)
    ax_bev.set_aspect("equal", adjustable="datalim")
    ax_bev.legend(
        handles=[
            Line2D([0], [0], color="black", linewidth=3, label="ego"),
            Line2D([0], [0], color="#d62728", linewidth=2.5, label="GT risky actor"),
            Line2D([0], [0], color="#ff8c00", linewidth=1.5, label="static obstacle"),
            Line2D([0], [0], marker="*", color="#111111", linestyle="", markersize=10, label="critical frame"),
        ],
        loc="best",
        fontsize=8,
    )

    for method in METHODS:
        values = []
        for frame in frames:
            frame_scores = score_data[method].get(str(frame), {})
            values.append(max([float(v) for v in frame_scores.values()] or [0.0]))
        ax_score.plot(frames, values, color=COLORS[method], linewidth=2, label=method)
    if critical is not None:
        ax_score.axvline(critical, color="#d62728", linestyle="--", linewidth=1.5, label="GT critical point")
    ax_score.set_title("Maximum predicted actor risk score by frame", fontsize=10)
    ax_score.set_xlabel("frame")
    ax_score.set_ylabel("continuous score")
    ax_score.set_ylim(-0.03, 1.03)
    ax_score.grid(alpha=0.25)
    if behavior_interval is not None:
        ax_score.axvspan(
            behavior_interval[0],
            behavior_interval[1],
            color="#d62728",
            alpha=0.10,
            label="GT positive interval",
        )
        ax_score.text(
            behavior_interval[0],
            0.94,
            "GT interval [%d, %d]" % behavior_interval,
            color="#b22222",
            fontsize=8,
            va="top",
        )
    ax_score.legend(ncol=4, fontsize=8, loc="upper right")

    os.makedirs(args.output_dir, exist_ok=True)
    out_png = os.path.join(args.output_dir, "scene_%s.png" % scene_key)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    summary = {
        "scene_key": scene_key,
        "data_type": args.data_type,
        "basic": args.basic,
        "variant": args.variant,
        "num_frames": len(frames),
        "frame_start": frames[0],
        "frame_end": frames[-1],
        "critical_frame": critical,
        "gt_positive_interval": list(behavior_interval) if behavior_interval else None,
        "gt_risk_ids": sorted(gt_ids),
        "ego_id": ego_id,
        "methods": {},
    }
    for method in METHODS:
        predicted = set()
        for frame_dict in roi_data[method].values():
            predicted.update(norm_id(k) for k, value in frame_dict.items() if value)
        summary["methods"][method] = {
            "frames_with_any_positive": sum(
                any(frame_dict.values()) for frame_dict in roi_data[method].values()
            ),
            "predicted_positive_ids": sorted(predicted),
            "max_score": max(
                [float(v) for frame_dict in score_data[method].values() for v in frame_dict.values()] or [0.0]
            ),
        }
    with open(os.path.join(args.output_dir, "scene_%s.json" % scene_key), "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return out_png, summary


def dataset_summary(args):
    rows = []
    for data_type in DATA_TYPES:
        root = os.path.join(args.data_root, data_type, data_type)
        if not os.path.isdir(root):
            root = os.path.join(args.data_root, data_type)
        for basic in sorted(os.listdir(root)):
            basic_path = os.path.join(root, basic)
            if not os.path.isdir(basic_path):
                continue
            variants_root = os.path.join(basic_path, "variant_scenario")
            for variant in sorted(os.listdir(variants_root)):
                variant_path = os.path.join(variants_root, variant)
                bbox_path = os.path.join(variant_path, "bbox.json")
                if not os.path.exists(bbox_path):
                    continue
                bbox = load_json(bbox_path)
                rows.append(
                    {
                        "data_type": data_type,
                        "basic": basic,
                        "variant": variant,
                        "scene_key": basic + "_" + variant,
                        "num_frames": len(bbox),
                        "is_test_prefix": basic.startswith(("10", "A6", "B3")),
                    }
                )
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "dataset_scene_inventory.csv")
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(row["data_type"] for row in rows)
    test_counts = Counter(row["data_type"] for row in rows if row["is_test_prefix"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(DATA_TYPES))
    width = 0.36
    ax.bar(x - width / 2, [counts[item] for item in DATA_TYPES], width, label="all archive", color="#9ecae1")
    ax.bar(x + width / 2, [test_counts[item] for item in DATA_TYPES], width, label="test-prefix view", color="#2171b5")
    ax.set_xticks(x)
    ax.set_xticklabels(DATA_TYPES, rotation=20, ha="right")
    ax.set_ylabel("number of scenario variants")
    ax.set_title("RiskBench archive inventory")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "dataset_split_counts.png"), dpi=180)
    plt.close(fig)

    summary = {
        "total_variants": len(rows),
        "all_counts": dict(counts),
        "test_prefix_counts": dict(test_counts),
        "frame_statistics": {
            "min": min(row["num_frames"] for row in rows),
            "median": float(np.median([row["num_frames"] for row in rows])),
            "max": max(row["num_frames"] for row in rows),
        },
    }
    with open(os.path.join(args.output_dir, "dataset_summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary


def official_reference_summary(args):
    source = args.official_metrics_root
    rows = []
    if not os.path.isdir(source):
        return {"available": False, "reason": "official metrics directory not found"}
    for method in sorted(os.listdir(source)):
        method_dir = os.path.join(source, method)
        if not os.path.isdir(method_dir):
            continue
        for data_type in DATA_TYPES:
            path = os.path.join(method_dir, data_type + ".json")
            if not os.path.isfile(path):
                continue
            result = load_json(path)
            matrix = result.get("confusion matrix", {})
            rows.append(
                {
                    "method": method,
                    "data_type": data_type,
                    "precision": result.get("precision", ""),
                    "recall": result.get("recall", ""),
                    "f1": result.get("f1-Score", ""),
                    "pic": result.get("PIC", ""),
                    "tp": matrix.get("TP", ""),
                    "fn": matrix.get("FN", ""),
                    "fp": matrix.get("FP", ""),
                    "tn": matrix.get("TN", ""),
                    "provenance": "author-provided predictions; recomputed with official evaluator",
                }
            )
    if not rows:
        return {"available": False, "reason": "no metric JSON files found"}

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "official_prediction_reference_metrics.csv")
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    interactive = [row for row in rows if row["data_type"] == "interactive" and row["f1"]]
    interactive.sort(key=lambda row: float(str(row["f1"]).rstrip("%")), reverse=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    methods = [row["method"] for row in interactive]
    f1 = [float(str(row["f1"]).rstrip("%")) for row in interactive]
    bars = ax.barh(methods[::-1], f1[::-1], color="#3b82a0")
    ax.set_xlim(0, max(f1 or [1]) * 1.18)
    ax.set_xlabel("F1 (%)")
    ax.set_title("Interactive split | author-provided prediction reference")
    ax.grid(axis="x", alpha=0.25)
    for bar, value in zip(bars, f1[::-1]):
        ax.text(value + 0.25, bar.get_y() + bar.get_height() / 2, "%.2f%%" % value, va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "official_prediction_reference_interactive_f1.png"), dpi=180)
    plt.close(fig)
    return {
        "available": True,
        "num_method_split_results": len(rows),
        "num_interactive_methods": len(interactive),
        "provenance": "author-provided predictions; recomputed with official evaluator; not new inference",
        "csv": csv_path,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--metadata-root", required=True)
    parser.add_argument("--score-root", required=True)
    parser.add_argument("--roi-root", required=True)
    parser.add_argument(
        "--official-metrics-root",
        default="artifacts/metrics/official_recomputed",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-type", default="interactive")
    parser.add_argument("--basic", default="10_i-1_1_c_f_f_1_rl")
    parser.add_argument("--variant", default="ClearSunset_low_")
    parser.add_argument("--skip-dataset-summary", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    out_png, scene_summary = plot_scene(args)
    dataset = None if args.skip_dataset_summary else dataset_summary(args)
    official = official_reference_summary(args)
    print(
        json.dumps(
            {
                "scene_plot": out_png,
                "scene": scene_summary,
                "dataset": dataset,
                "official_prediction_reference": official,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
