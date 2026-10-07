#!/usr/bin/env python
"""Create an official-code-compatible RiskBench dataset view.

The released archive has an extra nesting level such as
RiskBench_Dataset/interactive/interactive/<basic>. The baseline code expects
RiskBench_Dataset/interactive/<basic>. This script creates a lightweight
workspace view using symlinks and generated tracking.npy files; raw data stay
read-only under /data.
"""

from __future__ import print_function

import argparse
import json
import os
import shutil

import numpy as np


TYPES = ["interactive", "collision", "obstacle", "non-interactive"]
MIN_AREA = 100


def link(src, dst):
    if os.path.lexists(dst):
        return
    os.symlink(os.path.abspath(src), dst)


def load(path):
    with open(path) as handle:
        return json.load(handle)


def make_tracking(variant_src, tracking_dst):
    bbox = load(os.path.join(variant_src, "bbox.json"))
    attrs = load(os.path.join(variant_src, "actor_attribute.json"))
    ego_id = int(attrs["ego_id"]) % 65536
    rows = []
    for frame_name, objects in sorted(bbox.items()):
        frame = int(frame_name)
        for actor_id, box in objects.items():
            actor_id = int(actor_id) % 65536
            if actor_id == ego_id:
                continue
            width = float(box[2]) - float(box[0])
            height = float(box[3]) - float(box[1])
            if width * height < MIN_AREA:
                continue
            rows.append(
                [frame, actor_id, box[0], box[1], width, height, 1, -1, -1, -1]
            )
    np.save(tracking_dst, np.asarray(rows, dtype=np.float32))


def prepare_type(source, output, data_type):
    source_type = os.path.join(source, data_type, data_type)
    if not os.path.isdir(source_type):
        source_type = os.path.join(source, data_type)
    output_type = os.path.join(output, data_type)
    os.makedirs(output_type, exist_ok=True)

    basics = sorted(
        name for name in os.listdir(source_type)
        if os.path.isdir(os.path.join(source_type, name))
    )
    for basic in basics:
        src_basic = os.path.join(source_type, basic)
        dst_basic = os.path.join(output_type, basic)
        os.makedirs(dst_basic, exist_ok=True)
        for name in os.listdir(src_basic):
            if name != "variant_scenario":
                link(os.path.join(src_basic, name), os.path.join(dst_basic, name))

        src_variants = os.path.join(src_basic, "variant_scenario")
        dst_variants = os.path.join(dst_basic, "variant_scenario")
        os.makedirs(dst_variants, exist_ok=True)
        for variant in sorted(os.listdir(src_variants)):
            src_variant = os.path.join(src_variants, variant)
            dst_variant = os.path.join(dst_variants, variant)
            os.makedirs(dst_variant, exist_ok=True)
            for name in os.listdir(src_variant):
                link(
                    os.path.join(src_variant, name),
                    os.path.join(dst_variant, name),
                )
            make_tracking(src_variant, os.path.join(dst_variant, "tracking.npy"))
    return len(basics)


def prepare(args):
    os.makedirs(args.output, exist_ok=True)
    counts = {}
    for data_type in TYPES:
        counts[data_type] = prepare_type(args.source, args.output, data_type)

    metadata = os.path.join(args.metadata, "metadata")
    if not os.path.isdir(metadata):
        metadata = args.metadata
    for name in ["behavior", "state", "GT_risk", "GT_critical_point"]:
        src = os.path.join(metadata, name)
        if os.path.isdir(src):
            link(src, os.path.join(args.output, name))

    datasets_dir = os.path.join(args.output, "datasets")
    os.makedirs(datasets_dir, exist_ok=True)
    skip = os.path.join(datasets_dir, "skip_scenario.json")
    if not os.path.exists(skip):
        with open(skip, "w") as handle:
            json.dump([], handle)

    print(json.dumps({"output": args.output, "basic_counts": counts}, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
