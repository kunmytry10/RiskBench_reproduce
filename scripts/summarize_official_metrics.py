#!/usr/bin/env python
"""Write a compact CSV/Markdown summary for official offline metrics."""

import argparse
import csv
import json
import os


METHODS = ["Random", "Range", "Kalman filter"]
DATA_TYPES = ["interactive", "obstacle", "collision", "non-interactive"]


def main(args):
    rows = []
    for method in METHODS:
        for data_type in DATA_TYPES:
            path = os.path.join(args.metrics_root, method, data_type + ".json")
            if not os.path.exists(path):
                continue
            with open(path) as handle:
                metric = json.load(handle)
            rows.append({
                "method": method,
                "data_type": data_type,
                "precision": metric["precision"],
                "recall": metric["recall"],
                "f1": metric["f1-Score"],
                "PIC": metric["PIC"],
                "FA": metric["FA"],
                "consistency_1s": metric["Consistency_1s"],
                "consistency_2s": metric["Consistency_2s"],
                "consistency_3s": metric["Consistency_3s"],
            })
    aggregate = []
    for method in METHODS:
        totals = {key: 0 for key in ("TP", "FN", "FP", "TN")}
        for row in rows:
            if row["method"] != method:
                continue
            path = os.path.join(
                args.metrics_root, method, row["data_type"] + ".json"
            )
            with open(path) as handle:
                matrix = json.load(handle)["confusion matrix"]
            for key in totals:
                totals[key] += int(matrix[key])
        tp, fn, fp = totals["TP"], totals["FN"], totals["FP"]
        precision = tp / float(tp + fp) if tp + fp else 0.0
        recall = tp / float(tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )
        aggregate.append({
            "method": method,
            "TP": tp,
            "FN": fn,
            "FP": fp,
            "TN": totals["TN"],
            "precision": "%.2f%%" % (100 * precision),
            "recall": "%.2f%%" % (100 * recall),
            "f1": "%.2f%%" % (100 * f1),
        })
    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    with open(args.csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(args.markdown, "w") as handle:
        handle.write("| Method | Type | Precision | Recall | F1 | PIC | FA |\n")
        handle.write("|---|---|---:|---:|---:|---:|---:|\n")
        for row in rows:
            handle.write(
                "| {method} | {data_type} | {precision} | {recall} | "
                "{f1} | {PIC} | {FA} |\n".format(**row)
            )
        handle.write("\nAll-scenario aggregate (summed confusion matrix):\n\n")
        handle.write("| Method | Precision | Recall | F1 |\n")
        handle.write("|---|---:|---:|---:|\n")
        for row in aggregate:
            handle.write(
                "| {method} | {precision} | {recall} | {f1} |\n".format(**row)
            )
    with open(
        os.path.join(os.path.dirname(args.csv), "aggregate.json"), "w"
    ) as handle:
        json.dump(aggregate, handle, indent=2)
    print("wrote", args.csv)
    print("wrote", args.markdown)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-root", default="artifacts/official_offline_metrics")
    parser.add_argument(
        "--csv", default="artifacts/official_offline_metrics/summary.csv"
    )
    parser.add_argument(
        "--markdown", default="artifacts/official_offline_metrics/summary.md"
    )
    main(parser.parse_args())
