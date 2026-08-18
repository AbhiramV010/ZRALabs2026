"""Writes the graphs and tables out as files, for dropping into a document.

    python statistics/make_assets.py

Graphs go to `graphs/` as standalone SVG - vector, so they stay sharp at
any size in LaTeX or Word, and each one carries its own colours rather
than inheriting them from the report page. Tables go to `tables/` as CSV.

Both are generated from `bench_results.json`, the same file the report
page reads, so nothing here can drift away from the measurements.
"""

import csv

from pathlib import Path

from charts import chart_backbones, chart_profiles, chart_threads, chart_training
from common import RESULTS

HERE = Path(__file__).resolve().parent
GRAPHS = HERE / "graphs"
TABLES = HERE / "tables"

# A standalone SVG cannot reach the page's custom properties, so it
# carries them itself. The dark block is the same treatment the report
# uses, which keeps a figure readable if the document it lands in is dark.
SVG_STYLE = """<style>
  svg { --surface:#fbfcfa; --ink-2:#4b5551; --ink-3:#7a847f;
        --rule:#dde1da; --axis:#c2c9bf; --accent:#8a4b1f;
        --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a;
        background:#fbfcfa; }
  @media (prefers-color-scheme: dark) {
    svg { --surface:#181c1a; --ink-2:#a9b3ae; --ink-3:#7f8a85;
          --rule:#2a302d; --axis:#3a423e; --accent:#cf8a55;
          --s1:#3987e5; --s2:#d95926; --s3:#199e70;
          background:#181c1a; }
  }
  text { font-family: ui-monospace, Consolas, "DejaVu Sans Mono", monospace; }
  .tick { font-size:11.5px; fill:var(--ink-3); }
  .axis-title { font-size:11.5px; fill:var(--ink-3); letter-spacing:0.06em; }
  .bar-value { font-size:11.5px; fill:var(--ink-2); }
  .row-label { font-size:12.5px; fill:var(--ink-2); }
  .panel-head { font-size:11px; fill:var(--ink-3); letter-spacing:0.12em;
                text-transform:uppercase; }
  .note-in { font-size:11px; fill:var(--ink-3); }
  .end-label { font-size:12px; fill:var(--ink-2); }
  .muted-label { fill:var(--ink-3); }
</style>"""


def standalone(markup, title):
    """One chart as a file that renders on its own."""
    opening, rest = markup.split(">", 1)

    opening = opening.replace(
        "<svg ",
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'xmlns:xlink="http://www.w3.org/1999/xlink" ',
    )

    return (f'<?xml version="1.0" encoding="UTF-8"?>\n{opening}>'
            f"<title>{title}</title>{SVG_STYLE}{rest}\n")


def write_graphs():
    figures = [
        ("training-accuracy", chart_training,
         "Training and validation accuracy per epoch"),
        ("backbone-size-vs-speed", chart_backbones,
         "Backbone parameter count against measured latency"),
        ("scan-latency-by-profile", chart_profiles,
         "Scan latency by profile and frame size"),
        ("thread-scaling", chart_threads,
         "Speedup against thread count"),
    ]

    for name, builder, title in figures:
        path = GRAPHS / f"{name}.svg"
        path.write_text(standalone(builder(), title), encoding="utf-8")

        print(f"  graphs/{name}.svg  ({path.stat().st_size:,} bytes)")


def write_csv(name, headers, rows):
    path = TABLES / f"{name}.csv"

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)

    print(f"  tables/{name}.csv  ({len(rows)} rows)")


def write_tables():
    tr = RESULTS["training"]["main"]
    series = tr["series"]

    write_csv(
        "training-curve",
        ["epoch", "phase", "train_loss", "train_acc", "val_loss", "val_acc",
         "train_val_gap"],
        [
            [
                index + 1,
                "head_only" if index + 1 < tr["phase_boundary_epoch"]
                else "fine_tuning",
                series["train_loss"][index],
                series["train_acc"][index],
                series["val_loss"][index],
                series["val_acc"][index],
                round(series["train_acc"][index] - series["val_acc"][index], 4),
            ]
            for index in range(tr["epochs"])
        ],
    )

    specs = {b["architecture"]: b for b in RESULTS["backbones"]["backbones"]}
    var = RESULTS["variance"]["backbones_batch_32"]

    write_csv(
        "backbones",
        ["architecture", "parameters", "smaller_than_resnet18",
         "ms_per_image_batch32", "faster_than_resnet18", "spread_cv_percent",
         "float32_mb", "torchscript_mb", "trainable_head_only",
         "trainable_finetune"],
        [
            [
                arch,
                specs[arch]["parameters"],
                specs[arch]["vs_resnet18_parameters"],
                var[arch]["per_image_ms_mean"],
                var[arch]["speed_vs_resnet18_mean"],
                var[arch]["per_image_ms_cv_percent"],
                specs[arch]["float32_mb"],
                specs[arch]["torchscript_mb"],
                specs[arch]["trainable_head_only"],
                specs[arch]["trainable_finetune"],
            ]
            for arch in sorted(var, key=lambda a: -specs[a]["parameters"])
        ],
    )

    write_csv(
        "scan-windows",
        ["frame", "width", "height", "aspect_ratio", "megapixels",
         "windows_full", "windows_balanced", "windows_edge"],
        [
            [w["resolution"], w["width"], w["height"], w["aspect_ratio"],
             w["megapixels"], w["full"], w["balanced"], w["edge"]]
            for w in RESULTS["windows"]
        ],
    )

    write_csv(
        "scan-latency",
        ["frame", "profile", "windows", "milliseconds", "ms_per_window",
         "scans_per_second"],
        [
            [s["resolution"], s["profile"], s["windows"], s["headline_ms"],
             s["ms_per_window"], s["frames_per_second"]]
            for s in RESULTS["profiles"]["scans"]
        ],
    )

    write_csv(
        "thread-scaling",
        ["threads", "ms_per_batch_of_8", "ms_per_image", "speedup",
         "parallel_efficiency"],
        [
            [m["threads"], m["median_ms"], m["per_image_ms"],
             m["speedup_vs_one_thread"], m["parallel_efficiency"]]
            for m in RESULTS["threads"]["measurements"]
        ],
    )

    write_csv(
        "runtimes",
        ["runtime", "file", "graph_bytes", "sidecar_bytes", "total_bytes",
         "max_abs_logit_difference", "agrees_with_torch", "speedup_vs_torch"],
        [
            [r["runtime"], r["file"], r.get("bytes", ""),
             r.get("sidecar_bytes", 0),
             r.get("total_bytes", r.get("bytes", "")),
             r["max_abs_difference"], r["agrees_with_torch"],
             r["speedup_vs_torch"]]
            for r in RESULTS["runtimes"]["runtimes"]
        ],
    )

    write_csv(
        "sync-tiers",
        ["tier", "batch_size", "sends_thumbnails", "captures_in_batch",
         "json_bytes", "gzip_bytes", "compression_ratio",
         "gzip_bytes_per_capture"],
        [
            [t["tier"], t["batch_size"], t["thumbnails"],
             t["captures_in_batch"], t["json_bytes"], t["gzip_bytes"],
             t["compression_ratio"], t["gzip_bytes_per_capture"]]
            for t in RESULTS["sync"]["tiers"]
        ],
    )

    store = RESULTS["store"]

    write_csv(
        "device-storage",
        ["metric", "value", "unit"],
        [
            ["thumbnail_photo_like", store["thumbnail_bytes_photo_like"], "bytes"],
            ["thumbnail_noise_worst_case", store["thumbnail_bytes_noise"], "bytes"],
            ["thumbnail_flat_colour_floor", store["thumbnail_bytes_flat"], "bytes"],
            ["database_per_capture", store["database_bytes_per_capture"], "bytes"],
            ["captures_per_gigabyte", store["captures_per_gigabyte"], "captures"],
            ["write_rate", store["writes_per_second"], "per second"],
            ["ms_per_write", store["ms_per_write"], "milliseconds"],
            ["read_stats", store["reads"]["stats"]["median_ms"], "milliseconds"],
            ["read_captures_limit_50",
             store["reads"]["captures_limit_50"]["median_ms"], "milliseconds"],
            ["read_one_by_uuid",
             store["reads"]["get_one_by_uuid"]["median_ms"], "milliseconds"],
        ],
    )

    api = RESULTS["api"]

    write_csv(
        "api-latency",
        ["endpoint", "median_ms", "note"],
        [
            ["GET /v1/health", api["health"]["median_ms"], "store counts and uptime"],
            ["GET /v1/model", api["model_info"]["median_ms"], "first call builds the model"],
            ["GET /v1/captures?limit=50", api["captures_query"]["median_ms"],
             "50 rows with detections"],
            ["POST /v1/sync", api["sync"]["latency"]["median_ms"],
             "25 captures, gzipped, metadata only"],
            ["POST /v1/classify (1 image)",
             api["classify_edge_batch_1"]["median_ms"], "edge profile"],
            ["POST /v1/classify (4 images)",
             api["classify_edge_batch_4"]["median_ms"],
             f'{api["classify_edge_batch_4"]["per_image_ms"]} ms per image'],
        ],
    )

    tests = RESULTS["test_suite"]

    write_csv(
        "test-suite",
        ["module", "passed", "skipped", "subtests", "seconds"],
        [
            [name, data["passed"], data["skipped"], data.get("subtests", 0),
             data["seconds"]]
            for name, data in tests["modules"].items()
        ],
    )

    write_csv(
        "code-size",
        ["file", "total_lines", "code_lines", "comment_lines", "functions",
         "classes", "documented"],
        [
            [r["file"], r["total"], r["code"], r["comment"], r["funcs"],
             r["classes"], r["documented"]]
            for r in RESULTS["code"]["files"]
        ],
    )


def main():
    GRAPHS.mkdir(exist_ok=True)
    TABLES.mkdir(exist_ok=True)

    print("graphs")
    write_graphs()

    print("\ntables")
    write_tables()


if __name__ == "__main__":
    main()
