import subprocess
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import sys

EXPECTED_FIELD_COUNT = 4

def timestamp_tag():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def build_rocksdb(source_dir: Path, build_dir: Path):
    print("🛠 Compiling rocksdb...")
    build_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cmake", "-S", str(source_dir), "-B", str(build_dir), "-DCMAKE_BUILD_TYPE=Release", "-DROCKSDB_BUILD_SHARED=OFF"], check=True)
    subprocess.run(["cmake", "--build", str(build_dir), "--target", "rocksdb", "rocksdb_stress", "--parallel", "16"], check=True)

def build_benchmarker(source_dir: Path, build_dir: Path):
    print("🛠 Compiling benchmark tool...")
    build_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cmake", "-S", str(source_dir), "-B", str(build_dir)], check=True)
    subprocess.run(["cmake", "--build", str(build_dir)], check=True)

def run_exp(bench_path: Path, n, table_mode, bench_mode, d_dist, a_dist):
    cmd = [str(bench_path), str(n), table_mode, bench_mode, d_dist, a_dist]
    raw = subprocess.check_output(cmd).decode().strip()

    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != EXPECTED_FIELD_COUNT:
        raise RuntimeError(
            f"Unexpected benchmark output ({len(parts)} fields, expected {EXPECTED_FIELD_COUNT}): {raw}"
        )

    size, avg, p99, b_time = map(float, parts)
    return {
        "Keys": n,
        "TableMode": table_mode,
        "BenchMode": bench_mode,
        "DataDist": d_dist,
        "AccessPattern": a_dist,
        "SizeKB": size,
        "AvgLatUs": avg,
        "P99Us": p99,
        "BuildTimeSec": b_time,
        "Combo": f"Data={d_dist}, Access={a_dist}",
    }

def print_progress(done_keys: int, total_keys: int, label: str = ""):
    width = 40
    frac = 0.0 if total_keys == 0 else done_keys / total_keys
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)
    percent = frac * 100.0
    msg = f"\r[{bar}] {percent:6.2f}% ({done_keys}/{total_keys} keys)"
    if label:
        msg += f" | {label}"
    sys.stdout.write(msg)
    sys.stdout.flush()

def main():
    ts = timestamp_tag()

    source_dir = Path(__file__).resolve().parent.parent
    build_dir = source_dir / "build"
    build_rocksdb(source_dir, build_dir)

    source_dir = Path(__file__).resolve().parent
    build_dir = source_dir / "build"
    build_benchmarker(source_dir, build_dir)

    raw_dir = source_dir / "raw_data" / ts
    graphs_dir = source_dir / "graphs" / ts

    raw_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    bench_path = build_dir / "bench"
    if not bench_path.exists():
        raise FileNotFoundError(f"Benchmark binary not found: {bench_path}")

    n_values = [
        12345, 17891, 24567, 33321,
        45678, 61234, 78901, 104729,
        137913, 181337, 238901, 314159,
        412667, 543219, # 716543, 943717,
    ]

    table_modes = ["block", "cuckoo"]
    bench_modes = ["no_compact", "compact"]
    data_dists = ["uniform", "zipf", "normal"]
    access_dists = ["uniform", "zipf", "normal"]

    total_keys = (
        len(table_modes)
        * len(bench_modes)
        * len(data_dists)
        * len(access_dists)
        * sum(n_values)
    )
    done_keys = 0

    raw_results = []

    for n in n_values:
        for d_dist in data_dists:
            for a_dist in access_dists:
                for table_mode in table_modes:
                    for bench_mode in bench_modes:
                        label = f"{table_mode} | {bench_mode} | Data:{d_dist} | Access:{a_dist} | N:{n}"
                        print_progress(done_keys, total_keys, label=label)
                        raw_results.append(
                            run_exp(bench_path, n, table_mode, bench_mode, d_dist, a_dist)
                        )
                        done_keys += n
                        print_progress(done_keys, total_keys, label=label)

    sys.stdout.write("\n")

    df = pd.DataFrame(raw_results)
    csv_path = raw_dir / "final_bench_results.csv"
    df.to_csv(csv_path, index=False)

    build_df = df.groupby(["Keys", "TableMode", "BenchMode", "DataDist"]).agg({
        "BuildTimeSec": "mean",
        "SizeKB": "mean",
    }).reset_index()

    fig1, axes1 = plt.subplots(1, 2, figsize=(15, 6))
    for table_mode in ["block", "cuckoo"]:
        for bench_mode in ["no_compact", "compact"]:
            subset = build_df[
                (build_df["TableMode"] == table_mode) &
                (build_df["BenchMode"] == bench_mode) &
                (build_df["DataDist"] == "uniform")
            ]
            label = f"{table_mode} ({bench_mode})"
            ls = "-" if bench_mode == "compact" else "--"
            axes1[0].plot(subset["Keys"], subset["BuildTimeSec"], label=label, linestyle=ls, marker="o")
            axes1[1].plot(subset["Keys"], subset["SizeKB"], label=label, linestyle=ls, marker="s")

    axes1[0].set_title("Average Build Time")
    axes1[1].set_title("Average Disk Footprint")
    for ax in axes1:
        ax.set_xscale("log", base=10)
        ax.set_xlabel("Number of Keys")
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes1[0].set_ylabel("Seconds")
    axes1[1].set_ylabel("KB")
    plt.tight_layout()
    plt.savefig(graphs_dir / "metrics_build_and_size.png")
    plt.close(fig1)

    fig_lat = plt.figure(figsize=(10, 7))
    for table_mode in ["block", "cuckoo"]:
        for bench_mode in ["no_compact", "compact"]:
            m_df = df[(df["TableMode"] == table_mode) & (df["BenchMode"] == bench_mode)]
            for combo in m_df["Combo"].unique():
                c_df = m_df[m_df["Combo"] == combo]
                ls = "-" if bench_mode == "compact" else "--"
                plt.plot(
                    c_df["Keys"],
                    c_df["AvgLatUs"],
                    label=f"{table_mode} {bench_mode} {combo}",
                    linestyle=ls,
                    marker="v",
                )

    plt.title("Combined Average Latency Comparison")
    plt.xlabel("Number of Keys")
    plt.ylabel("Latency (us)")
    plt.xscale("log", base=10)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", ncol=1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(graphs_dir / "latency_combined.png")
    plt.close(fig_lat)

    fig2, axes2 = plt.subplots(3, 3, figsize=(18, 16))
    combos = [
        ("uniform", "uniform"), ("uniform", "zipf"), ("uniform", "normal"),
        ("zipf", "uniform"), ("zipf", "zipf"), ("zipf", "normal"),
        ("normal", "uniform"), ("normal", "zipf"), ("normal", "normal"),
    ]

    for (d_dist, a_dist), (r, c) in zip(combos, [(i, j) for i in range(3) for j in range(3)]):
        ax = axes2[r, c]
        for table_mode in ["block", "cuckoo"]:
            for bench_mode in ["no_compact", "compact"]:
                subset = df[
                    (df["TableMode"] == table_mode) &
                    (df["BenchMode"] == bench_mode) &
                    (df["DataDist"] == d_dist) &
                    (df["AccessPattern"] == a_dist)
                ]
                ax.plot(
                    subset["Keys"],
                    subset["AvgLatUs"],
                    label=f"{table_mode}-{bench_mode}",
                    marker="o",
                )

        ax.set_title(f"Data: {d_dist.upper()} | Access: {a_dist.upper()}")
        ax.set_xlabel("Keys")
        ax.set_ylabel("Avg Latency (us)")
        ax.set_xscale("log", base=10)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(graphs_dir / "latency_individual_combos.png")
    plt.close(fig2)

    print("\n✅ Success! Generated:")
    print(f" - {csv_path}")
    print(f" - {graphs_dir / 'metrics_build_and_size.png'}")
    print(f" - {graphs_dir / 'latency_combined.png'}")
    print(f" - {graphs_dir / 'latency_individual_combos.png'}")

if __name__ == "__main__":
    main()
