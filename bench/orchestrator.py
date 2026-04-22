import subprocess
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import sys
import argparse

EXPECTED_FIELD_COUNT = 3

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
    subprocess.run(["cmake", "-S", str(source_dir), "-B", str(build_dir), "-DCMAKE_BUILD_TYPE=Release"], check=True)
    subprocess.run(["cmake", "--build", str(build_dir)], check=True)

def run_exp(bench_path: Path, n, table_mode, bench_mode, d_dist, a_dist, duration_sec, miss_ratio, shuffle_keys):
    cmd = [str(bench_path), str(n), table_mode, bench_mode, d_dist, a_dist, str(duration_sec), str(miss_ratio), "true" if shuffle_keys else "false"]
    try:
        raw = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode().strip()
    except subprocess.CalledProcessError as e:
        print(f"\n  [ERROR] Benchmark failed for N={n} {table_mode}-{bench_mode}: {e.output.decode().strip() if e.output else ''}")
        return {
            "Keys": n,
            "TableMode": table_mode,
            "BenchMode": bench_mode,
            "DataDist": d_dist,
            "AccessPattern": a_dist,
            "SizeKB": float('nan'),
            "AvgLatUs": float('nan'),
            "BuildTimeSec": float('nan'),
            "Combo": f"Data={d_dist}, Access={a_dist}",
        }

    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != EXPECTED_FIELD_COUNT:
        raise RuntimeError(
            f"Unexpected benchmark output ({len(parts)} fields, expected {EXPECTED_FIELD_COUNT}): {raw}"
        )

    size, avg, b_time = map(float, parts)
    return {
        "Keys": n,
        "TableMode": table_mode,
        "BenchMode": bench_mode,
        "DataDist": d_dist,
        "AccessPattern": a_dist,
        "SizeKB": size,
        "AvgLatUs": avg,
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
    parser = argparse.ArgumentParser(description="Run or plot RocksDB benchmarks")
    parser.add_argument("--csv", type=str, help="Path to existing CSV to skip benchmarking and just plot")
    parser.add_argument("--duration", type=float, default=1.5, help="Duration in seconds for each read benchmark")
    parser.add_argument("--miss_ratio", type=float, default=0.2, help="Ratio of cache misses (0.0 to 1.0)")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle keys before insertion")
    args = parser.parse_args()

    source_dir = Path(__file__).resolve().parent

    if args.csv:
        csv_path = Path(args.csv).resolve()
        if not csv_path.exists():
            raise FileNotFoundError(f"Provided CSV not found: {csv_path}")
        print(f"Loading results from {csv_path}...")
        df = pd.read_csv(csv_path)
        results_dir = csv_path.parent
    else:
        ts = timestamp_tag()
        rocksdb_source_dir = source_dir.parent
        rocksdb_build_dir = rocksdb_source_dir / "build"
        build_rocksdb(rocksdb_source_dir, rocksdb_build_dir)

        build_dir = source_dir / "build"
        build_benchmarker(source_dir, build_dir)

        results_dir = source_dir / "results" / ts
        results_dir.mkdir(parents=True, exist_ok=True)

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
        data_dists = ["uniform", "zipf"]
        access_dists = ["uniform", "zipf"]

        total_keys = (
            len(table_modes)
            * len(bench_modes)
            * len(data_dists)
            * len(access_dists)
            * sum(n_values)
        )
        done_keys = 0
        csv_path = results_dir / "final_bench_results.csv"
        if csv_path.exists():
            csv_path.unlink()

        for n in n_values:
            for d_dist in data_dists:
                for a_dist in access_dists:
                    for table_mode in table_modes:
                        for bench_mode in bench_modes:
                            label = f"{table_mode} | {bench_mode} | Data:{d_dist} | Access:{a_dist} | N:{n}"
                            print_progress(done_keys, total_keys, label=label)
                            
                            res = run_exp(bench_path, n, table_mode, bench_mode, d_dist, a_dist, args.duration, args.miss_ratio, args.shuffle)
                            pd.DataFrame([res]).to_csv(csv_path, mode='a', header=not csv_path.exists(), index=False)
                            
                            done_keys += n
                            print_progress(done_keys, total_keys, label=label)

        sys.stdout.write("\n")
        df = pd.read_csv(csv_path)

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
    plt.savefig(results_dir / "metrics_build_and_size.png")
    plt.close(fig1)

    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
    combos = [
        ("uniform", "uniform"), ("uniform", "zipf"),
        ("zipf", "uniform"), ("zipf", "zipf"),
    ]

    for (d_dist, a_dist), (r, c) in zip(combos, [(0, 0), (0, 1), (1, 0), (1, 1)]):
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
    plt.savefig(results_dir / "latency_individual_combos.png")
    plt.close(fig2)

    print("\n✅ Success! Generated:")
    print(f" - {csv_path}")
    print(f" - {results_dir / 'metrics_build_and_size.png'}")
    print(f" - {results_dir / 'latency_individual_combos.png'}")

if __name__ == "__main__":
    main()
