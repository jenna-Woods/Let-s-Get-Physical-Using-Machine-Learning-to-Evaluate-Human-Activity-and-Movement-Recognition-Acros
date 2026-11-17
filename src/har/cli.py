import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="UCI HAR experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    download = sub.add_parser("download", help="Download original UCI archive")
    download.add_argument("--destination", default="data")
    for command in ["audit", "cv", "freeze", "final"]:
        p = sub.add_parser(command)
        p.add_argument("--data", default="data/UCI HAR Dataset")
        p.add_argument("--output", required=True)
        if command in {"cv", "freeze"}:
            p.add_argument("--config", default="configs/full.json")
        if command == "final":
            p.add_argument("--frozen", required=True)
    p = sub.add_parser("plot")
    p.add_argument("--results", required=True)
    args = parser.parse_args()
    if args.command == "download":
        from .data import download
        print(download(args.destination))
    elif args.command == "audit":
        from .data import Dataset, audit
        ds = Dataset(args.data)
        audit(ds, ds.load("train"), args.output)
    elif args.command == "plot":
        from .plots import plot_results
        plot_results(args.results)
    else:
        from .experiments import run_cv, freeze, evaluate_final
        if args.command == "final":
            print(evaluate_final(args.data, args.frozen, args.output))
        else:
            config = json.loads(Path(args.config).read_text())
            print((run_cv if args.command == "cv" else freeze)(args.data, config, args.output))


if __name__ == "__main__":
    main()
