import sys
import csv
import json
import argparse


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Summarise numeric CSV columns.")
    parser.add_argument("csv_file", help="CSV file with a header row")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--delimiter", default=",", help="Field delimiter (default comma)")
    parser.add_argument("--column", metavar="NAME", help="Show only this column")
    args = parser.parse_args()

    try:
        fh = open(args.csv_file, newline="", encoding="utf-8")
    except OSError as exc:
        print(f"Error: cannot read {args.csv_file}: {exc}", file=sys.stderr)
        sys.exit(1)

    with fh:
        reader = csv.reader(fh, delimiter=args.delimiter)
        header = next(reader, None)
        if not header:
            print("Error: the CSV file is empty", file=sys.stderr)
            sys.exit(1)
        header = [h.strip() for h in header]
        if args.column is not None and args.column not in header:
            print(f"Error: column '{args.column}' not found", file=sys.stderr)
            sys.exit(1)
        acc = {h: {"count": 0, "min": None, "max": None, "sum": 0.0, "bad": False} for h in header}
        for row in reader:
            for h, cell in zip(header, row):
                cell = cell.strip()
                if not cell:
                    continue
                a = acc[h]
                try:
                    v = float(cell)
                except ValueError:
                    a["bad"] = True
                    continue
                a["count"] += 1
                a["sum"] += v
                a["min"] = v if a["min"] is None or v < a["min"] else a["min"]
                a["max"] = v if a["max"] is None or v > a["max"] else a["max"]

    stats = {}
    for h in header:
        a = acc[h]
        if a["count"] and not a["bad"]:
            stats[h] = {"count": a["count"], "min": a["min"], "max": a["max"], "mean": a["sum"] / a["count"]}

    if args.column is not None:
        if args.column not in stats:
            print(f"Error: column '{args.column}' has no numeric values", file=sys.stderr)
            sys.exit(1)
        stats = {args.column: stats[args.column]}

    if args.json:
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        return

    headers = ["Column", "Count", "Min", "Max", "Mean"]
    rows = [[name, str(s["count"]), f"{s['min']:g}", f"{s['max']:g}", f"{s['mean']:.4f}"] for name, s in stats.items()]
    widths = [max([len(headers[i])] + [len(r[i]) for r in rows]) for i in range(len(headers))]
    print(" | ".join(f"{h:<{w}}" for h, w in zip(headers, widths)))
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        print(" | ".join(f"{c:<{w}}" for c, w in zip(r, widths)))


if __name__ == "__main__":
    main()
