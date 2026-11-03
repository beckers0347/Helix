# FILE: colreport.py

import argparse
import csv
import json
import sys

def main():
    parser = argparse.ArgumentParser(description='Analyze CSV file columns.')
    parser.add_argument('csv_file', type=str, help='Path to the CSV file')
    parser.add_argument('--json', action='store_true', help='Output results in JSON format')
    parser.add_argument('--delimiter', type=str, default=',', help='Delimiter for the CSV file (default is comma)')
    parser.add_argument('--column', type=str, help='Show only the specified column')
    args = parser.parse_args()

    try:
        with open(args.csv_file, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file, delimiter=args.delimiter)
            headers = next(reader)
            data = {header: [] for header in headers}

            for row in reader:
                for header, cell in zip(headers, row):
                    try:
                        value = float(cell.strip())
                        data[header].append(value)
                    except (ValueError, TypeError):
                        pass

            if args.column:
                if args.column not in data:
                    sys.stderr.write(f"Error: Column '{args.column}' not found in the CSV file\n")
                    sys.exit(1)
                data = {args.column: data[args.column]}

            if args.json:
                result = {}
                for header, values in data.items():
                    if values:
                        result[header] = {
                            'count': len(values),
                            'min': min(values),
                            'max': max(values),
                            'mean': sum(values) / len(values)
                        }
                print(json.dumps(result, indent=4))
            else:
                headers = list(data.keys())
                widths = [max(len(header), max(len(f"{v:.6f}") for v in values)) for header, values in data.items()]
                header_line = ' | '.join(f'{h:<{w}}' for h, w in zip(headers, widths))
                sep_line = '-+-'.join('-' * w for w in widths)
                data_lines = []

                for header, values in data.items():
                    if values:
                        data_lines.append(' | '.join(f'{v:<{widths[i]}}' for i, v in enumerate(values)))

                print(header_line)
                print(sep_line)
                if data_lines:
                    print('\n'.join(data_lines))

    except FileNotFoundError:
        sys.stderr.write(f"Error: Unable to read the CSV file '{args.csv_file}'\n")
        sys.exit(1)
    except Exception as e:
        sys.stderr.write(f"Error: {str(e)}\n")
        sys.exit(1)

if __name__ == '__main__':
    main()