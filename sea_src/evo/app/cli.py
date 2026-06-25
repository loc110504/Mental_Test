from __future__ import annotations

import argparse
import json

from .runner import run_case_directory, run_case_directory_filtered, run_case_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EvoToM-Mental on a case JSON file.")
    parser.add_argument("input_path", help="Path to a processed participant JSON file or a directory of JSON files.")
    parser.add_argument(
        "--mode",
        default="offline_train",
        choices=["offline_train", "dev_eval", "inference", "verified_update"],
        help="Execution mode.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Use terminal input for responses in inference mode.",
    )
    parser.add_argument(
        "--sample",
        help="Participant/sample id to run from a directory input, for example 300.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Run only the first N JSON files from a directory input.",
    )
    args = parser.parse_args()
    if args.input_path.endswith(".json"):
        result = run_case_file(args.input_path, mode=args.mode, interactive=args.interactive)
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        if args.sample or args.limit:
            df = run_case_directory_filtered(
                args.input_path,
                mode=args.mode,
                sample_id=args.sample,
                limit=args.limit,
            )
        else:
            df = run_case_directory(args.input_path, mode=args.mode)
        print(df.to_json(orient="records", force_ascii=False, indent=2))


if __name__ == "__main__":
    main()
