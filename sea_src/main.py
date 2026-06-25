import argparse
import os
import sys

from assessment import process_single_file
from data_load import load_chatprompt, load_scoring_standards
from logging_setup import (
    close_dialog_log,
    dialog_print,
    initialize_dialog_log,
    setup_logging,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run the original AgentMental pipeline on a single sample."
    )
    parser.add_argument(
        "--sample-id",
        default=os.getenv("AGENTMENTAL_SAMPLE_ID", "302"),
        help="Sample identifier without the .json suffix. Default: 302",
    )
    parser.add_argument(
        "--sample-path",
        default=os.getenv("AGENTMENTAL_SAMPLE_PATH"),
        help="Full path to a processed sample JSON file. Overrides --sample-id.",
    )
    parser.add_argument(
        "--data-dir",
        default=os.getenv("AGENTMENTAL_DATA_DIR"),
        help="Directory containing processed sample JSON files.",
    )
    parser.add_argument(
        "--csv-file",
        default=os.getenv("AGENTMENTAL_CSV_FILE"),
        help="Output CSV path for assessment results.",
    )
    parser.add_argument(
        "--scale",
        default="PHQ-8",
        choices=["PHQ-8"],
        help="Assessment scale to use.",
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Use manual interactive mode instead of automated test mode.",
    )
    parser.add_argument(
        "--skip-if-evaluated",
        action="store_true",
        help="Skip the sample if it already exists in the output CSV.",
    )
    return parser


def resolve_paths(args):
    src_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(src_dir)

    data_dir = args.data_dir or os.path.join(repo_root, "data", "processed_dev_daic_woz")
    scoring_path = os.path.join(repo_root, "scales", "scoring_standards.json")
    chatprompt_path = os.path.join(repo_root, "scales", "PHQ-8.json")
    csv_file = args.csv_file or os.path.join(repo_root, "evaluation", "agentmental_single_sample.csv")

    if args.sample_path:
        sample_path = args.sample_path
    else:
        sample_path = os.path.join(data_dir, f"{args.sample_id}.json")

    return {
        "repo_root": repo_root,
        "data_dir": data_dir,
        "scoring_path": scoring_path,
        "chatprompt_path": chatprompt_path,
        "csv_file": csv_file,
        "sample_path": sample_path,
    }


def validate_inputs(paths):
    if not os.path.isdir(paths["data_dir"]):
        raise FileNotFoundError(f"Data folder does not exist: {paths['data_dir']}")

    if not os.path.isfile(paths["scoring_path"]):
        raise FileNotFoundError(f"Scoring standards file not found: {paths['scoring_path']}")

    if not os.path.isfile(paths["chatprompt_path"]):
        raise FileNotFoundError(f"Chatprompt file not found: {paths['chatprompt_path']}")

    if not os.path.isfile(paths["sample_path"]):
        raise FileNotFoundError(f"Sample JSON file not found: {paths['sample_path']}")


if __name__ == "__main__":
    args = build_parser().parse_args()
    logger = setup_logging()
    initialize_dialog_log()

    try:
        logger.info("Psychological assessment program started.")
        paths = resolve_paths(args)
        validate_inputs(paths)

        selected_scale = args.scale
        scoring_standards = load_scoring_standards(paths["scoring_path"])
        chatprompt = load_chatprompt(paths["chatprompt_path"])

        automated = not args.manual
        mode_choice = "2" if automated else "1"

        logger.info("Running original AgentMental pipeline for a single sample.")
        dialog_print(f"Scale: {selected_scale}")
        dialog_print(f"Sample: {paths['sample_path']}")
        dialog_print(f"Mode: {'Automated test mode' if automated else 'Manual interactive mode'}")
        dialog_print(f"Result CSV: {paths['csv_file']}")

        process_single_file(
            file_path=paths["sample_path"],
            scoring_standards=scoring_standards,
            chatprompt=chatprompt,
            selected_scale=selected_scale,
            mode_choice=mode_choice,
            csv_file_path=paths["csv_file"],
            automated=automated,
            skip_if_evaluated=args.skip_if_evaluated,
        )
    except Exception as exc:
        logger.exception("Failed to run the AgentMental single-sample pipeline: %s", exc)
        dialog_print(f"Error: {exc}")
        sys.exit(1)
    finally:
        close_dialog_log()
