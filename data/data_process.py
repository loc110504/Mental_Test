import argparse
import csv
import json
from pathlib import Path

PHQ8_ITEM_COLUMNS = [
    "PHQ8_NoInterest",
    "PHQ8_Depressed",
    "PHQ8_Sleep",
    "PHQ8_Tired",
    "PHQ8_Appetite",
    "PHQ8_Failure",
    "PHQ8_Concentrating",
    "PHQ8_Moving",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert DAIC-WOZ score and transcript files into AgentMental JSON samples."
    )
    parser.add_argument(
        "--score-csv",
        required=True,
        help="CSV file containing Participant_ID, PHQ8_Score, PHQ8_Binary and item columns.",
    )
    parser.add_argument(
        "--transcript-dir",
        required=True,
        help="Directory containing transcript CSV files named like <Participant_ID>_TRANSCRIPT.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed_daic_woz",
        help="Directory where processed JSON files will be written.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail immediately if a transcript file is missing or malformed.",
    )
    return parser.parse_args()


def detect_delimiter(file_path):
    with open(file_path, "r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        return dialect.delimiter


def load_transcript(transcript_path):
    import pandas as pd

    delimiter = detect_delimiter(transcript_path)
    transcript_df = pd.read_csv(transcript_path, delimiter=delimiter, dtype=str)

    required_columns = {"speaker", "value"}
    missing_columns = required_columns.difference(transcript_df.columns)
    if missing_columns:
        raise ValueError(f"Transcript missing columns: {sorted(missing_columns)}")

    real_interview = []
    for _, transcript_row in transcript_df.iterrows():
        speaker = str(transcript_row["speaker"]).strip()
        content = transcript_row["value"]

        if pd.isna(content) or speaker not in {"Ellie", "Participant"}:
            continue

        content = str(content).strip()
        if not content:
            continue

        real_interview.append({"roleName": speaker, "content": content})

    return real_interview


def build_sample(row, real_interview):
    participant_id = str(int(float(row["Participant_ID"])))
    phq8_scores = {
        "PHQ8_Score": int(row["PHQ8_Score"]),
        "PHQ8_Binary": int(row["PHQ8_Binary"]),
        "items": {column: int(row[column]) for column in PHQ8_ITEM_COLUMNS},
    }
    return {
        "Participant_ID": participant_id,
        "real_interview": real_interview,
        "phq8_scores": phq8_scores,
    }


def process_phq8_dataset(score_csv_path, transcript_dir, output_dir, strict=False):
    import pandas as pd
    from tqdm import tqdm

    output_path = Path(output_dir)
    transcript_path = Path(transcript_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    scores_df = pd.read_csv(score_csv_path)
    missing_score_columns = {
        "Participant_ID",
        "PHQ8_Score",
        "PHQ8_Binary",
        *PHQ8_ITEM_COLUMNS,
    }.difference(scores_df.columns)
    if missing_score_columns:
        raise ValueError(f"Score CSV missing columns: {sorted(missing_score_columns)}")

    processed_count = 0
    missing_transcripts = []

    for _, row in tqdm(scores_df.iterrows(), total=len(scores_df), desc="Processing participants"):
        participant_id = str(int(float(row["Participant_ID"])))
        transcript_file = transcript_path / f"{participant_id}_TRANSCRIPT.csv"

        if not transcript_file.exists():
            message = f"Transcript not found: {transcript_file}"
            if strict:
                raise FileNotFoundError(message)
            print(f"skip  {message}")
            missing_transcripts.append(participant_id)
            continue

        try:
            real_interview = load_transcript(transcript_file)
        except Exception as exc:
            if strict:
                raise
            print(f"skip  transcript parse failed for {participant_id}: {exc}")
            missing_transcripts.append(participant_id)
            continue

        sample = build_sample(row, real_interview)
        sample_path = output_path / f"{participant_id}.json"
        with open(sample_path, "w", encoding="utf-8") as handle:
            json.dump(sample, handle, ensure_ascii=False, indent=2)
        processed_count += 1

    print(f"Processed samples: {processed_count}")
    print(f"Missing or invalid transcripts: {len(missing_transcripts)}")
    print(f"Output dir: {output_path.resolve()}")


def main():
    args = parse_args()
    process_phq8_dataset(
        score_csv_path=args.score_csv,
        transcript_dir=args.transcript_dir,
        output_dir=args.output_dir,
        strict=args.strict,
    )


if __name__ == "__main__":
    main()
