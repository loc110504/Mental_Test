import argparse
import csv
import os
import tempfile
import time
import zipfile
from pathlib import Path
from urllib.parse import urljoin, urlsplit, urlunsplit

DEFAULT_URL = "https://dcapswoz.ict.usc.edu/wwwdaicwoz/?C=D;O=A"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
)
DEFAULT_HEADERS = {"User-Agent": DEFAULT_USER_AGENT}
ZIP_SUFFIX = "_P.zip"
SPLIT_FILES = {
    "train": "train_split_Depression_AVEC2017.csv",
    "dev": "dev_split_Depression_AVEC2017.csv",
    "test": "test_split_Depression_AVEC2017.csv",
    "full_test": "full_test_split.csv",
}
\
KNOWN_PARTICIPANT_ID_MIN = 300
KNOWN_PARTICIPANT_ID_MAX = 492
MISSING_PARTICIPANT_IDS = {342, 394, 398, 460}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Download DAIC-WOZ files using a browser-like request and optionally "
            "extract transcripts directly from participant archives."
        )
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help="Dataset index page.",
    )
    parser.add_argument(
        "--manifest",
        help="Text file with one direct download URL per line.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/daic_woz_dataset",
        help="Directory where downloaded files will be stored.",
    )
    parser.add_argument(
        "--split",
        choices=sorted(SPLIT_FILES.keys()),
        help="Use a known local DAIC-WOZ split CSV to download participant zip archives.",
    )
    parser.add_argument(
        "--split-csv",
        help="Local split CSV used to filter participant archives by Participant_ID.",
    )
    parser.add_argument(
        "--include-split-csvs",
        action="store_true",
        help="Also download known split CSV files from the site.",
    )
    parser.add_argument(
        "--participant-zips-only",
        action="store_true",
        help="Download only participant packages like 300_P.zip.",
    )
    parser.add_argument(
        "--extract-transcripts-dir",
        help=(
            "If set, each participant zip is downloaded to a temp file and only "
            "the transcript CSV is extracted to this directory."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing downloaded files or extracted transcripts.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Timeout in seconds for HTTP requests.",
    )
    return parser.parse_args()


def load_participant_ids(csv_path):
    participant_ids = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames and "Participant_ID" in reader.fieldnames:
            for row in reader:
                value = row.get("Participant_ID")
                if value:
                    participant_ids.append(int(float(value)))
            return set(participant_ids)

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if row and row[0].strip():
                participant_ids.append(int(float(row[0])))
    return set(participant_ids)


def normalize_base_url(base_url):
    parts = urlsplit(base_url)
    clean_path = parts.path
    if not clean_path.endswith("/"):
        clean_path = clean_path.rsplit("/", 1)[0] + "/"
    return urlunsplit((parts.scheme, parts.netloc, clean_path, "", ""))


def remote_file_url(base_url, file_name):
    return urljoin(normalize_base_url(base_url), file_name)


def participant_zip_url(base_url, participant_id):
    return remote_file_url(base_url, f"{participant_id}{ZIP_SUFFIX}")


def participant_id_from_name(file_name):
    if not file_name.endswith(ZIP_SUFFIX):
        return None
    prefix = file_name[: -len(ZIP_SUFFIX)]
    if prefix.isdigit():
        return int(prefix)
    return None


def read_manifest(manifest_path):
    urls = []
    with open(manifest_path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                urls.append(stripped)
    return urls


def resolve_split_csv_path(split_name):
    if not split_name:
        return None

    candidate = Path(__file__).resolve().parent / SPLIT_FILES[split_name]
    if candidate.exists():
        return candidate
    return None


def filter_participant_links(links, participant_ids):
    filtered = []
    for url, file_name in links:
        participant_id = participant_id_from_name(file_name)
        if participant_id is not None and participant_id in participant_ids:
            filtered.append((url, file_name))
    return filtered


def build_participant_zip_links(base_url, participant_ids):
    return [
        (participant_zip_url(base_url, participant_id), f"{participant_id}{ZIP_SUFFIX}")
        for participant_id in sorted(set(participant_ids))
    ]


def known_participant_ids():
    return [
        participant_id
        for participant_id in range(KNOWN_PARTICIPANT_ID_MIN, KNOWN_PARTICIPANT_ID_MAX + 1)
        if participant_id not in MISSING_PARTICIPANT_IDS
    ]


def build_known_dataset_links(base_url, include_split_csvs=True, participant_zips_only=False):
    links = []
    if include_split_csvs and not participant_zips_only:
        links.extend(
            (remote_file_url(base_url, file_name), file_name)
            for file_name in SPLIT_FILES.values()
        )
    links.extend(build_participant_zip_links(base_url, known_participant_ids()))
    return links


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def download_to_path(url, destination_path, overwrite=False, timeout=60):
    import requests

    if destination_path.exists() and not overwrite:
        print(f"skip  {destination_path.name}")
        return True

    try:
        start_time = time.time()
        with requests.get(url, stream=True, headers=DEFAULT_HEADERS, timeout=timeout) as response:
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0) or 0)
            downloaded = 0

            with open(destination_path, "wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    downloaded += len(chunk)

        elapsed = max(time.time() - start_time, 0.001)
        size_mb = downloaded / 1024 / 1024
        speed_mb = size_mb / elapsed
        if total_size > 0:
            print(f"done  {destination_path.name} {size_mb:.1f}MB @ {speed_mb:.2f}MB/s")
        else:
            print(f"done  {destination_path.name} {size_mb:.1f}MB")
        return True
    except Exception as exc:
        if destination_path.exists():
            destination_path.unlink()
        print(f"fail  {destination_path.name}: {exc}")
        return False


def transcript_name_from_zip(zip_name):
    participant_id = zip_name[: -len(ZIP_SUFFIX)]
    return f"{participant_id}_TRANSCRIPT.csv"


def extract_transcript_from_zip(temp_zip_path, zip_name, transcript_dir, overwrite=False):
    transcript_name = transcript_name_from_zip(zip_name)
    output_path = Path(transcript_dir) / transcript_name
    if output_path.exists() and not overwrite:
        print(f"skip  transcript {transcript_name}")
        return True

    try:
        with zipfile.ZipFile(temp_zip_path, "r") as archive:
            names = archive.namelist()
            candidates = [
                name for name in names if os.path.basename(name) == transcript_name
            ]
            if not candidates:
                candidates = [
                    name for name in names if name.upper().endswith("_TRANSCRIPT.CSV")
                ]
            if not candidates:
                print(f"fail  {zip_name}: transcript not found inside archive")
                return False

            with archive.open(candidates[0]) as src, open(output_path, "wb") as dst:
                dst.write(src.read())
        print(f"done  transcript {transcript_name}")
        return True
    except zipfile.BadZipFile:
        print(f"fail  {zip_name}: bad zip file")
        return False


def download_and_extract_transcript(url, zip_name, transcript_dir, overwrite=False, timeout=60):
    transcript_name = transcript_name_from_zip(zip_name)
    output_path = Path(transcript_dir) / transcript_name
    if output_path.exists() and not overwrite:
        print(f"skip  transcript {transcript_name}")
        return True

    fd, temp_path = tempfile.mkstemp(prefix="daic_woz_", suffix=".zip")
    os.close(fd)
    temp_zip_path = Path(temp_path)

    try:
        if not download_to_path(url, temp_zip_path, overwrite=True, timeout=timeout):
            return False
        return extract_transcript_from_zip(
            temp_zip_path,
            zip_name,
            transcript_dir,
            overwrite=overwrite,
        )
    finally:
        if temp_zip_path.exists():
            temp_zip_path.unlink()


def build_download_list(args):
    split_csv_path = args.split_csv
    if not split_csv_path and args.split:
        split_csv_path = resolve_split_csv_path(args.split)
        if split_csv_path is None:
            raise SystemExit(
                f"Local split CSV not found for '{args.split}'. "
                f"Expected {SPLIT_FILES[args.split]} next to data_download.py, or pass --split-csv explicitly."
            )

    split_csv_urls = []
    if args.include_split_csvs:
        split_csv_urls.extend(
            (remote_file_url(args.url, file_name), file_name) for file_name in SPLIT_FILES.values()
        )

    if args.manifest:
        links = []
        for url in read_manifest(args.manifest):
            links.append((url, os.path.basename(url)))
    elif split_csv_path:
        participant_ids = load_participant_ids(split_csv_path)
        links = build_participant_zip_links(args.url, participant_ids)
    else:
        include_split_csvs = (
            not args.participant_zips_only and not args.extract_transcripts_dir
        )
        links = build_known_dataset_links(
            args.url,
            include_split_csvs=include_split_csvs,
            participant_zips_only=not include_split_csvs,
        )

    if args.split_csv and args.manifest:
        participant_ids = load_participant_ids(args.split_csv)
        links = filter_participant_links(links, participant_ids)

    dedup = {}
    for url, file_name in split_csv_urls + links:
        dedup[file_name] = url
    return [(url, file_name) for file_name, url in sorted(dedup.items())]


def main():
    args = parse_args()
    links = build_download_list(args)
    if not links:
        raise SystemExit("No matching files found to download.")

    if args.extract_transcripts_dir:
        ensure_dir(args.extract_transcripts_dir)
    else:
        ensure_dir(args.output_dir)

    success_count = 0
    print(f"Found {len(links)} files")

    for url, file_name in links:
        if args.extract_transcripts_dir and file_name.endswith(ZIP_SUFFIX):
            ok = download_and_extract_transcript(
                url,
                file_name,
                args.extract_transcripts_dir,
                overwrite=args.overwrite,
                timeout=args.timeout,
            )
        else:
            destination = Path(args.output_dir) / file_name
            ok = download_to_path(
                url,
                destination,
                overwrite=args.overwrite,
                timeout=args.timeout,
            )
        if ok:
            success_count += 1

    if args.extract_transcripts_dir:
        print(f"Completed {success_count}/{len(links)} downloads to transcripts in {Path(args.extract_transcripts_dir).resolve()}")
    else:
        print(f"Completed {success_count}/{len(links)} downloads to {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
