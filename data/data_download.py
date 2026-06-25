import argparse
import concurrent.futures
import os
import time
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen


DEFAULT_EXTENSIONS = {
    ".zip",
    ".tar",
    ".gz",
    ".7z",
    ".rar",
    ".csv",
    ".txt",
    ".json",
    ".xlsx",
}
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
)


class LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() != "a":
            return
        attr_map = dict(attrs)
        href = attr_map.get("href")
        if href:
            self.links.append(href)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Download dataset files from an authorized index page or from a "
            "manifest file containing one URL per line."
        )
    )
    parser.add_argument(
        "--url",
        help="Authorized dataset page containing downloadable links.",
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
        "--workers",
        type=int,
        default=4,
        help="Number of concurrent downloads.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download files even if they already exist.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Timeout in seconds for HTTP requests.",
    )
    return parser.parse_args()


def build_request(url):
    return Request(url, headers={"User-Agent": DEFAULT_USER_AGENT})


def read_manifest(manifest_path):
    urls = []
    with open(manifest_path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                urls.append(stripped)
    return urls


def discover_links(index_url, timeout):
    with urlopen(build_request(index_url), timeout=timeout) as response:
        html = response.read().decode("utf-8", errors="ignore")

    parser = LinkParser()
    parser.feed(html)

    discovered = []
    for href in parser.links:
        full_url = urljoin(index_url, href)
        parsed = urlparse(full_url)
        suffix = Path(parsed.path).suffix.lower()
        if suffix in DEFAULT_EXTENSIONS:
            discovered.append(full_url)

    unique_links = sorted(set(discovered))
    return unique_links


def filename_from_url(file_url, index):
    path_name = Path(urlparse(file_url).path).name
    if path_name:
        return path_name
    return f"dataset_file_{index:03d}.dat"


def download_file(file_url, destination_dir, overwrite=False, timeout=60):
    file_name = Path(destination_dir, filename_from_url(file_url, 0))
    if file_name.exists() and not overwrite:
        print(f"skip  {file_name.name}")
        return True

    try:
        start_time = time.time()
        with urlopen(build_request(file_url), timeout=timeout) as response:
            total_size = int(response.headers.get("Content-Length", "0") or 0)
            with open(file_name, "wb") as handle:
                downloaded = 0
                while True:
                    chunk = response.read(1024 * 64)
                    if not chunk:
                        break
                    handle.write(chunk)
                    downloaded += len(chunk)

        elapsed = max(time.time() - start_time, 0.001)
        size_mb = downloaded / 1024 / 1024
        speed_mb = size_mb / elapsed
        if total_size > 0:
            print(f"done  {file_name.name} {size_mb:.1f}MB @ {speed_mb:.2f}MB/s")
        else:
            print(f"done  {file_name.name} {size_mb:.1f}MB")
        return True
    except Exception as exc:
        if file_name.exists():
            file_name.unlink()
        print(f"fail  {file_name.name}: {exc}")
        return False


def main():
    args = parse_args()

    if not args.url and not args.manifest:
        raise SystemExit("Provide either --url or --manifest.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.manifest:
        download_links = read_manifest(args.manifest)
    else:
        download_links = discover_links(args.url, args.timeout)

    if not download_links:
        raise SystemExit("No downloadable links found.")

    print(f"Found {len(download_links)} files")
    success_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(args.workers, 1)) as executor:
        futures = [
            executor.submit(
                download_file,
                link,
                output_dir,
                args.overwrite,
                args.timeout,
            )
            for link in download_links
        ]
        for future in concurrent.futures.as_completed(futures):
            if future.result():
                success_count += 1

    print(f"Downloaded {success_count}/{len(download_links)} files to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
