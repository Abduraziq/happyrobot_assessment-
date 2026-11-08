#!/usr/bin/env python3
"""Send one audio file to the EoT prediction API."""

from __future__ import annotations

import argparse
import json
import mimetypes
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def multipart_body(path: Path) -> tuple[bytes, str]:
    boundary = f"----happyrobot-{uuid.uuid4().hex}"
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    payload = path.read_bytes()
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="audio"; filename="{path.name}"\r\n'
        f"Content-Type: {media_type}\r\n\r\n"
    ).encode("utf-8") + payload + f"\r\n--{boundary}--\r\n".encode("ascii")
    return body, boundary


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict EoT for an audio file through the serving API.")
    parser.add_argument("--audio", type=Path, required=True, help="Local WAV or FLAC file.")
    parser.add_argument("--url", default="http://127.0.0.1:8000/predict", help="Prediction endpoint URL.")
    parser.add_argument("--timeout", type=float, default=30.0, help="Request timeout in seconds.")
    args = parser.parse_args()
    if not args.audio.is_file():
        parser.error(f"Audio file does not exist: {args.audio}")

    body, boundary = multipart_body(args.audio)
    request = Request(
        args.url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=args.timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"API returned HTTP {exc.code}: {detail}") from exc
    except (TimeoutError, URLError) as exc:
        raise SystemExit(f"API request failed: {exc}") from exc
    total_ms = (time.perf_counter() - started) * 1000.0
    print(json.dumps({"prediction": result, "client_total_request_ms": total_ms}, indent=2))


if __name__ == "__main__":
    main()
