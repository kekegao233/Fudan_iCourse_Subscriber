"""Collect the four latest accessible optics replays for private review.

Reuses this repository's school login, OCR, and streaming ASR. No LLM API,
email, catalog crawl, or database publication is performed. Only an encrypted
bundle is written; the corresponding private key stays on the user's computer.
"""
from __future__ import annotations

import contextlib
import gzip
import io
import json
import os
from pathlib import Path
import re
import sys
import tarfile
from datetime import datetime, timezone, timedelta
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
COURSE_ID = "37988"
COUNT = 4


@contextlib.contextmanager
def private_output():
    # Dependency diagnostics may contain signed URLs. Never publish them.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def recent_candidates(lectures, today):
    """Sort by lesson date/period, then deduplicate actual lesson titles."""
    candidates = []
    for item in lectures:
        title = str(item.get("sub_title", ""))
        date_match = re.search(r"\d{4}-\d{2}-\d{2}", title)
        date = date_match.group() if date_match else str(item.get("date", ""))
        try:
            normalized = datetime.strptime(date, "%Y-%m-%d").date().isoformat()
        except ValueError:
            continue
        if not item.get("has_playback") or normalized > today:
            continue
        period = re.search(r"第(\d+)", title)
        sub_id = str(item.get("sub_id", ""))
        if not sub_id.isdigit():
            continue
        candidates.append((normalized, int(period.group(1)) if period else 0,
                           int(sub_id), dict(item, date=normalized, sub_id=sub_id)))
    candidates.sort(key=lambda entry: entry[:3], reverse=True)
    seen = set()
    result = []
    for *_, item in candidates:
        identity = (item["date"], item.get("sub_title") or item["sub_id"])
        if identity not in seen:
            seen.add(identity)
            result.append(item)
    return result


def transcript_usable(segments, duration_hint_s=0):
    if not segments or sum(len(s.get("text", "").strip()) for s in segments) < 200:
        return False
    end = 0
    for seg in sorted(segments, key=lambda s: s["start_ms"]):
        if seg["start_ms"] - end > 20 * 60_000:
            return False
        end = max(end, seg["end_ms"])
    return not duration_hint_s or duration_hint_s * 1000 - end <= 20 * 60_000


def seal(payload, public_pem, destination):
    from base64 import b64encode
    from Crypto.Cipher import AES, PKCS1_OAEP
    from Crypto.Hash import SHA256
    from Crypto.PublicKey import RSA
    from Crypto.Random import get_random_bytes

    key = get_random_bytes(32)
    cipher = AES.new(key, AES.MODE_EAX)
    compressed = gzip.compress(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    ciphertext, tag = cipher.encrypt_and_digest(compressed)
    wrapped = PKCS1_OAEP.new(RSA.import_key(public_pem), hashAlgo=SHA256).encrypt(key)
    envelope = {"format": "icourse-optics-rsa-oaep-sha256-aes256-eax-gzip-v1"}
    for label, value in (("key", wrapped), ("nonce", cipher.nonce),
                         ("tag", tag), ("ciphertext", ciphertext)):
        envelope[label] = b64encode(value).decode("ascii")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(envelope), encoding="utf-8")


def published_info(client, sub_id):
    """Honor the normal access response, including the release-time gate."""
    response = client.vpn.get(
        client.base_url + "/courseapi/v3/portal-home-setting/get-sub-info",
        params={"course_id": COURSE_ID, "sub_id": sub_id}, timeout=45,
    )
    response.raise_for_status()
    data = response.json()
    if data.get("code") != 0:
        return None, data.get("code")
    return data.get("data") or {}, 0


def visible_video_url(client, info):
    # Only normal, published player sources; no review-gate fallback.
    videos = info.get("video_list") or {}
    videos = videos.values() if isinstance(videos, dict) else videos
    for video in videos:
        url = video.get("preview_url") if isinstance(video, dict) else None
        if url:
            return client.sign_video_url(url, now=int(info.get("now") or 0) or None)
    for url in (info.get("playurl") or {}).values():
        if isinstance(url, str) and ".mp4" in url:
            return client.sign_video_url(url, now=int(info.get("now") or 0) or None)
    return None


def ensure_asr_models():
    from src.runtime import config

    model_name = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
    release = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    if not Path(config.ASR_MODEL_DIR).is_dir():
        archive = Path(model_name + ".tar.bz2")
        urlretrieve(release + archive.name, archive)
        with tarfile.open(archive) as tar:
            tar.extractall(path=".", filter="data")
        archive.unlink()
    if not Path(config.SILERO_VAD_PATH).is_file():
        urlretrieve(release + "silero_vad.onnx", config.SILERO_VAD_PATH)


def ppt_text(client, pages):
    from PIL import Image
    import imagehash
    from src.api.icourse import fetch_ppt_image
    from src.ai.ocr import ocr_image_text

    result, recent_hashes, previous_text, failed = [], [], None, 0
    for page in pages:
        try:
            with private_output():
                content = fetch_ppt_image(client, page)
            if not content:
                failed += 1
                continue
            with Image.open(io.BytesIO(content)) as img:
                fingerprint = imagehash.dhash(img)
            if any(fingerprint - old <= 2 for old in recent_hashes[-5:]):
                continue
            recent_hashes.append(fingerprint)
            with private_output():
                text = ocr_image_text(content).strip()
            if text and text != previous_text:
                result.append({"time_s": page.get("created_sec", 0), "text": text})
                previous_text = text
        except Exception:
            failed += 1
    return result, failed


def collect_one(client, lecture, info):
    sub_id = lecture["sub_id"]
    row = {"sub_id": sub_id, "title": lecture.get("sub_title", ""),
           "date": lecture["date"], "status": "pending", "issues": []}
    try:
        pages = client.get_ppt_list(COURSE_ID, sub_id)
    except Exception as exc:
        pages = []
        row["issues"].append("ppt_list:" + type(exc).__name__)
    duration_hint = max((p.get("created_sec", 0) for p in pages), default=0)
    try:
        segments = client.get_transcript_segments(sub_id) or []
    except Exception as exc:
        segments = []
        row["issues"].append("official_transcript:" + type(exc).__name__)
    if transcript_usable(segments, duration_hint):
        row["transcript_source"] = "official"
        row["segments"] = segments
    else:
        print(f"Lecture {sub_id}: requesting streaming ASR.", flush=True)
        video = visible_video_url(client, info)
        if not video:
            row["issues"].append("No published player URL for ASR")
        else:
            try:
                ensure_asr_models()
                from src.ai.transcriber import Transcriber
                url, headers = client.get_stream_params(video)
                with private_output():
                    transcriber = Transcriber()
                    text, asr_segments = transcriber.transcribe_url(url, http_headers=headers)
                row["transcript_source"] = "sensevoice_asr" if len(text.strip()) >= 200 else "asr_partial"
                row["transcript"] = text
                row["asr_segments"] = asr_segments
            except Exception as exc:
                # The upstream error may include a signed URL or cookies.
                row["issues"].append("asr:" + type(exc).__name__)
        if not row.get("transcript") and segments:
            row["segments"] = segments
            row["transcript_source"] = "official_partial"
            row["issues"].append("Official transcript has substantial gaps")
    print(f"Lecture {sub_id}: reading {len(pages)} slide snapshots.", flush=True)
    row["slides"], row["ppt_failed"] = ppt_text(client, pages)
    row["ppt_total"] = len(pages)
    row["duration_lower_bound_s"] = duration_hint
    source = row.get("transcript_source", "none")
    row["status"] = "complete" if source in ("official", "sensevoice_asr") else "partial"
    print(f"Lecture {sub_id}: {row['status']}; transcript={source}; "
          f"OCR pages={len(row['slides'])}; image errors={row['ppt_failed']}.", flush=True)
    return row


def run():
    from src.api.webvpn import WebVPNSession
    from src.api.icourse import ICourseClient

    public_pem = (ROOT / "scripts/optics_export_public.pem").read_text(encoding="ascii")
    destination = ROOT / "results/optics37988-latest4.encrypted.json"
    payload = {"course_id": COURSE_ID, "requested_count": COUNT,
               "collected_at": datetime.now(timezone.utc).isoformat(),
               "lectures": [], "skipped": []}
    print("Signing in to the existing school account.", flush=True)
    with private_output():
        vpn = WebVPNSession()
        vpn.login()
        vpn.authenticate_icourse()
    client = ICourseClient(vpn)
    detail = client.get_course_detail(COURSE_ID)
    payload["course_title"] = detail["title"]
    today = datetime.now(timezone(timedelta(hours=8))).date().isoformat()
    selected = []
    for lecture in recent_candidates(detail["lectures"], today):
        info, code = published_info(client, lecture["sub_id"])
        if info is None:
            payload["skipped"].append({"sub_id": lecture["sub_id"], "code": code})
            continue
        selected.append((lecture, info))
        if len(selected) == COUNT:
            break
    payload["selected"] = [{k: lecture[k] for k in ("sub_id", "sub_title", "date")}
                           for lecture, _ in selected]
    seal(payload, public_pem, destination)
    for lecture, info in reversed(selected):
        print(f"Collecting {lecture['date']} / {lecture['sub_id']}.", flush=True)
        try:
            row = collect_one(client, lecture, info)
        except Exception as exc:
            row = {"sub_id": lecture["sub_id"], "date": lecture["date"],
                   "status": "failed", "error_type": type(exc).__name__}
        payload["lectures"].append(row)
        seal(payload, public_pem, destination)
    complete = sum(row["status"] == "complete" for row in payload["lectures"])
    print(f"Complete transcripts: {complete}/{COUNT}. Encrypted result saved.", flush=True)
    return 0 if complete == COUNT else 2


if __name__ == "__main__":
    try:
        sys.exit(run())
    except Exception as exc:
        print("Collection stopped: " + type(exc).__name__ + ". No credentials logged.", flush=True)
        sys.exit(1)
