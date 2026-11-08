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
import time
import traceback
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


def safe_failure(exc, stage):
    """Expose fixed diagnostic labels and code locations, never response text."""
    message = str(exc)
    reason = "unclassified"
    for prefix, label in (
        ("Student ID and password are required", "missing_credentials"),
        ("WebVPN session cold", "cold_webvpn_session"),
        ("Failed to extract lck from CAS redirect chain", "icourse_cas_redirect"),
        ("Failed to extract lck from redirect", "webvpn_login_redirect"),
        ("Authentication failed", "webvpn_auth_rejected"),
        ("iCourse CAS auth failed", "icourse_auth_rejected"),
        ("Failed to get authChainCode", "webvpn_auth_method_missing"),
        ("No authChainCode found", "icourse_auth_method_missing"),
        ("Failed to establish WebVPN session", "webvpn_session_failed"),
        ("API error for course", "course_api_rejected"),
    ):
        if message.startswith(prefix):
            reason = label
            break
    result = {"stage": stage, "type": type(exc).__name__, "reason": reason}
    frames = traceback.extract_tb(exc.__traceback__)
    if frames:
        frame = frames[-1]
        result["location"] = f"{Path(frame.filename).name}:{frame.lineno}:{frame.name}"
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if isinstance(status, int):
        result["http_status"] = status
    return result


def login_session(vpn_factory, max_attempts=3, sleep=time.sleep):
    """Retry known transient sessions; do not retry rejected credentials."""
    from requests.exceptions import ConnectionError, Timeout

    for attempt in range(1, max_attempts + 1):
        vpn = vpn_factory()
        stage = "webvpn_login"
        try:
            print(f"School login attempt {attempt}/{max_attempts}.", flush=True)
            with private_output():
                vpn.login()
            stage = "icourse_login"
            print("WebVPN signed in; authenticating iCourse.", flush=True)
            with private_output():
                vpn.authenticate_icourse()
            print("iCourse sign-in completed.", flush=True)
            return vpn
        except Exception as exc:
            diagnostic = safe_failure(exc, stage)
            print("Login diagnostic: " + json.dumps(diagnostic), flush=True)
            retryable = isinstance(exc, (ConnectionError, Timeout)) or diagnostic["reason"] in {
                "cold_webvpn_session", "icourse_cas_redirect",
            }
            if not retryable or attempt == max_attempts:
                raise
            session = getattr(vpn, "session", None)
            if session is not None:
                session.close()
            sleep(5 * attempt)


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


def ensure_session(client, vpn_factory=None):
    """Renew an expired authenticated session before the next lecture."""
    if client.check_alive():
        return
    if vpn_factory is None:
        from src.api.webvpn import WebVPNSession
        vpn_factory = WebVPNSession
    print("Session expired; renewing school sign-in.", flush=True)
    client.vpn = login_session(vpn_factory)
    client._userinfo = None
    if not client.check_alive():
        raise RuntimeError("iCourse session verification failed")


def session_call(client, operation):
    """Retry one API read only when the authenticated session was lost."""
    try:
        return operation()
    except Exception:
        if client.check_alive():
            raise
        ensure_session(client)
        return operation()


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
    consecutive_failures, renewals = 0, 0
    for index, page in enumerate(pages):
        try:
            with private_output():
                content = fetch_ppt_image(client, page)
            if not content:
                raise ValueError("No slide image returned")
            with Image.open(io.BytesIO(content)) as img:
                fingerprint = imagehash.dhash(img)
            consecutive_failures = 0
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
            consecutive_failures += 1
            if consecutive_failures == 3:
                try:
                    if not client.check_alive():
                        if renewals >= 2:
                            failed += len(pages) - index - 1
                            break
                        ensure_session(client)
                        renewals += 1
                        consecutive_failures = 0
                except Exception:
                    failed += len(pages) - index - 1
                    break
            if consecutive_failures >= 8:
                failed += len(pages) - index - 1
                break
    return result, failed


def collect_one(client, lecture, info):
    ensure_session(client)
    sub_id = lecture["sub_id"]
    row = {"sub_id": sub_id, "title": lecture.get("sub_title", ""),
           "date": lecture["date"], "status": "pending", "issues": []}
    try:
        pages = session_call(client, lambda: client.get_ppt_list(COURSE_ID, sub_id))
    except Exception as exc:
        pages = []
        row["issues"].append("ppt_list:" + type(exc).__name__)
    duration_hint = max((p.get("created_sec", 0) for p in pages), default=0)
    try:
        segments = session_call(client, lambda: client.get_transcript_segments(sub_id)) or []
    except Exception as exc:
        segments = []
        row["issues"].append("official_transcript:" + type(exc).__name__)
    if transcript_usable(segments, duration_hint):
        row["transcript_source"] = "official"
        row["segments"] = segments
    else:
        print(f"Lecture {sub_id}: requesting streaming ASR.", flush=True)
        try:
            ensure_session(client)
            video = session_call(client, lambda: visible_video_url(client, info))
            if not video:
                row["issues"].append("No published player URL for ASR")
            else:
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
            row["issues"].append(safe_failure(exc, "asr"))
        if not row.get("transcript") and segments:
            row["segments"] = segments
            row["transcript_source"] = "official_partial"
            row["issues"].append("Official transcript has substantial gaps")
    row["slides"], row["ppt_failed"] = [], 0
    row["ppt_total"] = len(pages)
    row["duration_lower_bound_s"] = duration_hint
    source = row.get("transcript_source", "none")
    row["status"] = "complete" if source in ("official", "sensevoice_asr") else "partial"
    print(f"Lecture {sub_id}: {row['status']}; transcript={source}; slides deferred.", flush=True)
    return row, pages


def run():
    from src.api.webvpn import WebVPNSession
    from src.api.icourse import ICourseClient

    public_pem = (ROOT / "scripts/optics_export_public.pem").read_text(encoding="ascii")
    destination = ROOT / "results/optics37988-latest4.encrypted.json"
    payload = {"course_id": COURSE_ID, "requested_count": COUNT,
               "collected_at": datetime.now(timezone.utc).isoformat(),
               "lectures": [], "skipped": []}
    print("Signing in to the existing school account.", flush=True)
    vpn = login_session(WebVPNSession)
    client = ICourseClient(vpn)
    print("Reading optics course details.", flush=True)
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
    slide_jobs = []
    for lecture, info in reversed(selected):
        print(f"Collecting {lecture['date']} / {lecture['sub_id']}.", flush=True)
        try:
            row, pages = collect_one(client, lecture, info)
            slide_jobs.append((row, pages))
        except Exception as exc:
            row = {"sub_id": lecture["sub_id"], "date": lecture["date"],
                   "status": "failed", "error": safe_failure(exc, "lecture")}
        payload["lectures"].append(row)
        seal(payload, public_pem, destination)
    # Preserve every transcript before spending time on image processing.
    for row, pages in slide_jobs:
        print(f"Lecture {row['sub_id']}: reading {len(pages)} slide snapshots.", flush=True)
        try:
            ensure_session(client)
            row["slides"], row["ppt_failed"] = ppt_text(client, pages)
        except Exception as exc:
            row["issues"].append(safe_failure(exc, "slides"))
            row["ppt_failed"] = len(pages)
        seal(payload, public_pem, destination)
        print(f"Lecture {row['sub_id']}: OCR pages={len(row['slides'])}; "
              f"image errors={row['ppt_failed']}.", flush=True)
    complete = sum(row["status"] == "complete" for row in payload["lectures"])
    print(f"Complete transcripts: {complete}/{COUNT}. Encrypted result saved.", flush=True)
    return 0 if complete == COUNT else 2


if __name__ == "__main__":
    try:
        sys.exit(run())
    except Exception as exc:
        diagnostic = safe_failure(exc, "collection")
        print("Collection stopped: " + json.dumps(diagnostic), flush=True)
        try:
            seal({"course_id": COURSE_ID, "status": "failed", "error": diagnostic},
                 (ROOT / "scripts/optics_export_public.pem").read_text(encoding="ascii"),
                 ROOT / "results/optics37988-latest4.encrypted.json")
        except Exception:
            print("Could not save the encrypted diagnostic.", flush=True)
        sys.exit(1)
