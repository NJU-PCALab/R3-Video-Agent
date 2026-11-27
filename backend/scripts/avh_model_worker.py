#!/usr/bin/env python3
"""One isolated, real Qwen media inference. No persistent GPU service is assumed.

The parent serializes GPU work. This process exits after writing a durable JSON
response, including actual processor inputs. Install model dependencies in a
separate interpreter; do not upgrade the verified H3 environment in place.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
from fractions import Fraction
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback
import wave


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Capture the launched source, even if a deployment replaces the file mid-run.
WORKER_SOURCE_SHA256 = sha256(Path(__file__).resolve())


def write_response(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def gpu_lock_paths(device, project_root=None):
    if not re.fullmatch(r"\d+(,\d+)*", str(device)):
        raise ValueError("GPU worker device must name physical numeric card IDs")
    root = Path(project_root or Path(__file__).resolve().parents[1])
    return [root / "service_logs" / f"gpu_{index}.lock"
            for index in sorted({int(value) for value in str(device).split(",")})]


@contextmanager
def gpu_locks(device, response_path, project_root=None):
    """Same flock files as the H3 worker; acquire before importing GPU libraries."""
    handles = []
    status_path = Path(response_path).parent / "worker_status.json"
    paths = gpu_lock_paths(device, project_root)
    try:
        write_response(status_path, {"state": "waiting_gpu_lock", "device": str(device),
                                     "lock_paths": [str(path) for path in paths], "pid": os.getpid(), "time": time.time()})
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open("a+")
            handles.append(handle)
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        write_response(status_path, {"state": "running", "device": str(device),
                                     "lock_paths": [str(path) for path in paths], "pid": os.getpid(), "time": time.time()})
        yield
    finally:
        # Empty the allocator while the shared lock is still held. On failures,
        # traceback frames can otherwise retain a whole model until process exit.
        active_traceback = sys.exc_info()[2]
        if active_traceback:
            traceback.clear_frames(active_traceback)
        gc.collect()
        torch_module = sys.modules.get("torch")
        if torch_module is not None and torch_module.cuda.is_initialized():
            try:
                torch_module.cuda.synchronize()
                torch_module.cuda.empty_cache()
            except Exception:
                # CUDA failures must preserve the original exception/receipt.
                pass
        for handle in reversed(handles):
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()


def revision(model_path, request, model=None):
    configuration = Path(model_path) / "config.json"
    if not configuration.is_file():
        configuration = Path(model_path) / "model_index.json"
    fingerprint = sha256(configuration) if configuration.is_file() else "absent"
    requested = request.get("revision")
    prefix = "local-config-sha256:"
    if isinstance(requested, str) and requested.startswith(prefix):
        expected = requested[len(prefix):]
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("Configured local model revision must contain a valid SHA256")
        if fingerprint != expected:
            raise ValueError("Configured local model revision does not match the actual config/model_index SHA256")
    commit = getattr(getattr(model, "config", None), "_commit_hash", None)
    # A config fingerprint is explicit about its limited coverage; it is not
    # silently presented as a full checkpoint weight digest.
    return str(commit or request.get("revision") or f"local-config-sha256:{fingerprint}")


def runtime_metadata(model_path):
    result = {"model_path": str(Path(model_path).resolve()),
              "worker_source_sha256": WORKER_SOURCE_SHA256}
    for package in ("torch", "transformers", "diffusers", "qwen-omni-utils"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def character_age_condition(image, parent, request, folder):
    import cv2
    import numpy as np
    from PIL import Image

    mode = request["character_age_edit"]
    if mode not in ("younger", "older", "elderly") or not isinstance(parent, dict) or not parent.get("sha256"):
        raise ValueError("Character age editing requires an age direction and a verified image")
    portrait = image.convert("RGB")
    gray = cv2.cvtColor(np.asarray(portrait), cv2.COLOR_RGB2GRAY)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
    faces = sorted(detector.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=5, minSize=(64, 64)),
                   key=lambda face: face[2] * face[3], reverse=True) if not detector.empty() else []
    box = None
    if faces and (len(faces) == 1 or 2 * faces[1][2] * faces[1][3] < faces[0][2] * faces[0][3]):
        x, y, w, h = map(int, faces[0])
        side = min(round(1.6 * max(w, h)), *portrait.size)
        left = max(0, min(round(x + w / 2 - side / 2), portrait.width - side))
        top = max(0, min(round(y + h / 2 - side / 2), portrait.height - side))
        box = [left, top, left + side, top + side]
        portrait = portrait.crop(tuple(box)).resize((520, 520), Image.Resampling.LANCZOS)
    condition_path = folder / "character_age_condition.png"
    portrait.save(condition_path)
    return portrait, {
        "age_edit": mode, "source_sha256": parent["sha256"],
        "input_sha256": sha256(condition_path), "input_path": str(condition_path.resolve()),
        "crop_xyxy": box, "strategy": "dominant_face_crop" if box else "whole_canonical_portrait",
    }


def generate_image(model_path, request, response_path):
    import torch
    from PIL import Image
    from diffusers import QwenImage21Pipeline

    parameters = request.get("parameters", {})
    if parameters.get("pipeline_class", "QwenImage21Pipeline") != "QwenImage21Pipeline":
        raise ValueError("This documented worker supports the unified QwenImage21Pipeline only")
    model_index = json.loads((Path(model_path) / "model_index.json").read_text())
    if model_index.get("_class_name") != "QwenImage21Pipeline":
        raise ValueError("Checkpoint pipeline class does not match Qwen-Image 2.1")
    pipeline = QwenImage21Pipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16, local_files_only=True)
    if parameters.get("cpu_offload", False):
        pipeline.enable_model_cpu_offload(gpu_id=0)
    else:
        pipeline.to("cuda:0")
    parents = request.get("parent_images", [])
    parent_images = []
    for parent in parents:
        path = parent["path"] if isinstance(parent, dict) else parent
        expected = parent.get("sha256") if isinstance(parent, dict) else None
        if expected and sha256(path) != expected:
            raise ValueError("Parent image changed before editing")
        with Image.open(path) as image:
            parent_images.append(image.copy())
    if len(parent_images) > 10:
        raise ValueError("Published Qwen-Image 2.1 reference-image limit exceeded")
    options = {"prompt": request["prompt"], "width": int(request.get("width", 1024)),
               "height": int(request.get("height", 1024)),
               "num_inference_steps": int(parameters.get("num_inference_steps", 40)),
               "true_cfg_scale": float(parameters.get("true_cfg_scale", 1.0)),
               "generator": torch.Generator("cuda:0").manual_seed(int(request.get("seed", 42)))}
    condition = None
    age_edit = request.get("character_age_edit")
    if age_edit is not None:
        if len(parent_images) != 1:
            raise ValueError("Character age editing requires one verified canonical parent")
        portrait, condition = character_age_condition(parent_images[0], parents[0], request, Path(response_path).parent)
        parent_images = [portrait]
        negative = "waxy skin, shiny plastic skin, beauty filter, painted portrait, HDR, digital sharpening"
        age_negatives = {
            "younger": "exaggerated aging, deep aged wrinkles",
            "older": "",
            "elderly": "young adult, middle-aged adult, smooth unlined skin",
        }
        options["negative_prompt"] = ", ".join(filter(None, (negative, age_negatives[age_edit])))
        options["true_cfg_scale"] = 2.65 if age_edit == "elderly" else 2.15
    if parent_images:
        options["image"] = parent_images
    started = time.monotonic()
    with torch.inference_mode():
        image = pipeline(**options).images[0]
    output = Path(response_path).parent / "generated.png"
    image.save(output)
    return {"ok": True, "images": [{"path": str(output.resolve()), "sha256": sha256(output),
                                      "width": image.width, "height": image.height, "mode": image.mode}],
            "model": request.get("model") or Path(model_path).name,
            "revision": revision(model_path, request),
            "inputs": {"prompt": True, "parent_images": len(parent_images)},
            "parent_hashes": [sha256(p["path"] if isinstance(p, dict) else p) for p in parents],
            **({"character_age_condition": condition} if condition else {}),
            "sampling": {k: v for k, v in options.items() if k not in {"generator", "image", "prompt"}},
            "inference_seconds": time.monotonic() - started,
            "attention_processors_actual": sorted({type(p).__name__ for p in pipeline.transformer.attn_processors.values()})}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("image",), required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--response", required=True)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    # Set visibility before importing any GPU library; device 0 is the selected
    # physical card inside this worker.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    started = time.monotonic()
    try:
        request = json.loads(Path(args.request).read_text(encoding="utf-8"))
        # Reject a stale declared configuration before loading any GPU library.
        revision(args.model_path, request)
        with gpu_locks(args.device, args.response):
            # The model directory may have changed while waiting for the card.
            revision(args.model_path, request)
            result = generate_image(args.model_path, request, args.response)
        result.update(runtime=runtime_metadata(args.model_path), total_seconds=time.monotonic() - started)
        write_response(args.response, result)
        return 0
    except Exception as error:
        write_response(args.response, {"ok": False, "kind": args.kind, "error": str(error),
                                       "error_type": type(error).__name__, "traceback": traceback.format_exc(),
                                       "runtime": runtime_metadata(args.model_path),
                                       "total_seconds": time.monotonic() - started})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
