"""Exact, request-scoped AdaLN projection caching for the pinned H3 DiT.

No weights, denoising steps, timesteps or projection batch shapes change.  The
first model call supplies the actual packed sequence, conditioning and retake
masks.  We precompute the original modules' outputs for that finite schedule,
validate them, and discard only the now-redundant projection parameters.

Cached tensors intentionally are not registered buffers: DiffSynth's model
offloading must not move them, while the job-local pipeline keeps them alive.
This is inference-only and a pipeline must not be reused with a different
schedule or conditioning layout after enabling it.  Unsupported input fails
explicitly, rather than silently falling back to another execution path.
"""
from __future__ import annotations

import gc
import math
import struct
import time
from typing import Any

import torch
from torch import nn


def _tensor_key(value: torch.Tensor) -> tuple:
    if value.dtype != torch.float32 or value.ndim != 1 or not value.numel():
        raise ValueError("AdaLN timestep keys must be nonempty one-dimensional float32 tensors")
    values = value.detach().cpu().tolist()
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Nonfinite timesteps cannot be cached")
    return tuple(struct.pack("<f", v) for v in values)


def _scalar_key(value: torch.Tensor) -> bytes:
    if not isinstance(value, torch.Tensor) or value.numel() != 1 or value.dtype != torch.float32:
        raise ValueError("H3 scheduler timesteps must be scalar float32 tensors")
    return _tensor_key(value.reshape(1))[0]


def _patchify_video(value: torch.Tensor) -> torch.Tensor:
    # Same packing order as pinned diffsynth.models.minimax_h3_dit.patchify_video.
    b, c, ft, fh, fw = (int(v) for v in value.shape)
    packed = value.reshape(b, c, ft, 1, fh // 2, 2, fw // 2, 2)
    packed = torch.einsum("nctrhpwq->nthwcrpq", packed)
    return packed.reshape(b * ft * (fh // 2) * (fw // 2), c * 4).contiguous()


def _pack_audio(value: torch.Tensor) -> torch.Tensor:
    ac, ld, steps = (int(v) for v in value.shape)
    return value.permute(0, 2, 1).reshape(ac * steps, ld).contiguous()


def unique_timesteps_for_call(inputs: dict, timestep_video: torch.Tensor,
                              timestep_audio: torch.Tensor, *, dataward: bool = False) -> torch.Tensor:
    """Reproduce the pinned model_fn's float32 assignments in their exact order.

    Full packed positions are used so fully masked generation, zero-length
    modalities, reference audio/video, keyframes and coincident timesteps retain
    exactly the same unique-vector shape as the normal model invocation.
    """
    _scalar_key(timestep_video)
    _scalar_key(timestep_audio)
    device = inputs["video_latents"].device
    packed = inputs["packed"]
    img_pos, audio_pos = packed["img_pos"], packed["audio_pos"]
    if img_pos.device != device or audio_pos.device != device:
        raise ValueError("Packed positions and H3 latents must share a device")
    t_video = float(timestep_video) if dataward else 1.0 - float(timestep_video) / 1000
    t_audio = float(timestep_audio) if dataward else 1.0 - float(timestep_audio) / 1000
    anchor = inputs.get("ref_visual_anchor")
    if anchor is None:
        anchor = inputs.get("keyframe_cond_anchor")
    cond_rows = 0 if anchor is None else anchor.shape[0]
    audio_anchor = inputs.get("ref_audio_anchor")
    audio_rows = 0 if audio_anchor is None else audio_anchor.shape[0]
    timesteps = torch.full((packed["seq_len"],), t_video, dtype=torch.float32, device=device)
    timesteps[audio_pos] = t_audio
    if inputs.get("input_latents_video") is not None:
        mask = _patchify_video(inputs["denoise_mask_video"])[:, 0]
        timesteps[img_pos[cond_rows:][mask == 0]] = 1.0
    if inputs.get("input_latents_audio") is not None:
        mask = _pack_audio(inputs["denoise_mask_audio"])[:, 0]
        timesteps[audio_pos[audio_rows:][mask == 0]] = 1.0
    timesteps[img_pos[:cond_rows]] = max(t_video, inputs.get("imgvid_cond_noise_aug", 0.999))
    timesteps[audio_pos[:audio_rows]] = max(t_audio, inputs.get("audio_cond_noise_aug", 1.0))
    return torch.unique(timesteps, sorted=True)


def _layout_signature(inputs: dict) -> tuple:
    """Reject a changed request without copying the long token sequence to CPU."""
    def tensor_identity(value):
        if value is None:
            return None
        if not isinstance(value, torch.Tensor):
            raise TypeError("Conditioning and masks must be tensors")
        return (id(value), tuple(value.shape), str(value.device), str(value.dtype), value._version)

    packed = inputs["packed"]
    prompt = inputs["prompt_embeds"]
    return (
        id(packed), int(packed["seq_len"]),
        tuple(tensor_identity(packed[name]) for name in ("img_pos", "audio_pos", "text_pos")),
        tuple(tensor_identity(inputs.get(name)) for name in (
            "keyframe_cond_anchor", "ref_visual_anchor", "ref_audio_anchor",
            "denoise_mask_video", "denoise_mask_audio")),
        inputs.get("input_latents_video") is not None, inputs.get("input_latents_audio") is not None,
        float(inputs.get("imgvid_cond_noise_aug", 0.999)), float(inputs.get("audio_cond_noise_aug", 1.0)),
        str(prompt.dtype), str(prompt.device),
        tuple(inputs["video_latents"].shape), tuple(inputs["audio_latents"].shape),
        str(inputs["video_latents"].device), str(inputs["video_latents"].dtype),
    )


class ExactAdaLNLookup(nn.Module):
    """A parameter-free replacement that accepts only the active exact input."""
    def __init__(self, outputs: dict[tuple, tuple[torch.Tensor, ...]], context: dict,
                 stats: dict[str, Any]):
        super().__init__()
        self._outputs = outputs
        self._context = context
        self._stats = stats

    def forward(self, t_emb: torch.Tensor):
        if t_emb is not self._context.get("active_t_emb"):
            raise RuntimeError("AdaLN cache received an input outside the validated H3 embedding call")
        key = self._context.get("active_key")
        if key not in self._outputs:
            raise RuntimeError("AdaLN cache has no entry for the current timestep vector")
        result = self._outputs[key]
        if any(part.device != t_emb.device for part in result):
            raise RuntimeError("AdaLN cache device/dtype changed; reload the pipeline for this request")
        self._stats["cache_hits"] += 1
        return result


def enable_adaln_exact(pipe, *, validate: bool = True) -> dict[str, Any]:
    """Enable deferred exact caching and return live, JSON-serializable statistics.

    Call after from_pretrained/eval and before pipe(...).  Projection validation
    checks every unique timestep-vector entry for every transformer/final layer.
    Statistics mutate in place as cache construction and inference proceed.
    """
    if getattr(pipe, "_h3_adaln_exact_enabled", False):
        raise RuntimeError("Exact AdaLN caching is already enabled on this pipeline")
    dit = pipe.dit
    if dit.training:
        raise ValueError("Exact AdaLN caching requires an eval-mode DiT")
    if not hasattr(dit, "time_embedder") or not hasattr(dit, "_embed"):
        raise TypeError("Unsupported H3 DiT: time_embedder/_embed are required")
    stats: dict[str, Any] = {
        "profile": "adaln_exact", "built": False, "validation_enabled": bool(validate),
        "build_seconds": 0.0, "precompute_seconds": 0.0,
        "cache_bytes": 0, "freed_parameter_bytes": 0,
        "cache_hits": 0, "validation_max_abs_error": None,
        "projection_modules": 0, "schedule_steps": 0, "unique_timestep_vectors": 0,
    }
    original_fn = pipe.model_fn
    original_embed = dit._embed
    context: dict[str, Any] = {}

    def embed_checked(*args, **kwargs):
        unique = kwargs.get("unique_timesteps")
        if unique is None:
            raise RuntimeError("H3 _embed must pass unique_timesteps explicitly")
        key = _tensor_key(unique.reshape(-1))
        if key not in context["embeddings"]:
            raise RuntimeError("Unknown timestep vector: exact AdaLN cache cannot interpolate")
        embeddings, t_emb = original_embed(*args, **kwargs)
        expected = context["embeddings"][key]
        if (t_emb.shape != expected.shape or t_emb.dtype != expected.dtype
                or t_emb.device != expected.device or not torch.equal(t_emb, expected)):
            raise RuntimeError("H3 timestep embedding differs from the precomputed exact input")
        context["active_key"], context["active_t_emb"] = key, t_emb
        return embeddings, t_emb

    def build(inputs):
        started = time.perf_counter()
        video_steps = pipe.scheduler.timesteps
        audio_steps = pipe.scheduler_audio.timesteps
        if len(video_steps) != len(audio_steps) or not len(video_steps):
            raise ValueError("Video/audio schedules must have the same nonzero length")
        context["layout"] = _layout_signature(inputs)
        context["pairs"] = set()
        context["embeddings"] = {}
        device = inputs["video_latents"].device
        dtype = inputs["prompt_embeds"].dtype
        with torch.no_grad():
            for video_step, audio_step in zip(video_steps, audio_steps):
                tv = video_step.unsqueeze(0).to(dtype=torch.float32, device=device)
                ta = audio_step.unsqueeze(0).to(dtype=torch.float32, device=device)
                context["pairs"].add((_scalar_key(tv), _scalar_key(ta)))
                unique = unique_timesteps_for_call(inputs, tv, ta,
                    dataward=getattr(pipe, "_h3_timestep_convention", None) == "dataward")
                key = _tensor_key(unique)
                if key not in context["embeddings"]:
                    t_emb = dit.time_embedder(unique,
                        dtype=getattr(pipe, "_h3_time_embedding_dtype", dtype)).detach()
                    if t_emb.dtype not in (dtype, torch.float32) or t_emb.device != device or t_emb.ndim != 2:
                        raise RuntimeError("Unexpected H3 timestep embedding dtype/device/shape")
                    context["embeddings"][key] = t_emb
                    stats["cache_bytes"] += t_emb.numel() * t_emb.element_size()
            destinations = [(block, "adaln_proj") for block in dit.blocks]
            destinations.append((dit.final_layer, "adaln_proj"))
            max_error = 0.0
            for parent, attribute in destinations:
                projection = getattr(parent, attribute)
                if not isinstance(projection, nn.Module):
                    raise TypeError("Unsupported H3 AdaLN projection")
                param_bytes = sum(p.numel() * p.element_size() for p in projection.parameters())
                if not param_bytes:
                    raise ValueError("AdaLN projection must retain its original parameters until validation")
                outputs = {}
                layer_error = torch.zeros((), dtype=torch.float32, device=device)
                for key, t_emb in context["embeddings"].items():
                    output = projection(t_emb)
                    if not isinstance(output, tuple) or not output or not all(isinstance(v, torch.Tensor) for v in output):
                        raise TypeError("H3 AdaLN must return a nonempty tensor tuple")
                    if any(v.dtype != dtype or v.device != device for v in output):
                        raise RuntimeError("AdaLN returned an unexpected dtype/device")
                    cached = tuple(v.detach() for v in output)
                    if validate:
                        repeated = projection(t_emb)
                        if len(repeated) != len(cached) or any(a.shape != b.shape for a, b in zip(cached, repeated)):
                            raise RuntimeError("AdaLN projection output changed shape during validation")
                        for a, b in zip(cached, repeated):
                            layer_error = torch.maximum(layer_error, (a.float() - b.float()).abs().amax())
                        del repeated
                    outputs[key] = cached
                    stats["cache_bytes"] += sum(v.numel() * v.element_size() for v in cached)
                    del output, cached
                if validate:
                    error = float(layer_error)
                    if not math.isfinite(error) or error != 0.0:
                        raise RuntimeError(f"Exact AdaLN projection validation failed: max_abs_error={error}")
                    max_error = max(max_error, error)
                replacement = ExactAdaLNLookup(outputs, context, stats)
                setattr(parent, attribute, replacement)
                stats["freed_parameter_bytes"] += param_bytes
                stats["projection_modules"] += 1
                # The old wrapped linears are no longer traversed by DiffSynth.
                # Release CPU/GPU parameter references before loading the next layer.
                del projection, outputs, replacement, layer_error
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                if stats["projection_modules"] % 10 == 0 or stats["projection_modules"] == len(destinations):
                    print(f"Exact AdaLN cache: {stats['projection_modules']}/{len(destinations)} modules; "
                          f"released {stats['freed_parameter_bytes'] / 2**30:.2f} GiB of parameters", flush=True)
        dit._embed = embed_checked
        elapsed = time.perf_counter() - started
        stats.update(built=True, schedule_steps=len(video_steps),
                     unique_timestep_vectors=len(context["embeddings"]),
                     validation_max_abs_error=max_error if validate else None,
                     build_seconds=elapsed, precompute_seconds=elapsed)

    def model_fn_checked(*args, **kwargs):
        if args:
            raise TypeError("Exact H3 model_fn caching requires keyword-only pipeline inputs")
        if kwargs.get("dit") is not dit:
            raise RuntimeError("The H3 DiT changed after enabling exact acceleration")
        if kwargs.get("controlnet") is not None or kwargs.get("control_rows") is not None:
            raise ValueError("ControlNet is not supported by this exact AdaLN acceleration profile")
        if kwargs.get("use_gradient_checkpointing") or kwargs.get("use_gradient_checkpointing_offload"):
            raise ValueError("Exact AdaLN caching supports inference without gradient checkpointing")
        if not stats["built"]:
            build(kwargs)
        if _layout_signature(kwargs) != context["layout"]:
            raise RuntimeError("H3 conditioning/layout changed: reload the pipeline for a new request")
        pair = (_scalar_key(kwargs["timestep_video"]), _scalar_key(kwargs["timestep_audio"]))
        if pair not in context["pairs"]:
            raise RuntimeError("Unknown H3 scheduler timestep pair; exact cache has no fallback")
        context.pop("active_t_emb", None)
        context.pop("active_key", None)
        return original_fn(**kwargs)

    pipe.model_fn = model_fn_checked
    pipe._h3_adaln_exact_enabled = True
    return stats
