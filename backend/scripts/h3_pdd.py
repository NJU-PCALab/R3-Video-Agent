"""Adapt the matching published FL2VA/Ref2VA PDD-8 release to DiffSynth H3.

This implements 32 interval heads, four intervals per evaluation, and separate
video/audio shifted schedules. It is NOT ordinary LoRA loading or an arbitrary
eight-step Euler reduction. Runtime additive low-rank updates preserve the
published BF16 operation order. Original small FP32 base modules are restored
from existing local shards. The DiffSynth fused QKV and attention implementation
remain backend differences; this adapter does not claim bitwise Diffusers parity.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
import math
import struct
import textwrap
import time
from pathlib import Path
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F

PDD_STEPS, BLOCK_SIZE, NFE, RANK, ALPHA = 32, 4, 8, 64, 64.0
TARGETS = "to_q,to_k,to_v,to_out.0,ff.net.0.proj,ff.net.2,adaln_proj.linear"
PDD_REPOSITORY = "alibaba-pai/MiniMax-H3-Acc-LoRAs"
PDD_REVISION = "335001fb9e5455d68a0caa18ec2e319072150328"
PDD_RELEASES = {
    "fl2va": {"filename": "MiniMax-H3-FL2VA-Acc-8Step.safetensors",
              "sha256": "0b29be7042d883970eb0c20774a9ba03d95669ed80a721bb4d21be8ea0d0a196"},
    "ref2va": {"filename": "MiniMax-H3-Ref2VA-Acc-8Step.safetensors",
               "sha256": "111c82e669f6e20e628228172edf39395f1a9fc3ad049793895e542c0f55b18c"},
}


def validate_adapter_variant(sha256, base_variant, expected_sha256=None):
    """Bind variant to fixed release bytes; caller-supplied hashes cannot swap it."""
    if base_variant not in PDD_RELEASES:
        raise ValueError("PDD base_variant must be fl2va or ref2va")
    pinned = PDD_RELEASES[base_variant]["sha256"]
    if expected_sha256 is not None and expected_sha256 != pinned:
        raise ValueError("PDD checkpoint SHA256 mismatch with the selected published variant")
    if sha256 != pinned:
        raise ValueError("PDD checkpoint SHA256 mismatch with the selected published variant")


def validate_base_partition(base_weights, base_variant):
    path = Path(base_weights).expanduser().resolve()
    if base_variant not in PDD_RELEASES or path.name != "transformer" or path.parent.name.lower() != base_variant:
        raise ValueError("Raw H3 transformer directory must match the selected FL2VA/Ref2VA partition")
    return path


def validate_pdd_conditioning(base_variant, inputs):
    """Check actual encoded anchors and forbid mixing FL and Ref contracts."""
    if base_variant not in PDD_RELEASES:
        raise ValueError("PDD base_variant must be fl2va or ref2va")
    if inputs.get("control_rows") is not None or inputs.get("controlnet") is not None:
        raise ValueError("Published PDD does not support ControlNet")
    if inputs.get("input_latents_video") is not None or inputs.get("input_latents_audio") is not None:
        raise ValueError("Published PDD does not support Retake")
    if inputs.get("cfg_scale", 1.0) != 1.0:
        raise ValueError("PDD checkpoint validation is restricted to cfg_scale=1")
    visual, audio, keyframe = (inputs.get(name) for name in (
        "ref_visual_anchor", "ref_audio_anchor", "keyframe_cond_anchor"))
    for name, anchor in (("ref_visual_anchor", visual), ("ref_audio_anchor", audio), ("keyframe_cond_anchor", keyframe)):
        if anchor is not None and (not isinstance(anchor, torch.Tensor) or anchor.ndim != 2 or not anchor.numel()):
            raise ValueError(f"PDD requires a nonempty encoded tensor for {name}")
    if base_variant == "ref2va":
        if keyframe is not None or inputs.get("keyframes") or inputs.get("keyframe_indices"):
            raise ValueError("Ref2VA PDD cannot consume FL2VA keyframes")
        if visual is None and audio is None:
            raise ValueError("Ref2VA PDD requires actual visual or audio reference conditioning")
    else:
        if visual is not None or audio is not None or inputs.get("ref_blocks") is not None or inputs.get("references"):
            raise ValueError("FL2VA PDD cannot consume Ref2VA reference conditioning")
        indices = inputs.get("keyframe_indices")
        if keyframe is None:
            if inputs.get("keyframes") or indices:
                raise ValueError("FL2VA keyframe declarations require actual encoded keyframe anchors")
            # The pinned PAI predict_t2v.py explicitly uses this unconditioned path.
            return
        if (not isinstance(indices, (list, tuple)) or not 1 <= len(indices) <= 2
                or any(type(index) is not int or index not in (0, -1) for index in indices)
                or len(set(indices)) != len(indices)):
            raise ValueError("FL2VA PDD keyframes require unique first/last indices 0/-1")
        if inputs.get("keyframes") is not None and len(inputs["keyframes"]) != len(indices):
            raise ValueError("FL2VA keyframe declarations and indices have different lengths")
        latent = inputs.get("video_latents")
        if (not isinstance(latent, torch.Tensor) or latent.ndim != 5 or latent.shape[0] != 1
                or latent.shape[-2] % 2 or latent.shape[-1] % 2):
            raise ValueError("FL2VA PDD requires a valid video latent grid for its keyframe anchors")
        frame_rows = (latent.shape[-2] // 2) * (latent.shape[-1] // 2)
        if keyframe.shape != (len(indices) * frame_rows, latent.shape[1] * 4):
            raise ValueError("FL2VA encoded keyframe anchors do not match the actual latent grid")


def pdd_time_grid(shift: float, steps: int = PDD_STEPS):
    sigma = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64)
    return 1.0 - shift * sigma / (1 + (shift - 1) * sigma)


def pdd_plans(shift: float):
    sizes = pdd_time_grid(shift).diff()
    plans = torch.zeros(NFE, PDD_STEPS, dtype=torch.float64)
    for i in range(NFE):
        start = i * BLOCK_SIZE
        plans[i, start:start + BLOCK_SIZE] = sizes[start:start + BLOCK_SIZE] / sizes[start:start + BLOCK_SIZE].sum()
    return plans.float()


def qkv_interleave(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int):
    if q.shape != k.shape or q.shape != v.shape or q.shape[0] % heads:
        raise ValueError("Q/K/V row shapes must match and divide the head count")
    shape = (heads, q.shape[0] // heads, *q.shape[1:])
    return torch.stack((q.reshape(shape), k.reshape(shape), v.reshape(shape)), dim=1).flatten(0, 2)


def swiglu_swap(value: torch.Tensor):
    if value.shape[0] % 2:
        raise ValueError("SwiGLU projection must have an even output size")
    up, gate = value.chunk(2, dim=0)
    return torch.cat((gate, up), dim=0)


class AdditiveLoRA(nn.Module):
    """Original wrapped base linear plus BF16 B(Ax), without weight fusion."""
    def __init__(self, base, pairs, *, heads=None, swap=False):
        super().__init__()
        self.base = base
        self.heads, self.swap = heads, swap
        for i, (down, up) in enumerate(pairs):
            self.register_buffer(f"down_{i}", down, persistent=False)
            self.register_buffer(f"up_{i}", up, persistent=False)
        self.pair_count = len(pairs)
        self._device_pairs = {}

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, value):
        result = self.base(value)
        key = (value.device, value.dtype)
        if key not in self._device_pairs:
            self._device_pairs[key] = [(getattr(self, f"down_{i}").to(value.device, value.dtype),
                                        getattr(self, f"up_{i}").to(value.device, value.dtype))
                                       for i in range(self.pair_count)]
        updates = [F.linear(F.linear(value, down), up) for down, up in self._device_pairs[key]]
        if self.heads is not None:
            # Transpose token/feature axes before applying the weight-row permutation.
            delta = qkv_interleave(*(u.movedim(-1, 0) for u in updates), self.heads).movedim(0, -1)
        else:
            delta = updates[0]
            if self.swap:
                delta = swiglu_swap(delta.movedim(-1, 0)).movedim(0, -1)
        return result + (ALPHA / RANK) * delta.to(result.dtype)


class ParallelHead(nn.Module):
    """Pre-fused FP32 heads indexed by the validated model timestep, not a counter."""
    def __init__(self, weight, bias, plans, context):
        super().__init__()
        if weight.ndim != 3 or bias.shape != weight.shape[:2] or weight.shape[0] != PDD_STEPS:
            raise ValueError("PDD head requires 32 weight/bias directions")
        # Same complete plan einsum as the published head. CPU FP32 differs from
        # GPU einsum only by backend arithmetic; no BF16 head precision is lost.
        self.register_buffer("fused_weight", torch.einsum("pn,noi->poi", plans.float(), weight.float()), persistent=False)
        self.register_buffer("fused_bias", torch.einsum("pn,no->po", plans.float(), bias.float()), persistent=False)
        self.context = context
        self._device_heads = {}

    def forward(self, hidden):
        index = self.context.get("step_index")
        if index is None or not 0 <= index < NFE:
            raise RuntimeError("PDD parallel heads were not armed by a validated timestep")
        key = hidden.device
        if key not in self._device_heads:
            self._device_heads[key] = (self.fused_weight.to(key), self.fused_bias.to(key))
        weight, bias = self._device_heads[key]
        return F.linear(hidden.float(), weight[index], bias[index])


class PDDScheduler:
    """Diffusers H3 numeric schedule adapted to DiffSynth's negated velocity API."""
    def __init__(self, shift):
        self.shift = float(shift)
        self.num_train_timesteps = 1000
        self.training = False
        self.set_timesteps(NFE, shift=shift)

    def set_timesteps(self, num_inference_steps=NFE, denoising_strength=1.0, training=False, shift=None, **kwargs):
        if num_inference_steps != NFE or denoising_strength != 1.0 or training or kwargs:
            raise ValueError("Published PDD requires exactly eight evaluations and full denoising")
        if shift is not None and float(shift) != self.shift:
            raise ValueError("PDD video/audio shift changed from the validated published configuration")
        base = torch.linspace(1.0, 0.0, NFE + 1, dtype=torch.float32)
        self.sigmas = self.shift * base / (1 + (self.shift - 1) * base)
        self.timesteps = 1.0 - self.sigmas[:-1]

    def index_for_timestep(self, timestep):
        value = torch.as_tensor(timestep, dtype=torch.float32).detach().cpu().reshape(-1)
        if value.numel() != 1:
            raise ValueError("PDD timestep must be scalar")
        indices = torch.nonzero(self.timesteps == value[0]).flatten()
        if indices.numel() != 1:
            raise ValueError("Unknown PDD timestep; nearest-step fallback is forbidden")
        return int(indices[0])

    def step(self, model_output, timestep, sample, to_final=False, **kwargs):
        if to_final or kwargs:
            raise ValueError("PDD scheduler does not support alternate step options")
        index = self.index_for_timestep(timestep)
        timestep = torch.as_tensor(timestep).to(sample.device, sample.dtype)
        while timestep.ndim < sample.ndim:
            timestep = timestep.unsqueeze(-1)
        # DiffSynth's model_fn returns NEGATED data-ward H3 velocity.
        denoised = sample + (1 - timestep) * (-model_output)
        compute_dtype = torch.float32 if sample.dtype in (torch.float16, torch.bfloat16) else sample.dtype
        sigma = self.sigmas[index].to(sample.device, compute_dtype)
        next_sigma = self.sigmas[index + 1].to(sample.device, compute_dtype)
        ratio = next_sigma / sigma
        return (ratio * sample.to(compute_dtype) + (1 - ratio) * denoised.to(compute_dtype)).to(sample.dtype)


def _read_header(path):
    with Path(path).open("rb") as handle:
        length_bytes = handle.read(8)
        if len(length_bytes) != 8:
            raise ValueError("Truncated safetensors header")
        length = struct.unpack("<Q", length_bytes)[0]
        if not 2 <= length <= 16 * 1024 * 1024:
            raise ValueError("Invalid safetensors header size")
        return json.loads(handle.read(length))


def expected_shapes(dit):
    shapes = {"proj_out.weight": (PDD_STEPS, 96, 5376), "proj_out.bias": (PDD_STEPS, 96),
              "audio_proj_out.weight": (PDD_STEPS, 32, 5376), "audio_proj_out.bias": (PDD_STEPS, 32)}
    groups = [(f"transformer_blocks.{i}", block, True) for i, block in enumerate(dit.blocks)]
    groups += [(f"token_refiner.refiner_blocks.{i}", block, False) for i, block in enumerate(dit.token_refiner.blocks)]
    for prefix, block, adaln in groups:
        inner = block.attn.qkv_proj.weight.shape[0] // 3
        specs = [(f"attn.to_{kind}", inner, block.attn.qkv_proj.weight.shape[1]) for kind in "qkv"]
        specs += [("attn.to_out.0", *block.attn.out_proj.weight.shape),
                  ("ff.net.0.proj", *block.mlp.fc1.weight.shape),
                  ("ff.net.2", *block.mlp.fc2.weight.shape)]
        if adaln:
            specs.append(("adaln_proj.linear", *block.adaln_proj.linear.weight.shape))
        for name, out_features, in_features in specs:
            shapes[f"{prefix}.{name}.lora_down"] = (RANK, in_features)
            shapes[f"{prefix}.{name}.lora_up"] = (out_features, RANK)
    return shapes


def validate_checkpoint_header(header, shapes):
    metadata = header.get("__metadata__", {})
    required = {"lora_alpha": "64.0", "pdd_block_size": "4", "pdd_num_steps": "32",
                "lora_rank": "64", "lora_targets": TARGETS}
    if metadata != required:
        raise ValueError("PDD checkpoint metadata differs from the published rank64/grid32/block4 contract")
    actual = set(header) - {"__metadata__"}
    if actual != set(shapes):
        raise ValueError(f"PDD tensor keys mismatch: missing={sorted(set(shapes)-actual)[:3]}, unexpected={sorted(actual-set(shapes))[:3]}")
    for key, shape in shapes.items():
        if tuple(header[key]["shape"]) != tuple(shape) or header[key]["dtype"] != "BF16":
            raise ValueError(f"PDD tensor shape/dtype mismatch: {key}")


def _restore_fp32(dit, base_weights):
    from safetensors import safe_open
    names = ("video_patch_proj", "audio_patch_proj", "time_embedder.proj_in", "time_embedder.proj_out")
    desired = {f"{name}.{suffix}" for name in names for suffix in ("weight", "bias")}
    paths = sorted(Path(base_weights).glob("model-*.safetensors"))
    if not paths:
        raise FileNotFoundError("Existing raw FL2VA/Ref2VA transformer shards are required")
    found = {}
    for path in paths:
        header = _read_header(path)
        for key in desired:
            matches = [source for source in (key, "model.diffusion_model." + key) if source in header]
            if len(matches) > 1 or (matches and key in found):
                raise ValueError(f"Duplicate raw base tensor: {key}")
            if matches:
                source = matches[0]
                if header[source]["dtype"] != "F32":
                    raise ValueError(f"Raw base module must retain original FP32 precision: {source}")
                found[key] = (path, source, header[source]["shape"])
    if set(found) != desired:
        raise ValueError(f"Missing original FP32 base tensors: {sorted(desired-set(found))}")
    for key, (_, _, shape) in found.items():
        if tuple(shape) != tuple(dit.get_parameter(key).shape):
            raise ValueError(f"Original FP32 base tensor shape mismatch: {key}")
    sources = {}
    for name in names:
        module = dit.get_submodule(name)
        if getattr(module, "disk_offload", False):
            raise ValueError("PDD restoration requires CPU-resident raw base weights, not disk-offloaded modules")
        module.to(device="cpu", dtype=torch.float32)
        for dtype_name in ("offload_dtype", "onload_dtype", "preparing_dtype", "computation_dtype"):
            if hasattr(module, dtype_name):
                setattr(module, dtype_name, torch.float32)
        if hasattr(module, "state"):
            module.state = 0
        for suffix in ("weight", "bias"):
            key = f"{name}.{suffix}"
            path, source, _ = found[key]
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                value = handle.get_tensor(source)
                getattr(module, suffix).data.copy_(value)
            sources[key] = {"shard": path.name, "source_key": source, "dtype": "float32"}
    return sources


def _mixed_embed(self, *, x, audio_x, text_embeddings_selected, unique_timesteps, img_pos,
                 audio_pos, text_pos, refiner_cu_seqlens, refiner_max_seqlen, seq_len, device):
    dtype = text_embeddings_selected.dtype
    video = self.video_patch_proj(x.view(-1, x.shape[-1]).index_select(0, img_pos).float())
    audio = self.audio_patch_proj(audio_x.view(-1, audio_x.shape[-1]).index_select(0, audio_pos).float())
    text = self.condition_proj(text_embeddings_selected.to(device))
    text = self.token_refiner(text, cu_seqlens=refiner_cu_seqlens, max_seqlen=refiner_max_seqlen)
    embeddings = torch.zeros((seq_len, self.hidden_size), device=device, dtype=dtype)
    embeddings.index_add_(0, text_pos, text.to(dtype)[:text_pos.shape[0]])
    embeddings.index_add_(0, img_pos, video.to(dtype)[:img_pos.shape[0]])
    embeddings.index_add_(0, audio_pos, audio.to(dtype)[:audio_pos.shape[0]])
    return embeddings, self.time_embedder(unique_timesteps, dtype=torch.float32)


def _mixed_adaln(self, t_emb):
    output = self.linear(F.silu(t_emb).to(self.linear.weight.dtype))
    output = output.view(t_emb.shape[0] * self.modality_num, self.expand_ratio * self.hidden_size)
    return tuple(output.chunk(self.expand_ratio, dim=-1))


def _dataward_model_fn(original):
    """Patch precisely two checked AST assignments; all other vendor code stays intact."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    changed = set()
    for statement in tree.body[0].body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
            name = statement.targets[0].id
            if name in ("t_video", "t_audio"):
                modality = name.removeprefix("t_")
                expected = ast.parse(f"1.0 - float(timestep_{modality}) / 1000", mode="eval").body
                if ast.dump(statement.value) != ast.dump(expected):
                    raise ValueError("Pinned H3 model_fn timestep transformation changed")
                statement.value = ast.parse(f"float(timestep_{modality})", mode="eval").body
                changed.add(name)
    if changed != {"t_video", "t_audio"}:
        raise ValueError("Unsupported H3 model_fn: both timestep transformations are required")
    tree.body[0].decorator_list = []
    namespace = dict(original.__globals__)
    exec(compile(ast.fix_missing_locations(tree), "<H3-PDD-model-fn>", "exec"), namespace)
    return namespace[original.__name__]


def install_pdd(pipe, path, *, base_weights, base_variant="ref2va", video_shift=12.0,
                audio_shift=3.0, expected_sha256=None):
    """Install the local published PDD checkpoint before enable_adaln_exact(pipe).

    `base_weights` is the existing raw FL2VA or Ref2VA/transformer directory. Call the
    pipeline with num_inference_steps=8, cfg_scale=1, shifts 12/3. No download is
    performed and every checkpoint key/shape/dtype is validated before mutation.
    """
    started = time.perf_counter()
    if base_variant not in PDD_RELEASES or video_shift != 12.0 or audio_shift != 3.0:
        raise ValueError("Published PDD requires the matching original FL2VA/Ref2VA base and video/audio shifts 12/3")
    if getattr(pipe, "_h3_pdd_installed", False) or getattr(pipe, "_h3_adaln_exact_enabled", False):
        raise ValueError("Install PDD once, before enabling the exact AdaLN cache")
    dit = pipe.dit
    if dit.training or len(dit.blocks) != 50 or len(dit.token_refiner.blocks) != 2 or dit.hidden_size != 5376:
        raise ValueError("PDD requires the original eval-mode 50-layer/2-refiner H3 architecture")
    checkpoint = Path(path).expanduser().resolve(strict=True)
    if checkpoint.suffix != ".safetensors":
        raise ValueError("Only the published safetensors PDD checkpoint is supported")
    digest = hashlib.sha256()
    with checkpoint.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    sha = digest.hexdigest()
    validate_adapter_variant(sha, base_variant, expected_sha256)
    base_weights = validate_base_partition(base_weights, base_variant)
    shapes = expected_shapes(dit)
    if len(shapes) != 728:
        raise ValueError("Original PDD checkpoint must contain exactly 728 tensors")
    validate_checkpoint_header(_read_header(checkpoint), shapes)
    transformed_fn = _dataward_model_fn(pipe.model_fn)
    # Check every target before mutating any module, including existing adapters.
    targets = [block for block in dit.blocks] + [block for block in dit.token_refiner.blocks]
    for block in targets:
        linears = [block.attn.qkv_proj, block.attn.out_proj, block.mlp.fc1, block.mlp.fc2]
        if hasattr(block, "adaln_proj"):
            linears.append(block.adaln_proj.linear)
        for linear in linears:
            if linear.weight.dtype != torch.bfloat16 or linear.weight.device.type != "cpu":
                raise ValueError("PDD requires unquantized BF16 base target linears initially resident on CPU")
            if getattr(linear, "disk_offload", False) or getattr(linear, "lora_A_weights", []):
                raise ValueError("PDD requires an unmodified CPU base without existing hotloaded adapters")
    restored = _restore_fp32(dit, base_weights)
    from safetensors import safe_open
    context = {}
    adapter_bytes = 0
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        def pair(prefix):
            nonlocal adapter_bytes
            down, up = handle.get_tensor(prefix + ".lora_down"), handle.get_tensor(prefix + ".lora_up")
            adapter_bytes += (down.numel() + up.numel()) * 2
            if not bool(torch.isfinite(down).all()) or not bool(torch.isfinite(up).all()):
                raise ValueError(f"PDD adapter contains nonfinite values: {prefix}")
            return down, up
        groups = [(f"transformer_blocks.{i}", b, True) for i, b in enumerate(dit.blocks)]
        groups += [(f"token_refiner.refiner_blocks.{i}", b, False) for i, b in enumerate(dit.token_refiner.blocks)]
        for prefix, block, adaln in groups:
            block.attn.qkv_proj = AdditiveLoRA(block.attn.qkv_proj,
                [pair(f"{prefix}.attn.to_{kind}") for kind in "qkv"], heads=block.attn.num_heads)
            block.attn.out_proj = AdditiveLoRA(block.attn.out_proj, [pair(prefix + ".attn.to_out.0")])
            block.mlp.fc1 = AdditiveLoRA(block.mlp.fc1, [pair(prefix + ".ff.net.0.proj")], swap=True)
            block.mlp.fc2 = AdditiveLoRA(block.mlp.fc2, [pair(prefix + ".ff.net.2")])
            if adaln:
                block.adaln_proj.linear = AdditiveLoRA(block.adaln_proj.linear, [pair(prefix + ".adaln_proj.linear")])
        for attribute, prefix, shift in (("video_out", "proj_out", video_shift), ("audio_out", "audio_proj_out", audio_shift)):
            weight, bias = handle.get_tensor(prefix + ".weight"), handle.get_tensor(prefix + ".bias")
            if not bool(torch.isfinite(weight).all()) or not bool(torch.isfinite(bias).all()):
                raise ValueError("PDD heads contain nonfinite values")
            setattr(dit.final_layer, attribute, ParallelHead(weight, bias, pdd_plans(shift), context))
    dit._embed = MethodType(_mixed_embed, dit)
    for projection in [b.adaln_proj for b in dit.blocks] + [dit.final_layer.adaln_proj]:
        projection.forward = MethodType(_mixed_adaln, projection)
    pipe.scheduler, pipe.scheduler_audio = PDDScheduler(video_shift), PDDScheduler(audio_shift)
    def pdd_model_fn(*args, **kwargs):
        if args or kwargs.get("dit") is not dit:
            raise ValueError("PDD model must receive the installed DiT as keyword input")
        validate_pdd_conditioning(base_variant, kwargs)
        index = pipe.scheduler.index_for_timestep(kwargs["timestep_video"])
        if index != pipe.scheduler_audio.index_for_timestep(kwargs["timestep_audio"]):
            raise ValueError("PDD video/audio scheduler block indices do not match")
        context["step_index"] = index
        return transformed_fn(**kwargs)
    pipe.model_fn = pdd_model_fn
    pipe._h3_timestep_convention = "dataward"
    pipe._h3_time_embedding_dtype = torch.float32
    pipe._h3_pdd_installed = True
    return {"profile": "pdd8", "nfe": NFE, "interval_heads": PDD_STEPS, "block_size": BLOCK_SIZE,
            "lora_rank": RANK, "lora_alpha": ALPHA, "tensors": 728, "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha, "adapter_bytes": adapter_bytes, "base_variant": base_variant,
            "adapter_repository": PDD_REPOSITORY, "adapter_revision": PDD_REVISION,
            "adapter_variant": base_variant, "published_adapter_filename": PDD_RELEASES[base_variant]["filename"],
            "base_weights_path": str(base_weights),
            "base_partition_validation": "directory_name_and_original_fp32_source_keys",
            "fp32_restored": restored, "head_fusion_dtype": "float32", "head_fusion_device": "cpu",
            "lora_execution": "runtime_additive_bf16", "scheduler": "official_h3_float32_grid_and_blend",
            "backend_differences": ["DiffSynth fused base QKV GEMM", "DiffSynth packed attention backend",
                                   "CPU FP32 fused-head einsum versus published device einsum"],
            "install_seconds": time.perf_counter() - started}
