from __future__ import annotations
from pathlib import Path
from typing import Literal
import yaml
from pydantic import Field, model_validator
from .schemas import StrictModel


class Config(StrictModel):
    backend: str = "diffsynth"
    short_edge: int = 768
    export_width: int = 1344
    export_height: int = 768
    diffsynth_python: str = ""
    diffsynth_project: str = ""
    fl2va_weights: str = ""
    ref2va_weights: str = ""
    cuda_device: str = "0"
    inference_steps: int = 50
    acceleration_profile: Literal['native', 'adaln_exact', 'pdd8_exact', 'balanced'] = 'native'
    attention_backend: Literal['auto', 'flash_attn', 'flash_attn_3', 'torch_sdpa'] = 'auto'
    vram_reserve_gb: float = Field(default=5, ge=4)
    vram_limit_gb: float | None = Field(default=None, gt=0)
    pdd_lora_path: str | None = None
    pdd_lora_sha256: str | None = Field(default=None, pattern=r'^[0-9a-f]{64}$')
    pdd_base_variant: Literal['ref2va', 'fl2va'] = 'ref2va'
    fl2va_pdd_lora_path: str | None = None
    fl2va_pdd_lora_sha256: str | None = Field(default=None, pattern=r'^[0-9a-f]{64}$')

    @model_validator(mode='after')
    def validate_pdd_profile(self):
        if bool(self.fl2va_pdd_lora_path) != bool(self.fl2va_pdd_lora_sha256):
            raise ValueError('fl2va_pdd_lora_path and fl2va_pdd_lora_sha256 must be supplied together')
        if self.acceleration_profile in ('pdd8_exact', 'balanced'):
            expected_steps = 8 if self.acceleration_profile == 'pdd8_exact' else 50
            if self.inference_steps != expected_steps:
                raise ValueError(f'{self.acceleration_profile} requires exactly {expected_steps} steps in its configuration')
            if not self.pdd_lora_path or not self.pdd_lora_sha256:
                raise ValueError(f'{self.acceleration_profile} requires pdd_lora_path and its expected pdd_lora_sha256')
            if self.acceleration_profile == 'balanced' and self.backend != 'diffsynth':
                raise ValueError('balanced currently requires the DiffSynth backend')
            if self.acceleration_profile == 'balanced' and self.pdd_base_variant != 'ref2va':
                raise ValueError('balanced uses pdd_lora_path for Ref2VA; configure the FL2VA adapter separately')
        return self

    def supported_modes(self):
        if self.acceleration_profile == 'pdd8_exact':
            return ('ref2va',) if self.pdd_base_variant == 'ref2va' else ('t2va', 'fl2va')
        return ('t2va', 'fl2va', 'ref2va')

    def effective_config(self, mode):
        """Return a validated independent configuration for one H3 task mode."""
        if mode not in ('t2va', 'fl2va', 'ref2va'):
            raise ValueError('Unknown H3 task mode: '+str(mode))
        values = self.model_dump()
        if self.acceleration_profile == 'balanced':
            if mode == 'ref2va':
                values.update(acceleration_profile='pdd8_exact', inference_steps=8, pdd_base_variant='ref2va')
            elif self.fl2va_pdd_lora_path:
                values.update(acceleration_profile='pdd8_exact', inference_steps=8, pdd_base_variant='fl2va',
                              pdd_lora_path=self.fl2va_pdd_lora_path,
                              pdd_lora_sha256=self.fl2va_pdd_lora_sha256)
            else:
                values.update(acceleration_profile='adaln_exact', inference_steps=50)
        resolved = Config.model_validate(values)
        if mode not in resolved.supported_modes():
            permitted = 'Ref2VA' if resolved.pdd_base_variant == 'ref2va' else 'T2VA/FL2VA'
            raise ValueError(f'pdd8_exact {resolved.pdd_base_variant} adapter is supported only for {permitted} at exactly 8 steps')
        return resolved


def load_config(path=None):
    return Config.model_validate(yaml.safe_load(Path(path).read_text()) if path else {})
