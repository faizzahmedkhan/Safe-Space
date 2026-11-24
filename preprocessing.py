"""The model's input contract: how frames are cropped and how pixels are scaled.

A trained model only works on input prepared the way it was trained, so the
contract travels with the weights (the "preprocessing" key of the model's JSON
sidecar) and every scoring path reads it from there instead of assuming:

  crop       "frame" - every frame gets its own motion/centre crop
                       (frame_prep.prepare_for_model; the original pipeline).
             "clip"  - one crop box per 16-frame clip, computed from the
                       motion boxes of all its frames and applied to every
                       frame (frame_prep.observe_frame + build_clip), the way
                       Kinetics-400 clips were cropped. Removes the
                       crop jumping inside a clip that the model could
                       otherwise mistake for motion.
  normalize  "none"     - pixels / 255 only (the original pipeline).
             "kinetics" - pixels / 255, then the per-channel mean/std that
                       torchvision's R3D-18 Kinetics-400 weights were trained
                       with, so the pretrained (frozen) layers see the input
                       distribution they were built for.

Presets: "v1" = (frame, none), every model trained before this file existed
(and the default for a sidecar without the key); "v2" = (clip, kinetics).
The two fields are independent so either change can be ablated on its own.

Pure Python (no numpy/cv2/torch) so tests and tools can import it anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

CROP_MODES = ("frame", "clip")
NORMALIZE_MODES = ("none", "kinetics")

# torchvision R3D_18_Weights.KINETICS400_V1 transforms (RGB order).
KINETICS_MEAN = (0.43216, 0.394666, 0.37645)
KINETICS_STD = (0.22803, 0.22145, 0.216989)


@dataclass(frozen=True)
class PreprocSpec:
    crop: str = "frame"
    normalize: str = "none"

    def __post_init__(self) -> None:
        if self.crop not in CROP_MODES:
            raise ValueError(f"crop must be one of {CROP_MODES}, got {self.crop!r}")
        if self.normalize not in NORMALIZE_MODES:
            raise ValueError(f"normalize must be one of {NORMALIZE_MODES}, got {self.normalize!r}")

    @property
    def name(self) -> str:
        for preset, spec in PRESETS.items():
            if spec == self:
                return preset
        return f"crop={self.crop},normalize={self.normalize}"

    def to_dict(self) -> Dict[str, str]:
        return {"crop": self.crop, "normalize": self.normalize}

    def __str__(self) -> str:
        return self.name


PREPROC_V1 = PreprocSpec("frame", "none")
PREPROC_V2 = PreprocSpec("clip", "kinetics")
PRESETS: Dict[str, PreprocSpec] = {"v1": PREPROC_V1, "v2": PREPROC_V2}
# What a model without a recorded contract was trained with.
LEGACY_PREPROC = PREPROC_V1


def parse_preprocessing(value: Union[None, str, Dict[str, Any], PreprocSpec]) -> PreprocSpec:
    """Sidecar / CLI value -> PreprocSpec.  None means a legacy model (v1).
    Accepts a preset name ("v1", "v2"), a dict with "crop"/"normalize" (missing
    fields fall back to v1's), or a PreprocSpec."""
    if value is None:
        return LEGACY_PREPROC
    if isinstance(value, PreprocSpec):
        return value
    if isinstance(value, str):
        key = value.strip().lower()
        if key not in PRESETS:
            raise ValueError(f"unknown preprocessing preset {value!r}; use one of {sorted(PRESETS)}")
        return PRESETS[key]
    if isinstance(value, dict):
        unknown = set(value) - {"crop", "normalize", "name"}
        if unknown:
            raise ValueError(f"unknown preprocessing fields {sorted(unknown)}")
        return PreprocSpec(
            crop=str(value.get("crop", LEGACY_PREPROC.crop)),
            normalize=str(value.get("normalize", LEGACY_PREPROC.normalize)),
        )
    raise TypeError(f"cannot read preprocessing from {type(value).__name__}")


def sidecar_value(spec: PreprocSpec) -> Dict[str, str]:
    """What to write under "preprocessing" in a model sidecar."""
    return {"name": spec.name, **spec.to_dict()}


# Attribute used to attach the contract to a loaded model object, so every
# scoring path (live watcher, /analyze, calibration, evaluation) can read it
# from the model it was handed instead of trusting its caller.
MODEL_ATTR = "safespace_preprocessing"


def attach(model: Any, spec: PreprocSpec) -> Any:
    setattr(model, MODEL_ATTR, spec)
    return model


def of_model(model: Any) -> PreprocSpec:
    spec: Optional[PreprocSpec] = getattr(model, MODEL_ATTR, None)
    return spec if spec is not None else LEGACY_PREPROC
