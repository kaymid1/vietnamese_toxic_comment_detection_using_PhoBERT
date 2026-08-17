"""Portable references for application-owned artifacts.

This module is deliberately conservative: only managed-root paths are encoded.
URLs and unsupported URI schemes are opaque values and are never rewritten.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping
from typing import Optional, Union

from backend.runtime_paths import get_data_dir, get_model_options_dir, get_project_root, get_runtime_dir

PORTABLE_SCHEMES = {"data": get_data_dir, "runtime": get_runtime_dir, "model": get_model_options_dir}
PROTECTED_URI_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]")


@dataclass(frozen=True)
class ArtifactReference:
    value: str
    classification: str
    path: Optional[Path] = None
    warning: Optional[str] = None


def _is_absolute(value: str) -> bool:
    return bool(WINDOWS_ABSOLUTE_RE.match(value)) or value.startswith("/") or Path(value).is_absolute()


def _portable_parts(value: str) -> Optional[tuple[str, str]]:
    if "://" not in value:
        return None
    scheme, suffix = value.split("://", 1)
    if scheme not in PORTABLE_SCHEMES:
        return None
    candidate = Path(suffix.replace("\\", "/"))
    if candidate.is_absolute() or ".." in candidate.parts:
        return None
    return scheme, suffix


def _legacy_managed_ref(value: str) -> Optional[str]:
    """Map only recognised historical project layouts, without requiring existence."""
    normalized = value.replace("\\", "/")
    pieces = [part for part in normalized.split("/") if part]
    lowered = [part.lower() for part in pieces]
    for marker, scheme, required in ((".runtime", "runtime", {"model_registry", "kaggle_real_jobs"}), ("models", "model", {"options"}), ("data", "data", {"processed", "raw", "mlflow"})):
        for index, part in enumerate(lowered):
            if part != marker or index + 1 >= len(pieces):
                continue
            # Historical remapping is intentionally limited to known project
            # roots. Current configured roots are handled by containment above.
            # Skipping a renamed/unknown legacy root is safer than guessing.
            if index == 0 or lowered[index - 1] != "thesis":
                continue
            remainder = pieces[index + 1 :]
            if remainder and remainder[0].lower() in required:
                if scheme == "model":
                    remainder = remainder[1:]
                return f"{scheme}://{'/'.join(remainder)}"
    return None


def _roots(overrides: Optional[Mapping[str, Path]] = None) -> dict[str, Path]:
    resolved = {scheme: factory().resolve() for scheme, factory in PORTABLE_SCHEMES.items()}
    if overrides:
        unknown = set(overrides) - set(PORTABLE_SCHEMES)
        if unknown:
            raise ValueError(f"Unknown artifact root scheme(s): {', '.join(sorted(unknown))}")
        resolved.update({scheme: Path(path).expanduser().resolve() for scheme, path in overrides.items()})
    return resolved


def inspect_artifact_ref(
    value: Union[str, Path, None], *, roots: Optional[Mapping[str, Path]] = None
) -> ArtifactReference:
    raw = str(value or "").strip()
    if not raw:
        return ArtifactReference(raw, "empty")
    portable = _portable_parts(raw)
    if portable:
        scheme, suffix = portable
        if not suffix or ".." in Path(suffix.replace("\\", "/")).parts:
            return ArtifactReference(raw, "invalid", warning="invalid portable reference")
        return ArtifactReference(raw, "portable", _roots(roots)[scheme] / Path(suffix.replace("\\", "/")))
    if PROTECTED_URI_RE.match(raw):
        return ArtifactReference(raw, "protected_uri", warning="unsupported URL/URI preserved")
    # New writers encode by containment even when the configured root is not
    # named data/.runtime/models (for example an external volume on macOS).
    if _is_absolute(raw):
        candidate = Path(raw).expanduser().resolve()
        for scheme, root in _roots(roots).items():
            try:
                relative = candidate.relative_to(root)
            except ValueError:
                continue
            return ArtifactReference(raw, "managed_absolute", candidate, f"{scheme}://{relative.as_posix()}")
    if _is_absolute(raw):
        legacy = _legacy_managed_ref(raw)
        if legacy:
            return ArtifactReference(raw, "legacy_managed_absolute", Path(raw), warning=legacy)
        return ArtifactReference(raw, "external_absolute", Path(raw), "unmanaged external absolute path")
    candidate = get_project_root() / Path(raw)
    legacy = _legacy_managed_ref(str(candidate))
    if legacy:
        return ArtifactReference(raw, "legacy_managed_relative", candidate, warning=legacy)
    return ArtifactReference(raw, "relative_unmanaged", candidate, "unrecognised relative path")


def resolve_artifact_ref(
    value: Union[str, Path, None], *, roots: Optional[Mapping[str, Path]] = None
) -> Path:
    inspected = inspect_artifact_ref(value, roots=roots)
    if inspected.classification in {"portable", "managed_absolute", "legacy_managed_absolute", "legacy_managed_relative", "external_absolute", "relative_unmanaged"} and inspected.path is not None:
        return inspected.path.expanduser().resolve()
    raise ValueError(f"Artifact reference could not be resolved: {inspected.value!r} ({inspected.classification})")


def encode_artifact_ref(
    value: Union[str, Path, None], *, roots: Optional[Mapping[str, Path]] = None
) -> str:
    """Encode a known managed path, otherwise preserve the original string exactly."""
    raw = str(value or "").strip()
    if not raw:
        return raw
    inspected = inspect_artifact_ref(raw, roots=roots)
    if inspected.classification == "portable":
        return raw
    if inspected.classification in {"managed_absolute", "legacy_managed_absolute", "legacy_managed_relative"}:
        return str(inspected.warning)
    return raw
