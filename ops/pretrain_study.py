"""Study config lowering, TOML serialization, and cache validation for pretrain."""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

from ops.pretrain_base import CORPUS_ALIASES, MODEL_ALIASES, file_digest


def canonical_model(document: dict) -> None:
    document["model"] = MODEL_ALIASES.get(document.get("model"), document.get("model"))
    document["corpus"] = CORPUS_ALIASES.get(
        document.get("corpus"), document.get("corpus")
    )


def scalar_section(document: dict, section: str) -> dict:
    value = document.pop(section, None)
    if value is None:
        return {}
    if not isinstance(value, dict):
        document[section] = value
        return {}
    return dict(value)


def move_absent(document: dict, section: str, key: str, value) -> None:
    if key in document:
        raise ValueError(f"{section}.{key} conflicts with top-level {key}")
    document[key] = value


def single_subsection(section: dict, name: str):
    subtables = [key for key, value in section.items() if isinstance(value, dict)]
    if not subtables:
        return None
    if len(subtables) != 1:
        raise ValueError(f"{name} must select exactly one primitive")
    key = subtables[0]
    return key, dict(section.pop(key))


def normalize_primitive(name: str) -> str:
    return name.replace("-", "_")


def section_alias(document: dict, public: str, legacy: str) -> dict:
    public_value = scalar_section(document, public)
    legacy_value = scalar_section(document, legacy)
    if public_value and legacy_value:
        raise ValueError(f"{public} conflicts with {legacy}")
    return public_value or legacy_value


def _moveall(document: dict, section_name: str, fields: dict) -> None:
    for key, value in fields.items():
        move_absent(document, section_name, key, value)


def _lowertop(document: dict) -> None:
    pretrain = scalar_section(document, "pretrain")
    if pretrain:
        move_absent(document, "pretrain", "study", "pretrain")
        _moveall(document, "pretrain", pretrain)
    study = scalar_section(document, "study")
    if study:
        kind = study.pop("kind", None)
        if kind is not None:
            move_absent(document, "study", "study", kind)
        _moveall(document, "study", study)
    rounds = scalar_section(document, "rounds")
    if "count" in rounds:
        move_absent(document, "rounds", "rounds", rounds.pop("count"))
    if "target_ms" in rounds:
        move_absent(document, "rounds", "target_round_ms", rounds.pop("target_ms"))
    _moveall(document, "rounds", rounds)
    _moveall(document, "experiment", scalar_section(document, "experiment"))
    objective = scalar_section(document, "objective")
    reference = objective.pop("reference", None)
    if reference is not None:
        move_absent(document, "objective", "objective_reference", reference)
    _moveall(document, "objective", objective)


def _lowerperturb(document: dict) -> None:
    perturbation = scalar_section(document, "perturbation")
    distribution = perturbation.pop("distribution", None)
    if distribution is not None:
        move_absent(document, "perturbation", "perturbation", distribution)
    selected = single_subsection(perturbation, "perturbation")
    if selected is not None:
        name, fields = selected
        if fields:
            raise ValueError(f"perturbation.{name} does not accept fields")
        move_absent(document, "perturbation", "perturbation", name)
    _moveall(document, "perturbation", perturbation)
    proposal = scalar_section(document, "proposal")
    prop_dist = proposal.pop("distribution", None)
    if prop_dist is not None:
        move_absent(document, "proposal", "perturbation", prop_dist)
    _moveall(document, "proposal", proposal)


def _loweracq(document: dict) -> None:
    acquisition = scalar_section(document, "acquisition")
    method = acquisition.pop("method", None)
    if method is not None:
        move_absent(document, "acquisition", "acquisition", method)
    selected = single_subsection(acquisition, "acquisition")
    if selected is not None:
        name, fields = selected
        move_absent(document, "acquisition", "acquisition", name)
        _moveall(document, "acquisition", fields)
    kind = acquisition.pop("kind", None)
    if kind is not None:
        move_absent(document, "acquisition", "acquisition", kind)
    _moveall(document, "acquisition", acquisition)


def _movesurrmap(document: dict, fields: dict) -> None:
    mapping = {
        "neighbors": "k_neighbors",
        "candidates": "num_candidates",
        "samples": "num_samples",
    }
    for key, target in mapping.items():
        if key in fields:
            move_absent(document, "surrogate", target, fields.pop(key))
    _moveall(document, "surrogate", fields)


def _lowersurrsub(document: dict, selected: tuple[str, dict]) -> None:
    name, fields = selected
    if normalize_primitive(name) not in {"enn", "resident_enn"}:
        raise ValueError("surrogate must select method = 'enn'")
    fit = scalar_section(fields, "fit")
    for key, value in fit.items():
        target = {"candidates": "num_candidates", "samples": "num_samples"}.get(key)
        if target is None:
            raise ValueError(f"unknown field surrogate.enn.fit.{key}")
        move_absent(document, "surrogate.enn.fit", target, value)
    _movesurrmap(document, fields)


def _lowersurrogate(document: dict) -> None:
    surrogate = scalar_section(document, "surrogate")
    method = surrogate.pop("method", None)
    if method is not None and method not in {"enn", "resident_enn"}:
        raise ValueError("surrogate.method must be 'enn'")
    if "fit_candidates" in surrogate:
        move_absent(
            document, "surrogate", "num_candidates", surrogate.pop("fit_candidates")
        )
    if "fit_samples" in surrogate:
        move_absent(document, "surrogate", "num_samples", surrogate.pop("fit_samples"))
    selected = single_subsection(surrogate, "surrogate")
    if selected is not None:
        _lowersurrsub(document, selected)
    kind = surrogate.pop("kind", None)
    if kind is not None and kind != "resident_enn":
        raise ValueError("surrogate.kind must be 'resident_enn'")
    _movesurrmap(document, surrogate)


def _lowertrust(document: dict) -> None:
    trust_region = section_alias(document, "trust-region", "trust_region")
    reliability = trust_region.pop("reliability", None)
    if reliability is not None:
        if not isinstance(reliability, dict):
            raise ValueError("trust-region.reliability must be a table")
        move_absent(
            document, "trust-region.reliability", "reliability_controller", reliability
        )
    method = trust_region.pop("method", None)
    if method is not None:
        move_absent(document, "trust_region", "trust_region_kind", method)
    selected = single_subsection(trust_region, "trust_region")
    if selected is not None:
        name, fields = selected
        move_absent(
            document, "trust_region", "trust_region_kind", normalize_primitive(name)
        )
        sub_shape = fields.pop("shape", None)
        if sub_shape is not None:
            move_absent(document, "trust_region", "trust_region_shape", sub_shape)
        _moveall(document, "trust_region", fields)
    for key, target in (
        ("kind", "trust_region_kind"),
        ("shape", "trust_region_shape"),
    ):
        val = trust_region.pop(key, None)
        if val is not None:
            move_absent(document, "trust_region", target, val)
    if trust_region.pop("tensor_family", None) is not None:
        raise ValueError(
            "trust_region.tensor_family is reserved for learned/static shape values but is not configurable yet"
        )
    _moveall(document, "trust_region", trust_region)


def lower_study(raw: dict) -> dict:
    document = dict(raw)
    _lowertop(document)
    _lowerperturb(document)
    _loweracq(document)
    _lowersurrogate(document)
    _lowertrust(document)
    return document


def source_identity(root: Path) -> str:
    result = subprocess.run(
        ["jj", "log", "-r", "@", "--no-graph", "-T", "commit_id"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise ValueError("jj could not determine the working-copy source identity")
    return result.stdout.strip()


def toml_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) or (isinstance(value, float) and math.isfinite(value)):
        return str(value) if isinstance(value, int) else repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ", ".join(toml_value(item) for item in value) + "]"
    if isinstance(value, dict):
        fields = ", ".join(f"{key} = {toml_value(value[key])}" for key in sorted(value))
        return f"{{ {fields} }}"
    raise ValueError(f"unsupported pretrain configuration value: {value!r}")


def resolved_toml(document: dict, plan: dict) -> str:
    resolved = dict(document)
    if "episodes" in plan["recipe"]:
        resolved["generation"] = {
            **resolved["generation"],
            "episode_dataset": str((plan["corpus"] / "episodes.json").resolve()),
        }
    resolved["dataset"] = str((plan["corpus"] / "train.bauxptn").resolve())
    resolved["output"] = str(plan["output"].resolve())
    keys = ["version"] + sorted(key for key in resolved if key != "version")
    return "".join(f"{key} = {toml_value(resolved[key])}\n" for key in keys)


def _expectedfiles(
    corpus: Path, manifest: dict, recipe: dict, tokenizer: dict, splits: dict
) -> dict:
    expected = {
        "tokenizer.json": tokenizer.get("sha256"),
        **{
            f"{split}.bauxptn": splits.get(split, {}).get("sha256")
            for split in recipe["sequences"]
        },
    }
    episodes = recipe.get("episodes")
    if episodes is not None:
        record = manifest.get("episodes")
        if not isinstance(record, dict) or any(
            record.get(key) != value for key, value in episodes.items()
        ):
            raise ValueError(
                f"cached generation episodes do not match recipe: {corpus}"
            )
        expected["episodes.json"] = record.get("sha256")
    elif "episodes" in manifest:
        raise ValueError(f"cached corpus has unexpected generation episodes: {corpus}")
    return expected


def validate_cache(corpus: Path, manifest: dict, recipe: dict) -> None:
    if manifest.get("format") != "baux.pretraining.v1":
        raise ValueError(f"cached corpus has the wrong format: {corpus}")
    tokenizer = manifest.get("tokenizer")
    splits = manifest.get("splits")
    if not isinstance(tokenizer, dict) or not isinstance(splits, dict):
        raise ValueError(f"cached corpus manifest is incomplete: {corpus}")
    if (
        manifest.get("source") != recipe["source"]
        or manifest.get("split_policy") != recipe["split_policy"]
        or set(splits) != set(recipe["sequences"])
        or any(
            splits[split].get("sequences") != count
            for split, count in recipe["sequences"].items()
        )
    ):
        raise ValueError(f"cached corpus provenance or split sizes changed: {corpus}")
    expected = _expectedfiles(corpus, manifest, recipe, tokenizer, splits)
    for name, digest in expected.items():
        path = corpus / name
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or not path.is_file()
            or file_digest(path) != digest
        ):
            raise ValueError(f"cached corpus file is missing or corrupt: {path}")
