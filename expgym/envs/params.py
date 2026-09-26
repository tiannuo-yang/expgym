"""Flat hyperparameter search spaces: validation, prompts, schemas and sampling."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from expgym.envs.base import ToolInputError


@dataclass(frozen=True)
class Param:
    name: str
    kind: str  # "categorical", "integer" or "real"
    choices: Tuple[Any, ...] = ()
    lower: float = 0.0
    upper: float = 0.0
    log: bool = False
    default: Any = None

    @classmethod
    def categorical(cls, name: str, choices: Sequence[Any], default: Any = None) -> "Param":
        return cls(name, "categorical", tuple(choices), default=choices[0] if default is None else default)

    @classmethod
    def integer(cls, name: str, lower: int, upper: int, log: bool = False, default: Optional[int] = None) -> "Param":
        if default is None:
            default = int(round(math.exp((math.log(lower) + math.log(upper)) / 2))) if log \
                else int(round((lower + upper) / 2))
        return cls(name, "integer", lower=lower, upper=upper, log=log, default=default)

    @classmethod
    def real(cls, name: str, lower: float, upper: float, log: bool = False,
             default: Optional[float] = None) -> "Param":
        if default is None:
            default = math.exp((math.log(lower) + math.log(upper)) / 2) if log else (lower + upper) / 2
        return cls(name, "real", lower=float(lower), upper=float(upper), log=log, default=default)

    def describe(self) -> str:
        if self.kind == "categorical":
            return "- %s: choices=%r (default=%s)" % (self.name, list(self.choices), self.default)
        return "- %s: %s to %s%s (default=%s)" % (self.name, self.lower, self.upper,
                                                  " (log)" if self.log else "", self.default)

    def schema(self) -> Dict[str, Any]:
        if self.kind == "categorical":
            types = {"string" if isinstance(c, str) else "number" if isinstance(c, float) else "integer"
                     for c in self.choices}
            if "number" in types:
                types.discard("integer")
            return {"type": types.pop() if len(types) == 1 else sorted(types), "enum": list(self.choices)}
        schema = {"type": "integer" if self.kind == "integer" else "number",
                  "minimum": self.lower, "maximum": self.upper}
        if self.log:
            schema["description"] = "Log-scaled hyperparameter; provide its value within the stated bounds."
        return schema

    def check(self, value: Any) -> Tuple[Any, Optional[str]]:
        """Return ``(normalized value, error)``."""
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            return value, "%s invalid type" % self.name
        if isinstance(value, float) and not math.isfinite(value):
            return value, "%s must be finite" % self.name
        if self.kind == "categorical":
            for choice in self.choices:
                if isinstance(choice, str) == isinstance(value, str) and choice == value:
                    return choice, None
            return value, "%s out of range" % self.name
        if isinstance(value, str):
            return value, "%s invalid type" % self.name
        if self.kind == "integer":
            if value != int(value):
                return value, "%s must be integer" % self.name
            value = int(value)
        if not self.lower <= value <= self.upper:
            return value, "%s out of range" % self.name
        return value, None

    def sample(self, rng: random.Random) -> Any:
        if self.kind == "categorical":
            return rng.choice(self.choices)
        if self.log:
            value = math.exp(rng.uniform(math.log(self.lower), math.log(self.upper)))
        else:
            value = rng.uniform(self.lower, self.upper)
        return int(round(value)) if self.kind == "integer" else value


def validate_config(params: Sequence[Param], payload: Dict[str, Any],
                    ignored: Sequence[str] = ()) -> Dict[str, Any]:
    """Validate a complete flat configuration or raise :class:`ToolInputError`."""
    names = {p.name for p in params}
    unknown = sorted(k for k in payload if k not in names and k not in ignored)
    if unknown:
        raise ToolInputError("Unknown hyperparameter(s) %s" % unknown)
    missing = [p.name for p in params if p.name not in payload]
    if missing:
        raise ToolInputError("Invalid config: missing %s. You must specify ALL parameters." % ", ".join(missing))
    config, errors = {}, []
    for param in params:
        value, error = param.check(payload[param.name])
        config[param.name] = value
        if error:
            errors.append(error)
    if errors:
        raise ToolInputError("Invalid config: " + "; ".join(errors))
    return config


def params_from_configspace(space: Any) -> List[Param]:
    """Convert a flat ConfigSpace space (used by HPOBench) into :class:`Param` objects."""
    params = []
    for hp in space.get_hyperparameters():
        default = hp.default_value.item() if hasattr(hp.default_value, "item") else hp.default_value
        if hasattr(hp, "choices"):
            params.append(Param.categorical(hp.name, [c.item() if hasattr(c, "item") else c for c in hp.choices],
                                            default))
        elif hasattr(hp, "sequence"):
            params.append(Param.categorical(hp.name, list(hp.sequence), default))
        elif "Integer" in type(hp).__name__:
            params.append(Param.integer(hp.name, int(hp.lower), int(hp.upper), bool(hp.log), int(default)))
        else:
            params.append(Param.real(hp.name, float(hp.lower), float(hp.upper), bool(hp.log), float(default)))
    return params
