"""Policy-as-code enforcement (PHASE 5.2).

Built-in rules:
- GDPR: PII detection in metadata fields
- Mandatory DCAT fields
- Sensitive field flagging
- Custom YAML rules
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from dataclasses import field as dc_field
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from metadata_kg.core.metadata_schema import DCAT_MANDATORY_FIELDS


# ---------------------------------------------------------------------------
# PII detectors
# ---------------------------------------------------------------------------
_EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
_PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{2,4}\)?[\s-]?)?\d{3,4}[\s-]?\d{3,4}\b")
_THAI_ID_RE = re.compile(r"\b\d{1}\s?\d{4}\s?\d{5}\s?\d{2}\s?\d{1}\b")  # Thai national ID
_CC_RE = re.compile(r"\b(?:\d{4}[\s-]?){3}\d{4}\b")  # naive credit card
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")        # US SSN

PII_DETECTORS: dict[str, re.Pattern[str]] = {
    "email": _EMAIL_RE,
    "phone": _PHONE_RE,
    "thai_national_id": _THAI_ID_RE,
    "credit_card": _CC_RE,
    "us_ssn": _SSN_RE,
}


def detect_pii(value: str) -> list[str]:
    """Return a list of detected PII categories in a string."""
    if not isinstance(value, str):
        return []
    return [name for name, pat in PII_DETECTORS.items() if pat.search(value)]


# ---------------------------------------------------------------------------
# Rule containers
# ---------------------------------------------------------------------------
@dataclass
class PolicyRule:
    id: str
    description: str
    severity: str = "error"          # error | warning | info
    check: str = "mandatory_field"   # mandatory_field | pii | regex | forbid_value | custom
    field: str | None = None
    pattern: str | None = None
    forbid_values: list[str] = dc_field(default_factory=list)


@dataclass
class PolicyViolation:
    rule_id: str
    severity: str
    field: str | None
    message: str
    value_excerpt: str | None = None


# ---------------------------------------------------------------------------
# PolicyEngine
# ---------------------------------------------------------------------------
class PolicyEngine:
    """Loadable policy engine with built-in rules + YAML overrides."""

    def __init__(self) -> None:
        self.rules: list[PolicyRule] = []
        self._load_builtins()

    def _load_builtins(self) -> None:
        # Mandatory DCAT fields
        for f in DCAT_MANDATORY_FIELDS:
            self.rules.append(PolicyRule(
                id=f"DCAT_MAND_{f.upper()}",
                description=f"DCAT mandatory field '{f}' must be present",
                severity="error",
                check="mandatory_field",
                field=f,
            ))
        # GDPR — disallow PII in public fields
        for f in ("description", "title", "keyword", "publisher", "creator"):
            self.rules.append(PolicyRule(
                id=f"GDPR_PII_{f.upper()}",
                description=f"GDPR: field '{f}' must not contain PII",
                severity="error",
                check="pii",
                field=f,
            ))

    # ------------------------------------------------------------------
    def load_rules(self, rules_file: str | Path) -> None:
        """Load additional rules from YAML."""
        path = Path(rules_file)
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        for item in data.get("rules", []):
            self.rules.append(PolicyRule(**item))
        logger.info(f"Loaded {len(data.get('rules', []))} extra rules from {path}")

    # ------------------------------------------------------------------
    def check_compliance(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Apply all rules to a metadata dict. Returns pass/fail + violations."""
        violations: list[PolicyViolation] = []
        flat = self._flatten(metadata)

        for rule in self.rules:
            if rule.check == "mandatory_field":
                if rule.field and rule.field == "id":
                    continue  # 'id' is structural
                if rule.field and not self._has_field(flat, rule.field):
                    violations.append(PolicyViolation(
                        rule_id=rule.id,
                        severity=rule.severity,
                        field=rule.field,
                        message=f"Mandatory field '{rule.field}' missing",
                    ))

            elif rule.check == "pii":
                if rule.field:
                    val = self._get_field(flat, rule.field)
                    for v in self._as_strings(val):
                        cats = detect_pii(v)
                        if cats:
                            violations.append(PolicyViolation(
                                rule_id=rule.id,
                                severity=rule.severity,
                                field=rule.field,
                                message=f"PII detected ({', '.join(cats)}) in '{rule.field}'",
                                value_excerpt=v[:80],
                            ))

            elif rule.check == "regex" and rule.field and rule.pattern:
                val = self._get_field(flat, rule.field)
                pat = re.compile(rule.pattern)
                for v in self._as_strings(val):
                    if pat.search(v):
                        violations.append(PolicyViolation(
                            rule_id=rule.id,
                            severity=rule.severity,
                            field=rule.field,
                            message=f"Regex '{rule.pattern}' matched in '{rule.field}'",
                            value_excerpt=v[:80],
                        ))

            elif rule.check == "forbid_value" and rule.field:
                val = self._get_field(flat, rule.field)
                for v in self._as_strings(val):
                    if v in rule.forbid_values:
                        violations.append(PolicyViolation(
                            rule_id=rule.id,
                            severity=rule.severity,
                            field=rule.field,
                            message=f"Forbidden value '{v}' in '{rule.field}'",
                            value_excerpt=v,
                        ))

        errors = [v for v in violations if v.severity == "error"]
        return {
            "pass": len(errors) == 0,
            "violated_rules": [v.__dict__ for v in violations],
            "summary": {
                "error": len(errors),
                "warning": len([v for v in violations if v.severity == "warning"]),
                "info": len([v for v in violations if v.severity == "info"]),
            },
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _flatten(d: dict[str, Any]) -> dict[str, Any]:
        """Strip CURIE prefixes to bare field names for matching."""
        flat: dict[str, Any] = {}
        for k, v in d.items():
            key = k.split(":", 1)[-1] if isinstance(k, str) else str(k)
            key = key.split("#")[-1].split("/")[-1].lower()
            flat[key] = v
        return flat

    @staticmethod
    def _has_field(flat: dict[str, Any], field: str) -> bool:
        return field.lower() in flat and flat[field.lower()] not in (None, "", [])

    @staticmethod
    def _get_field(flat: dict[str, Any], field: str) -> Any:
        return flat.get(field.lower())

    @staticmethod
    def _as_strings(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple, set)):
            return [str(x) for x in value]
        if isinstance(value, dict):
            return [str(v) for v in value.values() if isinstance(v, str)]
        return [str(value)]


__all__ = ["PolicyEngine", "PolicyRule", "PolicyViolation", "detect_pii", "PII_DETECTORS"]
