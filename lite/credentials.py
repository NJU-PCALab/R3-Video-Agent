from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Mapping
from urllib.parse import urlsplit


class CredentialSourceError(ValueError):
    """Safe diagnostic messages never include source expressions or values."""


def _endpoint(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CredentialSourceError("A nonempty OpenAI-compatible base URL is required")
    value = value.rstrip("/")
    parsed = urlsplit(value)
    if (parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username
            or parsed.password or parsed.query or parsed.fragment or any(ord(char) < 32 for char in value)):
        raise CredentialSourceError("Provider base URL must not include credentials, queries or control characters")
    return value


def _key(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CredentialSourceError("Configured API credential is unavailable")
    value = value.strip()
    if value.startswith("Bearer "):
        value = value[len("Bearer "):]
    if not value or any(ord(char) < 32 for char in value):
        raise CredentialSourceError("Configured API credential has an invalid header form")
    return value


@dataclass(frozen=True)
class ResolvedCredentials:
    base_url: str
    api_key: str = field(repr=False)
    _provenance: dict = field(repr=False)

    def headers(self) -> dict[str, str]:
        return {"Authorization": "Bearer " + self.api_key, "Content-Type": "application/json"}

    def metadata(self) -> dict:
        return {"base_url": self.base_url, "header_name": "Authorization", "header_form": "Bearer",
                "credential_value_included": False, **self._provenance}


def resolve_openai_credentials(route: Mapping, environ: Mapping[str, str] | None = None) -> ResolvedCredentials:
    """Read a URL and a named environment secret at request time."""
    environment = os.environ if environ is None else environ
    base = route.get("base_url") or environment.get(route.get("base_url_env", ""), "")
    name = route.get("api_key_env", "")
    return ResolvedCredentials(_endpoint(base), _key(environment.get(name, "")),
                               {"credential_source_kind": "environment", "credential_environment": name})
