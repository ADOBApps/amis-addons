#!/usr/bin/env python3
"""Validate index.json and, when reachable, the packages it points at.

Runs on every pull request (see .github/workflows/validate.yml).  It needs no
third-party packages: the point is that a contributor can run the same check
locally with a stock Python.

    python .github/validate_catalog.py [index.json]
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.error
import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List

SCHEMA = 1
REQUIRED_ADDON = ("id", "name", "versions")
REQUIRED_VERSION = ("version", "api", "url", "sha256", "size")
HEX = set("0123456789abcdef")
MAX_PACKAGE = 512 * 1024 * 1024


class Problem(Exception):
    pass


def check_version(item: Dict[str, Any], addon_id: str, errors: List[str]) -> None:
    where = f"{addon_id} {item.get('version', '?')}"
    for key in REQUIRED_VERSION:
        if key not in item:
            errors.append(f"{where}: missing '{key}'")
            return
    sha = str(item["sha256"]).lower()
    if len(sha) != 64 or any(c not in HEX for c in sha):
        errors.append(f"{where}: sha256 must be 64 hex characters")
    if not str(item["url"]).startswith("https://"):
        errors.append(f"{where}: url must be https")
    if int(item["size"]) <= 0 or int(item["size"]) > MAX_PACKAGE:
        errors.append(f"{where}: size looks wrong ({item['size']})")
    if int(item["api"]) < 1:
        errors.append(f"{where}: api must be >= 1")
    if item.get("yanked") and not str(item.get("changelog", "")).strip():
        errors.append(f"{where}: a yanked version needs a changelog explaining why")


def check_package(item: Dict[str, Any], addon_id: str, errors: List[str]) -> None:
    """Download the archive and verify digest + manifest.  Network failure is a warning."""
    url = str(item["url"])
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            payload = response.read(MAX_PACKAGE + 1)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        print(f"  ..    cannot download {url}: {exc}")
        return
    if len(payload) > MAX_PACKAGE:
        errors.append(f"{addon_id}: package exceeds the size limit")
        return
    digest = hashlib.sha256(payload).hexdigest()
    if digest != str(item["sha256"]).lower():
        errors.append(f"{addon_id} {item['version']}: sha256 does not match the file")
        return
    if len(payload) != int(item["size"]):
        errors.append(f"{addon_id} {item['version']}: size does not match the file")
        return
    try:
        with zipfile.ZipFile(BytesIO(payload)) as archive:
            names = archive.namelist()
            for name in names:
                if name.startswith("/") or ".." in Path(name).parts:
                    errors.append(f"{addon_id}: unsafe member {name!r}")
                    return
            manifest_name = next((n for n in names if n.endswith("/manifest.json")
                                  and n.count("/") == 1), None)
            if manifest_name is None:
                errors.append(f"{addon_id}: package has no <id>/manifest.json")
                return
            manifest = json.loads(archive.read(manifest_name))
    except (zipfile.BadZipFile, json.JSONDecodeError) as exc:
        errors.append(f"{addon_id}: unreadable package ({exc})")
        return
    if manifest.get("id") != addon_id:
        errors.append(f"{addon_id}: manifest id is {manifest.get('id')!r}")
    if str(manifest.get("version")) != str(item["version"]):
        errors.append(f"{addon_id}: manifest version is {manifest.get('version')!r}")
    if int(manifest.get("api", 0)) != int(item["api"]):
        errors.append(f"{addon_id}: manifest api is {manifest.get('api')!r}")


def main(argv: List[str]) -> int:
    path = Path(argv[1] if len(argv) > 1 else "index.json")
    if not path.is_file():
        print(f"missing {path}")
        return 2
    data = json.loads(path.read_text(encoding="utf-8"))
    errors: List[str] = []

    if int(data.get("schema", 0)) != SCHEMA:
        errors.append(f"schema must be {SCHEMA}")
    addons = data.get("addons")
    if not isinstance(addons, list) or not addons:
        errors.append("'addons' must be a non-empty list")
        addons = []

    seen = set()
    for addon in addons:
        addon_id = str(addon.get("id", ""))
        if not addon_id:
            errors.append("an addon has no id")
            continue
        if addon_id in seen:
            errors.append(f"duplicate id {addon_id}")
        seen.add(addon_id)
        for key in REQUIRED_ADDON:
            if key not in addon:
                errors.append(f"{addon_id}: missing '{key}'")
        if not addon.get("license"):
            errors.append(f"{addon_id}: missing 'license'")
        versions = addon.get("versions") or []
        if not versions:
            errors.append(f"{addon_id}: no versions")
        version_keys = [str(v.get("version")) for v in versions]
        if len(version_keys) != len(set(version_keys)):
            errors.append(f"{addon_id}: duplicate version numbers")
        for item in versions:
            check_version(item, addon_id, errors)
            if verbosity:
                print(f"  ...   checking {addon_id} {item.get('version')}")
            check_package(item, addon_id, errors)

    if errors:
        print("\nproblems:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print(f"catalogue ok: {len(addons)} addons, "
          f"{sum(len(a.get('versions', [])) for a in addons)} versions")
    return 0


verbosity = 0
if __name__ == "__main__":
    sys.exit(main(sys.argv))
