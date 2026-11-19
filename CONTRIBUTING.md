# Contributing an addon

Additions are **pull requests** against this repository. `main` is protected;
the maintainer reviews and merges.

## 1. Build the package

From the AMIS source tree:

```python
from amis.market.installer import package_folder, package_sha256
zip_path = package_folder("path/to/my_addon", "out/my_addon-1.0.0.zip")
print(zip_path.name, zip_path.stat().st_size, package_sha256(zip_path))
```

The archive must contain a single top-level directory named after the addon,
with `manifest.json` at its root. `package_folder()` produces exactly that and
skips `__pycache__`.

## 2. Create a release

Upload the ZIP as a release asset on this repository, e.g. tag `my_addon-v1.0.0`.
The download URL is deterministic:

```
https://github.com/ADOBApps/amis-addons/releases/download/my_addon-v1.0.0/my_addon-1.0.0.zip
```

## 3. Add the catalogue entry

Append the version to `addons[]` in `index.json` (never edit or reuse an
existing version):

```json
{
  "version": "1.0.0",
  "api": 1,
  "min_amis": "0.1.0",
  "url": "https://github.com/.../my_addon-1.0.0.zip",
  "sha256": "…64 hex characters…",
  "size": 12345,
  "changelog": "What changed in this version",
  "published": "2026-11-19T00:00:00Z",
  "yanked": false
}
```

Update the addon's top-level fields too (`name`, `description`, `author`,
`license`, `category`, `tags`, `homepage`).

## 4. Validate and open the PR

```bash
python .github/validate_catalog.py
```

Then open a PR. CI runs the same validator.

## Rules

1. Submit only code you own or may redistribute.
2. Never commit images, patient data, tokens or credentials.
3. State the licence; AMIS itself is GPL-3.0-or-later, and anything distributed
   inside the application bundle must be GPL-3 compatible.
4. A new version needs a new semantic version; versions are immutable.
5. One addon per PR unless you own all of them.
6. Mark a broken release with `"yanked": true` instead of deleting it — existing
   installations keep working, new ones see the notice.
