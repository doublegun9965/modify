"""Check dependencies owned by this venv, ignoring unrelated system packages."""
import importlib.metadata as metadata
import sys
import sysconfig
from pathlib import Path

from packaging.requirements import Requirement


def check():
    paths = {str(Path(sysconfig.get_path(key)).resolve())
             for key in ("purelib", "platlib")}
    failures = []
    checked = 0
    for dist in metadata.distributions(path=sorted(paths)):
        name = dist.metadata["Name"]
        if not name:
            continue
        checked += 1
        for spec in dist.requires or []:
            requirement = Requirement(spec)
            if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
                continue
            try:
                installed = metadata.version(requirement.name)
            except metadata.PackageNotFoundError:
                failures.append(f"{name} {dist.version} requires {requirement}, which is missing")
                continue
            if requirement.specifier and installed not in requirement.specifier:
                failures.append(f"{name} {dist.version} requires {requirement}, found {installed}")
    if failures:
        raise RuntimeError("Project venv dependency conflicts:\n" + "\n".join(failures))
    print(f"Checked dependencies of {checked} project-venv packages; system-only packages excluded.")


if __name__ == "__main__":
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("Run with the project venv Python")
    check()
