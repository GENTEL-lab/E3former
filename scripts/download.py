"""Download and verify the released datasets."""

import argparse
import hashlib
import json
import re
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_path(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"Unsafe relative path: {name!r}")
    if not path.parts:
        raise ValueError("Empty relative path")
    return path


def destination_path(root, name):
    path = root.joinpath(*relative_path(name).parts)
    current = path
    while current != root:
        if current.is_symlink():
            raise ValueError(f"Refusing a symlink destination: {current}")
        current = current.parent
    return path


def verify(path, asset):
    if path.stat().st_size != asset["bytes"] or checksum(path) != asset["sha256"]:
        raise ValueError(f"Checksum or size mismatch: {path}")


def install_files(source, destination):
    paths = sorted(p for p in source.rglob("*") if p.is_file())
    pending = []
    for path in paths:
        target = destination_path(destination, path.relative_to(source).as_posix())
        if target.exists():
            if not target.is_file() or checksum(target) != checksum(path):
                raise FileExistsError(f"Existing file differs; nothing overwritten: {target}")
        else:
            pending.append((path, target))
    for path, target in pending:
        target.parent.mkdir(parents=True, exist_ok=True)
        path.replace(target)
    return len(paths)


def install_asset(path, asset, destination):
    verify(path, asset)
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".e3former-install-", dir=destination) as temp:
        staged = Path(temp)
        if asset.get("archive_root"):
            prefix = relative_path(asset["archive_root"])
            seen = set()
            with tarfile.open(path, "r:gz") as archive:
                for member in archive:
                    name = relative_path(member.name)
                    if not member.isfile() or name.parts[: len(prefix.parts)] != prefix.parts:
                        raise ValueError(f"Unexpected archive member: {member.name}")
                    if name in seen:
                        raise ValueError(f"Duplicate archive member: {member.name}")
                    seen.add(name)
                    target = staged / "data" / Path(*name.parts)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.extractfile(member) as src, target.open("wb") as dst:
                        shutil.copyfileobj(src, dst, length=1024 * 1024)
            if len(seen) != asset["members"]:
                raise ValueError(f"Archive member count mismatch: {path}")
        else:
            target = staged / Path(*relative_path(asset["target"]).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        return install_files(staged, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--namespace", help="Hugging Face username or organization")
    source.add_argument("--local-source", type=Path, help="Local Hugging Face staging directory")
    parser.add_argument("--destination", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path, default=ROOT / "resources.json")
    parser.add_argument(
        "--only",
        nargs="+",
        default=["all"],
        choices=("all", "cath", "cath-af", "ppbs", "ppbs-af"),
    )
    args = parser.parse_args()
    if args.namespace and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.namespace):
        parser.error("Use a username or organization name, not a URL or repository path.")
    manifest = json.loads(args.manifest.read_text())
    if manifest["format_version"] != 1:
        raise ValueError("Unsupported resource manifest version")
    assets = [a for a in manifest["assets"] if "all" in args.only or a["group"] in args.only]
    if not assets:
        raise ValueError("No matching resources")
    for asset in assets:
        relative_path(asset["filename"])
        if args.local_source:
            path = args.local_source / asset["repository"] / asset["filename"]
        else:
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as exc:
                raise SystemExit("Install download support: pip install -e '.[hub]'") from exc
            repository = manifest["repositories"][asset["repository"]]
            path = Path(
                hf_hub_download(
                    repo_id=f"{args.namespace}/{repository['name']}",
                    repo_type=repository["type"],
                    revision=repository["revision"],
                    filename=asset["filename"],
                )
            )
        count = install_asset(path, asset, args.destination)
        print(f"Verified {asset['filename']}; installed or matched {count} files.", flush=True)


if __name__ == "__main__":
    main()
