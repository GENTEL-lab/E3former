import contextlib
import hashlib
import importlib.util
import io
import json
import tarfile
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

SPEC = importlib.util.spec_from_file_location(
    "release_download", Path(__file__).resolve().parents[1] / "scripts/download.py"
)
download = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(download)


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.destination = self.root / "destination"

    def archive(self, names):
        path = self.root / "data.tar.gz"
        with tarfile.open(path, "w:gz") as archive:
            for name, content in names:
                info = tarfile.TarInfo(name)
                if content is None:
                    info.type = tarfile.SYMTYPE
                    info.linkname = "../../outside"
                    archive.addfile(info)
                else:
                    info.size = len(content)
                    archive.addfile(info, io.BytesIO(content))
        return path, {
            "bytes": path.stat().st_size,
            "sha256": download.checksum(path),
            "archive_root": "cath",
            "members": len(names),
        }

    def test_verified_extraction_and_repeated_install(self):
        path, asset = self.archive([("cath/processed/a.pt", b"protein")])
        for _ in range(2):
            self.assertEqual(download.install_asset(path, asset, self.destination), 1)
        self.assertEqual((self.destination / "data/cath/processed/a.pt").read_bytes(), b"protein")

    def test_corrupt_archive_is_rejected_before_install(self):
        path, asset = self.archive([("cath/processed/a.pt", b"protein")])
        asset["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            download.install_asset(path, asset, self.destination)
        self.assertFalse(self.destination.exists())

    def test_unsafe_members_are_rejected(self):
        for name, content in [
            ("../escape", b"x"),
            ("/absolute", b"x"),
            ("ppbs/processed/a.pt", b"x"),
            ("cath/link", None),
        ]:
            with self.subTest(name=name):
                path, asset = self.archive([(name, content)])
                with self.assertRaises(ValueError):
                    download.install_asset(path, asset, self.destination)
                self.assertFalse((self.destination / "data").exists())

    def test_duplicate_members_are_rejected(self):
        path, asset = self.archive([("cath/a", b"one"), ("cath/a", b"two")])
        with self.assertRaises(ValueError):
            download.install_asset(path, asset, self.destination)
        self.assertFalse((self.destination / "data").exists())

    def test_existing_files_are_not_overwritten(self):
        target = self.destination / "data/cath/z.pt"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"existing")
        path, asset = self.archive([("cath/a.pt", b"new"), ("cath/z.pt", b"different")])
        with self.assertRaises(FileExistsError):
            download.install_asset(path, asset, self.destination)
        self.assertEqual(target.read_bytes(), b"existing")
        self.assertFalse((target.parent / "a.pt").exists())

    def test_symlink_destinations_are_rejected(self):
        outside = self.root / "outside"
        outside.mkdir()
        self.destination.mkdir()
        (self.destination / "data").symlink_to(outside, target_is_directory=True)
        path, asset = self.archive([("cath/a.pt", b"new")])
        with self.assertRaises(ValueError):
            download.install_asset(path, asset, self.destination)
        self.assertEqual(list(outside.iterdir()), [])

    def test_checkpoint_install(self):
        path = self.root / "model.ckpt"
        path.write_bytes(b"weights")
        asset = {
            "bytes": 7,
            "sha256": hashlib.sha256(b"weights").hexdigest(),
            "target": "checkpoints/model.ckpt",
        }
        download.install_asset(path, asset, self.destination)
        self.assertEqual((self.destination / asset["target"]).read_bytes(), b"weights")

    def test_release_uses_one_repository(self):
        manifest = json.loads((download.ROOT / "resources.json").read_text())
        self.assertEqual(
            manifest["repositories"],
            {"e3former": {"name": "E3former_data", "type": "dataset", "revision": "v1.0"}},
        )
        self.assertEqual({a["repository"] for a in manifest["assets"]}, {"e3former"})
        self.assertEqual(
            {a["group"] for a in manifest["assets"]},
            {"cath", "cath-af", "ppbs", "ppbs-af"},
        )
        self.assertEqual(len(manifest["assets"]), 4)
        self.assertTrue(all("archive_root" in asset for asset in manifest["assets"]))

    def test_remote_assets_share_repository_and_revision(self):
        manifest = json.loads((download.ROOT / "resources.json").read_text())
        module = types.ModuleType("huggingface_hub")
        module.hf_hub_download = Mock(return_value=str(self.root / "cached"))
        with (
            patch("sys.argv", ["download.py", "--namespace", "owner"]),
            patch.dict("sys.modules", {"huggingface_hub": module}),
            patch.object(download, "install_asset", return_value=1) as install,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            download.main()
        expected = manifest["assets"]
        self.assertEqual(install.call_count, len(expected))
        self.assertEqual(module.hf_hub_download.call_count, len(expected))
        for call, asset in zip(module.hf_hub_download.call_args_list, expected):
            self.assertEqual(
                call.kwargs,
                {
                    "repo_id": "owner/E3former_data",
                    "repo_type": "dataset",
                    "revision": "v1.0",
                    "filename": asset["filename"],
                },
            )

    def test_local_selection_uses_nested_dataset_path(self):
        source = self.root / "huggingface"
        filename = "datasets/cath/cath.tar.gz"
        target = source / "e3former" / filename
        target.parent.mkdir(parents=True)
        path, asset = self.archive([("cath/processed/a.pt", b"protein")])
        path.replace(target)
        asset.update(group="cath", repository="e3former", filename=filename)
        manifest = self.root / "resources.json"
        manifest.write_text(
            json.dumps(
                {
                    "format_version": 1,
                    "assets": [
                        {"group": "checkpoints", "filename": "not-selected.ckpt"},
                        asset,
                    ],
                }
            )
        )
        with (
            patch(
                "sys.argv",
                [
                    "download.py",
                    "--local-source",
                    str(source),
                    "--manifest",
                    str(manifest),
                    "--destination",
                    str(self.destination),
                    "--only",
                    "cath",
                ],
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            download.main()
        self.assertEqual((self.destination / "data/cath/processed/a.pt").read_bytes(), b"protein")
        self.assertFalse((self.destination / "checkpoints").exists())


if __name__ == "__main__":
    unittest.main()
