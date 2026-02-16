"""Tests for ArtifactVerifier."""

from pathlib import Path

from src.artifact_verifier import ArtifactVerificationResult, ArtifactVerifier


class TestArtifactVerificationResult:
    def test_all_exist_when_no_missing(self):
        r = ArtifactVerificationResult(verified=["/a.py"], missing=[])
        assert r.all_exist

    def test_not_all_exist_when_missing(self):
        r = ArtifactVerificationResult(verified=[], missing=["/b.py"])
        assert not r.all_exist

    def test_empty_result_all_exist(self):
        r = ArtifactVerificationResult()
        assert r.all_exist


class TestExtractFilePaths:
    def setup_method(self):
        self.verifier = ArtifactVerifier(Path("/tmp"))

    def test_absolute_path(self):
        paths = self.verifier.extract_file_paths("Created /home/user/app.py")
        assert "/home/user/app.py" in paths

    def test_dot_slash_path(self):
        paths = self.verifier.extract_file_paths("See ./src/main.py for details")
        assert "./src/main.py" in paths

    def test_relative_path_with_extension(self):
        paths = self.verifier.extract_file_paths("Updated config.yaml file")
        assert "config.yaml" in paths

    def test_nested_relative_path(self):
        paths = self.verifier.extract_file_paths("Created src/utils/helper.py")
        assert "src/utils/helper.py" in paths

    def test_excludes_urls(self):
        paths = self.verifier.extract_file_paths(
            "See https://example.com/path/to/file.py for docs"
        )
        assert not any("example.com" in p for p in paths)

    def test_excludes_version_numbers(self):
        paths = self.verifier.extract_file_paths("Using Python 3.12")
        assert not paths

    def test_excludes_score_format(self):
        paths = self.verifier.extract_file_paths("Score: 0.5/1.0")
        assert not paths

    def test_multiple_paths(self):
        text = "Created /tmp/a.py and ./src/b.json and config.yaml"
        paths = self.verifier.extract_file_paths(text)
        assert "/tmp/a.py" in paths
        assert "./src/b.json" in paths
        assert "config.yaml" in paths

    def test_no_duplicates(self):
        text = "File app.py is at app.py"
        paths = self.verifier.extract_file_paths(text)
        assert paths.count("app.py") == 1

    def test_empty_text(self):
        paths = self.verifier.extract_file_paths("")
        assert paths == []

    def test_no_paths_in_text(self):
        paths = self.verifier.extract_file_paths("Hello world, no files here")
        assert paths == []

    def test_various_extensions(self):
        for ext in (".py", ".json", ".yaml", ".md", ".sh", ".ts", ".html", ".css"):
            paths = self.verifier.extract_file_paths(f"file{ext}")
            assert f"file{ext}" in paths, f"Failed for extension {ext}"


class TestVerify:
    def test_existing_file(self, tmp_path: Path):
        f = tmp_path / "hello.py"
        f.write_text("print('hi')")
        v = ArtifactVerifier(tmp_path)
        result = v.verify(["hello.py"])
        assert result.verified == ["hello.py"]
        assert result.missing == []
        assert result.all_exist

    def test_missing_file(self, tmp_path: Path):
        v = ArtifactVerifier(tmp_path)
        result = v.verify(["nonexistent.py"])
        assert result.verified == []
        assert result.missing == ["nonexistent.py"]
        assert not result.all_exist

    def test_mixed_existing_and_missing(self, tmp_path: Path):
        (tmp_path / "exists.py").write_text("")
        v = ArtifactVerifier(tmp_path)
        result = v.verify(["exists.py", "missing.py"])
        assert result.verified == ["exists.py"]
        assert result.missing == ["missing.py"]

    def test_absolute_path(self, tmp_path: Path):
        f = tmp_path / "abs.txt"
        f.write_text("data")
        v = ArtifactVerifier(tmp_path)
        result = v.verify([str(f)])
        assert result.verified == [str(f)]
        assert result.missing == []

    def test_directory_exists(self, tmp_path: Path):
        d = tmp_path / "subdir"
        d.mkdir()
        v = ArtifactVerifier(tmp_path)
        result = v.verify(["subdir"])
        assert result.verified == ["subdir"]

    def test_empty_paths(self, tmp_path: Path):
        v = ArtifactVerifier(tmp_path)
        result = v.verify([])
        assert result.all_exist
        assert result.verified == []
        assert result.missing == []

    def test_nested_relative_path(self, tmp_path: Path):
        nested = tmp_path / "src" / "lib"
        nested.mkdir(parents=True)
        (nested / "mod.py").write_text("")
        v = ArtifactVerifier(tmp_path)
        result = v.verify(["src/lib/mod.py"])
        assert result.verified == ["src/lib/mod.py"]

    def test_skip_verification_no_paths(self, tmp_path: Path):
        """Req 3.4: パスなしの場合は検証スキップ（空リストで呼ばれる）."""
        v = ArtifactVerifier(tmp_path)
        result = v.verify([])
        assert result.all_exist
