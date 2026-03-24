import ast
import difflib
import json
import shlex
import subprocess
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..exceptions import ReviewPromotionError
from ..provider import extract_python_source
from .constants import IMPLEMENT_DECORATOR_NAMES


class RuntimeReviewMixin:
    def approve_cache_entry(self, cache_key: str, *, reviewer: str = "manual") -> Dict[str, Any]:
        payload = self.cache.load(cache_key)
        if not payload:
            raise ReviewPromotionError("cache entry not found: {0}".format(cache_key))
        payload = self._normalize_cache_payload(payload, cache_key=cache_key)
        payload["approval_status"] = "approved"
        payload["approved_at"] = self._utcnow()
        payload["approved_by"] = reviewer
        self.cache.save(cache_key, payload)
        self._audit(
            "cache_approved",
            cache_key=cache_key,
            qualname=payload.get("qualname"),
            reviewer=reviewer,
            approval_status="approved",
        )
        return payload

    def export_review_artifacts(self, destination) -> Path:
        destination_path = Path(destination)
        destination_path.mkdir(parents=True, exist_ok=True)
        manifest = []
        for cache_file in sorted(self.config.cache_dir.glob("*.json")):
            payload = self.cache.load(cache_file.stem)
            if not payload:
                continue
            payload = self._normalize_cache_payload(payload, cache_key=cache_file.stem)
            slug = self._artifact_slug(payload)
            source_path = destination_path / "{0}.py".format(slug)
            patch_path = destination_path / "{0}.patch".format(slug)
            with source_path.open("w", encoding="utf-8") as handle:
                handle.write(payload["source"])
            with patch_path.open("w", encoding="utf-8") as handle:
                handle.write(self._build_review_patch(payload))
            manifest.append(
                {
                    "cache_key": cache_file.stem,
                    "module": payload.get("module"),
                    "qualname": payload.get("qualname"),
                    "mode": payload.get("mode"),
                    "source_file": str(source_path),
                    "patch_file": str(patch_path),
                    "target_source_path": payload.get("source_path"),
                    "source_lineno": payload.get("source_lineno"),
                    "approval_status": payload.get("approval_status"),
                    "context": payload.get("context", {}),
                }
            )
        manifest_path = destination_path / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
        return manifest_path

    def promote_cache_entry(self, cache_key: str, *, source_text: Optional[str] = None) -> Path:
        payload = self.cache.load(cache_key)
        if not payload:
            raise ReviewPromotionError("cache entry not found: {0}".format(cache_key))
        payload = self._normalize_cache_payload(payload, cache_key=cache_key)
        target_source_path = payload.get("source_path")
        if not target_source_path:
            raise ReviewPromotionError("cache entry {0} has no source path".format(cache_key))
        updated_source = source_text if source_text is not None else payload.get("source")
        if not updated_source:
            raise ReviewPromotionError("cache entry {0} has no generated source".format(cache_key))
        promoted_path = self._promote_source_into_file(
            Path(target_source_path),
            payload.get("qualname", ""),
            updated_source,
            payload.get("mode"),
        )
        payload["approval_status"] = "promoted"
        payload["promoted_at"] = self._utcnow()
        payload["promoted_path"] = str(promoted_path)
        self.cache.save(cache_key, payload)
        self._audit(
            "cache_promoted",
            cache_key=cache_key,
            qualname=payload.get("qualname"),
            promoted_path=str(promoted_path),
            approval_status="promoted",
        )
        return promoted_path

    def promote_review_artifacts(self, manifest_or_directory) -> Dict[str, str]:
        entries, _ = self._load_review_manifest_entries(manifest_or_directory)
        results = {}
        for entry in entries:
            cache_key = entry.get("cache_key")
            reviewed_source = self._reviewed_source_for_entry(entry)
            if cache_key is None or reviewed_source is None:
                continue
            promoted_path = self.promote_cache_entry(cache_key, source_text=reviewed_source)
            results[cache_key] = str(promoted_path)
        return results

    def interactive_review(
        self,
        manifest_or_directory,
        *,
        reviewer: str = "manual",
        input_func=input,
        output_func=print,
        default_action: str = "skip",
    ) -> Dict[str, str]:
        entries, _ = self._load_review_manifest_entries(manifest_or_directory)
        default = self._normalize_review_action(default_action, allow_quit=False)
        results = {}
        for entry in entries:
            cache_key = entry.get("cache_key")
            qualname = entry.get("qualname") or "<unknown>"
            output_func(
                "[{0}] {1}".format(
                    entry.get("approval_status") or "<unknown>",
                    qualname,
                )
            )
            if entry.get("source_file"):
                output_func("source: {0}".format(entry["source_file"]))
            if entry.get("patch_file"):
                output_func("patch: {0}".format(entry["patch_file"]))
            while True:
                response = input_func(
                    "Action [a=approve, p=promote, s=skip, q=quit] (default {0}): ".format(default)
                )
                action = self._normalize_review_action(response or default)
                if action is None:
                    output_func("Unrecognized action. Use approve, promote, skip, or quit.")
                    continue
                break
            if action == "quit":
                break
            if cache_key is None:
                results["<missing-cache-key>"] = action
                continue
            if action == "approve":
                self.approve_cache_entry(cache_key, reviewer=reviewer)
            elif action == "promote":
                reviewed_source = self._reviewed_source_for_entry(entry)
                if reviewed_source is None:
                    raise ReviewPromotionError(
                        "review entry {0} cannot be promoted because no reviewed source file is available".format(
                            qualname
                        )
                    )
                self.promote_cache_entry(cache_key, source_text=reviewed_source)
            results[cache_key] = action
        return results

    def export_git_review_bundle(
        self,
        destination,
        manifest_or_directory=None,
        *,
        base_ref: str = "HEAD",
        branch_name: Optional[str] = None,
    ) -> Path:
        destination_path = Path(destination)
        destination_path.mkdir(parents=True, exist_ok=True)
        if manifest_or_directory is None:
            manifest_path = self.export_review_artifacts(destination_path / "review")
        else:
            _, manifest_path = self._load_review_manifest_entries(manifest_or_directory)
        entries, manifest_path = self._load_review_manifest_entries(manifest_path)
        repo_root, current_branch = self._git_repo_context(self._review_repo_probe_path(entries, destination_path))
        self._verify_git_ref(repo_root, base_ref)
        review_branch = branch_name or self._default_review_branch_name(entries)
        patch_text, included_entries = self._build_git_review_patch(entries, repo_root)
        patch_path = destination_path / "paithon-review.patch"
        patch_path.write_text(patch_text, encoding="utf-8")
        apply_script_path = destination_path / "apply_review.sh"
        apply_script_path.write_text(
            self._build_git_apply_script(
                repo_root=repo_root,
                patch_path=patch_path,
                base_ref=base_ref,
                branch_name=review_branch,
            ),
            encoding="utf-8",
        )
        apply_script_path.chmod(0o755)
        bundle = {
            "repo_root": str(repo_root),
            "current_branch": current_branch,
            "base_ref": base_ref,
            "branch_name": review_branch,
            "patch_file": str(patch_path),
            "apply_script": str(apply_script_path),
            "review_manifest": str(manifest_path),
            "entry_count": len(included_entries),
            "entries": included_entries,
        }
        bundle_path = destination_path / "git_review_bundle.json"
        bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True), encoding="utf-8")
        self._audit(
            "git_review_bundle_exported",
            branch_name=review_branch,
            base_ref=base_ref,
            review_manifest=str(manifest_path),
            patch_file=str(patch_path),
            entry_count=len(included_entries),
        )
        return bundle_path

    @staticmethod
    def _artifact_slug(payload: Dict[str, Any]) -> str:
        parts = [payload.get("module", "unknown"), payload.get("qualname", "function"), payload.get("mode", "mode")]
        raw = "__".join(parts)
        return "".join(character if character.isalnum() or character in "._-" else "_" for character in raw)

    @staticmethod
    def _build_review_patch(payload: Dict[str, Any]) -> str:
        original = payload.get("template_source", "")
        updated = payload.get("source", "")
        from_name = payload.get("source_path") or "{0}.{1}".format(payload.get("module"), payload.get("qualname"))
        diff = difflib.unified_diff(
            original.splitlines(True),
            updated.splitlines(True),
            fromfile=from_name,
            tofile=from_name,
        )
        return "".join(diff) or "# No diff available\n"

    def _promote_source_into_file(self, source_path: Path, qualname: str, updated_source: str, mode: Optional[str]) -> Path:
        if not source_path.exists():
            raise ReviewPromotionError("source file not found: {0}".format(source_path))
        original_text = source_path.read_text(encoding="utf-8")
        promoted_text = self._replace_function_source(original_text, qualname, updated_source, mode=mode)
        source_path.write_text(promoted_text, encoding="utf-8")
        return source_path

    @staticmethod
    def _replace_function_source(file_text: str, qualname: str, updated_source: str, *, mode: Optional[str]) -> str:
        tree = ast.parse(file_text)
        qualname_parts = [part for part in qualname.split(".") if part != "<locals>"]
        target = RuntimeReviewMixin._find_qualname_node(tree, qualname_parts)
        if target is None or getattr(target, "lineno", None) is None or getattr(target, "end_lineno", None) is None:
            raise ReviewPromotionError("could not locate {0} in source file".format(qualname))
        lines = file_text.splitlines(True)
        indent = " " * getattr(target, "col_offset", 0)
        start_line = target.lineno
        prefix = ""
        if getattr(target, "decorator_list", None):
            start_line = min(decorator.lineno for decorator in target.decorator_list)
            kept = RuntimeReviewMixin._kept_decorators(target.decorator_list, mode=mode)
            prefix = "".join(
                "".join(lines[decorator.lineno - 1 : decorator.end_lineno])
                for decorator in kept
            )
        replacement = prefix + RuntimeReviewMixin._indent_function_source(updated_source, indent)
        return "".join(lines[: start_line - 1] + [replacement] + lines[target.end_lineno :])

    @staticmethod
    def _kept_decorators(decorator_list, *, mode: Optional[str]):
        if mode != "implement":
            return list(decorator_list)
        kept = []
        for decorator in decorator_list:
            if RuntimeReviewMixin._decorator_leaf_name(decorator) in IMPLEMENT_DECORATOR_NAMES:
                continue
            kept.append(decorator)
        return kept

    @staticmethod
    def _decorator_leaf_name(node) -> Optional[str]:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        if isinstance(node, ast.Call):
            return RuntimeReviewMixin._decorator_leaf_name(node.func)
        return None

    @staticmethod
    def _find_qualname_node(tree: ast.AST, qualname_parts: Sequence[str]):
        if not qualname_parts:
            return None
        current_nodes = list(getattr(tree, "body", ()))
        for part in qualname_parts:
            target = None
            for node in current_nodes:
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == part:
                    target = node
                    break
            if target is None:
                return None
            current_nodes = list(getattr(target, "body", ()))
        return target

    @staticmethod
    def _indent_function_source(source: str, indent: str) -> str:
        normalized = extract_python_source(source).rstrip("\n")
        if not indent:
            return normalized + "\n"
        return "\n".join((indent + line if line else line) for line in normalized.splitlines()) + "\n"

    @staticmethod
    def _normalize_review_action(action: str, *, allow_quit: bool = True) -> Optional[str]:
        normalized = (action or "").strip().lower()
        aliases = {
            "a": "approve",
            "approve": "approve",
            "p": "promote",
            "promote": "promote",
            "s": "skip",
            "skip": "skip",
        }
        if allow_quit:
            aliases.update({"q": "quit", "quit": "quit"})
        return aliases.get(normalized)

    def _load_review_manifest_entries(self, manifest_or_directory) -> Tuple[List[Dict[str, Any]], Path]:
        manifest_path = Path(manifest_or_directory)
        if manifest_path.is_dir():
            manifest_path = manifest_path / "manifest.json"
        if not manifest_path.exists():
            raise ReviewPromotionError("review manifest not found: {0}".format(manifest_path))
        with manifest_path.open("r", encoding="utf-8") as handle:
            entries = json.load(handle)
        if not isinstance(entries, list):
            raise ReviewPromotionError("review manifest must contain a list of entries: {0}".format(manifest_path))
        return entries, manifest_path

    @staticmethod
    def _reviewed_source_for_entry(entry: Dict[str, Any]) -> Optional[str]:
        reviewed_source_path = entry.get("source_file")
        target_source_path = entry.get("target_source_path")
        if not reviewed_source_path or not target_source_path:
            return None
        return Path(reviewed_source_path).read_text(encoding="utf-8")

    @staticmethod
    def _review_repo_probe_path(entries: Sequence[Dict[str, Any]], destination_path: Path) -> Path:
        for entry in entries:
            target_source_path = entry.get("target_source_path")
            if target_source_path:
                return Path(target_source_path).resolve().parent
        return destination_path.resolve()

    def _git_repo_context(self, start_path: Path) -> Tuple[Path, str]:
        repo_root_text = self._run_git(start_path, "rev-parse", "--show-toplevel")
        current_branch = self._run_git(Path(repo_root_text), "rev-parse", "--abbrev-ref", "HEAD")
        return Path(repo_root_text), current_branch

    def _verify_git_ref(self, repo_root: Path, ref_name: str) -> None:
        self._run_git(repo_root, "rev-parse", "--verify", ref_name)

    def _run_git(self, cwd: Path, *args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            raise ReviewPromotionError(
                "git command failed in {0}: git {1}: {2}".format(
                    cwd,
                    " ".join(args),
                    detail or "unknown git error",
                )
            )
        return completed.stdout.strip()

    def _build_git_review_patch(self, entries: Sequence[Dict[str, Any]], repo_root: Path) -> Tuple[str, List[Dict[str, Any]]]:
        diff_chunks = []
        included_entries = []
        for entry in entries:
            target_source_path = entry.get("target_source_path")
            reviewed_source_path = entry.get("source_file")
            qualname = entry.get("qualname")
            if not target_source_path or not reviewed_source_path or not qualname:
                continue
            target_path = Path(target_source_path)
            if not target_path.exists():
                raise ReviewPromotionError("source file not found for review entry: {0}".format(target_path))
            reviewed_source = Path(reviewed_source_path).read_text(encoding="utf-8")
            original_text = target_path.read_text(encoding="utf-8")
            promoted_text = self._replace_function_source(original_text, qualname, reviewed_source, mode=entry.get("mode"))
            if promoted_text == original_text:
                continue
            relative_path = self._repo_relative_path(target_path, repo_root)
            diff_chunks.append(
                "".join(
                    difflib.unified_diff(
                        original_text.splitlines(True),
                        promoted_text.splitlines(True),
                        fromfile="a/{0}".format(relative_path),
                        tofile="b/{0}".format(relative_path),
                    )
                )
            )
            included_entries.append(
                {
                    "cache_key": entry.get("cache_key"),
                    "qualname": qualname,
                    "target_source_path": str(target_path),
                    "reviewed_source_path": reviewed_source_path,
                }
            )
        if not diff_chunks:
            raise ReviewPromotionError("no promotable review entries were available for git export")
        return "".join(diff_chunks), included_entries

    @staticmethod
    def _repo_relative_path(target_path: Path, repo_root: Path) -> str:
        try:
            relative = target_path.resolve().relative_to(repo_root.resolve())
        except ValueError as exc:
            raise ReviewPromotionError(
                "review target is outside the git repository root: {0}".format(target_path)
            ) from exc
        return str(relative).replace("\\", "/")

    def _default_review_branch_name(self, entries: Sequence[Dict[str, Any]]) -> str:
        if entries:
            qualname = entries[0].get("qualname") or "review"
            slug = "".join(character if character.isalnum() else "-" for character in qualname.lower()).strip("-")
            slug = "-".join(part for part in slug.split("-") if part) or "review"
        else:
            slug = "review"
        timestamp = self._utcnow().replace(":", "").replace("+", "-").replace(".", "-").replace("T", "-")
        return "paithon-review/{0}-{1}".format(slug[:32], timestamp.lower())

    @staticmethod
    def _build_git_apply_script(*, repo_root: Path, patch_path: Path, base_ref: str, branch_name: str) -> str:
        return textwrap.dedent(
            """\
            #!/usr/bin/env bash
            set -eu
            cd {repo_root}
            git switch -c {branch_name} {base_ref}
            git apply {patch_path}
            """.format(
                repo_root=shlex.quote(str(repo_root)),
                branch_name=shlex.quote(branch_name),
                base_ref=shlex.quote(base_ref),
                patch_path=shlex.quote(str(patch_path.resolve())),
            )
        )
