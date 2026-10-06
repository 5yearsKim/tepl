"""Resolve repository metadata and source links for portable documentation builds."""

import os
from pathlib import Path
import re
import subprocess
from urllib.parse import quote, urlsplit

from mkdocs.exceptions import PluginError


def _git(root, *args):
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, check=False
        )
    except FileNotFoundError:
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _origin_url(root):
    remote = _git(root, "remote", "get-url", "origin").rstrip("/").removesuffix(".git")
    ssh = re.fullmatch(r"git@([^:]+):(.+)", remote)
    if ssh:
        return f"https://{ssh[1]}/{ssh[2]}"
    parsed = urlsplit(remote)
    if parsed.scheme == "ssh" and parsed.hostname:
        return f"https://{parsed.hostname}{parsed.path}"
    return remote if parsed.scheme in ("https", "http") else ""


def on_config(config):
    root = Path(config.config_file_path).resolve().parent
    repository = os.environ.get("GITHUB_REPOSITORY", "")
    server = os.environ.get("GITHUB_SERVER_URL", "https://github.com").rstrip("/")
    config.repo_url = (
        config.repo_url or (f"{server}/{repository}" if repository else _origin_url(root))
    ).rstrip("/")
    default_branch = _git(root, "symbolic-ref", "--short", "refs/remotes/origin/HEAD")
    source_ref = (
        os.environ.get("DOCS_SOURCE_REF")
        or default_branch.removeprefix("origin/")
        or _git(root, "branch", "--show-current")
        or "main"
    )
    config.extra["source_ref"] = quote(source_ref, safe="")
    if config.repo_url:
        config.repo_name = urlsplit(config.repo_url).path.strip("/")
        config.edit_uri = f"edit/{config.extra['source_ref']}/docs/"
    return config


_SOURCE_LINK = re.compile(r'\]\((?P<target>\.\./[^\s)]+)(?:\s+"(?P<title>[^"]*)")?\)')
_FENCE = re.compile(r"^\s*(`{3,}|~{3,})(.*)$")


def on_page_markdown(markdown, *, page, config, **kwargs):
    root = Path(config.config_file_path).resolve().parent
    docs = Path(config.docs_dir).resolve()

    def source_link(match):
        target = urlsplit(match["target"])
        source = (Path(page.file.abs_src_path).parent / target.path).resolve()
        if source.is_relative_to(docs):
            return match[0]
        if not source.is_relative_to(root) or not source.exists():
            raise PluginError(
                f"Missing repository source link in {page.file.src_uri}: {match['target']}"
            )
        if not config.repo_url:
            raise PluginError("Set DOCS_REPO_URL or configure a Git origin to resolve source links.")
        mode = "tree" if source.is_dir() else "blob"
        if match["title"] == "Download source":
            mode = "raw"
        path = quote(source.relative_to(root).as_posix(), safe="/")
        url = f"{config.repo_url}/{mode}/{config.extra['source_ref']}/{path}"
        if target.fragment:
            url += f"#{target.fragment}"
        title = f' "{match["title"]}"' if match["title"] else ""
        return f"]({url}{title})"

    # Preserve imports and illustrative links inside fenced code blocks.
    result, fence = [], None
    for line in markdown.splitlines(keepends=True):
        marker = _FENCE.match(line)
        if marker:
            if fence is None:
                fence = marker[1]
            elif (
                marker[1][0] == fence[0]
                and len(marker[1]) >= len(fence)
                and not marker[2].strip()
            ):
                fence = None
            result.append(line)
        else:
            result.append(line if fence else _SOURCE_LINK.sub(source_link, line))
    return "".join(result)
