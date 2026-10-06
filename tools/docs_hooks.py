"""Convert repository-relative source links for the published documentation."""

from pathlib import Path
import re
from urllib.parse import quote, unquote, urlsplit

from mkdocs.exceptions import PluginError


_SOURCE_LINK = re.compile(r'\]\((?P<target>\.\./[^\s)]+)(?:\s+"(?P<title>[^"]*)")?\)')
_FENCE = re.compile(r"^\s*(`{3,}|~{3,})(.*)$")


def on_page_markdown(markdown, *, page, config, **kwargs):
    root = Path(config.config_file_path).resolve().parent
    docs = Path(config.docs_dir).resolve()
    repository = config.repo_url.rstrip("/")
    branch = quote(config.extra["source_branch"], safe="")

    def source_link(match):
        target = urlsplit(match["target"])
        source = (Path(page.file.abs_src_path).parent / unquote(target.path)).resolve()
        if source.is_relative_to(docs):
            return match[0]
        if not source.is_relative_to(root) or not source.exists():
            raise PluginError(
                f"Missing repository source link in {page.file.src_uri}: {match['target']}"
            )
        mode = "tree" if source.is_dir() else "blob"
        if match["title"] == "Download source":
            mode = "raw"
        path = quote(source.relative_to(root).as_posix(), safe="/")
        url = f"{repository}/{mode}/{branch}/{path}"
        if target.query:
            url += f"?{target.query}"
        if target.fragment:
            url += f"#{target.fragment}"
        title = f' "{match["title"]}"' if match["title"] else ""
        return f"]({url}{title})"

    # Leave examples inside fenced code blocks unchanged.
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
