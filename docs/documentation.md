# Working on the documentation

The documentation site is built with [MkDocs](https://www.mkdocs.org/) and the
[Material theme](https://squidfunk.github.io/mkdocs-material/). Its Markdown
sources live in `docs/`; `mkdocs.yml` defines the navigation and site settings.

Run these commands from the repository root. Building the documentation only
needs Python 3.10 or newer; it does not require Bazel, Rust, or a C++ compiler.

## Preview locally

Create an environment and install the pinned documentation dependencies:

```sh
python3 -m venv .venv-docs
.venv-docs/bin/python -m pip install -r requirements-docs.txt
.venv-docs/bin/python -m mkdocs serve
```

Open the local URL printed by MkDocs. The server reloads when you edit Markdown,
assets, or the configuration. Stop it with `Ctrl+C`.
Restart the server after editing `tools/docs_hooks.py`.

On Windows, use `.venv-docs\Scripts\python.exe` in place of
`.venv-docs/bin/python`.

## Build and check

```sh
.venv-docs/bin/python -m mkdocs build --strict
```

The generated HTML, styles, scripts, and search index are written to `site/`.
The strict build rejects missing pages, broken internal links, invalid anchors,
and pages omitted from the navigation. It does not check external websites or
links inside raw HTML; check those when editing them.

The **Documentation** GitHub Actions workflow runs this build for documentation
changes on branches and in pull requests.

## Add or edit a page

1. Edit a Markdown file in `docs/`, or add a new one with a descriptive heading.
2. Add new pages to `nav` in `mkdocs.yml`.
3. Link to other documentation pages using relative `.md` paths. MkDocs converts
   these links to the generated page URLs.
4. Put images in `docs/assets/images/` and link to them relative to the page.
5. Preview the page and run the strict build before submitting it.

For source code and examples outside `docs/`, use relative repository paths,
such as `[grammar](../grammar/Tepl.g4)`. These links work when browsing Markdown
in GitHub. The build hook in `tools/docs_hooks.py` converts them into links to
the configured repository and branch. Files outside `docs/` are not included in
the site. Add the link title `"Download source"` when readers need the raw file.

Fenced code blocks support syntax highlighting and copy buttons. Use a
`mermaid` fence for diagrams. Markdown inside HTML elements such as `<details markdown="1">`
needs the `markdown="1"` attribute to render correctly.

## Publish to GitHub Pages

The repository includes a manual deployment in the **Documentation** workflow:

1. In the repository's **Settings → Pages**, set **Source** to **GitHub Actions**.
2. In **Actions → Documentation**, choose **Run workflow** on the default branch.
3. The workflow builds the site and deploys it to GitHub Pages. Its deployment
   job links to the published site.

Pushes and pull requests validate the documentation. Publishing requires running
the workflow manually on the default branch. GitHub Actions supplies the current
repository, default branch, and actual Pages URL, including a configured custom
domain. Forks and renamed repositories do not need edits to `mkdocs.yml`.

## Override deployment settings

| Variable | Default |
| --- | --- |
| `DOCS_SITE_URL` | Actual Pages URL during deployment; no canonical URL locally. |
| `DOCS_REPO_URL` | Current GitHub Actions repository, or local Git `origin`. |
| `DOCS_SOURCE_REF` | Repository default branch in Actions; local `origin/HEAD`, current branch, or `main`. |

Set these as repository variables in **Settings → Secrets and variables →
Actions → Variables**, or as environment variables for local builds. Source and
edit links use GitHub's URL structure. For an exported source tree without `.git`,
set `DOCS_REPO_URL` explicitly.

For example, build for a different source repository and a custom domain:

```sh
DOCS_SITE_URL=https://docs.example.org/ \
DOCS_REPO_URL=https://github.com/example/tepl \
DOCS_SOURCE_REF=stable \
  .venv-docs/bin/python -m mkdocs build --strict
```

You can also host the contents of `site/` with any static web server.
