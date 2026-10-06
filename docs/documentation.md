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
and pages omitted from the navigation. The source-link hook also rejects missing
repository files. The build does not check external websites or
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

For source code and examples outside `docs/`, use relative paths from the page:

```md
[Grammar](../grammar/Tepl.g4)
```

GitHub resolves these links to repository files. During a MkDocs build,
`tools/docs_hooks.py` converts them to GitHub URLs using `repo_url` and
`extra.source_branch` from `mkdocs.yml`. Links within `docs/` and fenced code
examples stay unchanged. Files outside `docs/` are not included in the site.

Add the link title `"Download source"` to get a raw download link on the site:

```md
[utils.rs](../../../labs/tutorial_rust/src/utils.rs "Download source")
```

On GitHub, the same link opens the source file, where readers can select **Raw**.

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
the workflow manually on the default branch. GitHub Actions supplies the actual
Pages URL, including a configured custom domain. Repository and edit links in
`mkdocs.yml` point to `5yearsKim/tepl` on `main`. For a fork or another source
branch, update `repo_url`, `edit_uri`, and `extra.source_branch` in that file.

## Override the site URL

`DOCS_SITE_URL` defaults to the actual Pages URL during deployment and is empty
for local builds. Override it as a repository variable in **Settings → Secrets
and variables → Actions → Variables**, or as an environment variable locally:

```sh
DOCS_SITE_URL=https://docs.example.org/ \
  .venv-docs/bin/python -m mkdocs build --strict
```

You can also host the contents of `site/` with any static web server.
