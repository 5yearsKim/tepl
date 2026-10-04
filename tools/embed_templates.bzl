"""Embed Rust runtime templates with a native tool, without invoking a shell."""

def _template_path(file):
    prefix = "templates/rust/"
    if not file.short_path.startswith(prefix):
        fail("Expected a Rust template under %s: %s" % (prefix, file.short_path))
    return file.short_path[len(prefix):]

def _embed_rust_templates_impl(ctx):
    templates = sorted(ctx.files.srcs, key = _template_path)
    args = ctx.actions.args()
    args.add(ctx.outputs.out)
    for template in templates:
        # Keep the output's logical path independent of the execution platform.
        args.add(_template_path(template))
        args.add(template)
    ctx.actions.run(
        executable = ctx.executable._tool,
        arguments = [args],
        inputs = templates,
        outputs = [ctx.outputs.out],
        mnemonic = "EmbedRustTemplates",
        progress_message = "Embedding Rust runtime templates",
    )
    return [DefaultInfo(files = depset([ctx.outputs.out]))]

embed_rust_templates = rule(
    implementation = _embed_rust_templates_impl,
    attrs = {
        "srcs": attr.label_list(allow_files = [".rs"]),
        "out": attr.output(mandatory = True),
        "_tool": attr.label(
            default = Label("//:embed_templates"),
            executable = True,
            cfg = "exec",
        ),
    },
)
