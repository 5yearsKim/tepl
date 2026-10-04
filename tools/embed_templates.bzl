"""Embed language runtime templates with a native tool, without a shell."""

def _template_path(file, language):
    prefix = "templates/" + language + "/"
    if not file.short_path.startswith(prefix):
        fail("Expected a template under %s: %s" % (prefix, file.short_path))
    return file.short_path[len(prefix):]

def _file_path(file):
    return file.short_path

def _embed_templates_impl(ctx):
    # Source label order is immaterial; sort by stable execution path.
    templates = sorted(ctx.files.srcs, key = _file_path)
    args = ctx.actions.args()
    args.add(ctx.outputs.out)
    args.add("tepl::codegen::" + ctx.attr.language)
    for template in templates:
        # Keep the output's logical path independent of the execution platform.
        args.add(_template_path(template, ctx.attr.language))
        args.add(template)
    ctx.actions.run(
        executable = ctx.executable._tool,
        arguments = [args],
        inputs = templates,
        outputs = [ctx.outputs.out],
        mnemonic = "EmbedTemplates",
        progress_message = "Embedding %s runtime templates" % ctx.attr.language,
    )
    return [DefaultInfo(files = depset([ctx.outputs.out]))]

embed_templates = rule(
    implementation = _embed_templates_impl,
    attrs = {
        "srcs": attr.label_list(allow_files = [".rs", ".h"]),
        "language": attr.string(default = "rust", values = ["rust", "cpp"]),
        "out": attr.output(mandatory = True),
        "_tool": attr.label(
            default = Label("//:embed_templates"),
            executable = True,
            cfg = "exec",
        ),
    },
)

# Existing Rust call sites retain their interface.
embed_rust_templates = embed_templates
