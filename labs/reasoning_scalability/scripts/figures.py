"""Publication figures for saved inference measurements; no benchmark execution."""


def render_figures(rows, out, chain_sizes, ranks):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator, NullLocator, StrMethodFormatter

    colors = {"explicit": "#356B8C", "partial": "#358579", "automatic": "#BC6743"}
    labels = {"explicit": "Explicit", "partial": "Partial auto", "automatic": "Full auto"}
    markers = {"explicit": "o", "partial": "s", "automatic": "D"}
    linestyles = {"explicit": "-", "partial": "--", "automatic": "-."}
    style = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arimo", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": 25,
        "axes.labelsize": 29,
        "axes.titlesize": 32,
        "axes.titleweight": "bold",
        "axes.titlepad": 18,
        "axes.labelpad": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.9,
        "axes.edgecolor": "#454545",
        "axes.labelcolor": "#252525",
        "axes.axisbelow": True,
        "xtick.labelsize": 25,
        "ytick.labelsize": 25,
        "xtick.color": "#454545",
        "ytick.color": "#454545",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.major.pad": 6,
        "ytick.major.pad": 6,
        "legend.fontsize": 25,
        "legend.frameon": False,
        "legend.handlelength": 2,
        "legend.labelspacing": 0.45,
        "legend.borderaxespad": 0.25,
        "lines.linewidth": 2,
        "lines.markersize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    }

    def base_axes(ax):
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.65, linestyle=(0, (2, 3)))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.margins(x=0.035, y=0.08)

    def curves(ax, family, ticks):
        group = [r for r in rows if r["family"] == family]
        for mode, label in labels.items():
            series = sorted((r for r in group if r["mode"] == mode), key=lambda r: r["size"])
            if not series:
                continue
            # Keep gaps at unsuccessful sizes rather than joining across them.
            x = [r["size"] for r in series]
            y = [r["median_total_ms"] if r["status"] == "proved" else float("nan") for r in series]
            successful = [r for r in series if r["status"] == "proved"]
            ax.plot(x, y, color=colors[mode], marker=markers[mode],
                    linestyle=linestyles[mode], label=label,
                    markerfacecolor="white", markeredgewidth=1.5, zorder=3, clip_on=False)
            ax.errorbar([r["size"] for r in successful], [r["median_total_ms"] for r in successful],
                        yerr=[[r["median_total_ms"] - r["min_total_ms"] for r in successful],
                              [r["max_total_ms"] - r["median_total_ms"] for r in successful]],
                        fmt="none", color=colors[mode], elinewidth=1, capsize=3, capthick=1, zorder=2)
            for r in series:
                if r["status"] != "proved":
                    ax.scatter(r["size"], r["median_total_ms"], marker="x", color=colors[mode], zorder=4)
        if any(r["status"] != "proved" for r in group):
            ax.plot([], [], linestyle="", marker="x", color="#454545", label="Unproved / mixed")
        ax.set_xticks(ticks, labels=[str(tick) for tick in ticks])
        ax.set_ylabel("Pipeline time (ms)")
        base_axes(ax)
        ax.legend(loc="upper left")

    def chain_panel(ax):
        curves(ax, "chain", chain_sizes)
        ax.set_xlabel("Dot operations per side")
        ax.set_ylim(bottom=0)

    def rank_panel(ax):
        ax.set_xscale("log", base=2)
        curves(ax, "rank", ranks)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True, steps=[1, 2, 5, 10]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel("Tensor rank")

    def phases_panel(ax):
        full = sorted((r for r in rows if r["family"] == "chain" and r["mode"] == "automatic"),
                      key=lambda r: r["size"])
        bottom = [0.0] * len(full)
        for field, label, color in (("construction_ms", "Construction", colors["explicit"]),
                                    ("verification_ms", "Verification", "#D59970"),
                                    ("other_ms", "Dispatch /\naccounting", "#B9BEC4")):
            values = [r["mean_" + field] for r in full]
            bars = ax.bar([r["size"] for r in full], values, width=0.68,
                          bottom=bottom, label=label, color=color,
                          edgecolor="white", linewidth=0.45, zorder=3)
            for bar, row in zip(bars, full):
                if row["status"] != "proved":
                    bar.set_hatch("///")
                    bar.set_edgecolor("#454545")
            bottom = [a + b for a, b in zip(bottom, values)]
        ax.set(xlabel="Dot operations per side", ylabel="Mean time (ms)", xticks=chain_sizes)
        base_axes(ax)
        ax.set_ylim(bottom=0)
        handles, names = ax.get_legend_handles_labels()
        if any(r["status"] != "proved" for r in full):
            handles.append(Patch(facecolor="white", edgecolor="#454545", hatch="///"))
            names.append("Unproved / mixed")
        ax.legend(handles, names, loc="upper left")

    panels = ((chain_panel, "(a) Chain scaling"),
              (rank_panel, "(b) Rank scaling"),
              (phases_panel, "(c) Time breakdown"))

    def save(fig, name):
        for suffix in ("pdf", "png", "svg"):
            fig.savefig(out / f"{name}.{suffix}", dpi=240, bbox_inches="tight", pad_inches=0.12)

    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 3, figsize=(19.2, 6.3), layout="constrained")
        fig.set_constrained_layout_pads(w_pad=0.14, h_pad=0.12, wspace=0.08)
        for ax, (draw, title) in zip(axes, panels):
            draw(ax)
            ax.set_title(title, loc="left")
        # The chain latency and phase totals describe the same workloads.
        chain_limit = max(axes[0].get_ylim()[1], axes[2].get_ylim()[1])
        axes[0].set_ylim(0, chain_limit)
        axes[2].set_ylim(0, chain_limit)
        save(fig, "scalability")
        plt.close(fig)
