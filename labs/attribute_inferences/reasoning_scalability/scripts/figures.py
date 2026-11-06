"""Three aligned panels, phase means and a matching total mean with range."""

def render_figures(rows, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    compile_phases = [('preparation', 'Preparation', '#4477AA'),
                      ('checking_construction', 'Checking /\nconstruction', '#EE9944'),
                      ('lowering_validation', 'Lowering /\nvalidation', '#228877'),
                      ('other', 'Other', '#C5CBD1')]
    runtime_phases = [('feasibility', 'Feasibility\nchecking', '#AA4499'),
                      ('attribute_construction', 'Attribute\nconstruction', '#66AACC'),
                      ('other', 'Other', '#C5CBD1')]
    text_scale = 2.4 * 0.8
    style = {'font.family': 'DejaVu Sans', 'font.size': 14 * text_scale,
             'axes.titlesize': 19 * text_scale, 'axes.labelsize': 16 * text_scale,
             'legend.fontsize': 12 * text_scale, 'legend.title_fontsize': 14 * text_scale,
             'xtick.labelsize': 14 * text_scale, 'ytick.labelsize': 14 * text_scale,
             'axes.labelpad': 4 * text_scale,
             'xtick.major.pad': 3.5 * text_scale, 'ytick.major.pad': 3.5 * text_scale, 'axes.spines.top': False,
             'axes.spines.right': False, 'axes.axisbelow': True, 'pdf.fonttype': 42,
             'svg.fonttype': 'none'}
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 3, figsize=(28, 11))
        fig.subplots_adjust(left=.065, right=.99, top=.84, bottom=.33, wspace=.4)
        for ax, panel, title, xlabel, ylabel in zip(axes, 'abc',
                ['(a) Dot-chain scaling', '(b) Additive-term scaling', '(c) Runtime rank scaling'],
                ['Dots per side', 'Additive terms', 'Tensor rank'],
                ['Compilation time\n(ms)', 'Compilation time\n(ms)', 'Matching execution\ntime (µs)']):
            series = [r for r in rows if r['panel'] == panel and r['successful']]
            if not series:
                raise ValueError('no successful cases for panel ' + panel)
            x = np.arange(len(series))
            bottom = np.zeros(len(series))
            phases = runtime_phases if panel == 'c' else compile_phases
            for field, label, color in phases:
                values = np.array([r['mean_' + field] for r in series])
                ax.bar(x, values, bottom=bottom, width=.66, label=label, color=color,
                       edgecolor='white', linewidth=.5)
                bottom += values
            total = np.array([r['mean_total'] for r in series])
            np.testing.assert_allclose(bottom, total, rtol=1e-10, atol=1e-8)
            error = [[r['mean_total'] - r['min_total'] for r in series],
                     [r['max_total'] - r['mean_total'] for r in series]]
            ax.errorbar(x, total, yerr=error, color='black', marker='o', markersize=4.5,
                        linewidth=1.6, elinewidth=.85, capsize=2.5, zorder=5,
                        label='_nolegend_')
            ax.set(xlabel=xlabel, ylabel=ylabel, xticks=x,
                   xticklabels=[str(r['x']) for r in series], ylim=(0, None))
            ax.set_title(title, loc='left', pad=16 * text_scale, fontweight='bold')
            ax.grid(axis='y', color='#DDE1E5', linestyle='--', linewidth=.6)
            ax.legend(loc='upper left', bbox_to_anchor=(-.015, -.24), ncol=2,
                      frameon=False, columnspacing=.8, handlelength=1.2,
                      title='Runtime phases' if panel == 'c' else 'Compilation phases')
        for suffix in ['png', 'pdf', 'svg']:
            fig.savefig(out / ('scalability.' + suffix), dpi=220, facecolor='white',
                        bbox_inches='tight', pad_inches=.15)
        plt.close(fig)
