import re
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches


def plot_dvh_moo_space(
    trial_df: pd.DataFrame,
    dvh_metric_goals: dict,
    path_out_svg: str,
    sampler_col: str = 'sampler_name_id',
    random_keyword: str = 'random',
    title: str = None):
    r"""
    ### Purpose:
    - Plot the DVH metric space explored during MOO/random optimization trials.
    For each DVH metric in `dvh_metric_goals` (other than the CTV target metric,
    D90% or D95%, which is used as the x-axis), a scatter subplot is produced with:
      - x axis = D90%(CTV) or D95%(CTV)
      - y axis = the other DVH metric
      - points colored blue for random-sampled trials, red for all other
        (e.g. MOBO-qNEHVI) generation methods
      - a green rectangle marking the region where BOTH the x and y metrics
        satisfy their clinical threshold, per `dvh_metric_goals`

    ### Inputs:
        - `trial_df`:= DataFrame where each row is a trial. Must contain the DVH metric
        columns referenced by `dvh_metric_goals` plus `sampler_col`. Other
        columns (penalty weights, dose_voxel_goal, hypervolume, acceptable,
        etc.) are ignored.
        - `dvh_metric_goals`:= dict mapping DVH metric column name -> [operator, threshold],
        e.g. {'D95%(CTV)':= ['>=', 95], 'D2cc(RECTUM)': ['<=', 66], ...}.
        Exactly one key must be the CTV target dose metric (D90%(CTV) or
        D95%(CTV)); it is used as the x-axis. All remaining keys are plotted
        as separate y-axis subplots.
        - `path_out_svg`:= output path for the saved SVG figure.
        - `sampler_col`:= column name identifying the generation method of each trial.
        - `random_keyword`:= substring (case-insensitive) identifying "random" sampling
        in `sampler_col`; any other value is treated as the optimizer (red).
        - `title`:= optional custom title; defaults to an auto-generated one.

    ### Output:
        - None := Saves the figure to `path_out_svg` and returns the matplotlib Figure.
    """

    x_pattern = re.compile(r'^D9[05]%\(.*\)$')
    x_candidates = [k for k in dvh_metric_goals if x_pattern.match(k)]
    if len(x_candidates) != 1:
        raise ValueError(
            f"Expected exactly one D90%/D95% CTV target metric in dvh_metric_goals, "
            f"found: {x_candidates}")
    x_metric = x_candidates[0]
    y_metrics = [k for k in dvh_metric_goals if k != x_metric]

    missing = [c for c in [x_metric, *y_metrics, sampler_col] if c not in trial_df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in trial_df: {missing}")

    is_random = trial_df[sampler_col].astype(str).str.contains(random_keyword, case=False, na=False)
    colors = np.where(is_random, 'b', 'r')

    other_labels = trial_df.loc[~is_random, sampler_col].astype(str).unique()
    red_label = other_labels[0] if len(other_labels) == 1 else (
        ', '.join(other_labels) if len(other_labels) else 'Optimizer')

    n = len(y_metrics)
    if n <= 3:
        fig, axes = plt.subplots(1, n, figsize=(10 * n, 7), squeeze=False)
    elif n == 4:
        fig, axes = plt.subplots(2, 2, figsize=(15, 15), squeeze=False)
    elif n <= 6:
        fig, axes = plt.subplots(2, 3, figsize=(30, 15), squeeze=False)
    else:
        ncols = 3
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(10 * ncols, 7.5 * nrows), squeeze=False)
    flat_axes = axes.flatten()

    fontsize = 20
    fig_title = title if title is not None else f"DVH space of {len(trial_df)} trials"

    def _region_bounds(op, threshold, lo, hi, margin_frac=0.15):
        span = hi - lo
        margin = span * margin_frac if span > 0 else abs(threshold) * 0.1 + 1
        if op in ('>=', '>'):
            return threshold, hi + margin
        elif op in ('<=', '<'):
            return lo - margin, threshold
        else:
            raise ValueError(f"Unsupported operator: {op}")

    red_patch = mpatches.Patch(color='red', label=red_label)
    blue_patch = mpatches.Patch(color='blue', label='Random')

    for i, y_metric in enumerate(y_metrics):
        ax = flat_axes[i]
        ax.scatter(trial_df[x_metric], trial_df[y_metric], c=colors, alpha=0.8)

        ax.set_title(fig_title, fontsize=fontsize)
        ax.set_xlabel(x_metric + ' [%]', fontsize=fontsize)
        ax.set_ylabel(y_metric + ' [%]', fontsize=fontsize)
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)

        legend = ax.legend(handles=[red_patch, blue_patch], loc='upper left',
                            title='Generation Method', fontsize=fontsize - 5)
        ax.add_artist(legend)

        x_op, x_thr = dvh_metric_goals[x_metric]
        y_op, y_thr = dvh_metric_goals[y_metric]

        x_lo, x_hi = trial_df[x_metric].min(), trial_df[x_metric].max()
        y_lo, y_hi = trial_df[y_metric].min(), trial_df[y_metric].max()

        rx0, rx1 = _region_bounds(x_op, x_thr, x_lo, x_hi)
        ry0, ry1 = _region_bounds(y_op, y_thr, y_lo, y_hi)

        ax.add_patch(Rectangle((rx0, ry0), rx1 - rx0, ry1 - ry0,
                                alpha=0.3, color='green'))
        ax.text((rx0 + rx1) / 2, (ry0 + ry1) / 2,
                s="clinically acceptable\nDVH metrics", fontsize=15,
                ha='center', va='center', color='black')

        margin_x = (x_hi - x_lo) * 0.15 if x_hi > x_lo else 1
        margin_y = (y_hi - y_lo) * 0.15 if y_hi > y_lo else 1
        ax.set_xlim(min(x_lo, rx0) - margin_x, max(x_hi, rx1) + margin_x)
        ax.set_ylim(min(y_lo, ry0) - margin_y, max(y_hi, ry1) + margin_y)

    for j in range(n, len(flat_axes)):
        flat_axes[j].axis('off')

    fig.tight_layout()
    fig.savefig(path_out_svg, bbox_inches='tight')
    return fig
