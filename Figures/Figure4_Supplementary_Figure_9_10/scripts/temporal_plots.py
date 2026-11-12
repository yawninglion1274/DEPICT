"""Motion plots from measurements at four reconstruction rates."""
from __future__ import annotations
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import plot_final_motion_speed_3groups as panel_e
import plot_motion_downsampled_examples as examples_plot
traj=examples_plot.traj
drawing_options=examples_plot.drawing_options
FACTORS=(1,2,4,20)
ANGLES=(12,24,48,240)
FPS=16.25

def set_rate_labels(ax):
    ax.set_yticklabels([f"{FPS / q:.5g} ({a} angles)" for q, a in zip(FACTORS, ANGLES)])

def plot_group_speeds(speeds,textless):
    # Use shared boxplot geometry, colors and whisker rules while allowing each
    # rate to retain its own identities and sample size.
    plt.rcParams.update({"pdf.fonttype":42,"font.family":"sans-serif",
        "font.sans-serif":["Arial","Helvetica","DejaVu Sans"]})
    if textless:
        fig=plt.figure(figsize=panel_e.TEXTLESS_SIZE,dpi=320)
        ax=fig.add_axes(panel_e.TEXTLESS_AXES)
    else:
        height=panel_e.OLD_H*panel_e.HEIGHT_SCALE
        bottom=panel_e.OLD_AXES[1]*panel_e.OLD_H
        top=(1-panel_e.OLD_AXES[1]-panel_e.OLD_AXES[3])*panel_e.OLD_H
        fig=plt.figure(figsize=(panel_e.OLD_W,height),dpi=320)
        ax=fig.add_axes((panel_e.OLD_AXES[0],bottom/height,panel_e.OLD_AXES[2],(height-bottom-top)/height))
    ax.set_axisbelow(True)
    ax.grid(axis="x",color=panel_e.GRID,linewidth=0.75)
    handles=[]
    right=max(66,float(speeds.mean_speed_um_min.max())+8)
    for group,color,offset in panel_e.GROUPS:
        values=[speeds.loc[speeds.motion_group.eq(group)&speeds.downsample_factor.eq(q),"mean_speed_um_min"].to_numpy() for q in FACTORS]
        if any(len(v)==0 for v in values):
            raise ValueError(f"No tracks for group {group}")
        positions=[c+offset for c in panel_e.ROW_CENTERS]
        artists=ax.boxplot(values,positions=positions,vert=False,widths=panel_e.BOX_WIDTH,
            whis=1.5,patch_artist=True,manage_ticks=False,
            boxprops={"edgecolor":panel_e.INK,"linewidth":1.3},medianprops={"linewidth":0},
            whiskerprops={"color":panel_e.INK,"linewidth":1.3},capprops={"color":panel_e.INK,"linewidth":1.3},
            flierprops={"marker":"o","markersize":2.4,"markerfacecolor":panel_e.INK,"markeredgecolor":"none","alpha":0.45})
        for box in artists["boxes"]:
            box.set_facecolor(color)
        for position,v in zip(positions,values):
            median=float(np.median(v))
            for ink,width,z in ((panel_e.INK,3.5/1.5,4),("white",2.1/1.5,5)):
                ax.plot([median,median],[position-panel_e.MEDIAN_HALF,position+panel_e.MEDIAN_HALF],color=ink,lw=width,zorder=z,solid_capstyle="round")
            if not textless:
                ax.text(right-0.8,position,f"n={len(v)}",ha="right",va="center",fontsize=6.5,color=panel_e.INK)
        handles.append(Patch(facecolor=color,edgecolor=panel_e.INK,label=group.capitalize()))
    ax.set_xlim(0,right)
    ax.set_ylim(0.45,4.55)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(12))
    ax.set_yticks(panel_e.ROW_CENTERS)
    if textless:
        ax.xaxis.set_major_formatter(mticker.NullFormatter())
        ax.yaxis.set_major_formatter(mticker.NullFormatter())
    else:
        set_rate_labels(ax)
        ax.set_xlabel(r"Mean track speed ($\mu$m min$^{-1}$)",fontsize=9,labelpad=6)
        ax.set_ylabel(r"Volume rate (volumes s$^{-1}$)",fontsize=9,labelpad=9)
        ax.legend(handles=handles,loc="lower center",bbox_to_anchor=(0.5,1.025),ncol=3,frameon=False,fontsize=8)
    ax.tick_params(direction="out",labelsize=8,width=1.1,length=4.5,color=panel_e.INK)
    for spine in ax.spines.values():
        spine.set_color(panel_e.INK)
        spine.set_linewidth(1.3)
    return fig

def choose_examples(groups, speeds):
    preferred = {"diffusing": [56, 143], "directional": [35, 54], "confined": [14, 8]}
    chosen = []
    for group in ("diffusing", "directional", "confined"):
        available = set(groups.loc[groups.final_group.eq(group), "track_id"])
        ids = [i for i in preferred[group] if i in available]
        ordered = speeds.loc[speeds.downsample_factor.eq(1) & speeds.track_id.isin(available)].sort_values(["mean_speed_um_min", "track_id"])
        if len(ordered) < 2:
            raise ValueError(f"Fewer than two matched {group} examples")
        # Unavailable preferred identities are replaced by fixed speed quartiles within
        # the reference group; coarse-resolution alpha never enters selection.
        for quantile in (0.25, 0.75):
            if len(ids) >= 2:
                break
            target = ordered.mean_speed_um_min.quantile(quantile)
            ranks = ordered.assign(distance=(ordered.mean_speed_um_min - target).abs()).sort_values(["distance", "track_id"])
            ids.append(next(i for i in ranks.track_id if i not in ids))
        for i in ids:
            chosen.append(dict(track_id=i, motion_group=group, original_example=i in preferred[group],
                               selection_rule="original example" if i in preferred[group] else "reference-speed quartile replacement"))
    return pd.DataFrame(chosen)

def example_panels(tracks, fits, selection):
    panels = {}
    opts = drawing_options(1.425)
    opts.clip_to_cube = True
    metrics = []
    for item in selection.itertuples():
        ref = tracks.loc[tracks.track_id.eq(item.track_id) & tracks.angle_group.eq(12)].sort_values("frame")
        fit = fits.loc[fits.track_id.eq(item.track_id) & fits.angle_group.eq(12)].iloc[0]
        full = traj.build_panel(fit, ref, opts)
        full["first_frame"] = 0
        panels[item.track_id] = {}
        for angle, q in zip(ANGLES, FACTORS):
            data = tracks.loc[tracks.track_id.eq(item.track_id) & tracks.angle_group.eq(angle)].sort_values("frame").copy()
            result = fits.loc[fits.track_id.eq(item.track_id) & fits.angle_group.eq(angle)].iloc[0]
            data["frame"] = data.time_s * FPS
            panel = examples_plot.sampled_panel(full, data, float(result.alpha))
            panels[item.track_id][q] = panel
            metrics.append(dict(track_id=item.track_id, motion_group=item.motion_group,
                angle_group=angle, downsample_factor=q, volume_rate_vps=FPS/q,
                n_retained=len(data), alpha=result.alpha, alpha_fit_r2=result.fit_r2_log,
                alpha_fit_lags=result.fit_lags, mean_speed_um_min=panel["mean_speed"] * 60,
                cube_um=1.425, n_points_outside_display_cube=int((~panel["sample_inside"]).sum()),
                original_example=item.original_example))
    return panels, pd.DataFrame(metrics)
