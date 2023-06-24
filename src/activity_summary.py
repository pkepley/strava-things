from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import joblib

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
import sqlite3

from config_reader import ( get_curve_clusterer_path, get_db_config,
                            get_start_loc, get_usual_route_file )
from lat_lon_tools import haversine_dist, estimate_pi
from ActivityAnimator import ActivityAnimator
from CurveClusterer import CurveClusterer


# TODO: this script probably should be converted into
#       a few functions, possibly bundled into a class

## Parse Command Line Configuration Parameters
parser = argparse.ArgumentParser(description='Summarize activities.')
parser.add_argument('--limit-count', dest='n_activity_max', action='store',
                    type=int, default=None)
parser.add_argument('--estimate-pi', dest='run_pi_estimation', action='store_const',
                    const=True, default=False,
                    help='estimate pi')
parser.add_argument('--skip-clustering', dest='skip_clustering', action='store_const',
                    const=True, default=False,
                    help='skip clustering')
parser.add_argument('--save-animation', dest='save_animation', action='store_const',
                    const=True, default=False,
                    help='save animation (this may take a while!)')
parser.add_argument('--save-figures', dest='save_figures', action='store_const',
                    const=True, default=False,
                    help='save figures (dumped to current directory)')
parser.add_argument('--run-headless', dest='run_headless', action='store_const',
                    const=True, default=False,
                    help='run headless (do not plt.show!)')
args = parser.parse_args()

# extract the parser parameters
n_activity_max    = args.n_activity_max
run_pi_estimation = args.run_pi_estimation
skip_clustering   = args.skip_clustering
save_animation    = args.save_animation
save_figures      = args.save_figures
run_headless      = args.run_headless

## Get Default Configuration Parameters
# the following config params *are* required
db_path = get_db_config()['db_path']
loc_conf = get_start_loc()
usual_start_lon = loc_conf['start_lon']
usual_start_lat = loc_conf['start_lat']

# the following config params are *not* required, and default to None if unset
usual_route_file = get_usual_route_file()
clusterer_path = get_curve_clusterer_path()


# fetch all of the activities. filter to the relevant ones.
with sqlite3.connect(db_path) as conn:
    dfe_summary = pd.read_sql(
        """
        select * from exercise_summary
        where date >= ?
        order by start_datetime_utc
        """,
        conn,
        params=('2022-12-30',)
    )
    dfe_summary.set_index('activity_id', inplace=True)

    if n_activity_max is not None:
        dfe_summary = dfe_summary.iloc[:n_activity_max]

    dfe_summary['start_datetime_utc'] = pd.to_datetime(
        dfe_summary['start_datetime_utc']
    )

    dist_to_usual_start = haversine_dist(
        dfe_summary.start_longitude,
        dfe_summary.start_latitude,
        usual_start_lon,
        usual_start_lat
    )

    # only keep activities that start near the usual start location
    dfe_summary = dfe_summary[dist_to_usual_start < 0.1]


# containers to hold objects we will pull
summ_stats = []
dfs = dict()

# pull all of the relevant activities
for activity_id in dfe_summary.index:
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql(
            """
            select * from exercise
            where activity_id=?
            order by time
            """,
            conn,
            params=(activity_id,)
        )
        df.drop(columns='activity_id',inplace=True)

        # convert distance from meters into to miles
        df['distance'] = 0.000621371 * df['distance']

    if df.shape[0] == 0:
        print(f"Activity {activity_id} had no records. Skipping.")
        continue

    activity_date = dfe_summary.loc[activity_id]['date']
    print(f"{activity_date}: {df.distance.isna().sum()}/{df.shape[0]}")

    df["pace"] = 60 / (df.velocity_smooth)
    df["minutes_in"] = df['time'] / 60
    df.set_index('time', inplace=True)

    dfs[activity_id] = df

    lbl = dfe_summary.loc[activity_id]['date']

    curr_datum = {
        'activity_id': activity_id,
        'name': lbl,
        'tot_dist': df.distance.iloc[-1],
        'duration_minutes': df.minutes_in.iloc[-1]
    }

    if run_pi_estimation:
        pi_est, n_loops, theta_tot = estimate_pi(
            df['longitude'].to_numpy(),
            df['latitude'].to_numpy()
        )
        curr_datum['pi_est'] = pi_est
        curr_datum['n_signed_loops'] = n_loops
        curr_datum['theta_tot'] = theta_tot

    summ_stats.append(curr_datum)

# let user know
print(f"Found {len(dfs)} activities")

# build a summary
df_summ = pd.DataFrame(summ_stats)
df_summ['name'] = df_summ['name'].apply(lambda x: "-".join(x.split("-")[-3:]))
df_summ['date'] = pd.to_datetime(df_summ['name'])
df_summ['avg_pace'] = df_summ['duration_minutes'] / df_summ['tot_dist']
df_summ['avg_speed'] = 60 * (1.0 / df_summ['avg_pace'])
df_summ = df_summ.merge(
    dfe_summary.reset_index()[['activity_id', 'start_datetime_utc']],
    on='activity_id'
)

# activities
activities = df_summ.sort_values('start_datetime_utc').activity_id.to_list()

# if we estimated pi for each run... let the user know the result!
if run_pi_estimation:
    pi_est = df_summ['pi_est'].mean()
    n_pi_est = np.sum(df_summ['n_signed_loops'] != 0)
    print(f"We estimated pi from {n_pi_est} activities")
    print(f"\t- to 20 decimal places pi estimate: {pi_est:0.20f}")
    print(f"\t- to 20 decimal places pi is      : {np.pi:0.20f}")
    print(f"\t- to 20 decimal places error is   : {pi_est-np.pi:0.20f}")

# deal with the clusterer - used to specify the color scheme, basically...
if clusterer_path is None or skip_clustering:
    if skip_clustering:
        skip_message = "The skip_clustering flag was set"
    else:
        skip_message = "The clusterer path was not provided"
    print(f"{skip_message}. We will skip clustering.")

    # T A B 20  A E S T H E T I C S
    cmap = mpl.colormaps.get_cmap('tab20')
    cs = [cmap(i % 20)  for i, _ in enumerate(activities)]

else:
    if not clusterer_path.exists():
        print(
            "The Curve Clusterer joblib file was not present.\n" +
            "\t - We will now cluster the curves (and save the result)\n" +
            "\t - This may take a while!"
        )
        cc = CurveClusterer(delta_dist=0.025)
        clusters = cc.fit_predict([dfs[a] for a in sorted(activities)])

        with open(clusterer_path, 'wb') as f_cc_out:
            joblib.dump(cc, f_cc_out)
    else:
        print("Loading curve clusterer.")
        cc = joblib.load(clusterer_path)
        clusters = cc.predict([dfs[a] for a in sorted(activities)])

        if cc.n_train >= len(activities):
            print(
                "The number of training activities matches (or exceeds) the number of activities.\n" +
                "\t- We assume that all curves are accounted for...\n"+
                "\t- We will NOT refit the clusterer."
            )
        else:
            n_new = len(activities) - cc.n_train
            newly_assigned = activities[-n_new:]
            dfs_new = [dfs[a] for a in sorted(newly_assigned)]

            if cc.refit_required(dfs_new):
                print(
                    "Newly found curves appear to belong to unobserved clusters.\n"+
                    "\t- We will now refit the curve clusterer (and save the results).\n"+
                    "\t- This may take a while!"
                )
                cc.fit(
                    [dfs[a] for a in sorted(activities)],
                    min_n_cluster=cc.n_cluster,
                    max_n_cluster=cc.n_cluster+len(newly_assigned)
                )
                clusters = cc.predict([dfs[a] for a in sorted(activities)])

                with open(clusterer_path, 'wb') as f_cc_out:
                    joblib.dump(cc, f_cc_out)
            else:
                print(
                    "New curves were observed, but they seem to belong to existing "+
                    "clusters.\n"+
                    "\t- If this is not the case, try deleting the clusterer joblib file\n" +
                    "\t  and re-running."
                )

                print(
                    f"The curve clusterer reports:\n" +
                    f"\t- I was trained on {cc.n_train} observations.\n" +
                    f"\t- I will cluster data into {cc.n_cluster} clusters."
                )

    # make a table of cluster info
    df_lbl = pd.DataFrame({
        'activity_id': activities,
        'cluster': clusters
       })
    df_lbl = df_lbl.set_index('activity_id')
    df_lbl = df_lbl.join(
        dfe_summary[['start_datetime_utc', 'date']]
       )
    df_lbl.sort_values(['cluster', 'start_datetime_utc'], inplace=True)
    df_lbl['cluster_rank'] = df_lbl.groupby('cluster')['start_datetime_utc'].rank()
    df_lbl.reset_index(inplace=True)

    # A E S T H E T I C S
    cmap = mpl.colormaps.get_cmap('cool')
    lbl_remap = dict()

    # since the hierarchical clusterer tends to place "similar" trajectories near
    # one another (resulting in similar clustering indices for similar trajectories)
    # we can flip back and forth between the high and low clustering indices to
    # put very distinct curves into 'similar' color ranges. the end goal is to make
    # similar (but different) trajectories more distinct visually
    for k in range(cc.n_cluster//2 + 1):
        lbl_remap[k] = 2 * k
        lbl_remap[cc.n_cluster - k] = 2 * k + 1

    cs_map = {a:c for a,c in zip(df_lbl.activity_id, df_lbl.cluster)}
    cs = [cmap(lbl_remap[cs_map[a]] / cc.n_cluster) for a in activities]


# Plot summary statistics
plt.style.use('dark_background')
n_ax = 5 if run_pi_estimation else 4
fig, axs = plt.subplots(n_ax, 1, figsize=(12, 3*n_ax))
ax_names = []

axs[0].plot(df_summ['date'], df_summ['tot_dist'], marker='.', c='lightgrey', zorder=-1)
axs[0].scatter(df_summ['date'], df_summ['tot_dist'], marker='.', c=cs, s=121, zorder=1)
axs[0].set_title("Total Distance")
axs[0].set_ylabel("Total Distance (mi)")
ax_names.append("total_distance")

axs[1].plot(df_summ['date'], df_summ['duration_minutes'], c='lightgrey', zorder=-1)
axs[1].scatter(df_summ['date'], df_summ['duration_minutes'], marker='.', c=cs, s=121, zorder=1)
axs[1].set_title("Total Duration")
axs[1].set_ylabel("Total Duration (min)")
ax_names.append("total_duration")

axs[2].plot(df_summ['date'], df_summ['avg_speed'], marker='.', c='lightgrey', zorder=-1)
axs[2].scatter(df_summ['date'], df_summ['avg_speed'], marker='.', c=cs, s=121, zorder=1)
axs[2].set_title("Average Speed")
axs[2].set_ylabel("Average Speed (mi/h)")
ax_names.append("average_speed")

axs[3].plot(df_summ['date'], df_summ['avg_pace'], marker='.', c='lightgrey', zorder=-1)
axs[3].scatter(df_summ['date'], df_summ['avg_pace'], marker='.', c=cs, s=121, zorder=1)
axs[3].set_title("Average Pace")
axs[3].set_ylabel("Average Pace (min/mi)")
ax_names.append("average_pace")

if run_pi_estimation:
    axs[4].plot(df_summ['date'], df_summ['n_signed_loops'], marker='.', c='lightgrey', zorder=-1)
    axs[4].scatter(df_summ['date'], df_summ['n_signed_loops'], marker='.', c=cs, s=121, zorder=1)
    axs[4].set_title("Signed Loops about a 'Central' Point")
    axs[4].set_ylabel("Signed Loops")
    ax_names.append("n_signed_loops")

plt.tight_layout()

if save_figures:
    print("Saving figures.")
    for i in range(n_ax):
        extent = axs[i].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig_name = f'fig_{i:02d}_{ax_names[i]}.png'
        fig.savefig(fig_name, bbox_inches=extent.expanded(1.1, 1.2))


if run_pi_estimation:
    fig, ax = plt.subplots()
    ax.hist(df_summ['pi_est'][df_summ['n_signed_loops'] != 0] - np.pi)
    ax.set_title(r"$\pi$ Estimate Error Histogram")
    ax.set_ylabel("Count")
    ax.set_xlabel("Estimation Error")

if run_pi_estimation and save_figures:
    fig.savefig(f'fig_{6:02d}_pi_error_histogram.png')


# animate
if usual_route_file is not None:
    if usual_route_file.exists():
        df_route = pd.read_csv(usual_route_file)
        print("'Usual' route file found. We will use it when animating the activities.")
    else:
        print("'Usual' route file did not exist. The 'usual' route will not be plotted.")
        df_route = None
else:
    print("'Usual' route file was not provided. The 'Usual' route will NOT be plotted.")
    df_route = None

activity_anim = ActivityAnimator(
    [dfs[a] for a in activities],
    activities,
    stride=3,
    trail=10,
    df_route=df_route,
    linecolors=cs
)

# these intervals work well(ish) on my machine
anim_interval = 50 if save_animation else 10

anim = FuncAnimation(
    activity_anim.fig,
    activity_anim.anim_update,
    init_func=activity_anim.anim_init,
    frames=activity_anim.nt,
    interval=anim_interval,
    blit=True
)

if save_animation:
    print("Saving animation to disk! This may take a while!")
    anim.save("activity_animation.mp4", dpi=600, bitrate=-1)
    print("Done!")

if not run_headless:
    plt.show()
