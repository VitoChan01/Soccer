"""
Soccer Trajectory Clustering
=============================
Two approaches:
  A) Multiscale DTW matching → spectral clustering on pairwise distance matrix
  B) Multivariate time series clustering (heading angle + relative angle to ball)

Expected data format
--------------------
players : dict[str, np.ndarray]
    key   = player id (e.g. "p1", "p7")
    value = array of shape (T, 2) — columns are (x, y) positions at each timestep

ball : np.ndarray, shape (T, 2)
    Ball position at each timestep

Both must be sampled at the same rate and cover the same time window.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import squareform
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import pandas as pd



# ─────────────────────────────────────────────
# SHARED UTILITIES
# ─────────────────────────────────────────────

def compute_velocity(positions: np.ndarray) -> np.ndarray:
    """Finite difference velocity, shape (T-1, 2)."""
    return np.diff(positions, axis=0)


def compute_heading(velocity: np.ndarray) -> np.ndarray:
    """
    Heading angle in radians at each timestep, shape (T-1,).
    Uses atan2 so result is in (-π, π].
    Stationary frames (speed < eps) inherit the previous heading.
    """
    angles = np.arctan2(velocity[:, 1], velocity[:, 0])
    speed = np.linalg.norm(velocity, axis=1)
    eps = 1e-6
    for i in range(1, len(angles)):
        if speed[i] < eps:
            angles[i] = angles[i - 1]
    return angles


def compute_relative_angle(player_heading: np.ndarray,
                            ball_heading: np.ndarray) -> np.ndarray:
    """
    Signed angular difference between player and ball headings, shape (T-1,).
    Result wrapped to (-π, π] so 0 = moving with ball, ±π = moving against it.
    """
    diff = player_heading - ball_heading
    return (diff + np.pi) % (2 * np.pi) - np.pi


def angular_distance(a: float, b: float) -> float:
    """Minimum angular distance between two angles (radians)."""
    d = abs(a - b) % (2 * np.pi)
    return min(d, 2 * np.pi - d)






# ─────────────────────────────────────────────
# APPROACH A — MULTISCALE DTW
# ─────────────────────────────────────────────

def downsample(signal: np.ndarray, factor: int) -> np.ndarray:
    """
    Downsample a 1-D signal by averaging non-overlapping windows.
    Trailing samples that don't fill a full window are dropped.
    """
    T = len(signal)
    T_trim = (T // factor) * factor
    return signal[:T_trim].reshape(-1, factor).mean(axis=1)


def dtw_angular(s: np.ndarray, t: np.ndarray) -> float:
    """
    Standard DTW with angular distance as the local cost.
    O(len(s) * len(t)) time and space.
    """
    n, m = len(s), len(t)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = angular_distance(s[i - 1], t[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return D[n, m]


def multiscale_dtw(s: np.ndarray, t: np.ndarray,
                   scales: list[int] = None) -> float:
    """
    Multiscale DTW distance between two heading-angle time series.

    The signal is compared at multiple temporal resolutions (scales).
    At each scale the signal is downsampled by that factor before DTW,
    so coarse structure (overall direction) and fine structure (quick turns)
    both contribute to the final distance.

    Parameters
    ----------
    s, t    : 1-D arrays of heading angles (radians)
    scales  : list of downsample factors, e.g. [1, 2, 4]
              scale=1 → original resolution
              scale=4 → each sample is the mean of 4 original samples

    Returns
    -------
    Weighted average DTW distance across scales (lower = more similar).
    """
    if scales is None:
        scales = [1, 2, 4]

    total, weight_sum = 0.0, 0.0
    for scale in scales:
        s_down = downsample(s, scale)
        t_down = downsample(t, scale)
        if len(s_down) == 0 or len(t_down) == 0:
            continue
        # Normalise by length so longer windows don't dominate
        d = dtw_angular(s_down, t_down) / max(len(s_down), len(t_down))
        # Finer scales get more weight (more detail)
        w = 1.0 / scale
        total += w * d
        weight_sum += w

    return total / weight_sum if weight_sum > 0 else 0.0


def build_multiscale_distance_matrix(signals: dict[str, np.ndarray],
                                     scales: list[int] = None) -> tuple:
    """
    Compute the N×N pairwise multiscale DTW distance matrix.

    Parameters
    ----------
    signals : dict player_id → heading angle array (1-D)
    scales  : passed through to multiscale_dtw

    Returns
    -------
    ids     : list of player ids (row/column order)
    D       : np.ndarray shape (N, N), symmetric distance matrix
    """
    ids = list(signals.keys())
    N = len(ids)
    D = np.zeros((N, N))

    for i in range(N):
        for j in range(i + 1, N):
            d = multiscale_dtw(signals[ids[i]], signals[ids[j]], scales)
            D[i, j] = d
            D[j, i] = d

    return ids, D


def cluster_multiscale(players: dict[str, np.ndarray],
                       ball: np.ndarray,
                       n_clusters: int = 3,
                       scales: list[int] = None,
                       use_relative_angle: bool = True) -> dict[str, int]:
    """
    Full multiscale DTW pipeline.

    Parameters
    ----------
    players           : dict player_id → position array (T, 2)
    ball              : ball position array (T, 2)
    n_clusters        : number of clusters for SpectralClustering
    scales            : multiscale factors (default [1, 2, 4])
    use_relative_angle: if True, cluster on angle relative to ball;
                        if False, cluster on absolute heading

    Returns
    -------
    dict player_id → cluster label (int)
    """
    ball_vel = compute_velocity(ball)
    ball_heading = compute_heading(ball_vel)

    signals = {}
    for pid, pos in players.items():
        vel = compute_velocity(pos)
        heading = compute_heading(vel)
        if use_relative_angle:
            signals[pid] = compute_relative_angle(heading, ball_heading)
        else:
            signals[pid] = heading

    ids, D = build_multiscale_distance_matrix(signals, scales)

    # Convert distance to affinity for spectral clustering
    # RBF kernel: sigma set to median distance (robust heuristic)
    sigma = np.median(D[D > 0]) if np.any(D > 0) else 1.0
    affinity = np.exp(-(D ** 2) / (2 * sigma ** 2))
    np.fill_diagonal(affinity, 1.0)

    sc = SpectralClustering(n_clusters=n_clusters,
                            affinity='precomputed',
                            random_state=42)
    labels = sc.fit_predict(affinity)

    return {ids[i]: int(labels[i]) for i in range(len(ids))}, D, ids

def get_multiscale_distance_matrix(players: dict[str, np.ndarray],
                                    ball: np.ndarray,
                                    scales: list[int] = None,
                                    use_relative_angle: bool = True) -> tuple:
    ball_vel = compute_velocity(ball)
    ball_heading = compute_heading(ball_vel)

    signals = {}
    for pid, pos in players.items():
        vel = compute_velocity(pos)
        heading = compute_heading(vel)
        if use_relative_angle:
            signals[pid] = compute_relative_angle(heading, ball_heading)
        else:
            signals[pid] = heading

    return build_multiscale_distance_matrix(signals, scales)









#MULTIVARIATE TIME SERIES
# def dtw_multivariate(S: np.ndarray, T_: np.ndarray,
#                      weights: np.ndarray = None) -> float:
#     """
#     DTW on multivariate time series using a weighted sum of per-dimension
#     angular distances.
 
#     Parameters
#     ----------
#     S, T_    : arrays of shape (time, D) where D = number of signals
#     weights  : 1-D array of length D; defaults to uniform
 
#     Returns
#     -------
#     Scalar DTW distance.
#     """
#     n, D = S.shape
#     m = T_.shape[0]
#     if weights is None:
#         weights = np.ones(D) / D
#     weights = np.array(weights) / weights.sum()
 
#     cost_matrix = np.zeros((n, m))
#     for i in range(n):
#         for j in range(m):
#             cost_matrix[i, j] = sum(
#                 weights[d] * angular_distance(S[i, d], T_[j, d])
#                 for d in range(D)
#             )
 
#     dtw_mat = np.full((n + 1, m + 1), np.inf)
#     dtw_mat[0, 0] = 0.0
#     for i in range(1, n + 1):
#         for j in range(1, m + 1):
#             dtw_mat[i, j] = cost_matrix[i - 1, j - 1] + min(
#                 dtw_mat[i - 1, j],
#                 dtw_mat[i, j - 1],
#                 dtw_mat[i - 1, j - 1]
#             )
#     return dtw_mat[n, m]

def dtw_multivariate(S: np.ndarray, T_: np.ndarray,
                     weights: np.ndarray = None,
                     max_frames: int = 100) -> float:

    n, D = S.shape
    m = T_.shape[0]
    if weights is None:
        weights = np.ones(D) / D
    weights = np.array(weights) / weights.sum()

    diff = np.abs(S[:, None, :] - T_[None, :, :])
    angular_diff = np.minimum(diff, 2 * np.pi - diff)
    cost_matrix = (angular_diff * weights[None, None, :]).sum(axis=2)

    dtw_mat = np.full((n + 1, m + 1), np.inf)
    dtw_mat[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dtw_mat[i, j] = cost_matrix[i - 1, j - 1] + min(
                dtw_mat[i - 1, j],
                dtw_mat[i, j - 1],
                dtw_mat[i - 1, j - 1]
            )
    return dtw_mat[n, m]

def build_multivariate_signals(players: dict[str, np.ndarray],
                                ball: np.ndarray) -> dict[str, np.ndarray]:
    """
    Build per-player multivariate signal array of shape (T-1, 2):
      col 0 : absolute heading angle θ(t)
      col 1 : relative angle to ball  θ_rel(t)

    Parameters
    ----------
    players : dict player_id → position array (T, 2)
    ball    : ball position array (T, 2)

    Returns
    -------
    dict player_id → array (T-1, 2)
    """
    ball_vel = compute_velocity(ball)
    ball_heading = compute_heading(ball_vel)

    mv_signals = {}
    for pid, pos in players.items():
        vel = compute_velocity(pos)
        heading = compute_heading(vel)
        rel_angle = compute_relative_angle(heading, ball_heading)
        mv_signals[pid] = np.stack([heading, rel_angle], axis=1)  # (T-1, 2)

    return mv_signals


def build_multivariate_distance_matrix(mv_signals: dict[str, np.ndarray],
                                        weights: np.ndarray = None) -> tuple:
    """
    N×N pairwise DTW distance matrix for multivariate signals.

    Returns
    -------
    ids : list of player ids
    D   : np.ndarray (N, N)
    """
    ids = list(mv_signals.keys())
    N = len(ids)
    D = np.zeros((N, N))

    for i in range(N):
        for j in range(i + 1, N):
            d = dtw_multivariate(mv_signals[ids[i]],
                                  mv_signals[ids[j]],
                                  weights)
            # Normalise by trajectory length
            d /= max(len(mv_signals[ids[i]]), len(mv_signals[ids[j]]))
            D[i, j] = d
            D[j, i] = d

    return ids, D


def cluster_multivariate(players: dict[str, np.ndarray],
                          ball: np.ndarray,
                          n_clusters: int = 3,
                          weights: np.ndarray = None) -> tuple:
    """
    Full multivariate pipeline (signals 1 & 3).

    Parameters
    ----------
    players    : dict player_id → position array (T, 2)
    ball       : ball position array (T, 2)
    n_clusters : number of clusters
    weights    : shape (2,) weights for [heading, relative_angle]
                 default = [0.5, 0.5]
                 set to [0, 1] to use only relative angle (signal 3)
                 set to [1, 0] to use only heading (signal 1)

    Returns
    -------
    cluster_labels : dict player_id → int
    D              : distance matrix
    ids            : player id list (row/col order)
    """
    if weights is None:
        weights = np.array([0.5, 0.5])

    mv_signals = build_multivariate_signals(players, ball)
    ids, D = build_multivariate_distance_matrix(mv_signals, weights)

    sigma = np.median(D[D > 0]) if np.any(D > 0) else 1.0
    affinity = np.exp(-(D ** 2) / (2 * sigma ** 2))
    np.fill_diagonal(affinity, 1.0)

    sc = SpectralClustering(n_clusters=n_clusters,
                            affinity='precomputed',
                            random_state=42)
    labels = sc.fit_predict(affinity)

    return {ids[i]: int(labels[i]) for i in range(len(ids))}, D, ids


def get_multivariate_distance_matrix(players: dict[str, np.ndarray],
                                      ball: np.ndarray,
                                      weights: np.ndarray = None) -> tuple:
    if weights is None:
        weights = np.array([0.5, 0.5])

    mv_signals = build_multivariate_signals(players, ball)
    return build_multivariate_distance_matrix(mv_signals, weights)



#dbscan

from sklearn.metrics import silhouette_score
def normalise_distance_matrix(D: np.ndarray) -> np.ndarray:
    max_d = D.max()
    if max_d == 0:
        return D
    return D / max_d
def auto_cluster_dbscan(D: np.ndarray, ids: list[str],
                         min_samples: int = 3,
                         eps_percentiles: list[float] = None) -> dict[str, int]:
    """
    Automatically pick eps per possession by maximising silhouette score.
    Searches over eps values derived from percentiles of the distance distribution.
    """
    if eps_percentiles is None:
        eps_percentiles = [15, 20, 25, 30, 35, 40, 50]

    nonzero = D[D > 0]
    candidates = [float(np.percentile(nonzero, p)) for p in eps_percentiles]

    best_score = -2
    best_labels = None
    best_eps = None

    for eps in candidates:
        db = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
        labels = db.fit_predict(D)

        n_clusters = len(set(labels) - {-1})
        n_outliers = sum(1 for l in labels if l == -1)

        # Skip degenerate solutions
        if n_clusters < 1 or n_clusters == len(ids):
            continue
        # Need at least 2 labels for silhouette (including -1 as a label)
        if len(set(labels)) < 2:
            continue

        score = silhouette_score(D, labels, metric='precomputed')

        if score > best_score:
            best_score = score
            best_labels = labels
            best_eps = eps

    if best_labels is None:
        print("  Warning: no valid clustering found, all assigned to cluster 0")
        best_labels = np.zeros(len(ids), dtype=int)
        print(f"  clusters={len(set(best_labels) - {-1})}, "
              f"outliers={sum(1 for l in best_labels if l == -1)}")
    else:
        print(f"  Auto-selected eps={best_eps:.4f}, "
              f"silhouette={best_score:.3f}, "
              f"clusters={len(set(best_labels) - {-1})}, "
              f"outliers={sum(1 for l in best_labels if l == -1)}")

    return {ids[i]: int(best_labels[i]) for i in range(len(ids))}






#plotting cluster
from shapely.geometry import LineString
import geopandas as gpd
from .soccer_plot import plot_background_ax, plot_trace, add_legend

def plot_clusters(players: dict[str, np.ndarray],
                  ball: np.ndarray,
                  cluster_labels: dict[str, int],
                  team_map: dict[str, str],
                  title: str = "Trajectory Clustering"):
    """
    Parameters
    ----------
    players       : dict player_id → position array (T, 2)
    ball          : ball position array (T, 2)
    cluster_labels: dict player_id → cluster int
    team_map      : dict player_id → "Home" or "Away"
    """
    n_clusters = max(cluster_labels.values()) + 1
    team_colors = {"Home": "blue", "Away": "red"}

    # One column per cluster, two rows (field | normalised)
    fig, axes = plt.subplots(2, n_clusters,
                             figsize=(7 * n_clusters, 14))

    # Build ball GeoDataFrame once
    ball_line = LineString(ball)
    ball_gdf = gpd.GeoDataFrame(geometry=[ball_line])

    for cluster_id in range(n_clusters):
        ax_field = axes[0, cluster_id]
        ax_norm  = axes[1, cluster_id]

        # ── Field plot (top row) ──────────────────────────────
        plot_background_ax(ax_field)
        plot_trace(ball_gdf, ax_field, color="green")

        # ── Normalised plot (bottom row) ──────────────────────
        ball_norm = ball - ball[0]
        ax_norm.plot(ball_norm[:, 0], ball_norm[:, 1],
                     color="green", linewidth=1.5, linestyle="--")
        ax_norm.plot(0, 0, "o", color="green", markersize=4)

        for pid, pos in players.items():
            if cluster_labels[pid] != cluster_id:
                continue

            color = team_colors.get(team_map.get(pid, "Home"), "blue")

            # Field: convert to GeoDataFrame and plot
            line = LineString(pos)
            gdf = gpd.GeoDataFrame(geometry=[line])
            plot_trace(gdf, ax_field, color=color)

            # Start marker on field
            ax_field.plot(pos[0, 0], pos[0, 1],
                          "o", color=color, markersize=5)
            ax_field.text(pos[0, 0], pos[0, 1], pid,
                          fontsize=7, color=color)

            # Normalised
            norm_pos = pos - pos[0]
            ax_norm.plot(norm_pos[:, 0], norm_pos[:, 1],
                         color=color, alpha=0.7, linewidth=1.5)
            ax_norm.plot(0, 0, "o", color=color, markersize=4)
            ax_norm.text(norm_pos[-1, 0], norm_pos[-1, 1],
                         pid, fontsize=7, color=color)

        add_legend(ax_field)
        ax_field.set_title(f"Cluster {cluster_id} — field positions")
        ax_norm.set_title(f"Cluster {cluster_id} — origin normalised")
        ax_norm.set_aspect("equal")
        ax_norm.axhline(0, color="grey", linewidth=0.5, linestyle=":")
        ax_norm.axvline(0, color="grey", linewidth=0.5, linestyle=":")

    fig.suptitle(title, fontsize=15, fontweight="bold")
    plt.tight_layout()
    return fig
def plot_distance_matrix(D: np.ndarray, ids: list[str], title: str = "Distance Matrix"):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(D, cmap='viridis')
    ax.set_xticks(range(len(ids)))
    ax.set_yticks(range(len(ids)))
    ax.set_xticklabels(ids, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(ids, fontsize=9)
    plt.colorbar(im, ax=ax, label='DTW distance')
    ax.set_title(title)
    plt.tight_layout()
    return fig

def analyse_clusters(cluster_labels: dict[str, int],
                     team_sheets_df: pd.DataFrame,
                     pid_col: str = "pID",
                     team_col: str = "Home_Away",
                     role_col: str = "role") -> None:
    """
    Analyse clustering results against team sheet metadata.

    Parameters
    ----------
    cluster_labels : dict player_id -> cluster int (-1 = outlier)
    team_sheets_df : DataFrame with columns pID, Home_Away, role
    """
    label_df = pd.DataFrame(list(cluster_labels.items()),
                            columns=[pid_col, "cluster"])
    df = label_df.merge(team_sheets_df[[pid_col, team_col, role_col]],
                        on=pid_col, how="left")

    cluster_order = sorted(df["cluster"].unique())
    cluster_names = {c: "Outlier" if c == -1 else f"Cluster {c}"
                     for c in cluster_order}
    df["cluster_name"] = df["cluster"].map(cluster_names)

    sep = "=" * 50

    # Team breakdown per cluster
    print(f"\n{sep}")
    print("TEAM BREAKDOWN PER CLUSTER")
    print(sep)
    team_breakdown = (df.groupby(["cluster_name", team_col])
                        .size()
                        .unstack(fill_value=0))
    team_breakdown["Total"] = team_breakdown.sum(axis=1)
    print(team_breakdown.to_string())

    # Role breakdown per cluster
    print(f"\n{sep}")
    print("ROLE BREAKDOWN PER CLUSTER")
    print(sep)
    role_breakdown = (df.groupby(["cluster_name", role_col])
                        .size()
                        .unstack(fill_value=0))
    print(role_breakdown.to_string())

    # Role breakdown per cluster, split by team
    print(f"\n{sep}")
    print("ROLE BREAKDOWN PER CLUSTER BY TEAM")
    print(sep)
    for team, team_df in df.groupby(team_col):
        print(f"\n  {team}")
        print("  " + "-" * 30)
        role_team = (team_df.groupby(["cluster_name", role_col])
                            .size()
                            .unstack(fill_value=0))
        print(role_team.to_string())

    print(f"\n{sep}\n")

def flag_players(cluster_labels: dict[str, int],
                 team_sheets_df: pd.DataFrame,
                 handling_players: list[str],
                 pid_col: str = "pID",
                 team_col: str = "Home_Away",
                 role_col: str = "role",
                 tid_col: str = "tID",
                 small_cluster_threshold: float = 0.3) -> pd.DataFrame:

    label_df = pd.DataFrame(list(cluster_labels.items()),
                            columns=[pid_col, "cluster"])
    df = label_df.merge(team_sheets_df[[pid_col, tid_col, team_col, role_col]],
                        on=pid_col, how="left")
    cluster_names = {c: "Outlier" if c == -1 else f"Cluster {c}"
                     for c in sorted(df["cluster"].unique())}
    df["cluster_name"] = df["cluster"].map(cluster_names)

    sep = "=" * 50
    size_cutoff = len(df) * small_cluster_threshold
    team_breakdown = (df.groupby(["cluster_name", team_col])
                        .size().unstack(fill_value=0))
    team_breakdown["Total"] = team_breakdown.sum(axis=1)
    small_clusters = (team_breakdown[team_breakdown["Total"] <= size_cutoff]
                      .index.tolist())

    #print(sep)
    #print("FLAGGED PLAYERS")
    #print(sep)

    flagged_rows = []
    for cluster_name in small_clusters:
        cluster_df = df[df["cluster_name"] == cluster_name]

        if list(cluster_df[role_col].unique()) == ["Goalkeeper"]:
            #print(f"\n  {cluster_name}: skipped (goalkeepers only)")
            continue

        #print(f"\n  {cluster_name} ({len(cluster_df)} players):")
        for _, row in cluster_df.iterrows():
            is_handling = row[pid_col] in handling_players
            #print(f"    {row[pid_col]} — {row[team_col]}, {row[role_col]}"
                  #f"{' [handler]' if is_handling else ''}")
            flagged_rows.append({
                "player":      row[pid_col],
                "tID":         row[tid_col],
                "team":        row[team_col],
                "role":        row[role_col],
                "cluster":     row["cluster"],
                "is_handling": is_handling
            })

    #if not flagged_rows:
        #print("\n  No flagged players.")
    #else:
    if flagged_rows:
        pids = [r["player"] for r in flagged_rows]
        #print(f"\n  Flagged player list: {pids}")

    #print(f"\n{sep}\n")

    return pd.DataFrame(flagged_rows,
                        columns=["player", "tID", "team", "role", "cluster", "is_handling"])

from .idsse_import import GameData as GD
def trajectory_anom_per_possession(GameData: GD, 
                               possessionID: str,
                               D_algo: str = 'MULTIVARIATE',
                               multivariate_weights: np.ndarray = np.array([0.5, 0.5]),
                               multiscale_scales: list[int] = None,
                               MS_use_relative_angle: bool = True, 
                               plotting: bool = False,
                               breakdown_report: bool = False,
                               min_frames: int = 50,
                               target_frames: int = 500
                               ):
    GameData.assign_roles(process_DF='TEAMSHEET')
    df=GameData.events.query('`case:concept:name`==@possessionID').copy()
    df['attribute:frame']=df['attribute:frame'].astype(int)
    start_frame, end_frame, session=np.min(df['attribute:frame']),np.max(df['attribute:frame']),np.unique(df['attribute:session'])[0]
    if (end_frame - start_frame) < min_frames:
        print(f"  Possession {possessionID} too short ({end_frame - start_frame} frames), skipping.")
        empty_df = pd.DataFrame(columns=["player", "tID", "team", "role", "cluster", "is_handling"])
        return empty_df, {}, None
    ball=GameData.positions.query('`Frame`>=@start_frame & `Frame`<@end_frame & Session==@session')
    extract_ball=np.unique(ball['Player'])[0]
    ball=ball.query('Player==@extract_ball')[['ball_x','ball_y', 'Frame']]
    ball.rename(columns={'ball_x':'x','ball_y':'y'}, inplace=True)

    pass_sub=df.query('ball.notna()')
    pass_sub = pass_sub[pass_sub["concept:name"].str.endswith(("Pass", "Cross", "Received", "Intercepted"), na=False)]
    pass_sub = pass_sub[pass_sub['Player'].notna()]
    handling_players=np.unique(pass_sub['Player'])

    session_df=GameData.positions[GameData.positions["Session"] == session]
    session_df = session_df.query('Frame<@end_frame & Frame>=@start_frame')
    session_df = session_df.sort_values(["Player", "Frame"])

    players = {
        pid: grp.sort_values("Frame")[["x", "y"]].values
        for pid, grp in session_df.groupby("Player")
    }
    ball_ar = ball.sort_values("Frame")[["x", "y"]].values
    expected_frames = ball_ar.shape[0]
    players = {pid: pos for pid, pos in players.items()
            if len(pos) == expected_frames}

    T = ball_ar.shape[0]
    if T > target_frames:
        factor = max(1, T // target_frames)
        players = {pid: pos[::factor] for pid, pos in players.items()}
        ball_ar = ball_ar[::factor]
        print(f"  Downsampled {T} → {ball_ar.shape[0]} frames (factor={factor})")


    if D_algo=="MULTIVARIATE":
        ids, D = get_multivariate_distance_matrix(players, ball_ar, weights=multivariate_weights)
    elif D_algo=="MULTISCALE":    
        ids, D = get_multiscale_distance_matrix(players, ball_ar, scales=multiscale_scales, use_relative_angle=MS_use_relative_angle)
    D_norm = normalise_distance_matrix(D)
    labels_db = auto_cluster_dbscan(D_norm, ids, min_samples=1)
    team_map = session_df.drop_duplicates("Player").set_index("Player")["Team"].to_dict()
    if plotting:
        fig = plot_clusters(players, ball_ar, labels_db, team_map,
                            title="Clusters")
        plt.show()
    if breakdown_report:
        analyse_clusters(labels_db, GameData.team_sheets_df)
    flagged = flag_players(labels_db, GameData.team_sheets_df, handling_players)
    return flagged, labels_db, D_norm

def report_flagged_summary(all_flagged: pd.DataFrame,
                           pid_col: str = "player",
                           team_col: str = "team",
                           role_col: str = "role",
                           possession_col: str = "possessionID",
                           top_n: int = 3) -> None:

    report_flagged_player(all_flagged, pid_col, team_col, role_col, top_n)
    report_flagged_roles(all_flagged, team_col, role_col, top_n)
    report_flagged_possession(all_flagged, team_col, role_col, possession_col, top_n)

def report_flagged_player(all_flagged: pd.DataFrame,
                           pid_col: str = "player",
                           team_col: str = "team",
                           role_col: str = "role",
                           top_n: int = 3) -> None:

    sep = "=" * 50

    print(sep)
    print(f"TOP {top_n} MOST FLAGGED PLAYERS PER TEAM")
    print(sep)
    player_counts = (all_flagged.groupby([team_col, pid_col, role_col])
                                .size()
                                .reset_index(name="count")
                                .sort_values([team_col, "count"], ascending=[True, False]))
    for team, team_df in player_counts.groupby(team_col):
        print(f"\n  {team}")
        print("  " + "-" * 30)
        for _, row in team_df.head(top_n).iterrows():
            print(f"    {row[pid_col]} ({row[role_col]}): {row['count']} times")

    print(f"\n{sep}\n")

def report_flagged_roles(all_flagged: pd.DataFrame,
                           team_col: str = "team",
                           role_col: str = "role",
                           top_n: int = 3) -> None:

    sep = "=" * 50

    print(sep)
    print(f"TOP {top_n} MOST FLAGGED ROLES PER TEAM")
    print(sep)
    role_counts = (all_flagged.groupby([team_col, role_col])
                              .size()
                              .reset_index(name="count")
                              .sort_values([team_col, "count"], ascending=[True, False]))
    for team, team_df in role_counts.groupby(team_col):
        print(f"\n  {team}")
        print("  " + "-" * 30)
        for _, row in team_df.head(top_n).iterrows():
            print(f"    {row[role_col]}: {row['count']} times")

    print(f"\n{sep}")

def report_flagged_possession(all_flagged: pd.DataFrame,
                           team_col: str = "team",
                           role_col: str = "role",
                           possession_col: str = "possessionID",
                           top_n: int = 3) -> None:

    sep = "=" * 50

    print(sep)
    print(f"TOP {top_n} POSSESSIONS WITH MOST FLAGGED PLAYERS")
    print(sep)
    possession_counts = (all_flagged.groupby(possession_col)
                                    .size()
                                    .reset_index(name="count")
                                    .sort_values("count", ascending=False))
    for _, row in possession_counts.head(top_n).iterrows():
        pid = row[possession_col]
        print(f"\n  {pid}: {row['count']} flagged players")

        poss_df = all_flagged[all_flagged[possession_col] == pid]

        # team composition + role breakdown per team in one line each
        print(f"    Composition:")
        role_comp = (poss_df.groupby([team_col, role_col])
                            .size()
                            .reset_index(name="count")
                            .sort_values([team_col, "count"], ascending=[True, False]))
        for team, team_df in role_comp.groupby(team_col):
            total = team_df["count"].sum()
            roles_str = ", ".join(f"{r[role_col]}x{r['count']}" for _, r in team_df.iterrows())
            print(f"      {team} ({total}): {roles_str}")

    print(f"\n{sep}\n")

def trajectory_anom_full_game(GameData):
    GameData.format_log('ALL')
    GameData.events=GameData.events.sort_values(['attribute:session','attribute:frame'])
    possession_ids, counts=np.unique(GameData.events['case:concept:name'], return_counts=True)
    results = []
    for pid, c in zip(possession_ids, counts):
        if c>1:
            print(f'Processing: {pid}')
            flagged, labels, D = trajectory_anom_per_possession(GameData, pid, target_frames=125)
            if not flagged.empty:
                flagged["possessionID"] = pid
                results.append(flagged)
    flagged_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    return flagged_df