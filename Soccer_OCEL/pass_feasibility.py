"""
Pass Feasibility Map — Python implementation
Based on: "Towards soccer pass feasibility maps: the role of players' orientation"
Arbués-Sangüesa et al., Journal of Sports Sciences, 2021.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
from . import utils
from .idsse_import import GameData
FIELD_LENGTH: float = 105.0
FIELD_WIDTH:  float = 68.0


SIGMA2_R: float = 1e3  
SIGMA2_P: float = 1e4  
SIGMA_A: float = 0.75
SIGMA0_D: float = 12.5
KAPPA: float = 6.0
RHO: float = FIELD_LENGTH / 20.0

@dataclass
class Player:
    """
    A single player on the pitch.

    Parameters
    ----------
    position : (x, y) in metres
    orientation : angle in radians (0 = facing right / east)
    role : one of 'goalkeeper', 'central_def', 'fullback', 'midfielder', 'forward'
    speed : metres per second (optional; used in ablation)
    name : display label (optional)
    """
    position: np.ndarray        # shape (2,)
    orientation: float          # radians
    speed: float = 0.0          # m/s
    name: str = ""
    role: str = ""

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=float)


@dataclass
class PassEvent:
    """One pass event snapshot."""
    event: str
    passer: Player
    receivers: list[Player]     # all offensive teammates (excluding passer)
    defenders: list[Player]     # all defending players
    ball_position: np.ndarray   # usually == passer.position at kick moment

    def __post_init__(self):
        self.ball_position = np.asarray(self.ball_position, dtype=float)


def _estimate_orientation_and_speed(
    tracking_df: pd.DataFrame,
    session: object,
    player_id: object,
    frame: int,
    fps: float,
    window: int = 5,
) -> tuple[float, float]:
    """
    Estimate a player's body orientation (radians) and speed (m/s) from the
    velocity vector computed over a small temporal window around `frame`.

    Parameters
    ----------
    tracking_df : full tracking dataframe
    session     : session identifier to filter on
    player_id   : Player column value
    frame       : the pass frame
    fps         : frames per second of the tracking data
    window      : half-window in frames for the velocity estimate

    Returns
    -------
    orientation : float, radians (0 = east / right)
    speed       : float, metres per second
    """
    mask = (
        (tracking_df["Session"] == session) &
        (tracking_df["Player"]  == player_id)
    )
    player_frames = tracking_df.loc[mask].set_index("Frame").sort_index()

    frames_present = player_frames.index.tolist()
    if not frames_present:
        return 0.0, 0.0

    before_frames = [f for f in frames_present if f <= frame]
    after_frames  = [f for f in frames_present if f >= frame]

    f_before = None
    f_after  = None
    for w in range(window, 0, -1):
        candidates_before = [f for f in before_frames if f <= frame - w]
        candidates_after  = [f for f in after_frames  if f >= frame + w]
        if candidates_before and candidates_after:
            f_before = max(candidates_before)
            f_after  = min(candidates_after)
            break

    if f_before is None and before_frames:
        f_before = max(before_frames)
    if f_after is None and after_frames:
        f_after = min(after_frames)

    if f_before is None or f_after is None or f_before == f_after:
        return 0.0, 0.0

    dx = player_frames.loc[f_after,  "x"] - player_frames.loc[f_before, "x"]
    dy = player_frames.loc[f_after,  "y"] - player_frames.loc[f_before, "y"]
    dt = (f_after - f_before) / fps  

    speed       = float(np.sqrt(dx**2 + dy**2) / dt) if dt > 0 else 0.0
    orientation = float(np.arctan2(dy, dx)) if (dx != 0 or dy != 0) else 0.0

    return orientation, speed

def build_pass_event(
    eID:                  str,
    tracking_df:          pd.DataFrame,
    team_sheets_df:       pd.DataFrame,
    session:              int,
    frame:                int,
    attacking_team:       str,
    player_in_possession: str,
    fps:                  float = 25.0,
    orientation_window:   int   = 5,
) -> PassEvent:
    """
    Build a PassEvent from your tracking and team-sheet dataframes.

    Parameters
    ----------
    tracking_df           : full tracking dataframe (all sessions, all players)
    team_sheets_df        : team-sheet dataframe with pID and position columns
    session               : session identifier (used to filter tracking_df)
    frame                 : frame number at the moment of the pass
    attacking_team        : Team value of the team in possession
    player_in_possession  : Player value of the player with the ball (the passer)
    fps                   : frames per second of your tracking data (default 25)
    orientation_window    : half-window (frames) for velocity estimation (default 5)

    Returns
    -------
    PassEvent ready to pass into feasibility_map()
    """
    frame_df = tracking_df[
        (tracking_df["Session"] == session) &
        (tracking_df["Frame"]   == frame)
    ].copy()

    if frame_df.empty:
        raise ValueError(f"No tracking data found for Session={session}, Frame={frame}")

    role_lookup = (
        team_sheets_df[["pID", "position"]]
        .drop_duplicates("pID")
        .set_index("pID")["position"]
        .to_dict()
    )

    passer_row = frame_df[frame_df["Player"] == player_in_possession]
    if passer_row.empty:
        raise ValueError(
            f"Player {player_in_possession!r} not found in frame {frame} "
            f"of session {session}."
        )

    ball_pos = np.array([
        float(passer_row["ball_x"].iloc[0]),
        float(passer_row["ball_y"].iloc[0]),
    ])

    def _make_player(row: pd.Series) -> Player:
        pid = row["Player"]

        orientation, speed = _estimate_orientation_and_speed(
            tracking_df, session, pid, frame, fps, orientation_window
        )

        return Player(
            position    = np.array([float(row["x"]), float(row["y"])]),
            orientation = orientation,
            speed       = speed,
            role        = role_lookup.get(pid, ""),
            name        = str(pid),
        )

    attacking_frame = frame_df[frame_df["Team"] == attacking_team]
    defending_frame = frame_df[frame_df["Team"] != attacking_team]

    # passer = _make_player(
    #     attacking_frame[attacking_frame["Player"] == player_in_possession].iloc[0]
    # )
    passer_match = attacking_frame[attacking_frame["Player"] == player_in_possession]
    if passer_match.empty:
        raise ValueError(
            f"Player {player_in_possession!r} not found in attacking team {attacking_team!r} "
            f"at Session={session}, Frame={frame}. "
            f"Players present: {attacking_frame['Player'].tolist()}"
        )
    passer = _make_player(passer_match.iloc[0])

    receivers = [
        _make_player(row)
        for _, row in attacking_frame.iterrows()
        if row["Player"] != player_in_possession
    ]

    defenders = [
        _make_player(row)
        for _, row in defending_frame.iterrows()
    ]

    return PassEvent(
        event         = eID,
        passer        = passer,
        receivers     = receivers,
        defenders     = defenders,
        ball_position = ball_pos,
    )





def make_grid(
    field_length: float = FIELD_LENGTH,
    field_width:  float = FIELD_WIDTH,
    resolution:   float = 1.0,          # metres per cell
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (X, Y) meshgrids covering the pitch.
    Shape: (n_rows, n_cols) where rows ↔ y, cols ↔ x.
    """
    xs = np.arange(0, field_length + resolution, resolution)
    ys = np.arange(0, field_width  + resolution, resolution)
    return np.meshgrid(xs, ys)


def angle_to_point(x: np.ndarray, y: np.ndarray,
                   ref: np.ndarray) -> np.ndarray:
    """
    Angle from ref to every grid point (x, y), in radians.
    ∠(x − ref) in the paper's notation.
    """
    return np.arctan2(y - ref[1], x - ref[0])


def angle_diff(a: np.ndarray, b: float) -> np.ndarray:
    """
    Signed angular difference (a − b) wrapped to (−π, π].
    Used for the orientation terms in Eq. 3 & 5.
    """
    diff = (a - b) % (2 * np.pi)
    diff[diff > np.pi] -= 2 * np.pi
    return diff



# Offensive maps  (Section III-A)
def receiver_map_single(
    X: np.ndarray, Y: np.ndarray,
    receiver: Player,
    sigma2_R: float = SIGMA2_R,
    sigma_a:  float = SIGMA_A,
) -> np.ndarray:
    """
    Single-receiver contribution M_Rj (Eq. 3).

    M_Rj(x) = exp(−‖x − r_j‖² / σ²_R)   [location prior g_Rj]
             × exp(−(∠(x−r_j) − α_Rj)²  / σ²_a)  [orientation prior g^a_Rj]
    """
    rj = receiver.position
    alpha_Rj = receiver.orientation

    # Location prior: distance² from receiver
    dist2 = (X - rj[0])**2 + (Y - rj[1])**2
    g_R = np.exp(-dist2 / sigma2_R)

    # Orientation prior: angular difference between grid direction and receiver facing
    phi = angle_to_point(X, Y, rj)
    diff = angle_diff(phi, alpha_Rj)
    g_a = np.exp(-(diff**2) / (sigma_a**2))

    return g_R * g_a


def receiver_map(
    X: np.ndarray, Y: np.ndarray,
    receivers: list[Player],
    sigma2_R: float = SIGMA2_R,
    sigma_a:  float = SIGMA_A,
) -> np.ndarray:
    """
    Aggregate receiver map M_R (Eq. 4): sum over all N_R receivers.
    """
    MR = np.zeros_like(X, dtype=float)
    for r in receivers:
        MR += receiver_map_single(X, Y, r, sigma2_R, sigma_a)
    return MR


def passer_map(
    X: np.ndarray, Y: np.ndarray,
    passer: Player,
    sigma2_P: float = SIGMA2_P,
    sigma_a:  float = SIGMA_A,
) -> np.ndarray:
    """
    Passer map M_P (Eq. 5).

    M_P(x) = exp(−‖x − p‖² / σ²_P)   [location prior g_P]
            × exp(−(∠(x−p) − α_P)²  / σ²_a)  [orientation prior g^a_P]
    """
    p       = passer.position
    alpha_P = passer.orientation

    dist2 = (X - p[0])**2 + (Y - p[1])**2
    g_P = np.exp(-dist2 / sigma2_P)

    phi  = angle_to_point(X, Y, p)
    diff = angle_diff(phi, alpha_P)
    g_aP = np.exp(-(diff**2) / (sigma_a**2))

    return g_P * g_aP


def offensive_map(
    X: np.ndarray, Y: np.ndarray,
    event: PassEvent,
    sigma2_R: float = SIGMA2_R,
    sigma2_P: float = SIGMA2_P,
    sigma_a:  float = SIGMA_A,
) -> np.ndarray:
    """
    Offensive map M_O (Eq. 2):  M_O(x) = M_P(x) × M_R(x)
    """
    MR = receiver_map(X, Y, event.receivers, sigma2_R, sigma_a)
    MP = passer_map(X, Y, event.passer, sigma2_P, sigma_a)
    return MP * MR


# Defensive map  (Section III-B)
def rotation_matrix(beta: float) -> np.ndarray:
    """2×2 rotation matrix R_β."""
    c, s = np.cos(beta), np.sin(beta)
    return np.array([[c, s], [-s, c]])


def _sigma_Di(defender: Player, ball: np.ndarray, sigma0_D: float) -> float:
    """
    Defender influence scale σ_Di (Eq. 8):
    σ_Di = ‖d_i − p‖ / σ⁰_D
    Scales with ball–defender distance so close defenders have small areas.
    """
    dist = np.linalg.norm(defender.position - ball)
    return max(dist / sigma0_D, 0.1)   # small floor to avoid division by zero


def defensive_map_single(
    X: np.ndarray, Y: np.ndarray,
    defender: Player,
    ball: np.ndarray,
    sigma0_D: float = SIGMA0_D,
) -> np.ndarray:
    """
    Single defender contribution M_Di (Eq. 6).

    After rotating the displacement (x − d_i) by β_i:
      • front side  (rotated_x ≥ 0): elongated ellipse
      • back side   (rotated_x < 0): isotropic circle (halved x-range)

    Returns *negative* values (defensive influence subtracts from feasibility).
    """
    di   = defender.position
    beta = defender.orientation
    R    = rotation_matrix(beta)

    sigma_Di = _sigma_Di(defender, ball, sigma0_D)

    # Rotated displacement for every grid point
    dx = X - di[0]
    dy = Y - di[1]

    # R @ [dx, dy] for each grid point simultaneously
    rx = R[0, 0] * dx + R[0, 1] * dy   # (R_β (x − d_i))_x
    ry = R[1, 0] * dx + R[1, 1] * dy   # (R_β (x − d_i))_y

    MDi = np.empty_like(X, dtype=float)

    # Back side: rx < 0  →  isotropic  (slower backward recovery)
    back  = rx < 0
    MDi[back]  = -np.exp(-(rx[back]**2 + ry[back]**2) / sigma_Di**2)

    # Front side: rx ≥ 0  →  halved x² so the ellipse stretches forward
    front = ~back
    MDi[front] = -np.exp(-(0.5 * rx[front]**2 + ry[front]**2) / sigma_Di**2)

    return MDi


def defensive_map(
    X: np.ndarray, Y: np.ndarray,
    event: PassEvent,
    sigma0_D: float = SIGMA0_D,
) -> np.ndarray:
    """
    Aggregate defensive map M_D (Eq. 7): sum over all N_D defenders.
    """
    MD = np.zeros_like(X, dtype=float)
    for d in event.defenders:
        MD += defensive_map_single(X, Y, d, event.ball_position, sigma0_D)
    return MD



# Speed extension (Section IV-B-2, ablation)
MEDIAN_RECEIVER_SPEED: float = 1.57   # m/s (from paper)
MEDIAN_DEFENDER_SPEED: float = 1.86   # m/s (from paper)

def receiver_map_with_speed(
    X: np.ndarray, Y: np.ndarray,
    receivers: list[Player],
    sigma2_R: float = SIGMA2_R,
    sigma_a:  float = SIGMA_A,
    median_speed: float = MEDIAN_RECEIVER_SPEED,
) -> np.ndarray:
    """
    Receiver map with speed modulation (Section IV-B-2).
    A faster-moving receiver has a wider effective orientation cone:
        ν_r = s_j / s̄_rec   →  σ²_a replaced by σ²_a · ν_r
    """
    MR = np.zeros_like(X, dtype=float)
    for r in receivers:
        nu_r = r.speed / median_speed if median_speed > 0 else 1.0
        # Avoid near-zero denominator
        nu_r = max(nu_r, 0.01)
        sigma_a_eff = sigma_a * np.sqrt(nu_r)    # wider cone for faster players

        rj = r.position
        dist2 = (X - rj[0])**2 + (Y - rj[1])**2
        g_R = np.exp(-dist2 / sigma2_R)

        phi  = angle_to_point(X, Y, rj)
        diff = angle_diff(phi, r.orientation)
        g_a  = np.exp(-(diff**2) / (sigma_a_eff**2))

        MR += g_R * g_a
    return MR


def defensive_map_with_speed(
    X: np.ndarray, Y: np.ndarray,
    event: PassEvent,
    sigma0_D: float = SIGMA0_D,
    median_speed: float = MEDIAN_DEFENDER_SPEED,
) -> np.ndarray:
    """
    Defensive map with speed modulation (Section IV-B-2).
    ν_d = max(s_d / s̄_def, 1) expands the front-side ellipse for fast defenders.
    """
    MD = np.zeros_like(X, dtype=float)
    for d in event.defenders:
        sigma_Di = _sigma_Di(d, event.ball_position, sigma0_D)
        beta     = d.orientation
        R        = rotation_matrix(beta)
        nu_d     = max(d.speed / median_speed, 1.0)

        dx = X - d.position[0]
        dy = Y - d.position[1]
        rx = R[0, 0] * dx + R[0, 1] * dy
        ry = R[1, 0] * dx + R[1, 1] * dy

        MDi = np.empty_like(X, dtype=float)

        back  = rx < 0
        MDi[back]  = -np.exp(-(rx[back]**2 + ry[back]**2) / sigma_Di**2)

        front = ~back
        # ν_d scales the front-side ellipse — faster defenders cover more ground ahead
        MDi[front] = -np.exp(-(0.5 * nu_d * rx[front]**2 + ry[front]**2) / sigma_Di**2)

        MD += MDi
    return MD

# Combined feasibility map  (Section III, Eq. 1)
def feasibility_map(
    X: np.ndarray, Y: np.ndarray,
    event: PassEvent,
    kappa:    float = KAPPA,
    sigma2_R: float = SIGMA2_R,
    sigma2_P: float = SIGMA2_P,
    sigma_a:  float = SIGMA_A,
    sigma0_D: float = SIGMA0_D,
    speed: bool = False,
    median_receiver_speed: float = MEDIAN_RECEIVER_SPEED,
    median_defender_speed: float = MEDIAN_DEFENDER_SPEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Full pass feasibility map M (Eq. 1):
        M(x) = κ · M_O(x) + M_D(x)

    Returns
    -------
    M  : combined feasibility map
    MO : offensive component
    MD : defensive component
    """
    if speed:
        MR = receiver_map_with_speed(
            X, Y, event.receivers, sigma2_R, sigma_a, median_receiver_speed
        )
        MP = passer_map(X, Y, event.passer, sigma2_P, sigma_a)
        MO = MP * MR
        MD = defensive_map_with_speed(X, Y, event, sigma0_D, median_defender_speed)
    else:        
        MO = offensive_map(X, Y, event, sigma2_R, sigma2_P, sigma_a)
        MD = defensive_map(X, Y, event, sigma0_D)
    M  = kappa * MO + MD
    return M, MO, MD


def compute_feasibility_for_frame(
    tracking_df:          pd.DataFrame,
    team_sheets_df:       pd.DataFrame,
    session:              int,
    frame:                int,
    attacking_team:       str,
    player_in_possession: str,
    eID:                  str = "", 
    fps:                  float = 25.0,
    orientation_window:   int   = 5,
    speed:                bool  = False,
    resolution:           float = 1.0,
    **map_kwargs,
) -> dict:
    """
    One-shot helper: build the event, run the feasibility map, rank receivers.

    Parameters
    ----------
    speed      : passed through to feasibility_map(speed=) — True enables
                 the speed-modulated variant (Section IV-B-2 ablation)
    resolution : grid resolution in metres (default 1 m)
    **map_kwargs : forwarded to feasibility_map()
                  (e.g. kappa=4, sigma0_D=10, sigma2_R=800)

    Returns
    -------
    dict with keys:
        'event'   : PassEvent
        'M'       : combined feasibility map (2-D array)
        'MO'      : offensive map
        'MD'      : defensive map
        'X', 'Y'  : meshgrid arrays
        'ranked'  : list of (score, Player) sorted best-first
        'scores'  : raw disk scores in receiver order
    """

    event = build_pass_event(
        eID, tracking_df, team_sheets_df,
        session, frame, attacking_team, player_in_possession,
        fps, orientation_window,
    )

    X, Y = make_grid(resolution=resolution)
    M, MO, MD = feasibility_map(X, Y, event, speed=speed, **map_kwargs)

    scores = evaluate_disk(M, X, Y, event.receivers)
    ranked = rank_receivers(scores, event.receivers)

    return {
        "event":  event,
        "M":  M, "MO": MO, "MD": MD,
        "X":  X, "Y":  Y,
        "ranked": ranked,
        "scores": scores,
    }

def evaluate_disk(
    M: np.ndarray, X: np.ndarray, Y: np.ndarray,
    receivers: list[Player],
    rho: float = RHO,
) -> list[float]:
    """
    Disk evaluation V1 (Eq. 9): average M over a disk of radius ρ around each receiver.
    Returns one feasibility score per receiver, in the same order.
    """
    scores = []
    for r in receivers:
        mask = (X - r.position[0])**2 + (Y - r.position[1])**2 <= rho**2
        if mask.any():
            scores.append(float(M[mask].mean()))
        else:
            scores.append(0.0)
    return scores


def rank_receivers(scores: list[float], receivers: list[Player]) -> list[tuple[float, Player]]:
    """Sort receivers by descending feasibility score."""
    ranked = sorted(zip(scores, receivers), key=lambda x: -x[0])
    return ranked


# Visualization
def _field_background(ax: plt.Axes,
                      field_length: float = FIELD_LENGTH,
                      field_width:  float = FIELD_WIDTH) -> None:
    """Draw a minimal pitch outline."""
    ax.set_facecolor("#3a7d44")   # grass green
    lw, lc = 1.2, "white"

    # Outer boundary
    ax.plot([0, field_length, field_length, 0, 0],
            [0, 0, field_width, field_width, 0], lw=lw, c=lc)

    # Centre line & circle
    ax.axvline(field_length / 2, lw=lw, c=lc)
    centre = plt.Circle((field_length / 2, field_width / 2), 9.15,
                        fill=False, lw=lw, color=lc)
    ax.add_patch(centre)

    # Penalty areas (simplified)
    for x_off in [0, field_length - 16.5]:
        rect = plt.Rectangle((x_off, (field_width - 40.3) / 2),
                              16.5, 40.3,
                              fill=False, lw=lw, edgecolor=lc)
        ax.add_patch(rect)

    ax.set_xlim(0, field_length)
    ax.set_ylim(0, field_width)
    ax.set_aspect("equal")
    ax.axis("off")


def _draw_orientation_arrow(ax: plt.Axes, player: Player, color: str,
                            arrow_len: float = 3.0) -> None:
    ax.annotate(
        "", xy=(player.position[0] + arrow_len * np.cos(player.orientation),
                player.position[1] + arrow_len * np.sin(player.orientation)),
        xytext=player.position,
        arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
    )


def plot_feasibility_map(
    event: PassEvent,
    M:  np.ndarray,
    MO: np.ndarray,
    MD: np.ndarray,
    X:  np.ndarray,
    Y:  np.ndarray,
    title: str = "Pass Feasibility Map",
    field_length: float = FIELD_LENGTH,
    field_width:  float = FIELD_WIDTH,
) -> plt.Figure:
    """
    Reproduce Figure 1 from the paper: offensive, defensive, and combined maps
    side-by-side, with player positions and orientations overlaid.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor("#1a1a2e")

    maps  = [MO, MD, M]
    names = ["Offensive map $M_O$", "Defensive map $M_D$",
             "Feasibility map $M$"]
    # Yellow = safe/high, blue = dangerous/low (matches paper colour scheme)
    cmaps = ["YlOrBr", "Blues_r", "RdYlGn"]

    for ax, data, name, cmap in zip(axes, maps, names, cmaps):
        _field_background(ax, field_length, field_width)

        # Normalise for display
        vmin, vmax = data.min(), data.max()
        if vmax > vmin:
            normed = (data - vmin) / (vmax - vmin)
        else:
            normed = np.zeros_like(data)

        ax.pcolormesh(X, Y, normed, cmap=cmap, alpha=0.75, shading="auto")

        # Passer  (red dot + arrow)
        p = event.passer
        ax.scatter(*p.position, c="red", s=80, zorder=5)
        _draw_orientation_arrow(ax, p, "red")
        if p.name:
            ax.text(p.position[0], p.position[1] + 2, p.name,
                    color="red", fontsize=7, ha="center")

        # Receivers (yellow dots + arrows)
        for r in event.receivers:
            ax.scatter(*r.position, c="yellow", s=60, zorder=5)
            _draw_orientation_arrow(ax, r, "yellow")
            if r.name:
                ax.text(r.position[0], r.position[1] + 2, r.name,
                        color="yellow", fontsize=7, ha="center")

        # Defenders (blue dots + arrows)
        for d in event.defenders:
            ax.scatter(*d.position, c="dodgerblue", s=60, zorder=5)
            _draw_orientation_arrow(ax, d, "dodgerblue")

        ax.set_title(name, color="white", fontsize=11, pad=6)

    fig.suptitle(title, color="white", fontsize=13, y=1.01)
    plt.tight_layout()
    return fig


def plot_component_breakdown(
    event: PassEvent,
    MO: np.ndarray,
    MD: np.ndarray,
    M:  np.ndarray,
    X:  np.ndarray,
    Y:  np.ndarray,
    ranked: list[tuple[float, Player]],
    field_length: float = FIELD_LENGTH,
    field_width:  float = FIELD_WIDTH,
) -> plt.Figure:
    """
    Extra diagnostic plot showing individual receiver contributions
    and the final ranking with disk scores.
    """
    n_recv = len(event.receivers)
    n_cols = min(n_recv + 1, 5)
    n_rows = (n_recv + 1 + n_cols - 1) // n_cols + 1   # receiver maps + combined

    fig, axes = plt.subplots(1 + 1, n_cols, figsize=(4 * n_cols, 9))
    fig.patch.set_facecolor("#1a1a2e")

    # Row 0: individual receiver maps
    for idx, r in enumerate(event.receivers):
        if idx >= n_cols:
            break
        ax  = axes[0, idx]
        MRj = receiver_map_single(X, Y, r)
        _field_background(ax, field_length, field_width)
        ax.pcolormesh(X, Y, MRj, cmap="YlOrBr", alpha=0.75, shading="auto")
        ax.scatter(*r.position, c="yellow", s=80, zorder=5)
        _draw_orientation_arrow(ax, r, "yellow")
        ax.scatter(*event.passer.position, c="red", s=60, zorder=5)
        label = r.name or f"R{idx+1}"
        ax.set_title(f"$M_{{R,{label}}}$", color="white", fontsize=10)

    for idx in range(len(event.receivers), n_cols):
        axes[0, idx].axis("off")

    # Row 1: combined map + ranking bar chart
    ax_main = axes[1, 0]
    _field_background(ax_main, field_length, field_width)
    vmin, vmax = M.min(), M.max()
    normed = (M - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(M)
    ax_main.pcolormesh(X, Y, normed, cmap="RdYlGn", alpha=0.75, shading="auto")
    ax_main.scatter(*event.passer.position, c="red", s=80, zorder=5)
    _draw_orientation_arrow(ax_main, event.passer, "red")
    for r in event.receivers:
        ax_main.scatter(*r.position, c="yellow", s=60, zorder=5)
        _draw_orientation_arrow(ax_main, r, "yellow")
    for d in event.defenders:
        ax_main.scatter(*d.position, c="dodgerblue", s=60, zorder=5)
    ax_main.set_title("$M$ (combined)", color="white", fontsize=10)

    # Ranking bar chart
    ax_bar = axes[1, 1]
    labels = [(r.name or f"R{i+1}") for i, (_, r) in enumerate(ranked)]
    scores = [s for s, _ in ranked]
    colors = ["gold" if i == 0 else "steelblue" for i in range(len(scores))]
    bars = ax_bar.barh(labels[::-1], scores[::-1], color=colors[::-1])
    ax_bar.set_facecolor("#1a1a2e")
    ax_bar.tick_params(colors="white")
    ax_bar.spines[:].set_color("#444")
    ax_bar.set_xlabel("Disk score", color="white")
    ax_bar.set_title("Receiver ranking", color="white", fontsize=10)
    ax_bar.bar_label(bars, fmt="%.3f", color="white", padding=3)
    for idx in range(2, n_cols):
        axes[1, idx].axis("off")

    plt.tight_layout()
    return fig



def get_k_enabled_passes(
    result: dict,
    k: int = 3,
    encode: str = 'role',
    prefix: str = 'Play_Pass',
    arg: dict = {
    'pitch':None,
    'xy_fields': (10, 10),
    'field_gdf':None
    },
) -> list:
    """
    Return a list of k label strings for the top-k feasible receivers.

    Parameters
    ----------
    result : dict returned by compute_feasibility_for_frame()
    k      : number of top receivers to include
    encode : which attribute to use 
             'role' → player.role 
             'name' → player.name 
             'zone' → field_gdf zone
             'tuple' → field xy tuple
             'label' → field label from xy_fields
    prefix : string prepended to each label
    para : dict of required inputs for
               label: GameData.pitch & GameDAta.settings['xy_fields']
               zone: GameData.field_gdf
               tuple: GameData.pitch
    """
    best_k = result['ranked'][:k]  

    if encode == 'role':
        return [f"{prefix}_{player.role}" for _, player in best_k]
    elif encode == 'name':
        return [f"{prefix}_{player.name}" for _, player in best_k]
    elif encode == 'tuple':
        if arg['pitch'] is None:
            raise ValueError("pitch must be provided when encode='tuple'.")
        labels = []
        for _, player in best_k:
            col, row = utils.get_field_position(player.position[0], player.position[1], arg['pitch'], arg['xy_fields'])
            labels.append(f"{prefix}_Grid_{col}_{row}")
        return labels
    elif encode == 'label':
        if arg['pitch'] is None:
            raise ValueError("pitch must be provided when encode='label'.")
        def _to_label(col, row):
            return f"{chr(ord('A') + col)}{row + 1}"
        return [
            f"{prefix}_{_to_label(*utils.get_field_position(player.position[0], player.position[1], arg['pitch'], arg['xy_fields']))}"
            for _, player in best_k
        ]

    elif encode == 'zone':
        if arg['field_gdf'] is None:
            raise ValueError("field_gdf must be provided when encode='zone'.")
        def _to_zone(x, y):
            from shapely.geometry import Point
            point = Point(x, y)
            zones = arg['field_gdf'][
                (arg['field_gdf']["name"] != "field") &
                arg['field_gdf'].geometry.contains(point)
            ]["name"].tolist()
            return ", ".join(zones) if zones else "open_play"
        return [
            f"{prefix}_{_to_zone(player.position[0], player.position[1])}"
            for _, player in best_k
        ]

    else:
        raise ValueError(
            f"encode must be 'role', 'name', 'tuple', 'label', or 'zone'. "
            f"Got {encode!r}."
        )
    
def _log_feasible_passes(
    GD: GameData, 
    speed: bool = False, 
    ) -> GameData:
    """
    Return a event log with feasibility analysis.
    """
    if 'feasible' not in GD.events.columns:
        GD.events['feasible'] = [{} for _ in range(len(GD.events))]

    mask = GD.events['eID'].str.contains('Shot|Pass|Cross', case=False, regex=True) & \
       ~GD.events['eID'].str.contains('Received|Intercepted|Saved|Blocked|Wide|WoodWork', case=False, regex=True)

    for idx in GD.events.index[mask]:
        e,s,f,p,t=GD.events.loc[idx][['eID','Session','Frame', 'pID', 'Team']]
        try:
            result = compute_feasibility_for_frame(
                tracking_df          = GD.positions,
                team_sheets_df       = GD.team_sheets_df,
                eID                  = e,
                session              = s,
                frame                = f,
                attacking_team       = t,
                player_in_possession = p,
                fps                  = float(GD.framerate),
                speed                = speed,
            )
            GD.events.at[idx, 'feasible'].update(result)
        except ValueError as e_err:
            print(f"Skipping event {e} (idx={idx}): {e_err}")
            continue

    return GD

def _log_enabled(
    GD: GameData, 
    k: int = 3, 
    encode: list = ['role', 'name', 'zone', 'typle', 'label']
    ) -> GameData:
    """
    Return a event log with enabled pass events.

    Parameters
    ----------
    GameData 
    k      : number of top receivers to include
    encode : which Player attribute to use as the label
             'role' → player.role 
             'name' → player.name 
             'zone' → field_gdf zone
             'tuple' → field xy tuple
             'label' → field label from xy_fields
    """
    for t in encode:
        if 'enabled_'+t not in GD.events.columns:
            GD.events['enabled_'+t] = [[] for _ in range(len(GD.events))]

    mask = GD.events['eID'].str.contains('Shot|Pass|Cross', case=False, regex=True) & \
       ~GD.events['eID'].str.contains('Received|Intercepted|Saved|Blocked|Wide|WoodWork', case=False, regex=True)
    arg= {
    'pitch':GD.pitch,
    'xy_fields': GD.settings['xy_fields'],
    'field_gdf': GD.field_gdf
    }
    

    for idx in GD.events.index[mask]:
        result=GD.events.loc[idx]['feasible']
        for t in encode:
            try:
                enabled = get_k_enabled_passes(result, k=k, encode=t, prefix=result['event'].event, arg=arg)
                GD.events.at[idx, 'enabled_'+t] += enabled
            except ValueError as e_err:
                print(f"Skipping event {result['event']} (idx={idx}): {e_err}")
                continue

    return GD