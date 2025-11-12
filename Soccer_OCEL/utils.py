import pandas as pd
import numpy as np
import math
import ast
def add_timestamps(df, frame0_time_half1, frame0_time_half2, frame_rate):
    df = df.copy()
    df["gameclock"] = df["Frame"] / frame_rate

    start_times = {
        1: frame0_time_half1,
        2: frame0_time_half2
    }

    df["timestamp"] = df.apply(
        lambda row: start_times.get(row["Session"], pd.NaT) 
                    + pd.to_timedelta(row["gameclock"], unit="s"),
        axis=1
    )

    return df#.drop(columns=["gameclock"])

def get_tID_from_pID(pID, team_sheets_df):
    """
    Get the team ID (tID) from a player ID (pID) using the team sheets DataFrame.
    """
    tID = team_sheets_df.loc[team_sheets_df['pID'] == pID, 'tID']
    if not tID.empty:
        return tID.values[0]
    else:
        return None
    

def get_teamside_from_pID(pID, team_sheets_df):
    """
    Get the team side (Home/Away) from a player ID (pID) using the team sheets DataFrame.
    """
    tID = team_sheets_df.loc[team_sheets_df['pID'] == pID, 'Home_Away']
    if not tID.empty:
        return tID.values[0]
    else:
        return None
    
def get_field_position(x, y,pitch, xy_fields=(10,10)):
    """
    Calculate the position of a field in a grid based on its coordinates.

    Example use on an event dataframe:
    >>> df = pd.DataFrame({'Start X': [0.1, 0.5, 0.9], 'Start Y': [0.2, 0.5, 0.8]})
    >>> df['field_position'] = df.apply(lambda row: get_field_position(row['Start X'], row['Start Y']), axis=1)

    Args:
        x (int): The x-coordinate of the field.
        y (int): The y-coordinate of the field.
        x_fields (int): The number of fields along the x-axis.
        y_fields (int): The number of fields along the y-axis.

    Returns:
        str: A position string in the format "A1", "B2", etc.
    """
    
    if x is None or y is None or pd.isna(x) or pd.isna(y):
        return None
    xblock = pitch.xlim[1]*2/ xy_fields[0]
    yblock = pitch.ylim[1]*2./ xy_fields[1]
    x_pos = math.floor(x / xblock)
    y_pos = math.floor(y / yblock)
    x_pos=np.clip(x_pos, 0, xy_fields[0] - 1)
    y_pos=np.clip(y_pos, 0, xy_fields[1] - 1)
    

    return (x_pos, y_pos)
def find_position(player_name, GD):
    row = GD.team_sheets_df.loc[GD.team_sheets_df["pID"] == player_name, "position"]
    return row.values[0] if len(row) > 0 else None

def parse_grid_position(pos):
    if isinstance(pos, str):
        return tuple(ast.literal_eval(pos))
    return pos
def calculate_teammate_pass_risk(df, ball_holder):
    df = df.copy()
    df['Grid Position'] = df['Grid Position'].apply(parse_grid_position)

    ball_team = df.loc[df['Player'] == ball_holder, 'Team'].iloc[0]
    ball_pos = np.array(df.loc[df['Player'] == ball_holder, 'Grid Position'].iloc[0])

    teammates = df[(df['Team'] == ball_team) & (df['Player'] != ball_holder)]
    opponents = df[df['Team'] != ball_team]

    def distance(a, b):
        return np.linalg.norm(a - b)

    scores = {}
    

    for _, t_row in teammates.iterrows():
        teammate_pos = np.array(t_row['Grid Position'])
        potential_interceptors=0
        dist_ball_teammate = distance(ball_pos, teammate_pos)

        valid_opponents = opponents[
            opponents['Grid Position'].apply(lambda p: distance(ball_pos, np.array(p)) < dist_ball_teammate)
        ]

        max_risk = 0
        bt_vec = teammate_pos - ball_pos
        bt_len = np.linalg.norm(bt_vec)

        if bt_len == 0:
            scores[t_row['Player']] = 0
            continue

        for _, o_row in valid_opponents.iterrows():
            opp_pos = np.array(o_row['Grid Position'])
            bo_vec = opp_pos - ball_pos
            if np.dot(bo_vec, bt_vec) <= 0:
                continue
            
            t = np.dot(opp_pos - ball_pos, bt_vec) / (bt_len ** 2)
            t = max(0, min(1, t))

            intersection = ball_pos + t * bt_vec
            dist_ball_intersection = distance(ball_pos, intersection)
            dist_opp_intersection = distance(opp_pos, intersection)

            if dist_opp_intersection > 0:
                if dist_ball_intersection/dist_opp_intersection >= 0.9:
                    potential_interceptors += 1
                max_risk= np.max([max_risk, dist_ball_intersection / dist_opp_intersection])
            

        scores[t_row['Player']] = max_risk*potential_interceptors*dist_ball_teammate

    return scores

# formatting
import pandas as pd

def formatting(GD):
    # Drop existing 'Player' column if it exists
    if 'Player' in GD.events.columns:
        GD.events = GD.events.drop(columns=['Player'])
    
    # Rename columns
    GD.events.rename(columns={
        'eID': 'concept:name',
        'pID': 'Player',
        'possessionID': 'case:concept:name',
        'timestamp': 'time:timestamp',
        'x': 'attribute:x',
        'y': 'attribute:y',
        'enabled':'attribute:enabled',
        'qualifier': 'attribute:qualifier',
        'gameclock': 'attribute:gameclock',
        'Session':'attribute:session',
        'Frame':'attribute:frame',
        'Team':'attribute:team',
        'outcome':'attribute:outcome',
        #'TeamLeft':'attribute:team_left',
        #'TeamRight':'attribute:team_right',
        'game':'attribute:game'
        # 'End X': 'attribute:end_x',
        # 'End Y': 'attribute:end_y'
    }, inplace=True)

    # Ensure columns exist before casting
    # df = GD.events
    # cast_columns = {
    #     'case:concept:name': str,
    #     'concept:name': str,
    #     #'time:timestamp': 'datetime64[ns]',
    #     'attribute:x': float,
    #     'attribute:y': float,
    #     'Player': str,
    #     'attribute:team': str,
    #     'attribute:session': str,
    #     'attribute:outcome': float,
    #     'attribute:frame': int,
    #     'attribute:gameclock': float,
    #     #'attribute:team_left': str,
    #     #'attribute:team_right': str,
    #     'attribute:game': str
    # }

    # for col, dtype in cast_columns.items():
    #     if col in df.columns:
    #         df[col] = df[col].astype(dtype)
    GD.events['time:timestamp'] = pd.to_datetime(GD.events['time:timestamp'], utc=True)

    return GD

def filter_players_involved_with_ball(df):
    """
    filtering out events where player was not involved in on ball action during the possession.
    """
    players_with_ball = (
        df.dropna(subset=["ball"])
        .groupby("case:concept:name")["Player"]
        .unique()
    )

    merged = df.merge(players_with_ball.rename("eligible_players"),
                      on="case:concept:name", how="left")

    mask = merged.apply(lambda r: r["Player"] in r["eligible_players"], axis=1)
    return merged[mask].drop(columns="eligible_players").reset_index(drop=True)
def idle_players_traces(df):
    """
    Activities of idel player.
    """
    players_with_ball = (
        df.dropna(subset=["ball"])
        .groupby("case:concept:name")["Player"]
        .unique()
    )

    merged = df.merge(players_with_ball.rename("eligible_players"),
                      on="case:concept:name", how="left")

    mask = merged.apply(lambda r: r["Player"] not in r["eligible_players"], axis=1)
    return merged[mask].drop(columns="eligible_players").reset_index(drop=True)

def add_pass_cross_sequences(df) :
 
    df = df.copy()
    mask_action = df['concept:name'].str.contains(r'(pass|cross)', case=False, na=False)
    mask_excluded = df['concept:name'].str.contains(r'(received|intercepted)', case=False, na=False)

    mask_main = mask_action & ~mask_excluded
    mask_secondary = mask_action & mask_excluded

    df.loc[mask_main, 'seq_main'] = (
        df.loc[mask_main]
        .groupby('case:concept:name')
        .cumcount() + 1
    )

    df.loc[mask_secondary, 'seq_recv'] = (
        df.loc[mask_secondary]
        .groupby('case:concept:name')
        .cumcount() + 1
    )

    df.loc[mask_main, 'concept:name'] = (
        df.loc[mask_main, 'concept:name'] + '_' + df.loc[mask_main, 'seq_main'].astype(int).astype(str)
    )
    df.loc[mask_secondary, 'concept:name'] = (
        df.loc[mask_secondary, 'concept:name'] + '_r' + df.loc[mask_secondary, 'seq_recv'].astype(int).astype(str)
    )

    df = df.drop(columns=['seq_main', 'seq_recv'], errors='ignore').reset_index(drop=True)
    return df
