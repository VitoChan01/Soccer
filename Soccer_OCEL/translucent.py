from . import utils
import pandas as pd
import numpy as np
def best_pass(current_player, Frame, Session, recipient, GD):
    bp=False
    passes=[]
    if not pd.isna(current_player):
        df=GD.positions.query('Frame == @Frame & Session == @Session')
        teammate_scores = utils.calculate_teammate_pass_risk(df, current_player)
        min_value = min(teammate_scores.values())
        min_keys = [k for k, v in teammate_scores.items() if v == min_value]
        if recipient:
            if recipient in min_keys:
                min_keys.remove(recipient)
                bp=True
        passes=list(set([utils.find_position(p, GD) for p in min_keys]))
        passes=['Play_Pass_'+t for t in passes]
    return passes, bp

# def add_pass_enabled(GD):
#     GD.events['enabled'] = [[] for _ in range(len(GD.events))]

#     mask = GD.events['eID'].str.contains('Shot|Pass|Cross', case=False, regex=True)

#     def add_best_pass(row):
#         if mask.loc[row.name]:
#             best = best_pass(
#                 row['pID'], row['Frame'], row['Session'],
#                 recipient=row.get('Recipient', None),
#                 GD=GD
#             )
#             row['enabled'] = row['enabled'] + best
#         return row

#     GD.events = GD.events.apply(add_best_pass, axis=1)
#     return GD
def add_pass_enabled(GD):
    if 'enabled' not in GD.events.columns:
        GD.events['enabled'] = [[] for _ in range(len(GD.events))]
    if 'best_pass' not in GD.events.columns:
        GD.events['best_pass'] = False


    #mask = GD.events['eID'].str.contains('Shot|Pass|Cross', case=False, regex=True)
    mask = GD.events['eID'].str.contains('Shot|Pass|Cross', case=False, regex=True) & \
       ~GD.events['eID'].str.contains('Received|Intercepted', case=False, regex=True)

    for idx in GD.events.index[mask]:
        recipient = GD.events.at[idx, 'Recipient'] if 'Recipient' in GD.events.columns else None
        best, bp = best_pass(GD.events.at[idx, 'pID'], GD.events.at[idx, 'Frame'], GD.events.at[idx, 'Session'], recipient, GD)
        GD.events.at[idx, 'enabled'] += best
        if bp:
            GD.events.at[idx, 'best_pass'] += bp

    return GD


def calculate_teammate_pass_risk(df, ball_holder):
    df = df.copy()
    df['Grid Position'] = df['Grid Position'].apply(utils.parse_grid_position)

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