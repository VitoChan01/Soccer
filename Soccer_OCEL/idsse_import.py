import pandas as pd
import numpy as np
import os
from floodlight.io.dfl import read_position_data_xml, read_event_data_xml, read_teamsheets_from_mat_info_xml
import Soccer_OCEL.utils as utils
from Soccer_OCEL.trace_events import position_events
from Soccer_OCEL.trace_voronoi import add_voronoi_area
import json
from pathlib import Path

#load
class GameData:
    def __init__(self, Game, events, positions, team_sheets_df,
                  pitch, possession, ballstatus, framerate, 
                  home_tID, away_tID, balltrace=None,
                  FH_start=None, SH_start=None, 
                  FH_TeamRight=None):
        self.Game = Game
        self.events = events
        self.positions = positions
        self.team_sheets_df = team_sheets_df
        self.pitch = pitch
        self.possession = possession
        self.ballstatus = ballstatus
        self.framerate = framerate
        self.home_tID = home_tID
        self.away_tID = away_tID
        self.balltrace = balltrace
        self.FH_start = FH_start
        self.SH_start = SH_start
        self.FH_TeamRight = FH_TeamRight

    def summary(self):
        """Quick summary of loaded data."""
        return {
            "Game ID": self.Game,
            "Home Team ID": self.home_tID,
            "Away Team ID": self.away_tID,
            "Frame Rate": self.framerate,
            "Num Events": len(self.events) if hasattr(self.events, "__len__") else "N/A",
            "Num Players": len(self.team_sheets_df),
        }
    
def load_game_data(path, Game, xy_fields=(10,10), encode_recipient_role=False, voronoi_area=False):
    Game_data = load_game_data_path(path, Game)

    events, team_sheets, pitch = read_event_data_xml(os.path.join(
        path, Game_data[1]), os.path.join(path, Game_data[0]))
    team_sheets_df = pd.concat([team_sheets["Home"].teamsheet.assign(Home_Away="Home"),
                                team_sheets["Away"].teamsheet.assign(Home_Away="Away")])
    positions, possession, ballstatus, _, pitch = read_position_data_xml(os.path.join(path, Game_data[2]), os.path.join(path, Game_data[0]))
    framerate=positions["firstHalf"]["Ball"].framerate
    Home_tID=team_sheets["Home"].teamsheet.loc[0,'tID']
    Away_tID=team_sheets["Away"].teamsheet.loc[0,'tID']
    GD = GameData(
        Game, events, positions, team_sheets_df, pitch,
        possession, ballstatus, framerate, Home_tID, Away_tID
    )
    GD = GameData_to_df(GD)
    # on-ball events
    eventdf=split_pass(GD.events, GD.framerate, GD.team_sheets_df)
    for i in eventdf[eventdf['eID']=='BallClaiming'].index:
        window = eventdf.loc[i-3:i]
        intercepted_idx = window[window['eID'].str.contains('Intercepted')].index
        eventdf = eventdf.drop(intercepted_idx)
    GD.events = eventdf.reset_index(drop=True)
    GD.events = split_shot(GD.events, GD.framerate, GD.team_sheets_df)
    positions_df = GD.positions.copy().rename(columns={'Player': 'pID'})
    GD.events = GD.events.merge( #.drop(columns=['x', 'y'])
        positions_df[['pID', 'Frame', 'Session', 'x', 'y']],
        on=['pID', 'Frame', 'Session'],
        how='left' 
    )

    #position events
    position_events_df=position_events(GD)
    position_events_df_with_timestamps = utils.add_timestamps(position_events_df, GD.FH_start, GD.SH_start, GD.framerate)
    GD.events=pd.concat([GD.events,position_events_df_with_timestamps]).sort_values(['Session', 'timestamp']).reset_index(drop=True)
    #marking Game
    GD.events['game']=Game
    #possession
    GD = label_full_possession(GD)
    GD = assign_possessionID(GD)
    #grid position
    GD.events['Grid Position'] = GD.events.apply(lambda row: utils.get_field_position(row['x'], row['y'], GD.pitch, xy_fields=xy_fields), axis=1)
    GD.positions['Grid Position'] = GD.positions.apply(lambda row: utils.get_field_position(row['x'], row['y'], GD.pitch, xy_fields=xy_fields), axis=1)
    #enabled events
    GD=add_pass_enabled(GD)
    #encode recipient role
    if encode_recipient_role:
        GD=encode_position(GD)
    if voronoi_area:
        GD.positions=add_voronoi_area(GD.positions, GD.pitch)

    
    return GD
def get_games(path):
    info_files = [x for x in os.listdir(path) if "matchinformation" in x]
    games = [file[-10:-4] for file in info_files]
    return games
def load_game_data_path(path, Game=None):
    info_files = [x for x in os.listdir(path) if "matchinformation" in x]
    event_files = sorted([x for x in os.listdir(path) if "events_raw" in x])
    position_files = [x for x in os.listdir(path) if "positions_raw" in x]
    if not Game:
        games=get_games(path)
        print("Please select one of the available games:")
        for i, game in enumerate(games):
            print(f'{i+1}: {game}')
        Game_i=int(input("Select a game by index: ")) - 1
        Game=games[Game_i]
        print(f"Selected: {Game}")
    Game_data=[]
    for files in [info_files, event_files, position_files]:
        for file in files:
            if Game in file:
                Game_data.append(file)
    return Game_data

def tracking_array_to_df(arr, player_ids, session, team):

    n_frames = arr.shape[0]

    teams = [team] * len(player_ids)
    # Reshape array: (n_frames, 20, 2)
    reshaped = arr.reshape(n_frames, arr.shape[1]//2, 2)

    records = [
        (frame, pid, team, x, y)
        for frame in range(n_frames)
        for (pid, team), (x, y) in zip(zip(player_ids, teams), reshaped[frame])
    ]
    df=pd.DataFrame(records, columns=["Frame", "Player", "Team", "x", "y"])
    df["Session"] = session
    return df

def ball_array_to_df(arr, session):
    records = [
        (frame, np.nan, "Ball", x, y, session)
        for frame, (x, y) in enumerate(arr)
    ]

    return pd.DataFrame(records, columns=["Frame", "Player", "Team", "x", "y", "Session"])

def GameData_to_df(GD):
    #positions df
    Home_player_ids = GD.team_sheets_df.query('Home_Away=="Home"')["pID"].tolist()
    Away_player_ids = GD.team_sheets_df.query('Home_Away=="Away"')["pID"].tolist()
    position_home = pd.concat([tracking_array_to_df(GD.positions["firstHalf"]["Home"].xy, Home_player_ids, 1, "Home"),
                           tracking_array_to_df(GD.positions["secondHalf"]["Home"].xy, Home_player_ids, 2, "Home")])
    position_away = pd.concat([tracking_array_to_df(GD.positions["firstHalf"]["Away"].xy, Away_player_ids, 1, "Away"),
                            tracking_array_to_df(GD.positions["secondHalf"]["Away"].xy, Away_player_ids, 2, "Away")])
    positions_df = pd.concat([position_home, position_away], ignore_index=True)
    positions_df=positions_df.sort_values(by=['Session',"Frame"])
    positions_df=positions_df.dropna(subset=["x", "y"]).reset_index(drop=True)
    positions_df['x']+= GD.pitch.xlim[1]
    positions_df['y']+= GD.pitch.ylim[1]


    ball_df = pd.concat([ball_array_to_df(GD.positions["firstHalf"]["Ball"].xy, 1),
                            ball_array_to_df(GD.positions["secondHalf"]["Ball"].xy, 2)], ignore_index=True)
    ball_df= ball_df.sort_values(by=['Session', "Frame"])
    ball_df = ball_df.dropna(subset=["x", "y"]).reset_index(drop=True)
    ball_df['x']+= GD.pitch.xlim[1]
    ball_df['y']+= GD.pitch.ylim[1]
    GD.pitch.xlim = (0, GD.pitch.xlim[1]*2)
    GD.pitch.ylim = (0, GD.pitch.ylim[1]*2)
    ball_pos = ball_df[['Session', 'Frame', 'x', 'y']].drop_duplicates(subset=['Session', 'Frame']).rename(
        columns={'x': 'ball_x', 'y': 'ball_y'}
    )
    positions_df = positions_df.merge(ball_pos, on=['Session', 'Frame'], how='left')
    positions_df['Dist_ball'] = np.sqrt((positions_df['x'] - positions_df['ball_x'])**2 + (positions_df['y'] - positions_df['ball_y'])**2)
    GD.positions = positions_df
    #events df
    eventdfFH=pd.concat([GD.events['firstHalf']['Home'].events,GD.events['firstHalf']['Away'].events])
    eventdfFH['Session']=1
    eventdfSH=pd.concat([GD.events['secondHalf']['Home'].events,GD.events['secondHalf']['Away'].events])
    eventdfSH['Session']=2
    eventdf=pd.concat([eventdfFH,eventdfSH])

    eventdf=eventdf.sort_values(['Session', 'timestamp'])
    eventdf=eventdf[eventdf['eID']!='Delete'].reset_index()
    eventdf=eventdf.drop(columns=['index'])

    qualifier_df = pd.json_normalize(eventdf['qualifier'])
    eventdf = eventdf.join(qualifier_df)#.drop(columns=['qualifier'])

    eventdf['Team'] = np.where(eventdf['tID'] == GD.home_tID, 'Home', 'Away')
    eventdf['Frame']= eventdf['gameclock'].apply(lambda x: int(round(x * GD.framerate)))
    eventdf = eventdf.drop_duplicates(subset=['eID', 'gameclock'], keep='first').reset_index(drop=True)
    try:
        GD.FH_TeamRight='Home' if GD.events['firstHalf']['Home'].events.loc[0,'qualifier']['TeamRight']==GD.home_tID else 'Away'
    except:
        try:
            GD.FH_TeamRight='Home' if GD.events['firstHalf']['Home'].events.loc[0,'qualifier']['Side'].lower()=='right' else 'Away'
        except:
            try:
                GD.FH_TeamRight='Home' if np.any(eventdf[eventdf['Session']==1]['TeamRight']==GD.away_tID) else 'Home'
            except:
                GD.FH_TeamRight='Home'
    GD.events = eventdf

    GD.FH_start=eventdf.query('Session==1').loc[:,'timestamp'].to_list()[0]
    GD.SH_start=eventdf.query('Session==2').loc[:,'timestamp'].to_list()[0]
    GD.positions=utils.add_timestamps(GD.positions, GD.FH_start, GD.SH_start, GD.framerate)
    GD.balltrace=utils.add_timestamps(ball_df, GD.FH_start, GD.SH_start, GD.framerate)
    return GD

#on-ball events
def split_pass(ocel_df, framerate, team_sheets_df):
    for ev in ['Pass', 'Cross']:
        pass_mask = ocel_df['eID'].str.endswith(ev)
        pass_events = ocel_df[pass_mask].copy()
        pass_received = pass_events.copy()
        successfulpass=pass_received['Evaluation'].str.startswith('successful')
        failedpass=pass_received['Evaluation'].str.startswith('unsuccessful') & pd.notna(pass_received['Recipient'])

        pass_received.loc[successfulpass, 'eID'] = (
            #pass_received.loc[successfulpass, 'eID'].astype(str) + '_Received'
            ev + '_Received'
        )
        pass_received.loc[failedpass, 'eID'] = (
            #pass_received.loc[failedpass, 'eID'].astype(str) + '_Intercepted'
            ev + '_Intercepted'
        )
        pass_received['pID']=pass_received['Recipient']
        #pass_received['attribute:start_grid'] = pass_received['end_grid']
        #pass_received['attribute:start_x'] = pass_received['attribute:end_x']
        #pass_received['attribute:start_y'] = pass_received['attribute:end_y']
        pass_received['timestamp'] = pass_received['timestamp'] + pd.to_timedelta(1/framerate, unit="s")#add one frame to the timestamp such that they do not happen at the same time
        pass_received['Frame']= pass_received['Frame'] + 1  # Increment frame by 1 to avoid same timestamp
        pass_received['gameclock'] = pass_received['gameclock'] + 1/framerate
        pass_received=pass_received[~pass_received['Recipient'].isna()]

        pass_events['eID']=ev
        ocel_df[pass_mask]=pass_events

        pass_received['tID']=pass_received['Recipient'].apply(lambda x: utils.get_tID_from_pID(x, team_sheets_df))
        pass_received['Recipient'] = float("nan")
        
        ocel_df = pd.concat([ocel_df, pass_received], ignore_index=True)
    
    return ocel_df.sort_values(['Session', 'timestamp']).reset_index(drop=True)

def split_shot(ocel_df, framerate, team_sheets_df):
    shot_mask = ocel_df['eID'].str.contains('ShotAtGoal')
    shot_events = ocel_df[shot_mask].copy()
    shot = shot_events.copy()
    successfulshot=shot['outcome']==1
    failedshot=shot['outcome']==0

    shot.loc[successfulshot, 'eID'] = 'Scored'
    shot.loc[failedshot, 'eID'] = (
        shot.loc[failedshot, 'eID'].astype(str).str.split('_').str[-1]
    )
    #shot['attribute:start_grid'] = shot['end_grid']
    #shot['attribute:start_x'] = shot['attribute:end_x']
    #shot['attribute:start_y'] = shot['attribute:end_y']

    #currently do not know if the timestamp is the time of the shot or the time of the goal, need to fix in the future
    shot['timestamp'] = shot['timestamp'] + pd.to_timedelta(1/framerate, unit="s")#add one frame to the timestamp such that they do not happen at the same time
    shot['Frame']= shot['Frame'] + 1  # Increment frame by 1 to avoid same timestamp
    shot['gameclock'] = shot['gameclock'] + 1/framerate
    

    #shot_events['concept:name'] = shot_events['concept:name'].apply(lambda x: insert_after_shot(x, 'Out'))
    ocel_df.loc[shot_mask, 'eID']=(ocel_df.loc[shot_mask, 'eID'].astype(str).str.split('_').str[:-1].str.join('_'))
    ocel_df = pd.concat([ocel_df, shot], ignore_index=True)
    #ocel_df = ocel_df_with_shot_dup.drop(columns=['To']).rename(columns={'From': 'Player'})
    
    return ocel_df.sort_values(['Session', 'timestamp']).reset_index(drop=True)

def get_player_location_by_frame(positions_df, pid, frame, Session):
    """
    Get the player's location (x, y) at a specific frame.
    """
    player_positions = positions_df[(positions_df['Player'] == pid) & (positions_df['Frame'] == frame) & (positions_df['Session'] == Session)]
    
    if not player_positions.empty:
        return player_positions[['x', 'y']].values[0]
    else:
        return None, None
def syn_location_to_events(eventdf, positions_df):
    xs, ys = get_player_location_by_frame(positions_df, eventdf['pID'], eventdf['Frame'], eventdf['Session'])
    
    eventdf['x'] = xs
    eventdf['y'] = ys
    
    return eventdf.drop(columns=['x_ball', 'y_ball'])

#possessions
def label_session_possession(arr, start_count=None, relative=True):
    arr = np.asarray(arr)
    
    change_points = np.diff(arr, prepend=arr[0]-1) != 0
    run_ids = np.cumsum(change_points) - 1 
    
    labels = []
    if not start_count:
        abs_count = 0
        rel_counts = {"Away": 0, "Home": 0}
    elif relative:
        rel_counts = {"Away": start_count[0]+1, "Home": start_count[1]+1}
    else:
        abs_count = start_count+1
    prev_run = None

    for i, (val, rid) in enumerate(zip(arr, run_ids)):
        prefix = "Away" if val == 2 else "Home"
        
        if rid != prev_run:
            prev_run = rid
            if relative:
                num = rel_counts[prefix]
                rel_counts[prefix] += 1
            else:
                num = abs_count
                abs_count += 1
        labels.append(f"{prefix}_{num}")
    
    return np.array(labels)
def label_full_possession(GD, relative=True):
    FH_possession=label_session_possession(GD.possession['firstHalf'].code, relative=relative)
    if relative:
        prefixes = np.char.split(FH_possession, '_')
        nums = np.array([int(p[1]) for p in prefixes])
        prefixes = np.array([p[0] for p in prefixes])
        

        away_max = np.max(nums[prefixes == 'Away']) if np.any(prefixes == 'Away') else None
        home_max = np.max(nums[prefixes == 'Home']) if np.any(prefixes == 'Home') else None
        start_count= [away_max, home_max]
    else:
        start_count=int(FH_possession[-1].split('_')[-1])
    SH_possession=label_session_possession(GD.possession['secondHalf'].code, start_count=start_count, relative=relative)
    GD.possession['firstHalf'].code=FH_possession
    GD.possession['secondHalf'].code=SH_possession
    return GD

def assign_possessionID(GD):
    df = GD.events.copy()
    Game = GD.Game
    possession_map = {
        1: np.array(GD.possession['firstHalf'].code),
        2: np.array(GD.possession['secondHalf'].code),
    }
    
    session_col = df['Session'].values
    frame_col = df['Frame'].values
    pid_col = df['pID'].values

    possession_ids = np.empty(len(df), dtype=object)
    attacking_team = np.full(len(df), np.nan, dtype=object)
    defending_team = np.full(len(df), np.nan, dtype=object)

    for i in range(len(df)):
        session = session_col[i]
        frame = frame_col[i]
        pid = pid_col[i]

        try:
            possession_code = possession_map[session][frame]
        except (KeyError, IndexError):
            #print(frame)
            #print(session)
            #print(len(possession_map[1]))
            #print(len(possession_map[2]))
            #print(df.loc[i,'eID'])
            #print(df.loc[i,'Session'])
            #print(df.loc[i,'Frame'])
            possession_code = possession_map[session][-1]
            e=df.loc[i,'eID']
            print(f'Warning: Event frame exceeded possession frame. Possession/Events may have missing/corrupted data. Session {session} Frame {frame} Event {e}.')

        possession_ids[i] = f"{Game}_{possession_code}"
        player_team_id = utils.get_tID_from_pID(pid, GD.team_sheets_df)
        Home_Away='Home' if player_team_id==GD.home_tID else 'Away'
        if possession_code[:4] == Home_Away:
            attacking_team[i] = pid
        else:
            defending_team[i] = pid

    df['possessionID'] = possession_ids
    df['attacking_team'] = attacking_team
    df['defending_team'] = defending_team

    GD.events = df

    return GD

# pass and cross recipient encoding
def encode_position(GD):
    #pass_mask = ocel_df['eID'].str.contains('Pass') | ocel_df['eID'].str.contains('Cross')
    pass_events = GD.events[
        GD.events['Recipient'].notna() &
        GD.events['Evaluation'].str.startswith('successful')
    ].copy()
    pass_events['player_position'] = pass_events.apply(
        lambda row: GD.team_sheets_df.loc[GD.team_sheets_df['pID'] == row['Recipient'], 'position'].values[0]
        ,
        axis=1
    )

    pass_events['eID'] = pass_events['eID'] + '_' + pass_events['player_position'].astype(str)
    outdf= GD.events.copy()
    outdf[
        GD.events['Recipient'].notna() &
        GD.events['Evaluation'].str.startswith('successful')
    ] = pass_events
    GD.events=outdf.sort_values(['Session', 'timestamp']).reset_index(drop=True)
    
    return GD

#enabled events
def best_pass(current_player, Frame, Session, recipient, GD):
    bp=False
    df=GD.positions.query('Frame == @Frame & Session == @Session')
    teammate_scores = utils.calculate_teammate_pass_risk(df, current_player)
    min_value = min(teammate_scores.values())
    min_keys = [k for k, v in teammate_scores.items() if v == min_value]
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


    mask = GD.events['eID'].str.contains('Shot|Pass|Cross', case=False, regex=True)

    for idx in GD.events.index[mask]:
        recipient = GD.events.at[idx, 'Recipient'] if 'Recipient' in GD.events.columns else None
        best, bp = best_pass(GD.events.at[idx, 'pID'], GD.events.at[idx, 'Frame'], GD.events.at[idx, 'Session'], recipient, GD)
        GD.events.at[idx, 'enabled'] += best
        if bp:
            GD.event.at[idx, 'best_pass'] += bp

    return GD







# player grouping
def assign_roles(GD, category='role', role_json_path=None):
    if not role_json_path:
        role_json_path=os.path.join(Path(__file__).resolve().parent, "role_groups.json")

    with open(role_json_path, "r", encoding="utf-8") as f:
        position_groups = json.load(f)
    if not category:
        keyset = list({k for v in position_groups.values() for k in v.keys()})
        for i,k in enumerate(keyset):
            print(f'{i}: {k}')
        selection=int(input('Select groupping by index: '))
        category=keyset[selection]

    def get_group(position_code: str, group_type: str) -> str:
        info = position_groups.get(position_code, {category: "Unknown"})
        return info.get(group_type, "Unknown")

    def find_position(player_name: str) -> str:
        row = GD.team_sheets_df.loc[GD.team_sheets_df["pID"] == player_name, "position"]
        return row.values[0] if len(row) > 0 else None

    def find_role(player_name: str) -> str:
        pos = find_position(player_name)
        return get_group(pos, category) if pos else "Unknown"

    GD.events[category] = GD.events["pID"].apply(find_role)
    roles = list({v[category] for v in position_groups.values() if category in v})
    for role in roles:
        GD.events[role] = GD.events["pID"].where(GD.events[category] == role, None)
    GD.events = GD.events.copy()
    return GD
def assign_multi_roles(GD,categories=None,role_json_path=None):
    if not role_json_path:
        role_json_path=os.path.join(Path(__file__).resolve().parent, "role_groups.json")

    with open(role_json_path, "r", encoding="utf-8") as f:
        position_groups = json.load(f)
    if not categories:
        categories = list({k for v in position_groups.values() for k in v.keys()})
    for category in categories:
        GD=assign_roles(GD, category, role_json_path)
    return GD



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

