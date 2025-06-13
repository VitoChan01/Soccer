from helpers import get_field_position
import pandas as pd
from pm4py.objects.ocel.util.log_ocel import log_to_ocel_multiple_obj_types
import numpy as np
import pm4py
import re

def soccer_ocel(df, tracking_data_home_df, tracking_data_away_df, x_fields=10, y_fields=10):
    final_df = soccer_ocel_df(df, tracking_data_home_df, tracking_data_away_df, x_fields=x_fields, y_fields=y_fields)

    ocel=soccer_df_to_ocel(final_df)
    return ocel
#load event data and preprocessing
def prepare_event_dataframe(df, x_fields=10, y_fields=10):
    '''
    Prepares the soccer event dataframe for further processing.
    - Calculates the duration of each event.
    - Calculates the travel distance for each event.
    - Determines the start and end grid positions based on the coordinates.
    - Identifies whether the event crosses grids.
    - Identify attack sessions and their success.
    - Renames columns to standardize the format.
    - Calculates team scores based on goals scored.

    Args:
        df (pd.DataFrame): The input event dataframe.
        x_fields (int): Number of fields along the x-axis. Default is 10.
        y_fields (int): Number of fields along the y-axis. Default is 10.

    Returns:
        pd.DataFrame: The processed event dataframe with additional attributes.
    '''
    df['timestamp'] = pd.to_datetime(df['Start Time [s]'], unit='s', origin='unix')
    df['attribute:duration'] = df['End Time [s]'] - df['Start Time [s]']
    #df['player']= df.apply(lambda row: [row["From"], row["To"]], axis=1)
    df['attribute:travel_distance'] = ((df['End X'] - df['Start X'])**2 + (df['End Y'] - df['Start Y'])**2)**0.5
    df['attribute:start_grid'] = df.apply(lambda row: get_field_position(row["Start X"], row["Start Y"], x_fields=x_fields, y_fields=y_fields), axis=1)
    df['end_grid'] = df.apply(lambda row: get_field_position(row["End X"], row["End Y"], x_fields=x_fields, y_fields=y_fields), axis=1)
    df['attribute:crossed_grid'] = df['attribute:start_grid'] != df['end_grid']
    df['attribute:attack_game'] = ((df['Type'] == 'SET PIECE') | (df['Type'] == 'RECOVERY')).cumsum()
    

    attack_id_away = 0
    attack_id_home = 0
    current_team = None
    attack_ids = []
    for i, row in df.iterrows():
        event = row['Type']
        if event in ['SET PIECE', 'RECOVERY']:
            current_team = row['Team']
            if current_team == 'Away':
                attack_id_away += 1
                attack_id = 'AA'+str(attack_id_away)
            else:
                attack_id_home += 1
                attack_id = 'HA'+str(attack_id_home)
        attack_ids.append(attack_id)

    df['attack_team'] = attack_ids

    df['attribute:attack_successful'] = False
    goal_indices = df[df['Subtype'].str.endswith('GOAL', na=False)].index
    successful_cases = df.loc[goal_indices, 'attack_team'].unique()
    df.loc[df['attack_team'].isin(successful_cases), 'attribute:attack_successful'] = True

    #rename columns
    df.rename(columns={
        'Type': 'concept:name',
        'attack_team': 'case:concept:name',
        'timestamp': 'time:timestamp',
        'Subtype': 'attribute:subtype',
        'Start X': 'attribute:start_x',
        'Start Y': 'attribute:start_y',
        'End X': 'attribute:end_x',
        'End Y': 'attribute:end_y'
    }, inplace=True)
    
    #join type and subtype
    #df['concept:name'] = df.apply(
    #    lambda row: f"{row['concept:name']}-{row['attribute:subtype']}" if pd.notnull(row['attribute:subtype']) else row['concept:name'],axis=1)

    df = team_scores(df)
    df = ball_obj(df)
    #df = pass_count(df)
    df = score_event(df)
    df = split_pass(df)
    df=df.sort_values(by=['time:timestamp', 'End Time [s]'],
                      ascending=[True, True],na_position='last' 
                      ).reset_index(drop=True)


    return df


def ball_obj(df):
    df['ball'] = df['concept:name'].apply(
        lambda x: 'ball_1' if not str(x).startswith(('CARD', 'CHALLENGE', 'FAULT RECEIVED')) else None
    )
    return df 

def team_scores(df):
    df['home_goal'] = (
        df['attribute:subtype'].str.endswith('GOAL', na=False) &
        df['case:concept:name'].str.startswith('HA')
    ).astype(int)

    df['away_goal'] = (
        df['attribute:subtype'].str.endswith('GOAL', na=False) &
        df['case:concept:name'].str.startswith('AA')
    ).astype(int)

    df['attribute:home_team_score'] = df['home_goal'].cumsum()
    df['attribute:away_team_score'] = df['away_goal'].cumsum()

    df.drop(columns=['home_goal', 'away_goal'], inplace=True)
    return df

def pass_count(df):
    is_pass = df['concept:name'] == 'PASS'
    pass_events = df[is_pass]
    sorted_passes = pass_events.sort_values(by=['case:concept:name', 'time:timestamp'])

    new_names = (
        sorted_passes
        .groupby('case:concept:name')
        .cumcount()
        .add(1)
        .astype(str)
        .radd('PASS')
    )

    df.loc[sorted_passes.index, 'concept:name'] = new_names
    return df

def score_event(df):
    goal_events_mask = df['attribute:subtype'].astype(str).str.endswith('GOAL', na=False)
    goal_events = df[goal_events_mask].copy()
    goal_events_dup = goal_events.copy()
    goal_events_dup['concept:name'] = 'Goal'
    goal_events_dup['attribute:start_grid'] = goal_events_dup['end_grid']
    goal_events_dup['attribute:start_x'] = goal_events_dup['attribute:end_x']
    goal_events_dup['attribute:start_y'] = goal_events_dup['attribute:end_y']
    goal_events_dup['time:timestamp'] = pd.to_datetime(goal_events_dup['End Time [s]'], unit='s', origin='unix')
    df = pd.concat([df, goal_events_dup], ignore_index=True)
    return df


def split_pass(ocel_df):
    def insert_after_pass(name, insert_text):
    # This will match 'Pass' optionally followed by digits, and capture both parts
        match = re.match(r'(PASS)(\d*)', name, re.IGNORECASE)
        if match:
            return f"{match.group(1)} {insert_text}{match.group(2)}".replace('  ', ' ')
        else:
            return name

    # Apply to the dataframe
    pass_mask = ocel_df['concept:name'].str.startswith('PASS')
    pass_events = ocel_df[pass_mask].copy()
    pass_received = pass_events.copy()
    pass_received['concept:name'] = pass_received['concept:name'].apply(lambda x: insert_after_pass(x, 'Received'))
    pass_received['attribute:start_grid'] = pass_received['end_grid']
    pass_received['attribute:start_x'] = pass_received['attribute:end_x']
    pass_received['attribute:start_y'] = pass_received['attribute:end_y']
    pass_received['time:timestamp'] = pd.to_datetime(pass_received['End Time [s]'], unit='s', origin='unix')
    pass_received['From'] = pass_received['To']

    pass_events['concept:name'] = pass_events['concept:name'].apply(lambda x: insert_after_pass(x, 'Out'))
    ocel_df_with_pass_dup = pd.concat([ocel_df[~pass_mask], pass_events, pass_received], ignore_index=True)
    ocel_df_with_pass_dup = ocel_df_with_pass_dup.drop(columns=['To']).rename(columns={'From': 'Player'})
    
    return ocel_df_with_pass_dup

# reshape the tracking data to long format (one row per player per time point)
def reshape_tracking(df, team_label):
    long_rows = []
    for col in df.columns:
        if col.startswith("Player"):
            x_col = col
            y_col = f"Unnamed: {int(df.columns.get_loc(col)) + 1}"
            for _, row in df.iterrows():
                if pd.notna(row[x_col]) and pd.notna(row[y_col]):
                    long_rows.append({
                        "Time [s]": row["Time [s]"],
                        "Frame": row["Frame"],
                        "Team": team_label.capitalize(),
                        #"Player": f"{team_label.capitalize()}_{col}",
                        "Player": f"{col}",
                        "X": row[x_col],
                        "Y": row[y_col]
                    })
    return pd.DataFrame(long_rows)

# identify only events when grid position of player changes
def get_grid_change_events(tracking_df):
    tracking_df = tracking_df.sort_values(by=['Player', 'Time [s]'])
    tracking_df['Prev Grid Position'] = tracking_df.groupby('Player')['Grid Position'].shift(1)
    grid_change_events = tracking_df[tracking_df['Grid Position'] != tracking_df['Prev Grid Position']]
    grid_change_events.rename(columns={'Grid Position': 'To Position', 
                                'Prev Grid Position': 'From Position'}, inplace=True
                       )
    return grid_change_events

# format dataframe for ocel
def format_ocel_df(ocel_df):
    ocel_df['concept:name'] = f"Player changes position"
    ocel_df['time:timestamp'] = pd.to_datetime(ocel_df['Time [s]'], unit='s')
    ocel_df['crossed_grid'] = True

    ocel_df.rename(columns={
    'X': 'attribute:end_x',
    'Y': 'attribute:end_y',
    'From Position': 'attribute:start_grid',
    'To Position': 'end_grid'
    }, inplace=True)
    
    return ocel_df

def merge_ball_and_player_event(ball_df, player_df):
    """
    Merges the ball event dataframe with the player movement event dataframe.

    Args:
        df (pd.DataFrame): The ball event dataframe.
        ocel_df (pd.DataFrame): The player movement event dataframe.

    Returns:
        pd.DataFrame: The merged dataframe with ball and tracking events.
    """
    ball_df = ball_df.sort_values('time:timestamp').reset_index(drop=True)
    player_df = player_df.sort_values('time:timestamp').reset_index(drop=True)

    # Columns to copy
    cols_to_fill = [
        'case:concept:name', 
        'attribute:attack_game', 
        'attribute:attack_successful',
        'attribute:home_team_score', 
        'attribute:away_team_score'
    ]

    for col in cols_to_fill:
        player_df[col] = None


    timestamps_large = ball_df['time:timestamp'].to_numpy(dtype='datetime64[ns]')

    for i, row in player_df.iterrows():
        ts = np.datetime64(row['time:timestamp']) 
        pos = timestamps_large.searchsorted(ts, side='right') - 1 
        if pos >= 0:
            for col in cols_to_fill:
                player_df.at[i, col] = ball_df.at[pos, col]
    final_df = pd.concat([ball_df, player_df], ignore_index=True).sort_values('time:timestamp')
    final_df.reset_index(drop=True, inplace=True)
    set_piece_time = final_df[final_df['concept:name'].str.startswith('SET PIECE')]['time:timestamp'].min()
    mask = final_df['case:concept:name'].isnull() & (final_df['time:timestamp'] < set_piece_time)
    final_df.loc[mask, 'case:concept:name'] = 'PRE_POSSESSION'
    return final_df

def soccer_df_to_ocel(df):
    ''' 
    Convert DataFrame to object centric event log
    Args:
        df (pd.DataFrame): The input event dataframe with columns:
            - 'concept:name': Activity name
            - 'time:timestamp': Timestamp of the event
            - 'From': Starting player
            - 'To': Ending player
            - 'attribute:subtype': Subtype of the event
            - 'attribute:start_x', 'attribute:start_y': Starting coordinates
            - 'attribute:end_x', 'attribute:end_y': Ending coordinates
            - 'attribute:duration': Duration of the event
            - 'attribute:travel_distance': Distance traveled during the event
            - 'attribute:start_grid', 'end_grid': Grid positions at start and end
            - 'attribute:crossed_grid': Whether the grid was crossed
            - 'attribute:attack_game': Attack session identifier
            - 'attribute:attack_successful': Whether the attack was successful
            - 'attribute:home_team_score', 'attribute:away_team_score': Scores of home and away teams
    Returns:
        ocel (pm4py.objects.ocel.obj.OCEL): The converted object-centric event log.

    '''
    #df = pm4py.format_dataframe(df, case_id='case:concept:name', activity_key='Activity', timestamp_key='Timestamp')
    event_log = pm4py.convert_to_event_log(df)

    # convert to ocel
    ocel= log_to_ocel_multiple_obj_types(event_log, activity_column='concept:name'
                                         , timestamp_column='time:timestamp'
                                         , obj_types=['ball','Goalkeeper','Attacker','Defender','Team','Player','Possession', 'end_grid']
                                         #, obj_types=['Goalkeeper','Attacker','Defender','Team','Player','Possession', 'end_grid', 'ball']
                                         ,additional_event_attributes=['attribute:subtype'
                                                                       , 'attribute:start_x', 'attribute:start_y'
                                                                       , 'attribute:end_x', 'attribute:end_y' 
                                                                       , 'attribute:duration', 'attribute:travel_distance'
                                                                       , 'attribute:start_grid', 'attribute:crossed_grid'
                                                                       , 'attribute:attack_game', 'attribute:attack_successful'
                                                                       , 'attribute:home_team_score', 'attribute:away_team_score'
                                                                       ])
    return ocel


def soccer_ocel_df(df, tracking_data_home_df, tracking_data_away_df, x_fields=10, y_fields=10):
    df = prepare_event_dataframe(df, x_fields=x_fields, y_fields=y_fields)

    tracking_long_home_df = reshape_tracking(tracking_data_home_df, "home")
    tracking_long_away_df = reshape_tracking(tracking_data_away_df, "away")
    tracking_long_df = pd.concat([tracking_long_home_df, tracking_long_away_df])

    # add grid position to tracking long
    tracking_long_df['Grid Position'] = tracking_long_df.apply(lambda row: get_field_position(row['X'], row['Y'], x_fields=x_fields, y_fields=y_fields), axis=1)
    tracking_grid_change_events_df = get_grid_change_events(tracking_long_df)

    # format the grid change events for ocel
    ocel_df = format_ocel_df(tracking_grid_change_events_df)

    # merge ball events and tracking events
    final_df = merge_ball_and_player_event(df, ocel_df)

    final_df=final_df.sort_values(by=['time:timestamp', 'End Time [s]'],
                                  ascending=[True, True],
                                  na_position='last'  # Put nulls at the end
                                  ).reset_index(drop=True)
    final_df['Possession']=final_df['case:concept:name'].copy()

    final_df=classify_player_role(final_df,tracking_data_home_df, tracking_data_away_df)

    return final_df

def soccer_ocel_no_tracking(df, x_fields=10, y_fields=10):
    df = prepare_event_dataframe(df, x_fields=x_fields, y_fields=y_fields)

    event_log = pm4py.convert_to_event_log(df)

    # convert to ocel
    ocel= log_to_ocel_multiple_obj_types(event_log, activity_column='concept:name'
                                         , timestamp_column='time:timestamp'
                                         , obj_types=['Team','Player','Possession', 'end_grid', 'ball','Goalkeeper','Attacker','Defender']
                                         ,additional_event_attributes=['attribute:subtype'
                                                                       , 'attribute:start_x', 'attribute:start_y'
                                                                       , 'attribute:end_x', 'attribute:end_y' 
                                                                       , 'attribute:duration', 'attribute:travel_distance'
                                                                       , 'attribute:start_grid', 'attribute:crossed_grid'
                                                                       , 'attribute:attack_game', 'attribute:attack_successful'
                                                                       , 'attribute:home_team_score', 'attribute:away_team_score'
                                                                       ])
    return ocel


def get_player_trajectory(df, player_col_name):

    x_idx = df.columns.get_loc(player_col_name)
    
    y_idx = x_idx + 1
    y_col_name = df.columns[y_idx]

    time_points = df['Time [s]'].values

    x_coords = df[player_col_name].values
    y_coords = df[y_col_name].values

    mask = (~pd.isnull(x_coords)) & (~pd.isnull(y_coords))
    time_points = time_points[mask]
    x_coords = x_coords[mask]
    y_coords = y_coords[mask]

    return time_points, x_coords, y_coords
def calculate_zone_fractions(x_coords, team):
    if team == 'Away':
        # Attacking direction is right (increasing x)
        defense_zone = (0.0, 0.5)
        #midfield_zone = (0.5, 0.75)
        attack_zone = (0.5, 1.0)
    else:  # Home team attacks left
        defense_zone = (0.5, 1.0)
        #midfield_zone = (0.25, 0.5)
        attack_zone = (0.0, 0.5)

    total = len(x_coords)
    if total == 0:
        return 0, 0, 0

    defense_frac = ((x_coords >= defense_zone[0]) & (x_coords < defense_zone[1])).sum() / total
    #midfield_frac = ((x_coords >= midfield_zone[0]) & (x_coords < midfield_zone[1])).sum() / total
    attack_frac  = ((x_coords >= attack_zone[0])  & (x_coords < attack_zone[1])).sum() / total

    return defense_frac, attack_frac
def classify_all_players(tracking_df, team_name):
    def_L=[]
    att_L=[]
    for col in tracking_df.columns:
        if col.startswith("Player"):
            _,x, y = get_player_trajectory(tracking_df, col)
            def_frac, att_frac = calculate_zone_fractions(x, team_name)
            role = np.argmax([def_frac, att_frac])
            if role==0:
                def_L.append(col)
            else:
                att_L.append(col)

    return def_L, att_L

def fraction_time_in_goal_area(x_coords, y_coords,team):
    if team=='Away':
        goal_x_min=0.93
        goal_x_max=1.2
    if team=='Home':
        goal_x_min=-0.2
        goal_x_max=0.06
    goal_y_min=0.4
    goal_y_max=0.6


    inside_goal = (
        (x_coords >= goal_x_min) & (x_coords <= goal_x_max) &
        (y_coords >= goal_y_min) & (y_coords <= goal_y_max)
    )
    
    # Fraction of time inside goal area
    frac = inside_goal.sum() / len(x_coords) if len(x_coords) > 0 else 0
    return frac

def find_goalkeepers(tracking_data_home_df, tracking_data_away_df):
    goalkeepers=[]
    col_L,frac_L=[],[]
    for col in tracking_data_away_df.columns:
        if col.startswith("Player"):
            _,x_trace,y_trace=get_player_trajectory(tracking_data_away_df,col)
            frac=fraction_time_in_goal_area(x_trace,y_trace, 'Away')
            col_L.append(col)
            frac_L.append(frac)
    goalkeepers.append(col_L[np.argmax(frac)])
    col_L,frac_L=[],[]
    for col in tracking_data_home_df.columns:
        if col.startswith("Player"):
            _,x_trace,y_trace=get_player_trajectory(tracking_data_home_df,col)
            frac=fraction_time_in_goal_area(x_trace,y_trace, 'Home')
            col_L.append(col)
            frac_L.append(frac)
    goalkeepers.append(col_L[np.argmax(frac)])
    return goalkeepers

def classify_player_role(df,tracking_data_home_df, tracking_data_away_df):
    away_def,away_att = classify_all_players(tracking_data_away_df, 'Away')
    home_def,home_att = classify_all_players(tracking_data_home_df, 'Home')
    att_players=away_att+home_att
    def_players=away_def+home_def
    goalkeepers=find_goalkeepers(tracking_data_home_df, tracking_data_away_df)
    att_players = [p for p in att_players if p not in goalkeepers]
    def_players = [p for p in def_players if p not in goalkeepers]
    df['Goalkeeper'] = None
    df['Attacker'] = None
    df['Defender'] = None

    df.loc[df['Player'].isin(att_players), 'Attacker'] = df.loc[df['Player'].isin(att_players), 'Player'].copy()
    df.loc[df['Player'].isin(goalkeepers), 'Goalkeeper'] = df.loc[df['Player'].isin(goalkeepers), 'Player'].copy()
    df.loc[df['Player'].isin(def_players), 'Defender'] = df.loc[df['Player'].isin(def_players), 'Player'].copy()

    return df