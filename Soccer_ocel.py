from helpers import get_field_position
import pandas as pd
from pm4py.objects.ocel.util.log_ocel import log_to_ocel_multiple_obj_types
import numpy as np
import pm4py


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
    df['attribute:start_grid'] = df.apply(lambda row: [get_field_position(row["Start X"], row["Start Y"], x_fields=x_fields, y_fields=y_fields)], axis=1)
    df['end_grid'] = df.apply(lambda row: [get_field_position(row["End X"], row["End Y"], x_fields=x_fields, y_fields=y_fields)], axis=1)
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

    #calculates team scores
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

# reshape the tracking data to long format (one row per player per time point)
def reshape_tracking(df, team_label):
    long_rows = []
    for col in df.columns:
        if col.startswith("Player"):
            x_col = col
            y_col = f"Unnamed: {int(df.columns.get_loc(col)) + 1}"
            for _, row in df.iterrows():
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
def format_ocel_df(tracking_df):
    ocel_df = tracking_df.copy()
    ocel_df['concept:name'] = f"Player changes position"
    ocel_df['time:timestamp'] = pd.to_datetime(ocel_df['Time [s]'], unit='s')
    ocel_df['crossed_grid'] = True
    ocel_df['To']=None

    ocel_df.rename(columns={
    'X': 'attribute:end_x',
    'Y': 'attribute:end_y',
    'From Position': 'attribute:start_grid',
    'To Position': 'end_grid',
    'Player': 'From',
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
    set_piece_time = final_df[final_df['concept:name'] == 'SET PIECE']['time:timestamp'].min()
    mask = final_df['case:concept:name'].isnull() & (final_df['time:timestamp'] < set_piece_time)
    final_df.loc[mask, 'case:concept:name'] = 'PRE_POSSESSION'
    return final_df

def soccer_ocel(df, tracking_data_home_df, tracking_data_away_df, x_fields=10, y_fields=10):
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

    # convert to event log
    #df = pm4py.format_dataframe(df, case_id='case:concept:name', activity_key='Activity', timestamp_key='Timestamp')
    event_log = pm4py.convert_to_event_log(final_df)

    # convert to ocel
    ocel= log_to_ocel_multiple_obj_types(event_log, activity_column='concept:name'
                                         , timestamp_column='time:timestamp'
                                         , obj_types=['Team','From', 'To','case:concept:name', 'end_grid']
                                         ,additional_event_attributes=['attribute:subtype'
                                                                       , 'attribute:start_x', 'attribute:start_y'
                                                                       , 'attribute:end_x', 'attribute:end_y' 
                                                                       , 'attribute:duration', 'attribute:travel_distance'
                                                                       , 'attribute:start_grid', 'attribute:crossed_grid'
                                                                       , 'attribute:attack_game', 'attribute:attack_successful'
                                                                       , 'attribute:home_team_score', 'attribute:away_team_score'
                                                                       ])
    return ocel

def soccer_ocel_no_tracking(df, x_fields=10, y_fields=10):
    df = prepare_event_dataframe(df, x_fields=x_fields, y_fields=y_fields)

    event_log = pm4py.convert_to_event_log(df)

    # convert to ocel
    ocel= log_to_ocel_multiple_obj_types(event_log, activity_column='concept:name'
                                         , timestamp_column='time:timestamp'
                                         , obj_types=['Team','From', 'To','case:concept:name', 'end_grid']
                                         ,additional_event_attributes=['attribute:subtype'
                                                                       , 'attribute:start_x', 'attribute:start_y'
                                                                       , 'attribute:end_x', 'attribute:end_y' 
                                                                       , 'attribute:duration', 'attribute:travel_distance'
                                                                       , 'attribute:start_grid', 'attribute:crossed_grid'
                                                                       , 'attribute:attack_game', 'attribute:attack_successful'
                                                                       , 'attribute:home_team_score', 'attribute:away_team_score'
                                                                       ])
    return ocel