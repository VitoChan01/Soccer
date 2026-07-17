import pandas as pd
import numpy as np
import os
from floodlight.io.dfl import read_position_data_xml, read_event_data_xml, read_teamsheets_from_mat_info_xml
from . import utils
from .translucent import add_pass_enabled
from .trace_events import position_events, movement_events
from .trace_voronoi import add_voronoi_area
from .role_objects import assign_roles as _assign_roles
from .role_objects import assign_roles_multi as _assign_roles_multi
from .pass_feasibility import get_k_enabled_passes, plot_feasibility_map, plot_component_breakdown, _log_feasible_passes, _log_enabled
import pm4py
from shapely.geometry import Polygon, LineString, Point
import geopandas as gpd


#load
class GameData:
    def __init__(self, Game, events, positions, team_sheets_df,
                  pitch, possession, ballstatus, framerate, 
                  home_tID, away_tID, settings=None, positional_events=None, movement_events=None, balltrace=None, FH_start=None, SH_start=None, 
                  FH_TeamRight=None):
        self.Game = Game
        self.events = events
        self.positional_events = positional_events
        self.movement_events = movement_events
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
        self.settings = settings
        

    def summary(self):
        """Quick summary of loaded data."""
        return {
            "Game ID": self.Game,
            "Home Team ID": self.home_tID,
            "Away Team ID": self.away_tID,
            "Frame Rate": self.framerate,
            "Num Events": len(self.events),
            "Num Players": len(self.team_sheets_df),
            'settings': self.settings
        }
    def join_events(self, log='ALL'):
        if log == 'ALL':
            df =  pd.concat([self.movement_events, self.positional_events])
        elif log == 'MOVEMENT':
            df = self.movement_events
        elif log == 'POSITIONAL':
            df = self.positional_events
        self.events=pd.concat([self.events, df])
        self.events=self.events.sort_values(["Session", "Frame"])
        return self
    def format_log(self, log='EVENTS', rename_dic=None):
        if log == 'ALL':
            self.events = utils.formatting(self.events, rename_dic)
            self.positional_events = utils.formatting(self.positional_events, rename_dic)
            self.movement_events = utils.formatting(self.movement_events, rename_dic)
        elif log == 'EVENTS':
            self.events = utils.formatting(self.events, rename_dic)
        elif log == 'POSITIONAL':
            self.positional_events = utils.formatting(self.positional_events, rename_dic)
        elif log == 'MOVEMENT':
            self.movement_events = utils.formatting(self.movement_events, rename_dic)
        return self
    def export_xes(self, log='EVENTS', export_dir=os.path.join('output', 'output_log'), ocel=False):
        if log == 'EVENTS':
            df = self.events
        elif log == 'POSITIONAL':
            df = self.positional_events
        elif log == 'MOVEMENT':
            df = self.movement_events
        export_path=os.path.join(export_dir,f'{self.Game}_{log}')
        if not ocel:
            event_log = pm4py.convert_to_event_log(df)
            pm4py.write_xes(event_log, export_path+'.xes')
            return event_log
    def encode_pass_direction(self):
        self.events=_encode_pass_direction(self.events)
        return self
    def encode_pass_distance(self):
        self.events=_encode_pass_distance(self.events)
        return self
    def encode_recipient_role(self, encode='role'):
        self=encode_position(self, encode=encode)
        return self
    def compute_voronoi(self):
        self.positions=add_voronoi_area(self.positions, self.pitch)
        return self
    def assign_roles(self, process_DF='ALL', team=False, category='role', role_json_path=None):
        self=_assign_roles(self, process_DF, team, category, role_json_path)
        return self
    def assign_roles_multi(self, process_DF='ALL', team=False, categories=None, role_json_path=None):
        self=_assign_roles_multi(self, process_DF, team, categories, role_json_path)
        return self
    def k_enabled(self, k=3, encode=['role', 'name', 'zone', 'tuple', 'label', 'role:role', 'role:zone', 'role:side', 'role:player_position']):
        self = _log_enabled(self, k=k, encode=encode)
        return self

    length = 105
    width = 68
    penalty_depth = 16.5
    penalty_width = 40.3
    goal_depth = 5.5
    goal_width = 18.32
    center_radius = 9.15
    penalty_spot = 11

    field = Polygon([
        (0,0),
        (length,0),
        (length,width),
        (0,width)
    ])

    left_half = Polygon([
        (0,0),
        (length/2,0),
        (length/2,width),
        (0,width)
    ])

    right_half = Polygon([
        (length/2,0),
        (length,0),
        (length,width),
        (length/2,width)
    ])

    center_line = LineString([
        (length/2,0),
        (length/2,width)
    ])

    center_circle = Point(length/2, width/2).buffer(center_radius)
    y1_goal = (width - goal_width)/2
    y2_goal = (width + goal_width)/2

    left_goal_area = Polygon([
        (0, y1_goal),
        (goal_depth, y1_goal),
        (goal_depth, y2_goal),
        (0, y2_goal)
    ])

    right_goal_area = Polygon([
        (length-goal_depth, y1_goal),
        (length, y1_goal),
        (length, y2_goal),
        (length-goal_depth, y2_goal)
    ])

    y1_pen = (width - penalty_width)/2
    y2_pen = (width + penalty_width)/2

    left_penalty_area = Polygon([
        (0, y1_pen),
        (penalty_depth, y1_pen),
        (penalty_depth, y2_pen),
        (0, y2_pen)
    ])

    right_penalty_area = Polygon([
        (length-penalty_depth, y1_pen),
        (length, y1_pen),
        (length, y2_pen),
        (length-penalty_depth, y2_pen)
    ])

    left_penalty_spot = Point(penalty_spot, width/2)
    right_penalty_spot = Point(length-penalty_spot, width/2)


    field_gdf = gpd.GeoDataFrame({
        "name":[
            "field",
            "left_half",
            "right_half",
            "center_line",
            "center_circle",
            "left_goal_area",
            "right_goal_area",
            "left_penalty_area",
            "right_penalty_area"
            #,
            #"left_penalty_spot",
            #"right_penalty_spot"
        ],
        "geometry":[
            field,
            left_half,
            right_half,
            center_line,
            center_circle,
            left_goal_area,
            right_goal_area,
            left_penalty_area,
            right_penalty_area
            #,
            #left_penalty_spot,
            #right_penalty_spot
        ]
    })



def load_game_data(path, Game, xy_fields=(10,10)
                   , encode_recipient_role=False, enabled_passes=False, enabled_with_speed=False, voronoi_area=False, event_based_possession=False, pass_direction=False, pass_distance=False
                   , get_position_events=True, get_movement_events=True
                   , movement_directions=None, movement_step=None, movement_max_gap=None, movement_noise_threshold=0.09):
    Game_data = load_game_data_path(path, Game)
    settings={'encode_recipient_role': encode_recipient_role, 'enabled_passes': enabled_passes, 'enabled_with_speed': enabled_with_speed, 'voronoi_area': voronoi_area, 'event_based_possession': event_based_possession, 'get_position_events': get_position_events, 'pass_direction': pass_direction, 'pass_distance': pass_distance, 'get_movement_events': get_movement_events, 'movement_directions': movement_directions, 'movement_step':movement_step, 'movement_max_gap': movement_max_gap, 'movement_noise_threshold':movement_noise_threshold, 'xy_fields': xy_fields}

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
        possession, ballstatus, framerate, Home_tID, Away_tID, settings
    )
    GD = GameData_to_df(GD)
    # on-ball events
    positions_df = GD.positions.copy().rename(columns={'Player': 'pID'})
    GD.events = GD.events.merge( 
        positions_df[['pID', 'Frame', 'Session', 'x', 'y']],
        on=['pID', 'Frame', 'Session'],
        how='left' 
    )
    GD.events=split_pass(GD.events, GD.framerate, GD.team_sheets_df, positions_df)
    if pass_distance:
        GD.events=_encode_pass_direction(GD.events)
    if pass_direction:
        GD.events=_encode_pass_distance(GD.events)

    intercepted_to_drop = []
    ballclaiming_idx = GD.events[GD.events['eID'] == 'BallClaiming'].index
    for i in ballclaiming_idx:
        window = GD.events.loc[i-3:i]
        intercepted_to_drop.extend(window[window['eID'].str.contains('Intercepted')].index.tolist())

    GD.events = GD.events.drop(intercepted_to_drop).reset_index(drop=True)
    GD.events = split_shot(GD.events, GD.framerate, GD.team_sheets_df)
    GD.events['ball']=Game
    GD.events['game']=Game
    GD.events['Frame']=GD.events['Frame'].astype(int)
    GD.events = GD.events.reset_index(drop=True)

    #assign possessionID
    if not event_based_possession:
        GD = label_full_possession(GD)
        GD = assign_possessionID(GD)
    else:
        GD.events=assign_possessionID_event_based(GD.events, Game, team_sheets_df)
    
    #position events
    if get_position_events:
        position_events_df=position_events(GD)
        position_events_df_with_timestamps = utils.add_timestamps(position_events_df, GD.FH_start, GD.SH_start, GD.framerate)
        GD.positional_events=position_events_df_with_timestamps
        GD.positional_events['game']=Game
        GD.positional_events['Frame']=GD.positional_events['Frame'].astype(int)
        #possession matching
        GD.positional_events=frame_possessionID_matching(GD.events, GD.positional_events)
        GD.positional_events['Grid Position'] = GD.positional_events.apply(lambda row: utils.get_field_position(row['x'], row['y'], GD.pitch, xy_fields=xy_fields), axis=1)
    #movement events
    if get_movement_events:
        GD.movement_events=movement_events(GD.positions, GD.FH_TeamRight, GD.pitch, movement_directions, movement_step, movement_max_gap, movement_noise_threshold)
        GD.movement_events = utils.add_timestamps(GD.movement_events, GD.FH_start, GD.SH_start, GD.framerate)
        GD.movement_events['game']=Game
        GD.positional_events['Frame']=GD.positional_events['Frame'].astype(int)
        #possession matching
        GD.movement_events=frame_possessionID_matching(GD.events, GD.movement_events)
        GD.movement_events['Grid Position'] = GD.movement_events.apply(lambda row: utils.get_field_position(row['x'], row['y'], GD.pitch, xy_fields=xy_fields), axis=1)

    #grid position
    GD.events['Grid Position'] = GD.events.apply(lambda row: utils.get_field_position(row['x'], row['y'], GD.pitch, xy_fields=xy_fields), axis=1)
    GD.positions['Grid Position'] = GD.positions.apply(lambda row: utils.get_field_position(row['x'], row['y'], GD.pitch, xy_fields=xy_fields), axis=1)
    #enabled events
    #GD=add_pass_enabled(GD)
    if enabled_passes:
        GD=_log_feasible_passes(GD, speed=enabled_with_speed)
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
    eventdf=eventdf[eventdf['eID']!='FinalWhistle'].reset_index()
    eventdf=eventdf.drop(columns=['index'])

    qualifier_df = pd.json_normalize(eventdf['qualifier'])
    eventdf = eventdf.join(qualifier_df)#.drop(columns=['qualifier'])

    eventdf['Team'] = eventdf['pID'].apply(lambda pid: utils.get_teamside_from_pID(pid, GD.team_sheets_df) if pd.notnull(pid) else None)
    #np.where(eventdf['tID'] == GD.home_tID, 'Home', 'Away')
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
def split_pass(ocel_df, framerate, team_sheets_df, positions_df,distance_threshold=0.2):
    ocel_df['pass_distance']=np.zeros(len(ocel_df))
    ocel_df['pass_angle']=np.zeros(len(ocel_df))
    for ev in ['Pass', 'Cross']:
        pass_mask = ocel_df['eID'].str.endswith(ev)
        pass_events = ocel_df[pass_mask].copy()
        pass_received = pass_events.copy()
        successfulpass=pass_received['Evaluation'].str.startswith('successful')
        failedpass=pass_received['Evaluation'].str.startswith('unsuccessful')# & pd.notna(pass_received['Recipient'])
        ocel_df.loc[successfulpass.index[successfulpass], 'outcome'] = 1
        ocel_df.loc[failedpass.index[failedpass], 'outcome'] = 0
        pass_events = ocel_df[pass_mask].copy()
        pass_received = pass_events.copy()

        # if simplified:
        pass_received.loc[successfulpass, 'eID'] = (
            ev + '_Received'
        )
        # else:
        #     pass_received.loc[successfulpass, 'eID'] = (
        #         pass_received.loc[successfulpass, 'eID'].astype(str) + '_Received'
        #     )
        pass_received.loc[failedpass, 'eID'] = (
            #pass_received.loc[failedpass, 'eID'].astype(str) + '_Intercepted'
            ev + '_Intercepted'
        )

        for idx, row in pass_received.iterrows():
            recipient = row['Recipient']
            if pd.notna(recipient):
                start_frame = row['Frame']
                session = row['Session']
                pos = pass_received.index.get_loc(idx)

                if pos < len(pass_received) - 1:
                    #next event frame
                    end_frame=pass_received.iloc[pos + 1]['Frame']
                else:
                    end_frame=start_frame+framerate*15
                
                recipient_positions = positions_df[(positions_df['pID'] == recipient) &
                                        (positions_df['Session'] == session)&
                                        (positions_df['Frame'] >= start_frame)&
                                        (positions_df['Frame'] <= end_frame)].copy()
            
            if not recipient_positions.empty:
                pass_received.loc[idx,'timestamp'] = pass_received.loc[idx,'timestamp'] + pd.to_timedelta(5/framerate, unit="s")#add one frame to the timestamp such that they do not happen at the same time
                pass_received.loc[idx,'Frame']= pass_received.loc[idx,'Frame'] + 5  # Increment frame by 5
                pass_received.loc[idx,'gameclock'] = pass_received.loc[idx,'gameclock'] + 5/framerate
                continue 
            # recipient_positions['distance_to_ball'] = np.sqrt(
            #     (recipient_positions['x'] - recipient_positions['ball_x'])**2 +
            #     (recipient_positions['y'] - recipient_positions['ball_y'])**2
            # )
            
            # received_event = recipient_positions[recipient_positions['distance_to_ball'] <= distance_threshold]
            received_event = recipient_positions[recipient_positions['Dist_ball'] <= distance_threshold]
            if received_event.empty:
                pass_received.loc[idx,'timestamp'] = pass_received.loc[idx,'timestamp'] + pd.to_timedelta(5/framerate, unit="s")#add one frame to the timestamp such that they do not happen at the same time
                pass_received.loc[idx,'Frame']= pass_received.loc[idx,'Frame'] + 5  # Increment frame by 5
                pass_received.loc[idx,'gameclock'] = pass_received.loc[idx,'gameclock'] + 5/framerate
                continue 

            received_row = received_event.iloc[0]#[received_event['Dist_ball'].idxmin()]
            #if row['Evaluation'].startswith('successful'):
            dx = received_row['ball_x'] - row['x']
            dy = received_row['ball_y'] - row['y']
            distance=np.sqrt(dx**2 + dy**2)
            pass_events.loc[idx,'pass_distance'] = distance
            pass_events.loc[idx,'pass_angle'] = np.degrees(np.arctan2(dy, dx))
            pass_received.loc[idx,'Frame'] = received_row['Frame']
            pass_received.loc[idx,'timestamp'] = received_row['timestamp'] 
            pass_received.loc[idx,'gameclock'] = received_row['gameclock'] 
            pass_received.loc[idx,'x'] = received_row['ball_x']
            pass_received.loc[idx,'y'] = received_row['ball_y']
        
        
        pass_received['pID']=pass_received['Recipient']
        #pass_received=pass_received[~pass_received['Recipient'].isna()]
        ocel_df[pass_mask]=pass_events

        pass_received['tID']=pass_received['Recipient'].apply(lambda x: utils.get_tID_from_pID(x, team_sheets_df))
        pass_received['Team']=pass_received['Recipient'].apply(lambda x: utils.get_teamside_from_pID(x, team_sheets_df))
        pass_received['Recipient'] = float("nan")

        temp_posession_lost = pass_received[pass_received['eID'].str.endswith('Intercepted')].copy()
        posession_lost = pass_events[failedpass].copy()
        posession_lost['eID']='Ball_lost'
        posession_lost['Frame']=(temp_posession_lost['Frame'].values-1).astype(int)
        posession_lost['timestamp']=temp_posession_lost['timestamp']-pd.to_timedelta(1/framerate, unit="s")
        posession_lost['gameclock']=temp_posession_lost['gameclock'].values-1/framerate

        
        ocel_df = pd.concat([ocel_df, posession_lost, pass_received], ignore_index=True)
    
    return ocel_df.sort_values(['Session', 'Frame']).reset_index(drop=True)

def _encode_pass_direction(pass_events: pd.DataFrame) -> pd.DataFrame:
    mask = pass_events['eID'].str.startswith('Play')

    def get_direction(angle):
        angle = angle % 360  # normalize to 0-360
        if angle <= 60 or angle >= 300:
            return 'Forward'
        elif 60 < angle < 120 or 240 < angle < 300:
            return 'Sideways'
        else:
            return 'Backward'

    pass_events.loc[mask, 'eID'] = pass_events[mask].apply(
        lambda row: get_direction(row['PlayAngle']) + '_' + row['eID'], axis=1
    )
    return pass_events


def _encode_pass_distance(pass_events: pd.DataFrame) -> pd.DataFrame:
    mask = pass_events['eID'].str.startswith('Play')

    pass_events.loc[mask, 'eID'] = pass_events[mask].apply(
        lambda row: row['eID'] + '_' + str(row['Distance']), axis=1
    )
    return pass_events

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
    shot.loc[failedshot, 'Team'] = shot.loc[failedshot, 'Team'].apply(
        lambda t: 'Away' if t == 'Home' else ('Home' if t == 'Away' else t)
    )
    shot.loc[failedshot, 'tID'] = shot.loc[failedshot, 'Team'].apply( lambda t: utils.get_tID_from_teamside(t, team_sheets_df))
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

def assign_possessionID_event_based(eventdf, Game, team_sheets_df):
    df = eventdf.copy()
    df = df.sort_values(["Session", "Frame"]).reset_index(drop=True)
    possession_ids = []
    attacking_team=[]
    defending_team=[]
    curr_id = 0
    team= df['Team'].iloc[0]  # Initialize with the first team's name
    teamID= df['tID'].iloc[0]
    for idx, row in df.iterrows():
        change = False
        eid=row['eID'].lower()
        
        for keyword in ['intercepted', 'ballclaiming', 'freekick'
                        , 'penalty', 'fairplay', 'goalkick'
                        , 'throwin', 'cornerkick', 'kickoff']:
            if keyword in eid:
                if ('notawarded' not in eid) and ('received' not in eid):
                    change = True
                    break
        if not change:
            if idx>0:
                peid=df.loc[idx-1,'eID'].lower()
                for keyword in ['scored', 'blockedshot', 'savedshot'
                                , 'shotwide','othershot', 'refereeball']:
                    if keyword in peid:
                        change = True
                        break
            if row['eID'] == 'TacklingGame' and str(row.get('PossessionChange', '')).lower() == 'true':
                if row['WinnerTeam'] != df.loc[idx - 1, 'tID']:
                    change = True

        if change:
            curr_id += 1
            team = row['Team']
            teamID=row['tID']
            if team not in ['Home', 'Away']:
                if teamID:
                     team = utils.get_teamside_from_tID(teamID, team_sheets_df)
                elif idx + 1 < len(df):
                    team = df.loc[idx + 1, 'Team']
                    teamID = df.loc[idx + 1, 'tID']

        possession_ids.append(f"{Game}_{team}_{curr_id}")
        if teamID==utils.get_tID_from_pID(row['pID'], team_sheets_df):
            attacking_team.append(row['pID'])
            defending_team.append(float("nan"))
        else:
            attacking_team.append(float("nan"))
            defending_team.append(row['pID'])


    df['possessionID'] = pd.Series(possession_ids)
    df['attacking_team']=attacking_team
    df['defending_team']=defending_team
    return df


def frame_possessionID_matching_singleDF(df):
    df = df.sort_values(["Session", "Frame"])
    possession_map = (
        df[df["possessionID"].notna()]
        .groupby(["Session", "possessionID"])["Frame"]
        .min()
        .reset_index()
        .rename(columns={"Frame": "start_frame"})
        .sort_values(["Session", "start_frame"])
    )

    def assign_possession(row):
        session_map = possession_map[possession_map["Session"] == row["Session"]]
        active = session_map[session_map["start_frame"] <= row["Frame"]]
        if active.empty:
            return None
        return active.iloc[-1]["possessionID"]

    df["possessionID"] = df.apply(assign_possession, axis=1)
    return df

def frame_possessionID_matching(df_withID, df_withoutID):
    possession_map = (
        df_withID[df_withID["possessionID"].notna()]
        .groupby(["Session", "possessionID"])["Frame"]
        .min()
        .reset_index()
        .rename(columns={"Frame": "start_frame"})
        .sort_values(["Session", "start_frame"])
    )

    def assign_possession(row):
        session_map = possession_map[
            possession_map["Session"] == row["Session"]
        ]

        active = session_map[
            session_map["start_frame"] <= row["Frame"]
        ]

        if active.empty:
            return None

        return active.iloc[-1]["possessionID"]

    result = df_withoutID.copy()
    result["possessionID"] = result.apply(assign_possession, axis=1)
    return result

# pass and cross recipient encoding
import json
from pathlib import Path
import os


def encode_position(GD, encode='role', role_json_path=None):
    """
    Append receiver position/role info to eID for successful pass events.

    Parameters
    ----------
    GD : GameData object
    encode : which attribute to use
             'role'            → raw position code from team_sheets_df (original behavior)
             'role:<category>' → look up the position code in role_groups.json, then use the <category> grouping for it
             'name'            → recipient's pID (player name/id) itself
             'zone'            → zone from field_gdf containing recipient's position
             'tuple'           → grid (col,row) tuple from recipient's position
             'label'           → letter-number field label (e.g. 'C4') from recipient's position
    role_json_path : path to role_groups.json (only used for 'role:<category>' / 'role:all')
    """
    def _get_recipient_xy(row):
        """Look up recipient's x, y from GD.positional_events via Session + Frame + pID."""
        session = row['Session']
        frame = row['Frame']
        recipient = row['Recipient']

        match = GD.positions.loc[
            (GD.positions['Session'] == session) &
            (GD.positions['Frame'] == frame) &
            (GD.positions['Player'] == recipient),
            ['x', 'y']  
        ]
        if match.empty:
            return None, None
        return match['x'].values[0], match['y'].values[0]
    pass_events = GD.events[
        GD.events['Recipient'].notna() &
        GD.events['Evaluation'].str.startswith('successful')
    ].copy()

    pass_events['player_position'] = pass_events.apply(
        lambda row: GD.team_sheets_df.loc[GD.team_sheets_df['pID'] == row['Recipient'], 'position'].values[0],
        axis=1
    )

    if encode == 'role':
        suffix = pass_events['player_position'].astype(str)

    elif encode.startswith('role:'):
        category = encode.split(':', 1)[1]

        path = role_json_path or os.path.join(Path(__file__).resolve().parent, "role_groups.json")
        with open(path, "r", encoding="utf-8") as f:
            position_groups = json.load(f)

        def _get_group(position_code, group_type):
            info = position_groups.get(position_code, {})
            return info.get(group_type, "Unknown") if info else "Unknown"

        
        suffix = pass_events['player_position'].apply(lambda pos: _get_group(pos, category))

    elif encode == 'name':
        suffix = pass_events['Recipient'].astype(str)

    elif encode in ('tuple', 'label', 'zone'):
        arg={
            'pitch':GD.pitch,
            'xy_fields': GD.settings['xy_fields'],
            'field_gdf': GD.field_gdf
        }
        if encode == 'tuple':
            if arg['pitch'] is None:
                raise ValueError("arg['pitch'] must be provided when encode='tuple'.")
            def _to_tuple(row):
                x, y = _get_recipient_xy(row)
                if x is None:
                    return "Unknown"
                col, r = utils.get_field_position(x, y, arg['pitch'], arg['xy_fields'])
                return f"Grid_{col}_{r}"
            suffix = pass_events.apply(_to_tuple, axis=1)
        elif encode == 'label':
            if arg['pitch'] is None:
                raise ValueError("arg['pitch'] must be provided when encode='label'.")
            def _to_label_str(col, r):
                return f"{chr(ord('A') + col)}{r + 1}"
            def _to_label(row):
                x, y = _get_recipient_xy(row)
                if x is None:
                    return "Unknown"
                col, r = utils.get_field_position(x, y, arg['pitch'], arg['xy_fields'])
                return _to_label_str(col, r)
            suffix = pass_events.apply(_to_label, axis=1)

        elif encode == 'zone':
            if arg['field_gdf'] is None:
                raise ValueError("arg['field_gdf'] must be provided when encode='zone'.")
            from shapely.geometry import Point
            def _to_zone(row):
                x, y = _get_recipient_xy(row)
                if x is None:
                    return "Unknown"
                point = Point(x, y)
                zones = arg['field_gdf'][
                    (arg['field_gdf']["name"] != "field") &
                    arg['field_gdf'].geometry.contains(point)
                ]["name"].tolist()
                return ", ".join(zones) if zones else "open_play"
            suffix = pass_events.apply(_to_zone, axis=1)


    else:
        raise ValueError(
            f"encode must be 'role' or 'role:<category>'. Got {encode!r}."
        )
    
    pass_events['eID'] = pass_events['eID'] + '_' + suffix

    outdf = GD.events.copy()
    outdf[
        GD.events['Recipient'].notna() &
        GD.events['Evaluation'].str.startswith('successful')
    ] = pass_events
    GD.events = outdf.sort_values(['Session', 'timestamp']).reset_index(drop=True)

    return GD
