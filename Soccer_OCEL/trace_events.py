from scipy.signal import savgol_filter, find_peaks
import numpy as np
import pandas as pd
from shapely.geometry import LineString
import geopandas as gpd

def position_events(GameData):
    field_y=GameData.pitch.ylim[1]
    field_x=GameData.pitch.xlim[1]
    all_repositions=[]
    for team in ['Home','Away']:
        FH_TeamSide= 'Right' if team==GameData.FH_TeamRight else 'Left'
        team_df=GameData.positions.query('Team == @team').copy().reset_index(drop=True)
        players=team_df['Player'].unique()
        for player in players:
            player_repos=player_repositioning(player, team_df, FH_TeamSide, field_y, field_x, team, GameData.framerate)
            all_repositions.append(player_repos)
    position_events_df=pd.concat(all_repositions).reset_index(drop=True)
    return position_events_df

def player_repositioning(player, positions_df, FH_TeamSide, field_y, field_x
                         , team, framerate=25, speed_smooth_window=10, metric='mean'
                         , speed_threshold=2.5, duration_threshold=5, Holding_speed_threshold=1.5):
    repositions=[]
    for session in range(1,3):    
        xy=positions_df.loc[(positions_df['Player']==player) & (positions_df['Session']==session),['x','y']].values
        movement=np.sqrt(np.sum(np.diff(xy,axis=0)**2,axis=1))*framerate
        #direction=np.arctan2(np.diff(xy[:,1]), np.diff(xy[:,0]))*180/np.pi
        frame=positions_df.loc[(positions_df['Player']==player) & (positions_df['Session']==session),'Frame'].values[1:]
        if len(movement) == 0 or len(movement) < framerate * duration_threshold:
            continue
        movement_flt=savgol_filter(movement, framerate*speed_smooth_window, 3)
        
        peaks, peak_prop= find_peaks(movement_flt, height=0)
        peak_height=peak_prop['peak_heights']
        ac_dc=np.diff(peak_height)
        significant_peaks=np.where(np.abs(ac_dc)>2)[0]
        zc=significant_peaks[np.where(np.diff(np.sign(ac_dc[significant_peaks])))[0]]
        zc_frame=peaks[zc]

        boundaries = np.concatenate(([0], np.asarray(zc_frame, dtype=int), [len(movement)-2]))

        seg_stats = []
        for i in range(len(boundaries) - 1):
            start, end = int(boundaries[i]), int(boundaries[i+1])
            if end-1 <= start:
                continue
            seg_vals = movement[start:end]
            mean_v = np.nanmean(seg_vals)
            median_v = np.nanmedian(seg_vals)
            try:
                y_travels, x_travels = xy[start:end, 1], xy[start:end, 0]
                y_travel = y_travels[-1] - y_travels[0]
                x_travel = x_travels[-1] - x_travels[0]
            except IndexError:
                continue
            
            movement_direction = np.arctan2(y_travel, x_travel) * 180 / np.pi
            distance = np.sqrt(y_travel**2 + x_travel**2)
            event=annotate_movement(x_travel, y_travels[0], y_travels[-1], x_travels[0], x_travels[-1], FH_TeamSide, movement_direction, field_y=field_y, field_x=field_x, mean_v=mean_v)
            seg_stats.append({'eID': event, 'pID': player, "Frame": frame[start], "end_frame": frame[end], 'n': frame[end]-frame[start]
                              , "mean_speed": mean_v, "median_speed": median_v, "direction": movement_direction
                              ,'travel':distance, 'Session':session, 'Team':team
                              , 'x':x_travels[0]
                              , 'y':y_travels[0]
                              , 'end_x':x_travels[-1]
                              , 'end_y':y_travels[-1]
                              , 'trajectory': LineString(zip(x_travels, y_travels))
                              })

        seg_df = pd.DataFrame(seg_stats)
        seg_df_flt=seg_df[(seg_df[metric+'_speed']>speed_threshold)&(seg_df['n']>=framerate*duration_threshold)]
        repositions.append(seg_df_flt)
        seg_df_hold=seg_df[(seg_df[metric+'_speed']<=Holding_speed_threshold)&(seg_df['n']>=framerate*duration_threshold)]
        repositions.append(seg_df_hold)
        FH_TeamSide= 'Right' if FH_TeamSide=='Left' else 'Left'  

    return pd.concat(repositions).reset_index(drop=True)

def annotate_movement(dx, start_y, end_y, start_x, end_x, TeamSide, movement_direction, field_y, field_x, mean_v):
    dx=-dx if TeamSide=='Right'else dx
    center_min = 0.3 * field_y
    center_max = 0.7 * field_y
    start_position = "Central" if center_min <= start_y <= center_max else "Wing"
    end_position = "Central" if center_min <= end_y <= center_max else "Wing"
    
    
    midfield_min = 0.3 * field_x
    midfield_max = 0.7 * field_x

    def x_block(x, TeamSide, midfield_min, midfield_max):
        action='Holding'
        if x<midfield_min:
            # if TeamSide=='Left':
            #     action='Holding'
            #     block='Defensive First'
            # else:
                #action='Securing'
            block='Attack Third' 
        elif midfield_min<=x<=midfield_max:
            #action='Holding'
            block='Midfield'
        else:
            # if TeamSide=='Left':
            #     action='Securing'
            #     block='Attack Third' 
            # else:
                #action='Holding'
            block='Defensive First'
        return action, block

        
    if mean_v<=1.5:
        action, block=x_block(start_x, TeamSide, midfield_min, midfield_max)
        return action + ' ' + start_position + ' ' + block

    if dx > 0:
        h_direction = " Advance"
    else:
        h_direction = " Retreat"

    if abs(movement_direction) < 30:
        return end_position+h_direction
    elif abs(movement_direction) < 75:
        if start_position!=end_position:
            return end_position+' Transition'
        else:
            return 'Diagonal Repositioning'
    else:
        if start_position!=end_position:
            return end_position+' Transition'
        else:
            return 'Lateral Repositioning'

def movement_events(positions_df, FH_TeamRight, pitch, movement_directions=None, step=3, max_gap=None, movement_noise_threshold=0.09):
    '''
    Step: subseting the df by n step. 
    '''

    x_min, x_max = pitch.xlim[0],pitch.xlim[1]
    y_min, y_max = pitch.ylim[0],pitch.ylim[1]
    tolerance=0

    positions_df = positions_df[
        (positions_df["x"] >= x_min-tolerance) & (positions_df["x"] <= x_max+tolerance) &
        (positions_df["y"] >= y_min-tolerance) & (positions_df["y"] <= y_max+tolerance)
    ]

    gdf = gpd.GeoDataFrame(
        positions_df,
        geometry=gpd.points_from_xy(positions_df["x"], positions_df["y"])
    )
    gdf = gdf.sort_values(["Player", "Session", "Frame"]).reset_index(drop=True)
    gdf['ball_loc']=gpd.GeoSeries.from_xy(gdf["ball_x"], gdf["ball_y"])
    gdf["Dist_ball"] = gdf.geometry.distance(gdf["ball_loc"])

    if step:
        gdf['Frame_drop']=gdf['Frame']%step
        gdf=gdf.query('Frame_drop==0').copy()
        gdf.drop(['x','y','ball_x','ball_y','Frame_drop'], axis=1, inplace=True)
    else:
        gdf.drop(['x','y','ball_x','ball_y'], axis=1, inplace=True)
    gdf = gdf.reset_index(drop=True)
    next_geom = gdf.groupby(["Player", "Session"])["geometry"].shift(-1)

    gdf["movement"] = gdf.geometry.distance(next_geom)
    gdf = gdf[(gdf["movement"].notna())].copy()
    gdf = gdf[gdf["movement"] > movement_noise_threshold]

    dx = next_geom.x - gdf.geometry.x
    dy = next_geom.y - gdf.geometry.y
    gdf["dx"] = dx
    gdf["dy"] = dy

    gdf["angle"] = np.arctan2(gdf["dy"], gdf["dx"])
    gdf["angle"] = (np.degrees(gdf["angle"]) + 360) % 360
    mask = (gdf["dx"] == 0) & (gdf["dy"] == 0)
    gdf.loc[mask, "angle"] = None

    mask = (
        ((gdf["Team"] == FH_TeamRight) & (gdf["Session"] == 1)) |
        ((gdf["Team"] != FH_TeamRight) & (gdf["Session"] == 2))
    )
    gdf["angle_attacking"] = gdf["angle"]
    gdf.loc[mask, "angle_attacking"] = (gdf.loc[mask, "angle_attacking"] + 180) % 360
    
    gdf = attack_angle_to_dir(gdf, movement_directions)
    gdf = label_aggregation(gdf,step , max_gap)
    return gdf

def attack_angle_to_dir(gdf, movement_directions=None):
    if movement_directions==16:
        bins = [0, 11.25,33.75,56.25,78.75,101.25,123.75,146.25,168.75,191.25,213.75,236.25,258.75,281.25,303.75,326.25,348.75,360]
        labels = [
            "Forward",
            "Slight Forward-Right",
            "Forward-Right",
            "Strong Forward-Right",
            "Right",
            "Strong Backward-Right",
            "Backward-Right",
            "Slight Backward-Right",
            "Backward",
            "Slight Backward-Left",
            "Backward-Left",
            "Strong Backward-Left",
            "Left",
            "Strong Forward-Left",
            "Forward-Left",
            "Slight Forward-Left",
            "Forward"
        ]

        gdf["eID"] = pd.cut(
            gdf["angle_attacking"],
            bins=bins,
            labels=labels,
            include_lowest=True,
            right=False,
            ordered=False
        )

        labels = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,0]

        gdf["dir_16_num"] = pd.cut(
            gdf["angle_attacking"],
            bins=bins,
            labels=labels,
            include_lowest=True,
            right=False,
            ordered=False
        ).astype("Int64")
    else:
        bins = [0, 22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5, 360]
        labels = [
            "Forward",
            "Forward-Right",
            "Right",
            "Backward-Right",
            "Backward",
            "Backward-Left",
            "Left",
            "Forward-Left",
            "Forward"
        ]

        gdf["eID"] = pd.cut(
            gdf["angle_attacking"],
            bins=bins,
            labels=labels,
            include_lowest=True,
            right=False,
            ordered=False
        )
        labels = [0,1,2,3,4,5,6,7,0]
        gdf["dir_8_num"] = pd.cut(
            gdf["angle_attacking"],
            bins=bins,
            labels=labels,
            include_lowest=True,
            right=False,
            ordered=False
        ).astype("Int64")
        gdf=gdf.rename(columns={"Player":"pID"})
    return gdf

def circular_mean(series):
    angles = series.dropna()
    if len(angles) == 0:
        return np.nan
    radians = np.radians(angles)
    return np.degrees(
        np.arctan2(np.mean(np.sin(radians)), np.mean(np.cos(radians)))
    ) % 360

def label_aggregation(gdf, step, max_gap=None):
    '''
    max_gap is prioritized over step. If max_gap is None, step will be used. If max_gap==-1(/<0), identical events will be aggregated regardless of frame brake.
    '''
    if max_gap:
        pass
    elif step:
        if step>5:
            max_gap=step
        else:
            max_gap=10
    else:
        max_gap=10
    gdf = gdf.sort_values(["pID", "Session", "Frame"]).copy()
    frame_gap = gdf.groupby(["pID", "Session"])["Frame"].diff()

    gdf["segment_id"] = (
        (gdf["eID"] != gdf["eID"].shift()) |
        (gdf["pID"] != gdf["pID"].shift()) |
        (gdf["Session"] != gdf["Session"].shift()) |
        (frame_gap > max_gap)
    ).cumsum()

    if max_gap<0:
        gdf["segment_id"] = (
            (gdf["eID"] != gdf["eID"].shift()) |
            (gdf["pID"] != gdf["pID"].shift()) |
            (gdf["Session"] != gdf["Session"].shift())
            ).cumsum()

    agg_dict = {
        "Frame": ["first", "last"],
        "movement": "sum",
        "dx": "sum",
        "dy": "sum",
        "angle": circular_mean,
        'angle_attacking': circular_mean
    }

    other_cols = [
        col for col in gdf.columns
        if col not in ["Frame", "movement", "dx", "dy", "angle",
                    "pID", "Session", "eID", "segment_id"]
    ]

    for col in other_cols:
        agg_dict[col] = "first"

    result = (
        gdf.groupby(
            ["pID", "Session", "eID", "segment_id"],
            observed=True,
            as_index=False,
            sort=False  # preserve segment order
        )
        .agg(agg_dict)
    )
    result.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in result.columns
    ]

    result = result.rename(columns={
        "Frame_first": "Frame",
        "Frame_last": "end_frame"
    })
    result["duration_frame"] = result["end_frame"] - result["Frame"]+1

    result = result.drop(columns="segment_id")
    result.columns = [col.replace("_first", "") for col in result.columns]
    result.columns = [col.replace("_sum", "") for col in result.columns]
    result.columns = [col.replace("_circular_mean", "") for col in result.columns]
    return result