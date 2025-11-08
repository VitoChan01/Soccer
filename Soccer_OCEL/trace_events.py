from scipy.signal import savgol_filter, find_peaks
import numpy as np
import pandas as pd

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
