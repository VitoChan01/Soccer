import pandas as pd

def filter_by_last_player(df):
    relevant_events = df[
        df['concept:name'].str.startswith('SHOT') |
        df['concept:name'].str.endswith('GOAL')
    ].set_index('case:concept:name')['From']
    
    # Step 3: Filter original DataFrame
    def include_row(row):
        if row['concept:name'] != 'Player changes position':
            return True
        last_player = relevant_events.get(row['case:concept:name'])
        return row['From'] == last_player
    
    return df[df.apply(include_row, axis=1)]

def filter_by_involved_player(df):
    grouped = df.groupby('case:concept:name')

    def include_row(row):
        if row['concept:name'] != 'Player changes position':
            return True
        
        case_id = row['case:concept:name']
        player = row['From']
        case_df = grouped.get_group(case_id)

        case_df_other = case_df[case_df['concept:name'] != 'Player changes position']

        involved = (
            (case_df_other['From'] == player) |
            (case_df_other['To'] == player)
        )
        return involved.any()
    return df[df.apply(include_row, axis=1)]

def movement_frequency_last_player(df):
    result = []

    for team in ['Home', 'Away']:
        if team:
            team_df = df[df['Team'] == team]
        else:
            team_df=df

        for case_id, group in team_df.groupby('case:concept:name'):
            group_sorted = group.sort_values(by='time:timestamp', ascending=True)

            if group_sorted['From'].dropna().empty:
                continue  # Skip if no From values

            last_player = group_sorted['From'].dropna().iloc[-1]

            position_change_count = group[
                (group['concept:name'] == 'Player changes position') &
                (group['From'] == last_player)
            ].shape[0]

            result.append({
                'case:concept:name': case_id,
                'team': team,
                'last_player': last_player,
                'position_changes': position_change_count
            })
    result=pd.DataFrame(result)
    
    return result
def movement_frequency_involved_players(df):
    result = []

    for team in ['Home', 'Away']:
        if team:
            team_df = df[df['Team'] == team]
        else:
            team_df=df

        for case_id, group in team_df.groupby('case:concept:name'):
            players = pd.concat([group['From'], group['To']]).dropna().unique()

            counts = group[
                (group['concept:name'] == 'Player changes position') &
                (group['From'].isin(players))
            ]['From'].value_counts()

            total = counts.sum()
            avg = counts.mean() if len(counts) > 0 else 0

            result.append({
                'case:concept:name': case_id,
                'team': team,
                'total_position_changes': total,
                'average_per_player': avg
            })

    result=pd.DataFrame(result)
    return result

def movement_frequency_report(last_df, involved_df):
    last_result=movement_frequency_last_player(last_df)
    involved_result=movement_frequency_involved_players(involved_df)
    result = pd.merge(
        last_result,
        involved_result,
        on=['case:concept:name', 'team'],
        how='outer'  # Use 'outer' in case any team/case appears in one but not the other
    )
    print(result)
    
    total=sum(result['position_changes'])
    average=(result['position_changes']).mean()
    total_t=sum(result['total_position_changes'])
    average_t=(result['total_position_changes']).mean()
    average_p=(result['average_per_player']).mean()
    shot_count=result['case:concept:name'].nunique()
    print(f'In this game, a total of {shot_count} shots were made.', 
            f'\nThe aggregated total shooter grid travel is {total}',
            f'\nThe average shooter grid travel per shot is {average}',
            f'\nThe aggregated total grid travel of all involved players is {total_t}',
            f'\nThe average grid travel of all attacks involved players per shot is {average_t}',
            f'\nThe average grid travel of all attacks involved players per player per shot is {average_p}\n\n')
    for team in ['Home','Away']:
        team_result=result[result['team']==team]
        total=sum(team_result['position_changes'])
        average=(team_result['position_changes']).mean()
        total_t=sum(team_result['total_position_changes'])
        average_t=(team_result['total_position_changes']).mean()
        average_p=(team_result['average_per_player']).mean()
        shot_count=team_result['case:concept:name'].nunique()
        print(team, f'team, made a total of {shot_count} shots.', 
              f'\nThe aggregated total shooter grid travel is {total}',
              f'\nThe average shooter grid travel per shot is {average}',
              f'\nThe aggregated total grid travel of all involved players is {total_t}',
              f'\nThe average grid travel of all attacks involved players per shot is {average_t}',
              f'\nThe average grid travel of all attacks involved players per player per shot is {average_p}\n\n')
    return result