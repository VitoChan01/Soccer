import os
import json
from pathlib import Path
import pandas as pd
# player grouping
def assign_roles(GD, process_DF='ALL', team=False, category='role', role_json_path=None):
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
    # Track unknown position codes
    unknown_positions = set()

    def get_group(position_code: str, group_type: str) -> str:
        info = position_groups.get(position_code, {category: "Unknown"})
        if info is None:
            unknown_positions.add(position_code)
        return info.get(group_type, "Unknown")

    def find_position(player_name: str) -> str:
        row = GD.team_sheets_df.loc[GD.team_sheets_df["pID"] == player_name, "position"]
        return row.values[0] if len(row) > 0 else None

    def find_role(player_name: str) -> str:
        pos = find_position(player_name)
        return get_group(pos, category) if pos else "Unknown"

    def processing(df, category):
        df[category] = df["pID"].apply(find_role)
        roles = list({v[category] for v in position_groups.values() if category in v})

        if team:
            for role in roles:
                try:
                    df[f"{role}_h"] = df["pID"].where(
                        (df[category] == role) & (df["Team"] == "Home"), None
                    )
                    df[f"{role}_a"] = df["pID"].where(
                        (df[category] == role) & (df["Team"] == "Away"), None
                    )
                except:
                    df[f"{role}_h"] = df["pID"].where(
                        (df[category] == role) & (df["Home_Away"] == "Home"), None
                    )
                    df[f"{role}_a"] = df["pID"].where(
                        (df[category] == role) & (df["Home_Away"] == "Away"), None
                    )
        for role in roles:
            df[role] = df["pID"].where(df[category] == role, None)
        return df
    if process_DF=='EVENTS' or process_DF == "ALL":
        GD.events=processing(GD.events, category)
    if process_DF=='POSITIONAL' or process_DF == "ALL":
        GD.positional_events=processing(GD.positional_events, category)
    if process_DF=='MOVEMENT' or process_DF == "ALL":
        GD.movement_events=processing(GD.movement_events, category)
    if process_DF=='TEAMSHEET' or process_DF == "ALL":
        GD.team_sheets_df=processing(GD.team_sheets_df, category)
    if unknown_positions:
        print(
            f"Warning: The following role(s) were not found in role_groups.json: {sorted(unknown_positions)}"
        )
    return GD
def assign_roles_multi(GD, process_DF='ALL', team=False, categories=None, role_json_path=None):
    if not role_json_path:
        role_json_path=os.path.join(Path(__file__).resolve().parent, "role_groups.json")

    with open(role_json_path, "r", encoding="utf-8") as f:
        position_groups = json.load(f)
    if not categories:
        categories = list({k for v in position_groups.values() for k in v.keys()})
    for category in categories:
        GD=assign_roles(GD, process_DF, team, category, role_json_path)
    return GD
