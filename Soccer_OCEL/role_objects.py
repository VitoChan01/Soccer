import os
import json
from pathlib import Path
import pandas as pd
# player grouping
def assign_roles(GD, team=False, category='role', role_json_path=None):
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

    if team:
        for role in roles:
            GD.events[f"{role}_h"] = GD.events["pID"].where(
                (GD.events[category] == role) & (GD.events["Team"] == "Home"), None
            )
            GD.events[f"{role}_a"] = GD.events["pID"].where(
                (GD.events[category] == role) & (GD.events["Team"] == "Away"), None
            )
    for role in roles:
        GD.events[role] = GD.events["pID"].where(GD.events[category] == role, None)
        
    #GD.events.drop(category, axis=1, inplace=True)

    
    # GD.events[category+'_r'] = GD.events["Recipient"].apply(find_role)
    # roles = list({v[category] for v in position_groups.values() if category in v})
    # for role in roles:
    #     GD.events[role+'_r'] = GD.events["Recipient"].where(GD.events[category+'_r'] == role, None)   
    #     GD.events[role] = GD.events.apply(
    #         lambda row: (
    #             # set([x for x in [row[role], row[role + '_r']] if pd.notna(x)])
    #             # if pd.notna(row[role]) or pd.notna(row[role + '_r'])
    #             # else None
    #             row[role]+';'+ row[role + '_r'] if pd.notna(row[role]) and pd.notna(row[role + '_r'])
    #             else row[role] if pd.notna(row[role]) 
    #             else row[role+'_r'] if pd.notna(row[role + '_r'])
    #             else None
    #             #row[role] if pd.notna(row[role])
    #             #else row[role + '_r'] if pd.notna(row[role + '_r'])
    #             #else None
    #         ),
    #         axis=1
    #     )
    #     GD.events.drop(role+'_r', axis=1, inplace=True)
    #GD.events = GD.events.copy()
    return GD
def assign_roles_multi(GD, team=False, categories=None, role_json_path=None):
    if not role_json_path:
        role_json_path=os.path.join(Path(__file__).resolve().parent, "role_groups.json")

    with open(role_json_path, "r", encoding="utf-8") as f:
        position_groups = json.load(f)
    if not categories:
        categories = list({k for v in position_groups.values() for k in v.keys()})
    for category in categories:
        GD=assign_roles(GD, team, category, role_json_path)
    return GD
