import json
position_groups = {
    # Goalkeeper
    "TW": {"role": "Goalkeeper", "zone": "defensive", "side": "central", "position": "keeper"},
    
    # Defenders
    "IVL": {"role": "Defender", "zone": "central", "side": "central", "position": "center-back"},
    "IVR": {"role": "Defender", "zone": "central", "side": "central", "position": "center-back"},
    "LV":  {"role": "Defender", "zone": "wing", "side": "left", "position": "fullback"},
    "RV":  {"role": "Defender", "zone": "wing", "side": "right", "position": "fullback"},
    
    # Defensive Midfielders
    "DML": {"role": "Midfielder", "zone": "central", "side": "left", "position": "defensive-mid"},
    "DMR": {"role": "Midfielder", "zone": "central", "side": "right", "position": "defensive-mid"},
    "DLM": {"role": "Midfielder", "zone": "central", "side": "central", "position": "defensive-mid"},
    
    # Attacking Midfielders / Wingers
    "OLM": {"role": "Midfielder", "zone": "wing", "side": "left", "position": "attacking-mid"},
    "ORM": {"role": "Midfielder", "zone": "wing", "side": "right", "position": "attacking-mid"},
    "ZO":  {"role": "Midfielder", "zone": "central", "side": "central", "type": "playmaker"},
    
    # Forwards
    "STL": {"role": "Forward", "zone": "wing", "side": "left", "position": "forward"},
    "STZ": {"role": "Forward", "zone": "central", "side": "central", "position": "forward-central"},
}
with open("Soccer_OCEL/role_groups.json", "w", encoding="utf-8") as f:
    json.dump(position_groups, f, indent=4)