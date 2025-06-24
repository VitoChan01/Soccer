# Transforming Football Data into Object-centric Event Logs with Spatial Context Information


Object-centric event logs expand the conventional single-case notion event log by considering multiple objects, allowing for the analysis of more complex and realistic process behavior. However, the number of real-world object-centric event logs remains limited, and further studies are needed to test their usefulness. The increasing availability of data from team sports can facilitate object-centric process mining, leveraging both real-world data and suitable use cases. In this paper, we present a framework for transforming football data into an object-centric event log, further enhanced with a spatial dimension. We demonstrate the effectiveness of our framework by generating object-centric event logs based on real-world football data and discuss the results for varying process representations. With our paper, we provide the first example for object-centric event logs in football analytics. Future work should consider variant analysis and filtering techniques to better handle variability.


This repository contains the implementation of "Transforming Football Data into Object-centric Event Logs with Spatial Context Information"

## Dependencies
* Python 3.11+
## Required packages
For required packages, please see [requirements.txt](requirements.txt).

To install all required packages: 
```
pip install -r requirements.txt
```
## Data
### Metrica Sports Sample Data 
This study utilize the football game sample data from [Metrica Sports Github](https://github.com/metrica-sports/sample-data?tab=readme-ov-file). The data shouold be stored in the [sample-data](sample-data) directory.

To pull the data:
```
git clone https://github.com/metrica-sports/sample-data.git
```
### Import of World Cup Dataset

The world cup dataset is only required for the analysis *analyze_alternative_datasets.ipynb*.

1. Download the dataset from https://fluxicon.com/blog/2019/10/process-mining-meets-football-how-does-a-football-team-possess-the-ball-on-the-pitch/ (see *download here*)
2. Unzip and place under 'data/02_world_cup_dataset'

Your folder structure should now be
data/02_world_cup_dataset/matches/...
data/02_world_cup_dataset/teams/...
## Codes
### Implementation
- [`convert_ocel.ipynb`](convert_ocel.ipynb): 
    - Example notebook to transform the the football data into object-centric event log of game-based events without the player movement events.
- [`import_tracking_data.ipynb`](import_tracking_data.ipynb): 
    - Example notebook to transform the the tracking data into object-centric event log of player movement events.
- [`measurements.ipynb`](measurements.ipynb): 
    - Experiments wiht performance measurements.
    - Statistical analysis.
    - Visualization of process instance on spatial map.
- [`ocel_with_movement.ipynb`](ocel_with_movement.ipynb): 
    - Example notebook to convert the the football data into object-centric event log of game-based events with the player movement events.
- [`ocel_with_movement_df.ipynb`](ocel_with_movement_df.ipynb): 
    - Example notebook to convert the the football data into a Pandas DataFrame structured for object-centric event log of game-based events with the player movement events.
    - Various filtering of event log and visualization with DFG.


### Modules
- [`Soccer_ocel.py`](Soccer_ocel.py): 
    - Functinos to transform the foodball data into an object-centric event log.
- [`filters_and_analysis.py`](filters_and_analysis.py): 
    - Functinos to filter the event log by the last player of a possession or by all players involved in a possession. 
    - Function to report movement statistics within a possession.
- [`helpers.py`](helpers.py): 
    - Functino to get the field position in grid coordinate system. 

## Event log
The event log has the following attributes:
 Object Type         | Description                                                           
 ------------------- | ---------------------------------------------------------------------
 `Team`              | The team associated with the event.                                   
 `Player`              | The player in possession of the ball or associated with the event.       |
 `Possession` | Unique ID for each case--an attack. 'AA' indicates away team attack, 'HA' indicates home team attack    
 `Ball` | The football.
 `end_grid`          | The grid zone where the event ends.                              

The generated event log has the following attributes:
 Attribute                      | Description      
  ------------------- | ---------------------------------------------------------------------                                         
 `attribute:subtype`            | Event subtype (e.g. `"PASS"`, `"GOAL"`, `"SHOT"`). For details please refer to the data documentation.     
 `attribute:start_x`, `start_y` | X/Y coordinates of the ball at the start of the event.        
 `attribute:end_x`, `end_y`     | X/Y coordinates of the ball at the end of the event.          
 `attribute:duration`           | Duration (in seconds) of the event.                           
 `attribute:travel_distance`    | Distance traveled by the ball.                                
 `attribute:start_grid`         | Grid ID where the event begins.                               
 `attribute:crossed_grid`       | If the event start in one grid and ends in another.           
 `attribute:attack_game`        | Identifier for attack sequence. Combined count of both team. 
 `attribute:attack_successful`  | Boolean flag — `True` if the attack led to a goal.            
 `attribute:home_team_score`    | Cumulative score of the home team at this point in the match.
 `attribute:away_team_score`    | Cumulative score of the away team at this point in the match. 

## License 
[LICENSE](LICENSE)

