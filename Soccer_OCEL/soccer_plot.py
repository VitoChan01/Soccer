from shapely.geometry import Polygon, LineString, Point
import geopandas as gpd
from adjustText import adjust_text
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
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

def fastplot(gdf):
    base=field_gdf.plot(facecolor="none", edgecolor="black", linewidth=2)
    #base = field_gdf.boundary.plot(color="black", linewidth=2)
    gdf.plot(ax=base)
def plot_background():
    base=field_gdf.plot(facecolor="none", edgecolor="black", linewidth=2)
    return base
def plot_background_ax(ax):
    field_gdf.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=2)
def plot_trace(gdf, base, color):
    gdf.plot(ax=base, edgecolor=color)

def add_legend(base):

    legend_elements = [
        Line2D([0], [0], color="blue", linewidth=1, label="Home"),
        Line2D([0], [0], color="red", linewidth=1, label="Away"),
        Line2D([0], [0], color="green", linewidth=1, label="Ball"),
    ]

    base.legend(handles=legend_elements, loc="upper right")


def add_event(df, ax, color):
    texts = []
    for _, row in df.iterrows():
        t = ax.annotate(
            row["concept:name"],
            xy=(row["attribute:x"], row["attribute:y"]),
            bbox=dict(boxstyle="round,pad=0.2", facecolor=color, edgecolor="black", alpha=0.7),
            fontsize=6
        )
        texts.append(t)
    
    adjust_text(texts, ax=ax)


def vis_possession(GameData, possessionID):
    df=GameData.events.query('`case:concept:name`==@possessionID')
    start_frame, end_frame, session=np.min(df['attribute:frame']),np.max(df['attribute:frame']),np.unique(df['attribute:session'])[0]
    print(start_frame, end_frame, session)
    ball=GameData.positions.query('`Frame`>=@start_frame & `Frame`<=@end_frame')
    extract_ball=np.unique(ball['Player'])[0]
    ball=ball.query('Player==@extract_ball')[['ball_x','ball_y']]
    ball = gpd.GeoDataFrame(
        ball,
        geometry=gpd.points_from_xy(ball["ball_x"], ball["ball_y"])
    )
    ball_line = gpd.GeoDataFrame(
        geometry=[LineString(zip(ball["ball_x"], ball["ball_y"]))]
    )
    h_df=gpd.GeoDataFrame(df.query('`attribute:team`=="Home"'),
        geometry=GameData.events['trajectory'])
    a_df=gpd.GeoDataFrame(df.query('`attribute:team`=="Away"'),
        geometry=GameData.events['trajectory'])

    frames = end_frame-start_frame
    session_df=GameData.positions[GameData.positions["Session"] == session]

    fig, ax = plt.subplots()
    def update(frame):
        ax.clear()
        ax.plot(ax=ax)
        field_gdf.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=2)

        past = session_df[session_df["Frame"] < frame]
        if len(past)!=0:
            home = past[past["Team"] == "Home"]
            away = past[past["Team"] == "Away"]
            
            ax.scatter(home["x"], home["y"], color="steelblue", s=0.1)
            ax.scatter(away["x"], away["y"], color="indianred", s=0.1)
            ax.scatter(home["ball_x"], home["ball_y"], color="green", s=0.1)
        
        
        current = session_df[session_df["Frame"] == frame]
        
        home = current[current["Team"] == "Home"]
        away = current[current["Team"] == "Away"]
        
        ax.scatter(home["x"], home["y"], color="steelblue", s=50)
        ax.scatter(away["x"], away["y"], color="indianred", s=50)
        ax.scatter(current["ball_x"].iloc[0], current["ball_y"].iloc[0], edgecolor="black", color="green", s=30)
        add_legend(ax)
        a_events=a_df.query('`attribute:frame`<=@frame & ball.notna()')
        h_events=h_df.query('`attribute:frame`<=@frame & ball.notna()')
        if len(a_events)!=0:
            add_event(a_events, ax, "#a84848")
        if len(h_events)!=0:
            add_event(h_events, ax, "#4878a8")
        ax.set_title(f"Frame {frame}")

    ani = FuncAnimation(fig, update, frames=frames, interval=90)
    ani.save("match.gif", writer="pillow")