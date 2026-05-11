from shapely.geometry import Polygon, LineString, Point
import geopandas as gpd
from adjustText import adjust_text
from matplotlib.lines import Line2D
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