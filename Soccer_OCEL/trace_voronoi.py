import imageio.v2 as imageio
import glob
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Polygon as ShapelyPolygon
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import numpy as np
import os


def finite_voronoi_polygons_2d(vor, radius):
    """
    Reconstruct infinite Voronoi regions in a 2D diagram to finite
    polygons clipped by a bounding box.
    """
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)

    # Map ridge vertices to ridges
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Construct polygons for each region
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]
        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue

        # Infinite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0 and v2 >= 0:
                continue  # both finite

            # Compute the missing endpoint at infinity
            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_vertices.append(far_point.tolist())
            new_region.append(len(new_vertices) - 1)

        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = [v for _, v in sorted(zip(angles, new_region))]
        new_regions.append(new_region)

    return new_regions, np.asarray(new_vertices)
def frame_voronoi(positions_df, pitch, frame, session):
    frame_df=positions_df[(positions_df['Frame']==frame) & (positions_df['Session']==session)] 
    frame_ball=frame_df[['ball_x', 'ball_y']].iloc[0].values
    frame_points=np.dstack([frame_df['x'], frame_df['y']])[0]

    vor = Voronoi(frame_points)
    mask = np.array(frame_df['Team'] == 'Home')  # True for Home, False for Away

    regions, vertices = finite_voronoi_polygons_2d(vor, max(pitch.xlim[1], pitch.ylim[1]))
    return regions, vertices, mask, frame_ball, frame_points
    
def frame_voronoi_plot(positions_df, pitch, frame, session):
    regions, vertices, mask, frame_ball, frame_points=frame_voronoi(positions_df, pitch, frame, session)
    patches = []
    colors = []
    for idx, region in enumerate(regions):
        polygon = vertices[region]
        # Clip polygon to pitch boundaries
        polygon[:, 0] = np.clip(polygon[:, 0], 0, pitch.xlim[1])
        polygon[:, 1] = np.clip(polygon[:, 1], 0, pitch.ylim[1])
        patches.append(Polygon(polygon, closed=True))
        colors.append('blue' if mask[idx] else 'red')
    fig, ax = plt.subplots(figsize=(8,5))
    pitch.plot(ax=ax)
    ax.set_facecolor("lightgrey")

    collection = PatchCollection(patches, facecolor=colors, edgecolor='k', alpha=0.4)
    ax.add_collection(collection)


    ax.scatter(frame_points[:, 0], frame_points[:, 1], c=['blue' if t else 'red' for t in mask], s=40, edgecolor='k', zorder=3)
    ax.scatter(frame_ball[0], frame_ball[1], c='green', edgecolor='k', zorder=4, label='Ball')
#plotting frame
def save_voronoi_frames(positions_df, pitch, session=1, out_dir="frames", step=1):
    os.makedirs(out_dir, exist_ok=True)
    max_frame = positions_df[positions_df['Session'] == session]['Frame'].max()
    min_frame = 0

    for frame in range(min_frame, max_frame + 1, step):
        frame_df = positions_df[(positions_df['Frame'] == frame) & (positions_df['Session'] == session)]
        frame_ball=frame_df[['ball_x', 'ball_y']].iloc[0].values

        if frame_df.empty or frame_ball.empty:
            continue

        frame_points = np.dstack([frame_df['x'], frame_df['y']])[0]
        vor = Voronoi(frame_points)
        mask = np.array(frame_df['Team'] == 'Home')

        regions, vertices = finite_voronoi_polygons_2d(vor, radius=max(pitch.xlim[1], pitch.ylim[1]))

        patches = []
        colors = []
        for idx, region in enumerate(regions):
            polygon = vertices[region]
            polygon[:, 0] = np.clip(polygon[:, 0], 0, pitch.xlim[1])
            polygon[:, 1] = np.clip(polygon[:, 1], 0, pitch.ylim[1])
            patches.append(Polygon(polygon, closed=True))
            colors.append('blue' if mask[idx] else 'red')

        fig, ax = plt.subplots(figsize=(8,5))
        pitch.plot(ax=ax)
        ax.set_facecolor("lightgrey")
        #ax.set_xlim(0, pitch.xlim[1]*2)
        #ax.set_ylim(0, pitch.ylim[1]*2)
        #ax.set_aspect('equal')
        #ax.set_axis_off()
        ax.set_title(f"Session {session} - Frame {frame}")

        collection = PatchCollection(patches, facecolor=colors, edgecolor='k', alpha=0.4)
        ax.add_collection(collection)

        ax.scatter(frame_points[:, 0], frame_points[:, 1],
                   c=['blue' if t else 'red' for t in mask],
                   s=40, edgecolor='k', zorder=3)
        ax.scatter(frame_ball['x'], frame_ball['y'],
                   c='green', edgecolor='k', s=60, zorder=4, label='Ball')

        # Save and immediately close figure to free memory
        plt.savefig(f"{out_dir}/frame_{frame:05d}.png", dpi=120, bbox_inches='tight')
        plt.close(fig)
#animation
def make_animation_from_frames(out_dir="frames", output="voronoi_animation.mp4", fps=1):
    frames = sorted(glob.glob(f"{out_dir}/frame_*.png"))
    with imageio.get_writer(output, fps=fps) as writer:
        for f in frames:
            writer.append_data(imageio.imread(f))
    print(f"Saved animation to {output}")



def add_voronoi_area(positions_df, pitch):
    """
    Compute Voronoi area for each player at each frame & session.
    Adds a column 'voronoi_area' to positions_df.
    """
    df = positions_df.copy()
    df['voronoi_area'] = np.nan
    max_area=pitch.xlim[1]*pitch.ylim[1]

    sessions = df['Session'].unique()
    for session in sessions:
        frames = df.loc[df['Session'] == session, 'Frame'].unique()
        for frame in frames:
            frame_df = df[(df['Frame'] == frame) & (df['Session'] == session)]
            if frame_df.empty:
                continue

            try:
                regions, vertices, mask, frame_ball, frame_points = frame_voronoi(positions_df, pitch, frame, session)
            except Exception as e:
                continue

            areas = []
            for region in regions:
                polygon = np.array(vertices)[region]
                polygon[:, 0] = np.clip(polygon[:, 0], 0, pitch.xlim[1])
                polygon[:, 1] = np.clip(polygon[:, 1], 0, pitch.ylim[1])
                poly = ShapelyPolygon(polygon)
                areas.append(poly.area)


            df.loc[frame_df.index, 'voronoi_area'] = areas
    df.loc[df['voronoi_area'] > max_area, 'voronoi_area'] = np.nan
    return df
