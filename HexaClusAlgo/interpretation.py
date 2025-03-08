import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import geopandas as gpd

def animate_polygon_merges(polygons, history, points=None, save_path=None):
    """
    Creates an animation showing the sequence of active polygons during merges.
    
    Parameters:
    - polygons: List of all shapely Polygon objects
    - history: List of lists, where each sublist contains the indices of active polygons for that step
    - points: List of shapely Point objects (optional)
    - save_path: File path to save the animation (optional)
    Returns:
    - anim: The animation object to keep alive
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    orginal_nr = len(history[0])
    
    def update(frame):
        ax.clear()
        active_indices = history[frame]
        active_polygons = [polygons[i] for i in active_indices]
        
        # Create GeoDataFrame for plotting active polygons only
        gdf_active = gpd.GeoDataFrame({'geometry': active_polygons})
        
        # Plot active polygons
        colors = ['#cce3de' if i > orginal_nr  else '#fefcfb' for i in active_indices[:-1]] + ['#e29578']
        gdf_active.plot(ax=ax, color=colors, edgecolor='black', label='Active')
        
        # Plot points if provided
        if points:
            x = [point.x for point in points]
            y = [point.y for point in points]
            ax.scatter(x, y, color='blue', marker='o', s=10, label='Points')
        
        # Add title and legend
        ax.set_title(f"Merge Step {frame}")
        ax.axis('off')
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
    
    # Create the animation and return it to prevent deletion
    anim = FuncAnimation(fig, update, frames=len(history), interval=300, repeat=False)
    # Save or display the animation
    if save_path:
        anim.save(save_path, writer='pillow')
    else:
        plt.show()
    return anim  # Return the animation object

def plot_single_frame(polygons, history, frame = -1, 
                      landmarks=None, save_path=None, colors =None, title = "London Map with Famous Landmarks", 
                      factor =-1.15,interval =[0.2, 0.05]):
    fig, ax = plt.subplots(figsize=(10, 10))
    orginal_nr = len(history[0])
    ax.clear()
    active_indices = history[frame]
    active_polygons = [polygons[i] for i in active_indices]
    # Create GeoDataFrame for plotting active polygons only
    gdf_active = gpd.GeoDataFrame({'geometry': active_polygons, 'colors': colors})
    # Plot active polygons
    if colors is None:
        colors = ['#cce3de' if i > orginal_nr  else '#fefcfb' for i in active_indices]
        gdf_active.plot(ax=ax, color=colors, edgecolor='black', label='Active')
    else:
        gdf_active['colors'] = colors
        gdf_active.plot(column='colors', cmap='coolwarm', ax=ax, legend=False, edgecolor='black', label='Active')
    
    # Plot points if provided
    if landmarks:
      # Plot landmarks as points
        landmark_points = gpd.GeoDataFrame({'geometry': list(landmarks.values()), 'name': list(landmarks.keys())})
        landmark_points.plot(ax=ax, color='red', marker='*', markersize=100, label='Landmarks')

       # Plot landmark names outside the map with arrows pointing to actual locations
        for i, (name, point) in enumerate(landmarks.items()):
            # Define the offset position for each label
            label_x = point.x + interval[0]  # Move label horizontally to the right
            label_y = point.y + interval[1] * ( (factor) ** i)  # Alternate above and below to avoid overlap

            # Draw annotation with an arrow
            ax.annotate(
                name, xy=(point.x, point.y), xytext=(label_x, label_y),
                arrowprops=dict(facecolor='black', arrowstyle='->', lw=1),
                fontsize=10, ha='left', va='center', color='black', weight='bold'
            )
    
    # Add title and legend
    ax.set_title(title)
    ax.axis('off')
    #plt.show()
    #handles, labels = ax.get_legend_handles_labels()
    #by_label = dict(zip(labels, handles))
    #ax.legend(by_label.values(), by_label.keys())
    if save_path is not None:
        plt.savefig(save_path)

