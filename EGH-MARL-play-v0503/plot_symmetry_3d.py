import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from matplotlib.patches import Patch

# Load data from data.txt (each line: x y a b)
data = np.loadtxt('model_test_log.txt')
x_data = data[:, 0]
y_data = data[:, 1]
a_data = data[:, 2]
b_data = data[:, 3]

# Create grid for surface plotting
xi = np.linspace(x_data.min(), x_data.max(), 100)
yi = np.linspace(y_data.min(), y_data.max(), 100)
xi, yi = np.meshgrid(xi, yi)

# Interpolate a and b data to generate regular grid surfaces
zi_a = griddata((x_data, y_data), a_data, (xi, yi), method='cubic')
zi_b = griddata((x_data, y_data), b_data, (xi, yi), method='cubic')

# Create a single figure with both surfaces overlaid
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Find indices for the specified coordinates
x_coords = [-0.5, 0.5]
y_coords = [0.5, -0.5]

# Get z values for surface A at the specified coordinates
z_a_points = []
z_b_points = []
for x, y in zip(x_coords, y_coords):
    # Find the closest grid point
    x_idx = np.argmin(np.abs(xi[0, :] - x))
    y_idx = np.argmin(np.abs(yi[:, 0] - y))
    if zi_a[y_idx, x_idx] > 0: 
        z_a_points.append(zi_a[y_idx, x_idx]+0.004)
    else:
        z_a_points.append(zi_a[y_idx, x_idx]-0.004)
    if zi_b[y_idx, x_idx] > 0: 
        z_b_points.append(zi_b[y_idx, x_idx]+0.004)
    else:
        z_b_points.append(zi_b[y_idx, x_idx]-0.004)
print("z_a_points:", z_a_points)
print("z_b_points:", z_b_points)

# Plot surface A and B (lower zorder)
surf1 = ax.plot_surface(xi, yi, zi_a, facecolor='blue', alpha=0.5, linewidth=0.5, edgecolor='cyan', antialiased=True,zorder=1)
surf2 = ax.plot_surface(xi, yi, zi_b, facecolor='red', alpha=0.6, linewidth=0.5, antialiased=True,zorder=2)

# Draw vertical lines between points and extend to bottom
z_bottom = min(zi_a.min(), zi_b.min())
for i in range(len(x_coords)):
    x = x_coords[i]
    y = y_coords[i]
    z_top = max(z_a_points[i], z_b_points[i])
    z_bottom_line = [z_top, z_bottom]
    ax.plot([x, x], [y, y], z_bottom_line, color='gray', linewidth=1, linestyle='--', zorder=50)
    
    # Mark the bottom intersection point
    ax.scatter([x], [y], [z_bottom], color='black', s=50, marker='^', zorder=50)
    
    # Add coordinate annotation
    annotation_text = f'({x:.1f}, {y:.1f})'
    ax.text(x, y, z_bottom, annotation_text, fontsize=10, color='black', zorder=20)

# Plot points on surfaces (highest zorder to ensure visibility)
ax.scatter(x_coords, y_coords, z_a_points, color='black', s=50, marker='^', label='Surface A Points', zorder=100, alpha=1.0)
ax.plot(x_coords, y_coords, z_a_points, color='gray', linewidth=1, linestyle='--', zorder=50)

ax.scatter(x_coords, y_coords, z_b_points, color='black', s=50, marker='^', label='Surface B Points', zorder=100, alpha=1.0)
ax.plot(x_coords, y_coords, z_b_points, color='gray', linewidth=1, linestyle='--', zorder=50)

# Add data value annotations for points on surfaces
for i in range(len(x_coords)):
    # Annotate vx (blue surface) points
    ax.text(x_coords[i], y_coords[i], z_a_points[i], f'vx={z_a_points[i]:.2f}', 
            fontsize=10, color='black', zorder=150, ha='left', va='bottom')
    # Annotate vy (red surface) points
    ax.text(x_coords[i], y_coords[i], z_b_points[i], f'vy={z_b_points[i]:.2f}', 
            fontsize=10, color='black', zorder=150, ha='left', va='bottom')



# Set labels and title
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_zlabel('vel', fontsize=12)
ax.set_title('Symmetry Visualization', fontsize=14, pad=20)

# Add legend
legend_elements = [Patch(facecolor='blue', alpha=0.5, edgecolor='cyan', linewidth=1, label='vx - blue surface with edges'),
                   Patch(facecolor='red', alpha=0.6, label='vy - red surface without edges')]
ax.legend(handles=legend_elements, loc='upper left')

# Set viewing angle
ax.view_init(elev=25, azim=45)

# Save the figure instead of displaying
plt.savefig('surface_plot.png', dpi=300, bbox_inches='tight')
plt.close()