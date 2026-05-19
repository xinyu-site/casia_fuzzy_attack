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

# Plot surface A and B
surf1 = ax.plot_surface(xi, yi, zi_a, cmap='viridis', alpha=0.7, linewidth=0, antialiased=True)
surf2 = ax.plot_surface(xi, yi, zi_b, cmap='plasma', alpha=0.7, linewidth=0, antialiased=True)

# Add color bars
cbar1 = fig.colorbar(surf1, ax=ax, shrink=0.5, aspect=20, pad=0.1)
cbar1.set_label('Value A')
cbar2 = fig.colorbar(surf2, ax=ax, shrink=0.5, aspect=20, pad=0.15)
cbar2.set_label('Value B')

# Set labels and title
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_zlabel('vel', fontsize=12)
ax.set_title('Symmetry Visualization', fontsize=14, pad=20)

# Add legend
legend_elements = [Patch(facecolor='green', alpha=0.7, label='vel_x'),
                   Patch(facecolor='orange', alpha=0.7, label='vel_y')]
ax.legend(handles=legend_elements, loc='upper left')

# Set viewing angle
ax.view_init(elev=25, azim=45)

# Save the figure instead of displaying
plt.savefig('surface_plot.png', dpi=300, bbox_inches='tight')
plt.close()