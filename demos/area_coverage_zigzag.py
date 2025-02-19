import numpy as np
import matplotlib.pyplot as plt

def generate_straight_line_path(num_points=100, init_alt=100.0):
    """
    Generates a straight, forward-slanting path in 3D.
    For simplicity, from (x, y, z) = (0, 0, init_alt) to (50, 0, 0).
    """
    t = np.linspace(0, 1, num_points)
    x = 50 * t                  # move 50 units in x
    y = 0 * t                   # no lateral displacement
    z = init_alt * (1 - t)      # linearly descend from init_alt to 0
    return x, y, z

def generate_zigzag_path(num_points=100, init_alt=100.0):
    """
    Generates a zigzag path in 3D.
    Moves in y, zigzags in x, and descends in z.
    """
    t = np.linspace(0, 1, num_points)
    y = 50 * t                  # overall forward motion in y from 0 -> 50
    # lateral zigzag in x (amplitude=10, ~3 cycles)
    x = 10 * np.sin(6 * np.pi * t)
    z = init_alt * (1 - t)      # descend linearly
    return x, y, z

def generate_spiral_path(num_points=100, init_alt=100.0):
    """
    Generates a simple spiral (helical) path in 3D.
    Moves in a circle in x-y while descending in z.
    """
    t = np.linspace(0, 1, num_points)
    # Radius = 20, ~2 full revolutions (4π) as it goes down
    R = 20120
    x = R * np.cos(4 * np.pi * t)
    y = R * np.sin(4 * np.pi * t)
    z = init_alt * (1 - t)      # descend linearly from init_alt to 0
    return x, y, z

def compute_coverage_map(x_path, y_path, z_path, fov_degs=120,
                         grid_size=50, world_extent=60):
    """
    Computes a discrete 'coverage count' map on the ground (z=0 plane)
    for the given drone path and downward-pointing camera FOV.

    - x_path, y_path, z_path: arrays describing the drone's 3D path
    - fov_degs: full field of view (e.g., 120 => half-angle of 60 degrees)
    - grid_size: number of grid steps in each dimension for coverage
    - world_extent: half-width of the area to consider in x and y ([-extent, +extent])
    """
    half_angle = np.radians(fov_degs / 2.0)

    xs = np.linspace(-world_extent, world_extent, grid_size)
    ys = np.linspace(-world_extent, world_extent, grid_size)
    Xg, Yg = np.meshgrid(xs, ys)

    coverage = np.zeros_like(Xg)

    for (xd, yd, zd) in zip(x_path, y_path, z_path):
        if zd <= 0:
            # Drone at or below ground level => skip
            continue
        dist_horizontal = np.sqrt((Xg - xd)**2 + (Yg - yd)**2)
        angle = np.arctan2(dist_horizontal, zd)  # angle from vertical
        mask = (angle <= half_angle)
        coverage[mask] += 1

    return Xg, Yg, coverage

def compute_hazard_map(Xg, Yg):
    """
    Example hazard function. In reality, you might:
      - Use real terrain elevation data to calculate slope hazards
      - Incorporate known obstacles or no-fly zones
      - Look for a flat region or minimal hazard
    Here, we'll define hazard as 'distance from origin' for demonstration,
    so the center is considered safer (lower hazard).
    """
    hazard = np.sqrt(Xg**2 + Yg**2)
    return hazard

def find_safe_landing_spot(Xg, Yg, coverage, hazard, coverage_threshold=10):
    """
    Finds a potential 'safe' landing spot.
    1) Coverage >= coverage_threshold (drone saw it enough times)
    2) Among those cells, pick the one with minimal hazard

    Returns (x_best, y_best, coverage_best).
    If no cell meets coverage requirement, returns None.
    """
    # Indices where coverage meets or exceeds threshold
    indices = np.where(coverage >= coverage_threshold)
    if len(indices[0]) == 0:
        return None

    # Evaluate hazard at those indices
    i_candidates, j_candidates = indices
    sub_hazard = hazard[i_candidates, j_candidates]
    # Pick the index of minimal hazard
    min_idx = np.argmin(sub_hazard)
    i_best = i_candidates[min_idx]
    j_best = j_candidates[min_idx]

    x_best = Xg[i_best, j_best]
    y_best = Yg[i_best, j_best]
    c_best = coverage[i_best, j_best]

    return (x_best, y_best, c_best)

def plot_three_strategies():
    # Generate paths
    x_s, y_s, z_s = generate_straight_line_path()
    x_z, y_z, z_z = generate_zigzag_path()
    x_sp, y_sp, z_sp = generate_spiral_path()

    # Compute coverage
    Xg_s, Yg_s, cov_s = compute_coverage_map(x_s, y_s, z_s)
    Xg_z, Yg_z, cov_z = compute_coverage_map(x_z, y_z, z_z)
    Xg_sp, Yg_sp, cov_sp = compute_coverage_map(x_sp, y_sp, z_sp)

    # Compute hazard maps (same size for each)
    haz_s = compute_hazard_map(Xg_s, Yg_s)
    haz_z = compute_hazard_map(Xg_z, Yg_z)
    haz_sp = compute_hazard_map(Xg_sp, Yg_sp)

    # Find a "safe" landing spot for each strategy
    # (arbitrary coverage threshold = 10)
    best_s = find_safe_landing_spot(Xg_s, Yg_s, cov_s, haz_s, coverage_threshold=10)
    best_z = find_safe_landing_spot(Xg_z, Yg_z, cov_z, haz_z, coverage_threshold=10)
    best_sp = find_safe_landing_spot(Xg_sp, Yg_sp, cov_sp, haz_sp, coverage_threshold=10)

    fig = plt.figure(figsize=(15, 10))

    # ============== ROW 0: STRAIGHT LINE ==============
    # (0,0) -> 2D path
    ax1 = fig.add_subplot(3, 3, 1)
    ax1.plot(x_s, z_s, label="Straight Descent")
    ax1.set_xlabel('X')
    ax1.set_ylabel('Z (Altitude)')
    ax1.set_title('Straight - 2D (X vs Z)')
    ax1.grid(True)
    ax1.legend()

    # (0,1) -> 3D path
    ax2 = fig.add_subplot(3, 3, 2, projection='3d')
    ax2.plot(x_s, y_s, z_s, 'b', label="Straight Path")
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Straight - 3D Path')
    ax2.legend()

    # (0,2) -> Coverage heatmap (3D surface)
    ax3 = fig.add_subplot(3, 3, 3, projection='3d')
    ax3.plot_surface(Xg_s, Yg_s, cov_s, cmap='hot', edgecolor='none')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Coverage Count')
    ax3.set_title('Straight - Coverage')
    # Mark the chosen landing spot (if found)
    if best_s is not None:
        x_best, y_best, c_best = best_s
        ax3.scatter(x_best, y_best, c_best, color='white', s=80, marker='*', label='Landing Spot')
        ax3.legend()

    # ============== ROW 1: ZIGZAG ==============
    # (1,0) -> 2D path
    ax4 = fig.add_subplot(3, 3, 4)
    ax4.plot(x_z, z_z, 'r', label="Zigzag Descent")
    ax4.set_xlabel('X')
    ax4.set_ylabel('Z (Altitude)')
    ax4.set_title('Zigzag - 2D (X vs Z)')
    ax4.grid(True)
    ax4.legend()

    # (1,1) -> 3D path
    ax5 = fig.add_subplot(3, 3, 5, projection='3d')
    ax5.plot(x_z, y_z, z_z, 'r', label="Zigzag Path")
    ax5.set_xlabel('X')
    ax5.set_ylabel('Y')
    ax5.set_zlabel('Z')
    ax5.set_title('Zigzag - 3D Path')
    ax5.legend()

    # (1,2) -> Coverage heatmap (3D surface)
    ax6 = fig.add_subplot(3, 3, 6, projection='3d')
    ax6.plot_surface(Xg_z, Yg_z, cov_z, cmap='hot', edgecolor='none')
    ax6.set_xlabel('X')
    ax6.set_ylabel('Y')
    ax6.set_zlabel('Coverage Count')
    ax6.set_title('Zigzag - Coverage')
    # Mark the chosen landing spot (if found)
    if best_z is not None:
        x_best, y_best, c_best = best_z
        ax6.scatter(x_best, y_best, c_best, color='white', s=80, marker='*', label='Landing Spot')
        ax6.legend()

    # ============== ROW 2: SPIRAL (RECOMMENDED) ==============
    # (2,0) -> 2D path
    ax7 = fig.add_subplot(3, 3, 7)
    ax7.plot(x_sp, z_sp, 'g', label="Spiral Descent")
    ax7.set_xlabel('X')
    ax7.set_ylabel('Z (Altitude)')
    ax7.set_title('Spiral - 2D (X vs Z)')
    ax7.grid(True)
    ax7.legend()

    # (2,1) -> 3D path
    ax8 = fig.add_subplot(3, 3, 8, projection='3d')
    ax8.plot(x_sp, y_sp, z_sp, 'g', label="Spiral Path")
    ax8.set_xlabel('X')
    ax8.set_ylabel('Y')
    ax8.set_zlabel('Z')
    ax8.set_title('Spiral - 3D Path')
    ax8.legend()

    # (2,2) -> Coverage heatmap (3D surface)
    ax9 = fig.add_subplot(3, 3, 9, projection='3d')
    ax9.plot_surface(Xg_sp, Yg_sp, cov_sp, cmap='hot', edgecolor='none')
    ax9.set_xlabel('X')
    ax9.set_ylabel('Y')
    ax9.set_zlabel('Coverage Count')
    ax9.set_title('Spiral - Coverage')
    # Mark the chosen landing spot (if found)
    if best_sp is not None:
        x_best, y_best, c_best = best_sp
        ax9.scatter(x_best, y_best, c_best, color='white', s=80, marker='*', label='Landing Spot')
        ax9.legend()

    # compare_coverages(cov_s, cov_z, cov_sp, coverage_threshold=1)

    plt.tight_layout()
    plt.show()
    
    
def analyze_coverage(name, coverage, coverage_threshold=1):
    """
    Analyzes a coverage map and returns:
    - total_coverage: Sum of coverage counts (how many times all cells combined were viewed)
    - coverage_area: Count of grid cells with coverage >= coverage_threshold
    - average_coverage: Mean coverage across all cells
    """
    total_coverage = np.sum(coverage)
    coverage_area  = np.sum(coverage >= coverage_threshold)
    average_coverage = np.mean(coverage)
    
    # Print or store results
    print(f"Coverage Analysis for {name}:")
    print(f"  Sum of coverage counts   = {total_coverage:.1f}")
    print(f"  # cells >= {coverage_threshold} coverage = {coverage_area}")
    print(f"  Average coverage         = {average_coverage:.2f}\n")
    
    return total_coverage, coverage_area, average_coverage

def compare_coverages(cov_s, cov_z, cov_sp, coverage_threshold=1):
    """
    Compares the coverage arrays for Straight, Zigzag, and Spiral paths.
    Prints which path has the greatest total coverage, the widest coverage area,
    and their average coverage.
    """
    # Analyze each coverage map
    s_results = analyze_coverage("Straight", cov_s, coverage_threshold)
    z_results = analyze_coverage("Zigzag",   cov_z, coverage_threshold)
    sp_results = analyze_coverage("Spiral",  cov_sp, coverage_threshold)

    # Unpack
    s_total, s_area, s_avg   = s_results
    z_total, z_area, z_avg   = z_results
    sp_total, sp_area, sp_avg = sp_results

    # Decide which path has greatest total coverage
    max_total = max(s_total, z_total, sp_total)
    if max_total == s_total:
        best_total = "Straight"
    elif max_total == z_total:
        best_total = "Zigzag"
    else:
        best_total = "Spiral"

    # Decide which path has the widest coverage area
    max_area = max(s_area, z_area, sp_area)
    if max_area == s_area:
        best_area = "Straight"
    elif max_area == z_area:
        best_area = "Zigzag"
    else:
        best_area = "Spiral"

    print("Summary of Which Path is Best:")
    print(f" - Greatest TOTAL coverage:   {best_total}")
    print(f" - Widest coverage AREA:      {best_area}")



if __name__ == "__main__":
    plot_three_strategies()
