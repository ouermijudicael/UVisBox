import numpy as np
from uvisbox.Modules.UncertaintyTube.uncertainty_tubes_stats import project_points_onto_line
from uvisbox.Modules.UncertaintyTube.uncertainty_tubes import uncertainty_tubes_2D
from uvisbox.Datasets.flow2d import double_gyre

def test_project_points_onto_line():
    # Define a line from (0,0) to (1,0)
    point0 = np.array([0.0, 0.0])
    point1 = np.array([1.0, 0.0])
    # Points to project: above, below, and on the line
    points = np.array([
        [0.5, 1.0],
        [0.5, -1.0],
        [0.5, 0.0]
    ])
    projections = project_points_onto_line(point0, point1, points)
    expected = np.array([
        [1, 1.0],
        [1, -1.0],
        [1, 0.0]
    ])
    projection_error = np.linalg.norm(projections - expected, axis=1)
    if not np.all(projection_error < 1e-6):
        print("Projections:\n", projections)
        print("Expected:\n", expected)
        print("Errors:\n", projection_error)
        print("test_project_points_onto_line FAILED.")
    else:
        print("test_project_points_onto_line PASSED.")

def test_uncertainty_tube_2D():
    # Create synthetic trajectory data
    n_trajectories = 3
    n_time_steps = 100
    n_ensemble_members = 10

    # random place 3 trajectories in the domain [0,2] x [0,1]
    np.random.seed(42)
    trajectories = np.zeros((n_trajectories, n_time_steps, n_ensemble_members, 2))
    for i in range(n_trajectories):
        start_x = np.random.uniform(0, 2)
        start_y = np.random.uniform(0, 1)
        for j in range(n_ensemble_members):
            trajectories[i, 0, j] = [start_x + np.random.normal(0, 0.05), start_y + np.random.normal(0, 0.05)]
        for t in range(1, n_time_steps):
            for j in range(n_ensemble_members):
                u, v = double_gyre(trajectories[i, t-1, j, 0], trajectories[i, t-1, j, 1], t*0.1)
                trajectories[i, t, j] = trajectories[i, t-1, j] + np.array([u, v]) * 0.1 + np.random.normal(0, 0.01, size=2)

    ax = uncertainty_tubes_2D(trajectories)

if __name__ == "__main__":
    test_project_points_onto_line()
    test_uncertainty_tube_2D()
    print("All tests passed.")

  