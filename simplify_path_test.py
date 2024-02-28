import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from shapely.geometry import LineString
import tqdm
def adjust_path_points(simplified_points, target_points):
    while len(simplified_points) < target_points:
        # Find indices of the furthest consecutive points
        furthest_idx = max(range(len(simplified_points) - 1),
                           key=lambda i: np.linalg.norm(np.array(simplified_points[i]) - np.array(simplified_points[i + 1])))
        # Insert a midpoint
        midpoint = tuple(np.mean([simplified_points[furthest_idx], simplified_points[furthest_idx + 1]], axis=0))
        simplified_points.insert(furthest_idx + 1, midpoint)

    while len(simplified_points) > target_points:
        # Find indices of the closest consecutive points
        closest_idx = min(range(len(simplified_points) - 1),
                          key=lambda i: np.linalg.norm(np.array(simplified_points[i]) - np.array(simplified_points[i + 1])))
        # Remove one of the closest points
        del simplified_points[closest_idx]

def simplify_path_to_target_points(path, target_points, tolerance=0.1, tolerance_increment=0.05):
    if len(path) < 2 or target_points >= len(path):
        return path

    line = LineString(path)
    simplified_line = line.simplify(tolerance)
    simplified_points = list(simplified_line.coords)

    while len(simplified_points) != target_points:
        if len(simplified_points) > target_points:
            tolerance += tolerance_increment
        else:
            tolerance -= tolerance_increment
            tolerance_increment /= 2

        simplified_line = line.simplify(tolerance)
        simplified_points = list(simplified_line.coords)

        if tolerance <= 0 or tolerance_increment < 1e-5:
            break

    # Directly adjust the number of points to match the target
    adjust_path_points(simplified_points, target_points)

    return simplified_points


class PathSimplifierApp:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.25)
        self.path, = plt.plot([], [], 'r-', lw=2)  # Path drawn by the user
        self.simplified_path, = plt.plot([], [], 'go-', lw=2)  # Simplified path

        self.drawing = False
        self.xs = list()
        self.ys = list()

        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)

        self.ax_simplify = plt.axes([0.7, 0.05, 0.1, 0.075])
        self.btn_simplify = Button(self.ax_simplify, 'Simplify')
        self.btn_simplify.on_clicked(self.simplify_path)

        self.ax_slider = plt.axes([0.1, 0.05, 0.5, 0.03], facecolor='lightgoldenrodyellow')
        self.slider = Slider(self.ax_slider, 'Num Points', 2, 100, valinit=5, valstep=1)
        self.slider.on_changed(self.update_simplification)

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        self.drawing = True
        self.xs.append(event.xdata)
        self.ys.append(event.ydata)

    def on_motion(self, event):
        if not self.drawing or event.inaxes != self.ax:
            return
        self.xs.append(event.xdata)
        self.ys.append(event.ydata)
        self.path.set_data(self.xs, self.ys)
        self.fig.canvas.draw_idle()

    def on_release(self, event):
        self.drawing = False

    def simplify_path(self, event):
        target_points = int(self.slider.val)
        path = list(zip(self.xs, self.ys))
        simplified_path = simplify_path_to_target_points(path, target_points)
        xs, ys = zip(*simplified_path) if simplified_path else ([], [])

        self.simplified_path.set_data(xs, ys)
        self.fig.canvas.draw_idle()

    def update_simplification(self, val):
        # Automatically update the simplification if desired when slider changes
        pass


def generate_random_path(num_points):
    """
    Generate a random continuous path with the specified number of points.
    The path is generated in 2D space with random walks.
    """
    # Generate random x and y offsets
    x_offsets = np.random.randn(num_points).cumsum()
    y_offsets = np.random.randn(num_points).cumsum()

    # Convert the offsets to a list of (x, y) points
    path = list(zip(x_offsets, y_offsets))
    return path


def test_simplification():
    num_tests = 100  # Number of test cases
    for i in tqdm.tqdm(range(num_tests),desc="test cases"):
        # Generate a random path with at least 100 points
        path = generate_random_path(100)

        # Test simplification from 2 to 100 target points
        for target_points in range(2, 101):
            simplified_path = simplify_path_to_target_points(path, target_points)

            # Check if the simplified path has the correct length
            assert len(
                simplified_path) == target_points, f"Failed for target_points={target_points} with result length={len(simplified_path)}"

    print("All test cases passed successfully.")


# Ensure the simplify_path_to_target_points function is defined as per the previous discussions.
# Call the test function
test_simplification()

# app = PathSimplifierApp()
# plt.show()