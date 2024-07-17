from nuplan_extent.planning.training.modeling.models.utils import shift_down
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any
from adjustText import adjust_text
import matplotlib
matplotlib.use('Agg')
from nuplan_extent.planning.training.modeling.models.utils import shift_down


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    sigmoid function with numpy
    :param x: input of numpy array
    :return: sigmoid(x)
    """
    return 1 / (1 + np.exp(-x))


def softmax(x: np.ndarray) -> np.ndarray:
    """
    Softmax function with numpy
    :param x: input of numpy array
    :return: softmax(x)
    """
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def render_and_save_features(
        input_dict: Dict,
        filename: str,
        bev_range: List[float]):
    """
    save intermidiate features for visualization.
    :param features: dict of features to be saved
    :param filename: path to save features
    :param bev_range: range of BEV
    """
    y_range = [-bev_range[2], -bev_range[0]]
    x_range = [bev_range[1], bev_range[3]]
    features = {}
    for k, v in input_dict.items():
        if isinstance(v, np.ndarray):
            features[k] = v
        else:
            try:
                features[k] = v.data.float().cpu().numpy()
            except BaseException:
                features[k] = v

    # Create a new figure with no border
    fig = plt.figure(figsize=(10, 10), frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    # Display the image using imshow
    # reset drivable_area_map, route_raster from center coordinate to rear axle coordinate, shift then down by self.base_l_shift
    # features['raster'][0, 2] = shift_down(features['raster'][0, 2], shift_pixels=int(1.416/0.5))
    # features['raster'][0, 3] = shift_down(features['raster'][0, 3], shift_pixels=int(1.416/0.5))
    # features['raster'][0, 4] = shift_down(features['raster'][0, 4], shift_pixels=int(1.416/0.5))
    # features['raster'][0, 6] = shift_down(features['raster'][0, 6], shift_pixels=int(1.416/0.5))
    raster_show = features['raster'][0][[1,2,4]].transpose([1,2,0])
    raster_show[:, :, 0:1] = raster_show[:, :, 0:1] + features['raster'][0][[0]].transpose([1,2,0])
    raster_show[:, :, 0:1] = raster_show[:, :, 0:1] + features['raster'][0][[3]].transpose([1,2,0])
    ax.imshow(raster_show, extent=[-104, 104, -104, 104])
    if 'collision_flatten' in features:
        x = features['collision_flatten'].reshape(-1, 3)[:, 0]
        y = features['collision_flatten'].reshape(-1, 3)[:, 1]
        ax.scatter(-y, x, s=1, c='yellow', alpha=0.3)
    if 'heatmap_flatten' in features:
        x = features['heatmap_flatten'].reshape(-1, 3)[:, 0]
        y = features['heatmap_flatten'].reshape(-1, 3)[:, 1]
        ax.scatter(-y, x, s=1, c='red', alpha=0.3)
    if 'pred_heatmap' in features:
        ax.imshow(
            sigmoid(features['pred_heatmap'][0]).sum(0),
            extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
            alpha=0.65,
            cmap='plasma')
    if 'before_smoother_trajectory' in features:
        x = features['before_smoother_trajectory'][0, :, 0]
        y = features['before_smoother_trajectory'][0, :, 1]
        ax.scatter(-y, x, s=3, c='blue', alpha=0.4)
    if 'before_emergency_break_trajectory' in features:
        x = features['before_emergency_break_trajectory'][0, :, 0]
        y = features['before_emergency_break_trajectory'][0, :, 1]
        ax.scatter(-y, x, s=1, c='yellow', alpha=0.4)

    if 'multimode_trajectory' in features:
        # (batch, mode, seq_len, state_size)
        texts = []  # List to store text objects for adjustText

        multimode_trajectory = features['multimode_trajectory']
        pred_log_prob = features['scores']  # (batch, mode)
        selected_mode = np.argsort(pred_log_prob)[::-1][:6]
        for i in range(multimode_trajectory.shape[1]):

            x = multimode_trajectory[0, i, :, 0]
            y = multimode_trajectory[0, i, :, 1]
            # Draw the trajectory line with a very light color
            ax.plot(-y, x, color='lightgray', linewidth=0.5)
            ax.scatter(-y[-1], x[-1], s=25, c='red', marker='X', alpha=0.4)
            if i not in selected_mode:
                continue
            if abs(y[-1]) < 104 and abs(x[-1]) < 104:
                texts.append(
                    ax.text(-y[-1], x[-1], f'{pred_log_prob[i]:.2f}', color='red'))

        # Use adjustText to prevent overlapping
        adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))

    if 'pred_latitudinal_bin' in features:
        pred_latitudinal_bin = features['pred_latitudinal_bin'][0]

        # Apply softmax to obtain probabilities
        probabilities = softmax(pred_latitudinal_bin)

        # Calculate the circle coordinates
        circle_x = 104 - 2
        circle_y = 104 - 2

        # Calculate the circle radius
        circle_radius = 8

        # Define the colormap
        colormap = plt.cm.get_cmap('coolwarm')

        for i, probability in enumerate(probabilities):
            # Calculate the color based on the probability
            color = colormap(probability)

            # Draw the circle with the corresponding color
            circle = plt.Circle((circle_x, circle_y),
                                circle_radius,
                                color=color,
                                fill=True)

            # Add the circle to the plot
            ax.add_patch(circle)

            # Print the probability text inside the circle
            text_x = circle_x - circle_radius / 2 - 2
            text_y = circle_y - circle_radius / 2 - 2
            ax.text(text_x, text_y, f'{probability:.2f}', color='black')
            # Update the circle position for the next iteration
            circle_x -= 2 * circle_radius

    if 'pred_longitudinal_bin' in features:
        pred_longitudinal_bin = features['pred_longitudinal_bin'][0]

        # Apply softmax to obtain probabilities
        probabilities = softmax(pred_longitudinal_bin)

        # Calculate the circle coordinates
        circle_x = -56 + 2
        circle_y = 56 - 2

        # Calculate the circle radius
        circle_radius = 8

        # Define the colormap
        colormap = plt.cm.get_cmap('coolwarm')

        for i, probability in enumerate(probabilities):
            # Calculate the color based on the probability
            color = colormap(probability)

            # Draw the circle with the corresponding color
            circle = plt.Circle((circle_x, circle_y),
                                circle_radius,
                                color=color,
                                fill=True)

            # Add the circle to the plot
            ax.add_patch(circle)

            # Print the probability text inside the circle
            text_x = circle_x - circle_radius / 2 + 2
            text_y = circle_y - circle_radius / 2 + 2
            ax.text(text_x, text_y, f'{probability:.2f}', color='black')
            # Update the circle position for the next iteration
            circle_y -= 2 * circle_radius

    x = features['trajectory'][0, :, 0]
    y = features['trajectory'][0, :, 1]
    ax.scatter(-y, x, s=20, c='green', alpha=0.4)
    ax.scatter(-y[-1], x[-1], s=30, c='green', marker='X', alpha=0.4)
    ax.scatter(-y[5], x[5], s=30, c='green', marker='X', alpha=0.4)
    # ax.scatter(-y[9], x[9], s=30, c='green', marker='X', alpha=0.4)

    ax.set_xlim([x_range[0], x_range[1]])
    ax.set_ylim([y_range[0], y_range[1]])
    plt.margins(0, 0)

    # Save the figure to a JPEG file
    # if features['enable_emergency_break'][0]:
    fig.savefig(filename, dpi=100, bbox_inches='tight', pad_inches=0)

    # Close the figure to release memory
    plt.close(fig)
