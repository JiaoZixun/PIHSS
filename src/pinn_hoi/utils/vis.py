from __future__ import annotations

from typing import Optional

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch

from pinn_hoi.common.rot import pose7_to_points
from pinn_hoi.utils.io import ensure_dir


def set_axes_equal(ax):
    x_limits = ax.get_xlim3d(); y_limits = ax.get_ylim3d(); z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0]); x_mid = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0]); y_mid = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0]); z_mid = np.mean(z_limits)
    plot_radius = 0.5 * max([x_range, y_range, z_range, 1e-6])
    ax.set_xlim3d([x_mid - plot_radius, x_mid + plot_radius])
    ax.set_ylim3d([y_mid - plot_radius, y_mid + plot_radius])
    ax.set_zlim3d([z_mid - plot_radius, z_mid + plot_radius])


def render_hand_object_prediction_video(
    hand_endpoints: np.ndarray,
    obj_points_world_gt: np.ndarray,
    obj_points_canonical: np.ndarray,
    obj_pose7_gt: np.ndarray,
    pred_pose7: Optional[np.ndarray],
    contact_label: Optional[np.ndarray],
    out_path: str,
    fps: int = 12,
    max_frames: int = 160,
) -> None:
    ensure_dir('/'.join(out_path.split('/')[:-1]))
    T = min(hand_endpoints.shape[0], obj_pose7_gt.shape[0], max_frames)
    pred_pts = None
    if pred_pose7 is not None:
        with torch.no_grad():
            can = torch.from_numpy(obj_points_canonical).float()
            pred_pts = pose7_to_points(can, torch.from_numpy(pred_pose7[:T]).float()).cpu().numpy()

    frames = []
    for t in range(T):
        fig = plt.figure(figsize=(7, 7), dpi=120)
        ax = fig.add_subplot(111, projection='3d')
        og = obj_points_world_gt[t]
        og = og[::max(1, og.shape[0] // 384)]
        ax.scatter(og[:, 0], og[:, 1], og[:, 2], s=2, label='GT object')
        if pred_pts is not None and t < pred_pts.shape[0]:
            op = pred_pts[t]
            op = op[::max(1, op.shape[0] // 384)]
            ax.scatter(op[:, 0], op[:, 1], op[:, 2], s=2, marker='^', label='Pred object')
        l = hand_endpoints[t, 0]
        r = hand_endpoints[t, 1]
        if contact_label is not None:
            cl = contact_label[t, 0] > 0.5
            cr = contact_label[t, 1] > 0.5
            ax.scatter(l[:, 0], l[:, 1], l[:, 2], s=12, label='Left endpoints')
            ax.scatter(r[:, 0], r[:, 1], r[:, 2], s=12, label='Right endpoints')
            if cl.any():
                lc = l[cl]
                ax.scatter(lc[:, 0], lc[:, 1], lc[:, 2], s=42, marker='o', label='Left contact')
            if cr.any():
                rc = r[cr]
                ax.scatter(rc[:, 0], rc[:, 1], rc[:, 2], s=42, marker='o', label='Right contact')
        else:
            ax.scatter(l[:, 0], l[:, 1], l[:, 2], s=12, label='Left endpoints')
            ax.scatter(r[:, 0], r[:, 1], r[:, 2], s=12, label='Right endpoints')
        center = obj_pose7_gt[t, 4:7]
        radius = 0.35
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        set_axes_equal(ax)
        ax.set_title(f'PI-CATS frame {t}')
        ax.legend(loc='upper right', fontsize=7)
        ax.view_init(elev=20, azim=-65)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
        frames.append(img)
        plt.close(fig)
    imageio.mimsave(out_path, frames, fps=fps)
