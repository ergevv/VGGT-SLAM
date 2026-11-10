"""读取保存的点云并通过 Viser 查看，无需启动 SLAM 或加载模型。"""

import argparse
from pathlib import Path
import time

import numpy as np
import open3d as o3d
import viser


def load_point_cloud(path: Path, voxel_size: float):
    """读取点云、过滤无效坐标并降采样，保留颜色且不修改原文件。"""
    if not path.is_file():
        raise FileNotFoundError(f"点云文件不存在：{path}")
    print(f"正在读取：{path}", flush=True)
    cloud = o3d.io.read_point_cloud(
        str(path), remove_nan_points=True, remove_infinite_points=True
    )
    if cloud.is_empty():
        raise ValueError(f"点云为空或无法读取：{path}")
    print(f"原始点数：{len(cloud.points):,}", flush=True)
    if voxel_size > 0:
        cloud = cloud.voxel_down_sample(voxel_size)
    if not cloud.has_colors():
        cloud.paint_uniform_color([0.3, 0.6, 0.9])
    print(f"显示点数：{len(cloud.points):,}", flush=True)
    points = np.asarray(cloud.points, dtype=np.float32)
    colors = (np.clip(np.asarray(cloud.colors), 0, 1) * 255).astype(np.uint8)
    return points, colors


def main():
    """解析文件和显示参数，启动独立的点云查看服务。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "pcd_path", nargs="?", type=Path,
        default=Path(__file__).resolve().parent / "logs/office_loop_poses_points.pcd",
        help="点云文件路径，默认读取 office_loop 建图结果",
    )
    parser.add_argument("--voxel_size", type=float, default=0.05,
                        help="显示用体素大小，默认 0.05；设为 0 显示全部点")
    parser.add_argument("--port", type=int, default=8081,
                        help="查看器端口，默认 8081")
    args = parser.parse_args()
    if not np.isfinite(args.voxel_size) or args.voxel_size < 0:
        parser.error("voxel_size 必须是非负的有限数值")
    if not 1 <= args.port <= 65535:
        parser.error("port 必须在 1 到 65535 之间")
    try:
        points, colors = load_point_cloud(args.pcd_path, args.voxel_size)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    # 独立显示点云，不创建建图模块或加载神经网络。
    server = viser.ViserServer(host="127.0.0.1", port=args.port)
    server.gui.add_markdown(f"**{args.pcd_path.name}**\n\n显示点数：{len(points):,}")
    server.scene.add_point_cloud(
        "/saved_point_cloud", points=points, colors=colors,
        point_size=0.005, point_shape="circle",
    )
    # 使用点云的主要范围设置视角，避免极少数离群点导致地图看起来过小。
    lower, upper = np.percentile(points, [0.5, 99.5], axis=0)
    center = (lower + upper) / 2
    radius = max(float(np.linalg.norm(upper - lower)), 0.1)

    @server.on_client_connect
    def set_camera(client):
        """浏览器连接时自动对准点云，并使用 SLAM 的相机坐标系朝向。"""
        client.camera.up_direction = (0.0, -1.0, 0.0)
        client.camera.position = center + radius * np.array([0.7, -0.5, -0.7])
        client.camera.look_at = center

    port = server.get_port()
    print(f"查看地址：http://127.0.0.1:{port}", flush=True)
    print(f"软件渲染浏览器：./open_viewer.sh {port}（Ctrl+C 停止服务）", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()
