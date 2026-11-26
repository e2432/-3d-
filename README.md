# 参数化可见性几何与 3D 信息矩阵最小实验

本实验实现文档中的第一阶段计算，不使用真实照片、CAD 文件或神经网络。
它用参数化长方体或圆柱体生成表面采样点和法向量，再计算相机视角下的可见性、点到平面观测 Jacobian，以及 6×6 几何 Fisher 信息矩阵。

## 实验内容

程序会完成以下检查。

1. 生成完整物体表面点云和法向量。
2. 根据视场、表面朝向、物体自遮挡和可选外部遮挡物计算二值可见性。
3. 对每个可见点计算

   ```text
   J_i = n_i^T [ I  -[p_i]_x ]
   ```

4. 累加

   ```text
   Lambda = sum_i v_i J_i^T R_i^-1 J_i
   ```

5. 用中心有限差分计算数值 Jacobian，比较解析 FIM 和数值 FIM。
6. 比较完整可见和被遮挡时的特征值，观察哪些位姿方向的约束变弱。

## Ubuntu 运行

```bash
cd visibility_fim
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python run_experiment.py --shape cuboid --output outputs/cuboid
python run_experiment.py --shape cylinder --output outputs/cylinder

# 自定义尺寸，单位为米
python run_experiment.py --shape cuboid --cuboid-size 0.50 0.30 0.40 --output outputs/custom_cuboid
python run_experiment.py --shape cylinder --cylinder-radius 0.18 --cylinder-height 0.45 --output outputs/custom_cylinder
```

如果只安装了系统 Python 和 NumPy，也可以直接运行。程序不依赖 GPU。

## 输出文件

- `summary.json` 保存采样数量、可见数量、FIM、特征值和有限差分误差。
- `surface_points.csv` 保存物体坐标系和相机坐标系中的表面采样点、法向量与可见性。
- `fim.npz` 保存完整可见与遮挡情况下的矩阵和特征分解结果。

## 坐标约定

- 物体坐标系中的点经过 `T_co` 变换到相机坐标系。
- 相机位于原点，`+Z` 方向朝向场景。
- 位姿微扰采用左乘形式 `T' = Exp(delta) T`。
- `delta = [delta_tx, delta_ty, delta_tz, delta_wx, delta_wy, delta_wz]`。

当前实现使用点到平面标量观测和固定标量噪声方差。后续接入 RGB-D 图像时，可以把网络或几何拟合模块输出的点、法向量、可见性置信度和噪声替换到同一计算层。

## RGB-D 挤出体基元拟合数据集 v1

本仓库同时发布一份 RGB-D 数据集：200 个案例，包含彩色图、深度图、目标裁剪块、类别标签、包围盒与超椭球拟合参数。说明见 [`datasets/rgbd_extrusion_primitive_fitting_dataset_v1/README.md`](datasets/rgbd_extrusion_primitive_fitting_dataset_v1/README.md)。

原始 ROS bag（200 个文件，约 25.7 GB）托管在 [Release `rgbd-dataset-v1`](https://github.com/e2432/-3d-/releases/tag/rgbd-dataset-v1)。
