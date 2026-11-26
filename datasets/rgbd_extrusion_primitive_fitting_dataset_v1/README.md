# RGB-D 挤出体基元拟合数据集 v1

本目录包含 200 个 RGB-D 案例：彩色图、深度图、目标裁剪块、类别标签、包围盒，以及部分案例的超椭球（superellipsoid）拟合参数。

原始 ROS bag（200 个文件，约 25.7 GB）超过 GitHub 单文件 100 MB 的限制，未放入本仓库，托管在
[Release `rgbd-dataset-v1`](https://github.com/e2432/-3d-/releases/tag/rgbd-dataset-v1)。

## 目录结构

| 路径 | 内容 | 文件数 |
| --- | --- | --- |
| `color/color_case_{ID}.png` | 640×480 RGB 彩色图 | 200 |
| `depth/depth_case_{ID}.png` | 640×480 8-bit 灰度深度图 | 200 |
| `color_crops/crop_case_{ID}_id_{N}.png` | 目标裁剪块 | 808 |
| `labels_color_crops/crop_case_{ID}_id_{N}.csv` | 裁剪块类别标签，每行一个类别名（无表头） | 801 |
| `bboxes/case_{ID}.csv` | 包围盒，表头 `ID,Top-left x,Top-left y,Bottom-right x,Bottom-right y`（像素坐标） | 200（另有 175 个 `(1)`、36 个 `(2)` 副本和 1 个 `.xlsx`） |
| `labels/case_{ID}.csv` | 场景类别标签，表头 `ID,Label`；仅案例 1–25 | 25（另有 1 个 `.xlsx`） |
| `fitted_superellipsoids/case_{ID}.csv` | 超椭球拟合参数，表头 `ID,a1,a2,a3,e1,e2,x,y,z,roll,pitch,yaw`；仅案例 1–25 | 25（另有 2 个 `.xlsx`） |
| `bags_manifest.csv` | Release 中 200 个 bag 的文件名与字节数 | — |
| `MANIFEST.sha256` | 上述图像/标注文件的 SHA-256 | — |
| Release `rgbd-dataset-v1` | `case_{ID}.bag` ROS bag | 200（约 25.7 GB） |

本目录数据文件合计约 109 MB。

## 案例 ID

200 个案例的 ID 为：`1–85`、`201–232`、`1721–1732`、`1737–1747`、`1941–2000`。
`color/`、`depth/`、`color_crops/`、`bboxes/` 覆盖全部 200 个案例；
`labels/` 与 `fitted_superellipsoids/` 仅覆盖案例 `1–25`。

## 获取原始 bag

推荐使用 [GitHub CLI](https://cli.github.com/) 下载全部 bag：

```bash
gh release download rgbd-dataset-v1 --repo e2432/-3d- --dir bags --pattern '*.bag'
```

也可以从 [Release 页面](https://github.com/e2432/-3d-/releases/tag/rgbd-dataset-v1) 手动下载。
bag 文件未做任何格式转换；文件名与 `bags_manifest.csv` 中的记录一致。

## 已知问题

- 源数据中 `case_210` 有 3 份字节完全相同的 bag；Release 只保留了一份 `case_210.bag`，因此 Release 为 200 个文件，其余非 bag 目录原样保留。
- `bboxes/` 中有 175 个 `case_{ID}(1).csv` 与 36 个 `case_{ID}(2).csv` 重复命名副本；`labels/`、`fitted_superellipsoids/` 各含 1–2 个 `.xlsx` 副本。本次上传保留这些文件，未做去重或内容校验。
- `color_crops/` 有 808 个裁剪块，`labels_color_crops/` 只有 801 个标签文件，缺少：
  `crop_case_1_id_5`、`crop_case_2_id_5`、`crop_case_3_id_5`、`crop_case_3_id_6`、`crop_case_4_id_5`、`crop_case_5_id_5`、`crop_case_6_id_5`。
- 深度图为 8-bit 灰度 PNG，文件中未附带相机内参或深度单位说明。
- 数据集未附带许可证声明；使用前请与数据提供方确认授权。

## 校验

在本目录执行：

```bash
sha256sum -c MANIFEST.sha256
```
