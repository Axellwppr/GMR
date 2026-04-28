# IK 配置简化生成（只计算 quat offset）

这个目录现在支持一条简化流程：
- 不做 `human_scale` 优化
- 不做 `pos_offset` 优化（统一写 0）
- 只基于 T-pose 一次性计算 `quat offset`
- 输入是一个“无 quat/pos_offset 的简化 mapping 配置”

## 新脚本

`ik_config_manager/build_ik_from_tpose.py`

输入：
- 机器人名称（可选，仅日志用：`--robot_name`）
- 机器人模型文件（`--robot_xml`）
- 机器人 T-pose 初始姿态（`--pose_init`）
- 人体 T-pose（`--src_human smplx|bvh` + 对应文件）
- 简化 mapping 配置（`--mapping_config`）

输出：
- 可直接被 GMR 使用的完整 IK 配置 JSON（包含计算好的 `ik_match_table1/2[*][4]` 四元数偏移）
- 可选：生成完成后直接打开 MuJoCo 预览机器人 T-pose 与映射轴

## 简化 mapping 配置格式

```json
{
  "human_root_name": "pelvis",
  "robot_root_name": "pelvis",
  "ground_height": 0.0,
  "human_height_assumption": 1.8,
  "use_ik_match_table1": true,
  "use_ik_match_table2": true,
  "root_quat_offset_override_wxyz": [0.5, -0.5, -0.5, -0.5],
  "root_quat_extra_wxyz": [1.0, 0.0, 0.0, 0.0],
  "human_scale_table": {
    "pelvis": 1.0
  },
  "mappings": [
    {
      "robot_link": "pelvis",
      "human_bone": "pelvis",
      "table1_pos": 100,
      "table1_rot": 10,
      "table2_pos": 100,
      "table2_rot": 5
    }
  ]
}
```

说明：
- `mappings` 不包含 `pos_offset`、`quat_offset`
- `table2_pos/table2_rot` 可省略，默认与 table1 相同
- 如果不提供 `human_scale_table`，脚本会按映射骨骼自动补成 `1.0`
- 可选 `root_quat_offset_override_wxyz`：直接覆盖 root link 的 quat offset（推荐用于修正 root 90° 偏差）
- 可选 `root_quat_extra_wxyz`：在自动计算 root quat 上额外左乘一个旋转
- 也支持在单条 mapping 里配置 `quat_offset_override_wxyz` 或 `quat_extra_wxyz`（优先级高于全局 root 配置）

## 使用示例

SMPLX:
```bash
python ik_config_manager/build_ik_from_tpose.py \
  --robot_name unitree_g1 \
  --robot_xml assets/unitree_g1/g1_mocap_29dof.xml \
  --src_human smplx \
  --smplx_file ik_config_manager/SMPLX_TPOSE_UNIFIED_AMASS.npz \
  --pose_init ik_config_manager/pose_inits/unitree_g1_tpose.json \
  --mapping_config ik_config_manager/simple_configs/smplx_to_g1_simple.json \
  --output general_motion_retargeting/ik_configs/smplx_to_g1_from_tpose.json
```

BVH:
```bash
python ik_config_manager/build_ik_from_tpose.py \
  --robot_name unitree_g1 \
  --robot_xml assets/unitree_g1/g1_mocap_29dof.xml \
  --src_human bvh \
  --bvh_file ik_config_manager/TPOSE.bvh \
  --pose_init ik_config_manager/pose_inits/unitree_g1_tpose.json \
  --mapping_config ik_config_manager/simple_configs/bvh_to_g1_simple.json \
  --output general_motion_retargeting/ik_configs/bvh_to_g1_from_tpose.json
```

## 生成后可视化轴

在命令末尾加上：

```bash
  --preview \
  --preview_seconds 120 \
  --preview_axis_scale 0.08 \
  --preview_show_human_raw_axes \
  --preview_show_corrected_axes
```

说明：
- 机器人 link 轴：实色 RGB（X/Y/Z）
- `--preview_show_human_raw_axes`：显示对齐后的人体原始轴（半透明）
- `--preview_show_corrected_axes`：显示应用 scale + quat/pos offset 后的人体轴（半透明，使用真实矫正后位置）

## 示例简化配置

- `ik_config_manager/simple_configs/smplx_to_g1_simple.json`
- `ik_config_manager/simple_configs/bvh_to_g1_simple.json`

这两个文件只保留 mapping+权重，不包含 quat/pos offset。
