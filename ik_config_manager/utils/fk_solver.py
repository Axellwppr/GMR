import mujoco
import numpy as np
import xml.etree.ElementTree as ETree
import torch
import copy
from collections import OrderedDict


class MuJoCoFK:
    """
    通过 MuJoCo 解析机器人 XML，获取所有链接信息，用于前向运动学计算各 link 的姿态
    """
    def __init__(self, asset_file: str, device=torch.device("cpu")):
        self.mjcf_file = asset_file
        self.model = mujoco.MjModel.from_xml_path(self.mjcf_file)
        self.data  = mujoco.MjData(self.model)

        self.dof_axis = []
        self.has_freejoint = False
        self._joint_order = []
        for jid in range(self.model.njnt):
            jname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            jtype = int(self.model.jnt_type[jid])
            if jtype == int(mujoco.mjtJoint.mjJNT_FREE):
                self.has_freejoint = True
                continue
            if jtype == int(mujoco.mjtJoint.mjJNT_BALL):
                raise NotImplementedError(
                    f"Ball joint '{jname}' is not supported by MuJoCoFK qpos builder."
                )
            if jname is None:
                jname = f"joint_{jid}"
            self._joint_order.append(jname)
            self.dof_axis.append(self.model.jnt_axis[jid].tolist())

        self.num_dof = len(self._joint_order)
        self.num_extend_dof = self.num_dof

        self.mjcf_data = mjcf_data = self.from_mjcf(self.mjcf_file)
        self.body_names = copy.deepcopy(mjcf_data["node_names"])
        self._parents = mjcf_data["parent_indices"]
        self._proper_kinematic_structure = copy.deepcopy(mjcf_data["node_names"])
        self._offsets = mjcf_data["local_translation"][None,].to(device)
        self._local_rotation = mjcf_data["local_rotation"][None,].to(device)
        self.actuated_joints_idx = np.arange(len(self._joint_order), dtype=np.int32)
        self.dof_axis = torch.tensor(self.dof_axis, dtype=torch.float32)
        self.num_bodies = len(self.body_names)
        # Map parsed body names to actual MuJoCo body IDs.
        # MuJoCo body id 0 is always "world", while parsed node_names starts from robot root body.
        # Using numeric range directly may cause an off-by-one mismatch (root -> world).
        self.body_ids = []
        for body_name in self.body_names:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id < 0:
                raise ValueError(f"Body '{body_name}' not found in MuJoCo model.")
            self.body_ids.append(int(body_id))

        if mjcf_data["joints_range"].shape[0] == self.num_dof:
            self.joints_range = mjcf_data["joints_range"].to(device)
        else:
            joint_ranges = []
            for jid in range(self.model.njnt):
                jtype = int(self.model.jnt_type[jid])
                if jtype == int(mujoco.mjtJoint.mjJNT_FREE):
                    continue
                if jtype in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
                    joint_ranges.append(np.array(self.model.jnt_range[jid], dtype=np.float32))
            self.joints_range = torch.from_numpy(np.array(joint_ranges, dtype=np.float32)).to(device)
        # 添加关节顺序列表
        self.joint_order = self.get_joint_order()
    
    def from_mjcf(self, path):
        # function from Poselib:
        tree = ETree.parse(path)
        xml_doc_root = tree.getroot()
        xml_world_body = xml_doc_root.find("worldbody")
        if xml_world_body is None:
            raise ValueError("MJCF parsed incorrectly please verify it.")
        # assume this is the root
        xml_body_root = xml_world_body.find("body")
        if xml_body_root is None:
            raise ValueError("MJCF parsed incorrectly please verify it.")

        xml_joint_root = xml_body_root.find("joint")

        node_names = []
        parent_indices = []
        local_translation = []
        local_rotation = []
        joints_range = []
        body_to_joint = OrderedDict()

        # recursively adding all nodes into the skel_tree
        def _add_xml_node(xml_node, parent_index, node_index):
            node_name = xml_node.attrib.get("name")
            # parse the local translation into float list
            pos = np.fromstring(
                xml_node.attrib.get("pos", "0 0 0"), dtype=float, sep=" "
            )
            quat = np.fromstring(
                xml_node.attrib.get("quat", "1 0 0 0"), dtype=float, sep=" "
            )
            node_names.append(node_name)
            parent_indices.append(parent_index)
            local_translation.append(pos)
            local_rotation.append(quat)
            curr_index = node_index
            node_index += 1
            all_joints = xml_node.findall(
                "joint"
            )  # joints need to remove the first 6 joints
            if len(all_joints) == 6:
                all_joints = all_joints[6:]

            for joint in all_joints:
                if not joint.attrib.get("range") is None:
                    joints_range.append(
                        np.fromstring(joint.attrib.get("range"), dtype=float, sep=" ")
                    )
                else:
                    if not joint.attrib.get("type") == "free":
                        joints_range.append([-np.pi, np.pi])
            for joint_node in xml_node.findall("joint"):
                body_to_joint[node_name] = joint_node.attrib.get("name")

            for next_node in xml_node.findall("body"):
                node_index = _add_xml_node(next_node, curr_index, node_index)

            return node_index

        _add_xml_node(xml_body_root, -1, 0)
        if self.num_dof > 0 and len(joints_range) not in (0, self.num_dof):
            print(
                f"[WARN] Parsed joints_range({len(joints_range)}) != num_dof({self.num_dof}), "
                "will fallback to MuJoCo model joint ranges."
            )
        return {
            "node_names": node_names,
            "parent_indices": torch.from_numpy(
                np.array(parent_indices, dtype=np.int32)
            ),
            "local_translation": torch.from_numpy(
                np.array(local_translation, dtype=np.float32)
            ),
            "local_rotation": torch.from_numpy(
                np.array(local_rotation, dtype=np.float32)
            ),
            "joints_range": torch.from_numpy(np.array(joints_range)),
            "body_to_joint": body_to_joint,
        }
    
    def get_joint_order(self):
        """获取求解器的关节顺序列表"""
        return list(self._joint_order)
    
    def fk(self, qpos_full: np.ndarray):
        self.data.qpos[:] = qpos_full
        mujoco.mj_forward(self.model, self.data)
        centers, Rs = [], []
        for i in self.body_ids:
            p = np.array(self.data.xpos[i], dtype=np.float32)
            Rm = np.array(self.data.xmat[i], dtype=np.float64).reshape(3, 3)
            centers.append(p)
            Rs.append(Rm)
        return np.stack(centers, axis=0), np.stack(Rs, axis=0)
