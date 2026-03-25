import argparse
import copy
import json
import pathlib

from scipy.spatial.transform import Rotation as R


HERE = pathlib.Path(__file__).resolve().parent
GMR_ROOT = HERE.parent
IK_CONFIG_ROOT = GMR_ROOT / "general_motion_retargeting" / "ik_configs"


def load_json(path: pathlib.Path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(data, path: pathlib.Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
        f.write("\n")


def canonical_human_name(name: str) -> str:
    return name.lower()


def xrobot_human_name(name: str) -> str:
    return "_".join(part.capitalize() for part in name.split("_"))


def wxyz_to_rot(quat_wxyz):
    w, x, y, z = quat_wxyz
    return R.from_quat([x, y, z, w])


def rot_to_wxyz(rot: R):
    x, y, z, w = rot.as_quat()
    return [float(w), float(x), float(y), float(z)]


def infer_left_delta(ref_smplx: dict, ref_xrobot: dict) -> R:
    deltas = []
    for table_name in ("ik_match_table1", "ik_match_table2"):
        common_keys = set(ref_smplx[table_name]) & set(ref_xrobot[table_name])
        for robot_body in sorted(common_keys):
            quat_s = ref_smplx[table_name][robot_body][4]
            quat_x = ref_xrobot[table_name][robot_body][4]
            rs = wxyz_to_rot(quat_s)
            rx = wxyz_to_rot(quat_x)
            deltas.append(rx * rs.inv())

    ref = deltas[0]
    for idx, delta in enumerate(deltas[1:], start=1):
        err_deg = (delta.inv() * ref).magnitude() * 180.0 / 3.141592653589793
        if err_deg > 1e-6:
            raise ValueError(f"Inconsistent delta at sample {idx}: {err_deg:.6f} deg")
    return ref


def build_overrides(ref_smplx: dict, ref_xrobot: dict):
    overrides = {}
    for table_name in ("ik_match_table1", "ik_match_table2"):
        overrides[table_name] = {}

        smplx_by_human = {
            canonical_human_name(entry[0]): entry
            for entry in ref_smplx[table_name].values()
        }
        xrobot_by_human = {
            canonical_human_name(entry[0]): entry
            for entry in ref_xrobot[table_name].values()
        }

        common_humans = set(smplx_by_human) & set(xrobot_by_human)
        for human_name in sorted(common_humans):
            entry_s = smplx_by_human[human_name]
            entry_x = xrobot_by_human[human_name]
            change = {}
            if entry_s[1:3] != entry_x[1:3]:
                change["weights"] = copy.deepcopy(entry_x[1:3])
            if entry_s[3] != entry_x[3]:
                change["pos_offset"] = copy.deepcopy(entry_x[3])
            if change:
                overrides[table_name][human_name] = change
    return overrides


def convert_config(target_smplx: dict, left_delta: R, overrides: dict):
    out = copy.deepcopy(target_smplx)
    out["human_root_name"] = xrobot_human_name(target_smplx["human_root_name"])
    out["human_scale_table"] = {
        xrobot_human_name(name): value
        for name, value in target_smplx["human_scale_table"].items()
    }

    for table_name in ("ik_match_table1", "ik_match_table2"):
        new_table = {}
        for robot_body, entry in target_smplx[table_name].items():
            human_name, pos_w, rot_w, pos_offset, rot_offset = entry
            human_key = canonical_human_name(human_name)

            new_entry = [
                xrobot_human_name(human_name),
                float(pos_w),
                float(rot_w),
                copy.deepcopy(pos_offset),
                rot_to_wxyz(left_delta * wxyz_to_rot(rot_offset)),
            ]

            if human_key in overrides[table_name]:
                override = overrides[table_name][human_key]
                if "weights" in override:
                    new_entry[1] = override["weights"][0]
                    new_entry[2] = override["weights"][1]
                if "pos_offset" in override:
                    new_entry[3] = copy.deepcopy(override["pos_offset"])

            new_table[robot_body] = new_entry
        out[table_name] = new_table
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ref-smplx",
        type=pathlib.Path,
        default=IK_CONFIG_ROOT / "smplx_to_g1.json",
    )
    parser.add_argument(
        "--ref-xrobot",
        type=pathlib.Path,
        default=IK_CONFIG_ROOT / "xrobot_to_g1.json",
    )
    parser.add_argument(
        "--target-smplx",
        type=pathlib.Path,
        default=IK_CONFIG_ROOT / "smplx_to_l7.json",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=IK_CONFIG_ROOT / "xrobot_to_l7.json",
    )
    args = parser.parse_args()

    ref_smplx = load_json(args.ref_smplx)
    ref_xrobot = load_json(args.ref_xrobot)
    target_smplx = load_json(args.target_smplx)

    left_delta = infer_left_delta(ref_smplx, ref_xrobot)
    overrides = build_overrides(ref_smplx, ref_xrobot)
    output = convert_config(target_smplx, left_delta, overrides)
    save_json(output, args.output)

    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
