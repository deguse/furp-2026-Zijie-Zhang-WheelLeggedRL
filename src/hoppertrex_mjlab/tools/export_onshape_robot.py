#!/usr/bin/env python3
"""Export the HopperTrex Onshape assembly to URDF and MJCF assets."""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from dataclasses import dataclass
import itertools
import json
import math
import os
import re
import shutil
import sys
from urllib.parse import urlparse
import xml.dom.minidom
import xml.etree.ElementTree as ET
from pathlib import Path


DEFAULT_ONSHAPE_URL = (
    "https://cad.onshape.com/documents/eba1be29ba3a0caf3e640bd9/"
    "w/f954e7178f5bf956382f7efa/e/ae1c6caf19c0b947e3e3dcf9"
)
BASE_LINK_NAME = "chassis_base"

# Leg motors: Damiao DM-J6248P. Wheel motors: Lingkong/RMD-L-9025-35T.
DM_J6248P_PEAK_TORQUE_NM = 97.0
DM_J6248P_NO_LOAD_SPEED_RAD_PER_SEC = 60.0 * 2.0 * math.pi / 60.0
RMD_L_9025_35T_PEAK_TORQUE_NM = 5.8
RMD_L_9025_35T_NO_LOAD_SPEED_RAD_PER_SEC = 280.0 * 2.0 * math.pi / 60.0

JOINT_MAX_EFFORT_NM = DM_J6248P_PEAK_TORQUE_NM
JOINT_MAX_VELOCITY_RAD_PER_SEC = RMD_L_9025_35T_NO_LOAD_SPEED_RAD_PER_SEC
JOINT_NAME_MAP = {
    "thigh_left": "thigh_left_01",
    "thigh_right": "thigh_right_01",
}
LEG_JOINT_LIMITS_RAD = {
    "thigh_left_01": (-2.443, 0.698),
    "thigh_right_01": (-0.698, 2.443),
    "knee_left": (-0.698, 2.094),
    "knee_right": (-2.094, 0.698),
}
WHEEL_JOINTS = {"wheel_left", "wheel_right"}

# Use Onshape mass properties by default. Put part-name overrides here if the CAD
# model is intentionally lighter/heavier than the physical robot.
LINK_MASS_OVERRIDES_KG: dict[str, float] = {}


@dataclass(frozen=True)
class OnshapeTarget:
    document_id: str
    workspace_id: str
    element_id: str


def parse_onshape_url(url: str) -> OnshapeTarget:
    parts = [part for part in urlparse(url).path.split("/") if part]
    try:
        document_id = parts[parts.index("documents") + 1]
        workspace_id = parts[parts.index("w") + 1]
        element_id = parts[parts.index("e") + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError(
            "Onshape URL must contain /documents/<document>/w/<workspace>/e/<element>"
        ) from exc

    return OnshapeTarget(
        document_id=document_id,
        workspace_id=workspace_id,
        element_id=element_id,
    )


def read_onshape_key(path: Path) -> tuple[str, str]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    lines = [line for line in lines if line]
    if len(lines) != 2:
        raise RuntimeError(f"{path} must contain access key and secret key on two lines")
    return lines[0], lines[1]


def configure_onshape_env(key_path: Path) -> None:
    access_key, secret_key = read_onshape_key(key_path)
    os.environ["ONSHAPE_API"] = "https://cad.onshape.com"
    os.environ["ONSHAPE_ACCESS_KEY"] = access_key
    os.environ["ONSHAPE_SECRET_KEY"] = secret_key


def patch_exporter_cache(cache_dir: Path) -> None:
    from onshape_urdf_exporter.onshape_api.client import Client

    def get_cache_path() -> Path:
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    Client.get_cache_path = staticmethod(get_cache_path)


def _sanitize_joint_name(name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_]+", "_", name.strip())
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    return sanitized or "joint"


def _has_two_mated_occurrences(feature_data: dict) -> bool:
    entities = feature_data.get("matedEntities", [])
    if len(entities) != 2:
        return False
    return all(entity.get("matedOccurrence") for entity in entities)


def _mated_top_occurrence_id(mated_entity: dict) -> str | None:
    occurrence = mated_entity.get("matedOccurrence")
    if not occurrence:
        return None
    return str(occurrence[0])


def _instance_name_by_id(root: dict) -> dict[str, str]:
    return {
        str(instance.get("id")): str(instance.get("name", ""))
        for instance in root.get("instances", [])
    }


def _choose_dof_tree_root(root: dict, dof_features: list[dict]) -> str | None:
    connected_ids: set[str] = set()
    degree: defaultdict[str, int] = defaultdict(int)
    for feature in dof_features:
        entities = feature.get("featureData", {}).get("matedEntities", [])
        ends = [_mated_top_occurrence_id(entity) for entity in entities]
        if ends[0] is None or ends[1] is None:
            continue
        connected_ids.update(ends)
        degree[ends[0]] += 1
        degree[ends[1]] += 1

    names = _instance_name_by_id(root)
    root_name_patterns = ("base", "body", "trunk", "chassis", "frame")
    named_candidates = [
        occurrence_id
        for occurrence_id in connected_ids
        if any(pattern in names.get(occurrence_id, "").lower() for pattern in root_name_patterns)
    ]
    if len(named_candidates) == 1:
        return named_candidates[0]
    if named_candidates:
        return max(named_candidates, key=lambda occurrence_id: degree[occurrence_id])
    if degree:
        return max(degree, key=degree.get)
    return None


def _orient_dof_mates_as_tree(root: dict) -> None:
    """Order DOF mate entities as child, then parent for onshape-urdf-exporter."""

    dof_features = []
    for feature in root.get("features", []):
        if feature.get("featureType") != "mate" or feature.get("suppressed"):
            continue
        data = feature.get("featureData", {})
        if not data.get("name", "").startswith("dof_"):
            continue
        if not _has_two_mated_occurrences(data):
            continue
        dof_features.append(feature)

    root_id = _choose_dof_tree_root(root, dof_features)
    if root_id is None:
        return

    adjacency: defaultdict[str, list[tuple[str, dict]]] = defaultdict(list)
    for feature in dof_features:
        entities = feature["featureData"]["matedEntities"]
        first = _mated_top_occurrence_id(entities[0])
        second = _mated_top_occurrence_id(entities[1])
        if first is None or second is None:
            continue
        adjacency[first].append((second, feature))
        adjacency[second].append((first, feature))

    desired_parent_child: dict[int, tuple[str, str]] = {}
    visited = {root_id}
    queue: deque[str] = deque([root_id])
    while queue:
        parent = queue.popleft()
        for child, feature in adjacency[parent]:
            if child in visited:
                continue
            visited.add(child)
            queue.append(child)
            desired_parent_child[id(feature)] = (parent, child)

    for feature in dof_features:
        desired = desired_parent_child.get(id(feature))
        if desired is None:
            continue
        parent, child = desired
        data = feature["featureData"]
        entities = data["matedEntities"]
        first = _mated_top_occurrence_id(entities[0])
        second = _mated_top_occurrence_id(entities[1])
        if first == parent and second == child:
            entities[0], entities[1] = entities[1], entities[0]


def patch_exporter_revolute_dofs() -> None:
    """Treat active Onshape revolute mates as continuous DOFs.

    onshape-urdf-exporter only exports mates whose names start with ``dof_``.
    The HopperTrex CAD model already contains the revolute mates, but they are
    named after the wheel/roller instances. Patch the assembly payload in memory
    so the exporter keeps the existing CAD structure without requiring Onshape
    edits. The exporter also assumes mated entities are ordered child first,
    parent second, so orient the resulting DOFs as a tree rooted at the body/base.
    """

    from onshape_urdf_exporter.onshape_api.client import Client

    original_get_assembly = Client.get_assembly

    def get_assembly_with_revolute_dofs(self, *args, **kwargs):
        assembly = original_get_assembly(self, *args, **kwargs)
        root = assembly.get("rootAssembly", {})
        features = root.get("features", [])
        for feature in features:
            if feature.get("featureType") != "mate" or feature.get("suppressed"):
                continue

            data = feature.get("featureData", {})
            if data.get("mateType") != "REVOLUTE":
                continue
            if not _has_two_mated_occurrences(data):
                continue

            original_name = _sanitize_joint_name(str(data.get("name", "")))
            if original_name.startswith("dof_"):
                continue

            parts = original_name.split("_")
            if "wheel" in parts or "continuous" in parts:
                data["name"] = f"dof_{original_name}"
            else:
                data["name"] = f"dof_{original_name}_continuous"

        _orient_dof_mates_as_tree(root)
        return assembly

    Client.get_assembly = get_assembly_with_revolute_dofs


def find_assembly_name(target: OnshapeTarget) -> str:
    from onshape_urdf_exporter.onshape_api.client import Client

    client = Client(logging=False)
    elements = client.list_elements(target.document_id, target.workspace_id).json()
    target_element = next(
        (element for element in elements if element.get("id") == target.element_id),
        None,
    )
    assemblies = [element for element in elements if element.get("type") == "Assembly"]

    if target_element is not None and target_element.get("type") == "Assembly":
        return str(target_element["name"])

    if len(assemblies) == 1:
        assembly = assemblies[0]
        print(
            "Target element is not an assembly; using the only assembly "
            f"{assembly['name']!r} ({assembly['id']})."
        )
        return str(assembly["name"])

    available = ", ".join(f"{entry['name']} ({entry['id']})" for entry in assemblies)
    raise RuntimeError(
        f"Could not resolve assembly element {target.element_id}. "
        f"Available assemblies: {available}"
    )


def write_exporter_config(
    output_dir: Path,
    robot_name: str,
    assembly_name: str,
    target: OnshapeTarget,
) -> None:
    config = {
        "document_id": target.document_id,
        "workspace_id": target.workspace_id,
        "assembly_name": assembly_name,
        "robot_name": robot_name,
        "draw_frames": False,
        "draw_collisions": False,
        "use_fixed_links": False,
        "configuration": "default",
        "ignore_limits": False,
        "joint_max_effort": JOINT_MAX_EFFORT_NM,
        "joint_max_velocity": JOINT_MAX_VELOCITY_RAD_PER_SEC,
        "no_dynamics": False,
        "simplify_stls": False,
        "use_collisions_configurations": True,
        "package_name": "",
        "add_dummy_base_link": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.yaml").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )


def run_urdf_exporter(output_dir: Path) -> None:
    from onshape_urdf_exporter.__main__ import main

    previous_argv = sys.argv[:]
    try:
        sys.argv = ["onshape-urdf-exporter", str(output_dir)]
        main()
    finally:
        sys.argv = previous_argv


def prettify_xml(path: Path) -> None:
    raw = path.read_text(encoding="utf-8")
    pretty = xml.dom.minidom.parseString(raw.encode("utf-8")).toprettyxml(indent="  ")
    cleaned = "\n".join(line for line in pretty.splitlines() if line.strip())
    path.write_text(cleaned + "\n", encoding="utf-8")


def format_float(value: float) -> str:
    return f"{value:.12g}"


def format_vector(values: list[float] | tuple[float, ...]) -> str:
    return " ".join(format_float(value) for value in values)


def parse_vector(value: str | None, default: tuple[float, float, float]) -> list[float]:
    if not value:
        return list(default)
    parts = value.split()
    if len(parts) != 3:
        return list(default)
    try:
        return [float(part) for part in parts]
    except ValueError:
        return list(default)


def rotation_matrix_from_rpy(rpy: list[float]) -> list[list[float]]:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def transform_point(
    point: tuple[float, float, float] | list[float],
    origin_xyz: list[float],
    rotation: list[list[float]],
) -> list[float]:
    return [
        origin_xyz[row]
        + rotation[row][0] * point[0]
        + rotation[row][1] * point[1]
        + rotation[row][2] * point[2]
        for row in range(3)
    ]


def transformed_aabb(
    bounds_min: list[float],
    bounds_max: list[float],
    origin_xyz: list[float],
    origin_rpy: list[float],
) -> tuple[list[float], list[float]]:
    rotation = rotation_matrix_from_rpy(origin_rpy)
    corners = [
        transform_point(corner, origin_xyz, rotation)
        for corner in itertools.product(*zip(bounds_min, bounds_max))
    ]
    out_min = [min(corner[axis] for corner in corners) for axis in range(3)]
    out_max = [max(corner[axis] for corner in corners) for axis in range(3)]
    center = [(out_min[axis] + out_max[axis]) / 2 for axis in range(3)]
    extents = [max(out_max[axis] - out_min[axis], 1e-6) for axis in range(3)]
    return center, extents


def box_inertia_diagonal(mass: float, extents: list[float]) -> tuple[float, float, float]:
    x, y, z = extents
    return (
        max(mass * (y * y + z * z) / 12.0, 1e-9),
        max(mass * (x * x + z * z) / 12.0, 1e-9),
        max(mass * (x * x + y * y) / 12.0, 1e-9),
    )


def find_link_mesh(link: ET.Element) -> tuple[str, ET.Element | None]:
    for element_name in ("collision", "visual"):
        for element in link.findall(element_name):
            mesh = element.find("./geometry/mesh")
            filename = mesh.get("filename") if mesh is not None else None
            if filename:
                return filename, element.find("origin")
    raise RuntimeError(f"Link {link.get('name')!r} has no mesh geometry")


def estimate_link_inertial(
    link: ET.Element,
    mass: float,
    urdf_dir: Path,
) -> tuple[list[float], tuple[float, float, float]]:
    import trimesh

    mesh_filename, origin = find_link_mesh(link)
    mesh_path = urdf_dir / mesh_filename
    mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
    if mesh.bounds is None:
        raise RuntimeError(f"Could not calculate bounds for {mesh_path}")

    origin_xyz = parse_vector(origin.get("xyz") if origin is not None else None, (0.0, 0.0, 0.0))
    origin_rpy = parse_vector(origin.get("rpy") if origin is not None else None, (0.0, 0.0, 0.0))
    center, extents = transformed_aabb(
        [float(value) for value in mesh.bounds[0]],
        [float(value) for value in mesh.bounds[1]],
        origin_xyz,
        origin_rpy,
    )
    return center, box_inertia_diagonal(mass, extents)


def set_link_inertial(
    link: ET.Element,
    mass: float,
    center: list[float],
    inertia: tuple[float, float, float],
) -> None:
    inertial = link.find("inertial")
    if inertial is None:
        inertial = ET.SubElement(link, "inertial")

    origin = inertial.find("origin")
    if origin is None:
        origin = ET.SubElement(inertial, "origin")
    origin.set("xyz", format_vector(center))
    origin.set("rpy", "0 0 0")

    mass_element = inertial.find("mass")
    if mass_element is None:
        mass_element = ET.SubElement(inertial, "mass")
    mass_element.set("value", format_float(mass))

    inertia_element = inertial.find("inertia")
    if inertia_element is None:
        inertia_element = ET.SubElement(inertial, "inertia")
    ixx, iyy, izz = inertia
    inertia_element.set("ixx", format_float(ixx))
    inertia_element.set("ixy", "0")
    inertia_element.set("ixz", "0")
    inertia_element.set("iyy", format_float(iyy))
    inertia_element.set("iyz", "0")
    inertia_element.set("izz", format_float(izz))


def apply_inertial_overrides(urdf_path: Path) -> None:
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    applied: list[str] = []
    for link in root.findall("link"):
        name = link.get("name")
        mass = LINK_MASS_OVERRIDES_KG.get(name or "")
        if mass is None:
            continue
        center, inertia = estimate_link_inertial(link, mass, urdf_path.parent)
        set_link_inertial(link, mass, center, inertia)
        applied.append(f"{name}={format_float(mass)}kg")

    tree.write(urdf_path, encoding="unicode")
    prettify_xml(urdf_path)
    print(f"Applied inertial overrides: {', '.join(applied)}")


def organize_urdf_assets(output_dir: Path) -> Path:
    mesh_dir = output_dir / "meshes"
    metadata_dir = output_dir / "metadata"
    mesh_dir.mkdir(exist_ok=True)
    metadata_dir.mkdir(exist_ok=True)

    for stl_path in output_dir.glob("*.stl"):
        stl_path.replace(mesh_dir / stl_path.name)

    for metadata_path in output_dir.glob("*.part"):
        metadata_path.replace(metadata_dir / metadata_path.name)

    urdf_path = output_dir / "robot.urdf"
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename")
        if not filename:
            continue
        mesh.set("filename", f"meshes/{Path(filename).name}")
    for joint in root.findall(".//joint"):
        name = joint.get("name")
        if joint.get("type") == "continuous" and name and name.endswith("_continuous"):
            joint.set("name", name[: -len("_continuous")])
        if name in JOINT_NAME_MAP:
            joint.set("name", JOINT_NAME_MAP[name])
            name = JOINT_NAME_MAP[name]
        if name in LEG_JOINT_LIMITS_RAD:
            joint.set("type", "revolute")
            limit = joint.find("limit")
            if limit is None:
                limit = ET.SubElement(joint, "limit")
            lower, upper = LEG_JOINT_LIMITS_RAD[name]
            limit.set("lower", format_float(lower))
            limit.set("upper", format_float(upper))
        if name in WHEEL_JOINTS:
            joint.set("type", "continuous")
    tree.write(urdf_path, encoding="unicode")
    prettify_xml(urdf_path)
    return urdf_path


def simplify_large_stls(mesh_dir: Path, max_faces: int = 180_000) -> None:
    import trimesh
    from onshape_urdf_exporter.stl_utils import simplify_stl

    for stl_path in sorted(mesh_dir.glob("*.stl")):
        mesh = trimesh.load(str(stl_path), force="mesh", process=False)
        face_count = len(mesh.faces)
        if face_count <= max_faces:
            continue
        print(f"Simplifying {stl_path.name}: {face_count} faces > {max_faces}")
        simplify_stl(stl_path)


def find_urdf_mesh_references(urdf_path: Path) -> set[str]:
    root = ET.parse(urdf_path).getroot()
    refs: set[str] = set()
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename")
        if filename:
            refs.add(filename)
    return refs


def read_urdf_inertials(urdf_path: Path) -> dict[str, dict[str, str]]:
    root = ET.parse(urdf_path).getroot()
    inertials: dict[str, dict[str, str]] = {}
    for link in root.findall("link"):
        name = link.get("name")
        inertial = link.find("inertial")
        if not name or inertial is None:
            continue
        origin = inertial.find("origin")
        mass = inertial.find("mass")
        inertia = inertial.find("inertia")
        if mass is None or inertia is None:
            continue
        inertials[name] = {
            "pos": origin.get("xyz", "0 0 0") if origin is not None else "0 0 0",
            "mass": mass.get("value", "0"),
            "diaginertia": format_vector(
                [
                    _float_attr(inertia, "ixx"),
                    _float_attr(inertia, "iyy"),
                    _float_attr(inertia, "izz"),
                ]
            ),
        }
    return inertials


def _set_body_pos_z(root: ET.Element, body_name: str, z_offset: float) -> None:
    body = root.find(f".//body[@name='{body_name}']")
    if body is None:
        return
    pos = parse_vector(body.get("pos"), (0.0, 0.0, 0.0))
    pos[2] += z_offset
    body.set("pos", format_vector(pos))


def _set_visual_geom_defaults(
    geom: ET.Element,
    mesh_name: str | None,
    used_names: set[str],
) -> None:
    base_name = geom.get("name") or (f"{mesh_name}_visual" if mesh_name else "visual")
    name = base_name
    suffix = 2
    while name in used_names:
        name = f"{base_name}_{suffix}"
        suffix += 1
    used_names.add(name)
    geom.set("name", name)
    geom.set("class", "visual")
    geom.set("contype", "0")
    geom.set("conaffinity", "0")
    geom.set("group", "2")


def _add_geom(parent: ET.Element, **attrs: str) -> ET.Element:
    return ET.SubElement(parent, "geom", {key: str(value) for key, value in attrs.items()})


def _add_hoppertrex_collision_geoms(root: ET.Element) -> None:
    chassis = root.find(".//body[@name='chassis_base']")
    thigh_left = root.find(".//body[@name='thigh_left']")
    thigh_right = root.find(".//body[@name='thigh_right']")
    calf_left = root.find(".//body[@name='calf_left']")
    calf_right = root.find(".//body[@name='calf_right']")
    wheel_left = root.find(".//body[@name='wheel']")
    wheel_right = root.find(".//body[@name='wheel_2']")

    if chassis is not None:
        _add_geom(
            chassis,
            name="chassis_base_collision",
            type="box",
            pos="0.055 0.126 0.153",
            size="0.18 0.16 0.08",
            **{"class": "collision"},
        )
    for side, thigh in (("left", thigh_left), ("right", thigh_right)):
        if thigh is None:
            continue
        _add_geom(
            thigh,
            name=f"thigh_{side}_collision",
            type="capsule",
            fromto="0 0 0 0.16 0.16 0",
            size="0.028",
            **{"class": "collision"},
        )
    for side, calf in (("left", calf_left), ("right", calf_right)):
        if calf is None:
            continue
        _add_geom(
            calf,
            name=f"calf_{side}_collision",
            type="capsule",
            fromto="0 0 0 0 0.18 0",
            size="0.026",
            **{"class": "collision"},
        )
    for side, wheel in (("left", wheel_left), ("right", wheel_right)):
        if wheel is None:
            continue
        # MuJoCo cylinders are aligned along local Z. The Onshape wheel joint also
        # rotates around local Z in the wheel body, so this creates a round rolling
        # contact around the actuated axis.
        _add_geom(
            wheel,
            name=f"wheel_{side}_collision",
            type="cylinder",
            pos="0.000105 -0.000075 0.01384",
            size="0.100 0.018",
            **{"class": "wheel_collision"},
        )


def _float_attr(element: ET.Element, name: str, default: float = 0.0) -> float:
    try:
        return float(element.get(name, str(default)))
    except ValueError:
        return default


def write_mujoco_import_urdf(urdf_path: Path) -> Path:
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    for link in root.findall(".//link"):
        inertial = link.find("inertial")
        if inertial is None:
            inertial = ET.SubElement(link, "inertial")
            ET.SubElement(inertial, "origin", {"xyz": "0 0 0"})

        mass = inertial.find("mass")
        if mass is None:
            mass = ET.SubElement(inertial, "mass")
        if _float_attr(mass, "value") <= 0.0:
            mass.set("value", "0.001")

        inertia = inertial.find("inertia")
        if inertia is None:
            inertia = ET.SubElement(inertial, "inertia")
        for attr in ("ixx", "iyy", "izz"):
            if _float_attr(inertia, attr) <= 0.0:
                inertia.set(attr, "1e-6")
        for attr in ("ixy", "ixz", "iyz"):
            inertia.set(attr, inertia.get(attr, "0"))

    import_path = urdf_path.with_name(f".{urdf_path.stem}.mujoco_import.urdf")
    tree.write(import_path, encoding="unicode")
    return import_path


def patch_mjcf(mjcf_path: Path, urdf_path: Path) -> None:
    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    compiler.set("angle", "radian")
    compiler.set("meshdir", ".")
    compiler.set("autolimits", "true")

    default = root.find("default")
    if default is None:
        default = ET.Element("default")
        asset = root.find("asset")
        insert_index = list(root).index(asset) if asset is not None else 1
        root.insert(insert_index, default)
    visual = ET.SubElement(default, "default", {"class": "visual"})
    ET.SubElement(
        visual,
        "geom",
        {
            "type": "mesh",
            "contype": "0",
            "conaffinity": "0",
            "group": "2",
        },
    )
    collision = ET.SubElement(default, "default", {"class": "collision"})
    ET.SubElement(
        collision,
        "geom",
        {
            "contype": "1",
            "conaffinity": "1",
            "condim": "3",
            "group": "3",
            "friction": "0.8 0.003 0.0001",
            "solref": "0.01 1",
            "solimp": "0.9 0.95 0.001",
        },
    )
    wheel_collision = ET.SubElement(default, "default", {"class": "wheel_collision"})
    ET.SubElement(
        wheel_collision,
        "geom",
        {
            "contype": "1",
            "conaffinity": "1",
            "condim": "6",
            "group": "3",
            "friction": "1.6 0.02 0.002",
            "solref": "0.005 1",
            "solimp": "0.95 0.99 0.001",
        },
    )

    for mesh in root.findall(".//mesh"):
        filename = mesh.get("file")
        if filename:
            mesh.set("file", f"meshes/{Path(filename).name}")

    inertials = read_urdf_inertials(urdf_path)
    base_inertial = inertials.get(BASE_LINK_NAME)
    worldbody = root.find("worldbody")
    if base_inertial is not None and worldbody is not None:
        base_body = worldbody.find(f"./body[@name='{BASE_LINK_NAME}']")
        if base_body is None:
            base_body = ET.Element("body", {"name": BASE_LINK_NAME})
            children = list(worldbody)
            for child in children:
                worldbody.remove(child)
                base_body.append(child)
            worldbody.append(base_body)

        inertial = base_body.find("inertial")
        if inertial is None:
            inertial = ET.Element("inertial")
            base_body.insert(0, inertial)
        inertial.set("pos", base_inertial["pos"])
        inertial.set("mass", base_inertial["mass"])
        inertial.set("diaginertia", base_inertial["diaginertia"])
        if base_body.find("freejoint") is None and base_body.find("joint[@type='free']") is None:
            freejoint = ET.Element("freejoint", {"name": "floating_base_joint"})
            insert_at = 1 if base_body.find("inertial") is not None else 0
            base_body.insert(insert_at, freejoint)

    for joint in root.findall(".//joint"):
        name = joint.get("name")
        if name in JOINT_NAME_MAP:
            joint.set("name", JOINT_NAME_MAP[name])
            name = JOINT_NAME_MAP[name]
        if name in LEG_JOINT_LIMITS_RAD:
            lower, upper = LEG_JOINT_LIMITS_RAD[name]
            joint.set("range", f"{format_float(lower)} {format_float(upper)}")
            joint.set("limited", "true")
            joint.set("actuatorfrcrange", f"{format_float(-DM_J6248P_PEAK_TORQUE_NM)} {format_float(DM_J6248P_PEAK_TORQUE_NM)}")
        elif name in WHEEL_JOINTS:
            joint.attrib.pop("range", None)
            joint.attrib.pop("limited", None)
            joint.set("actuatorfrcrange", f"{format_float(-RMD_L_9025_35T_PEAK_TORQUE_NM)} {format_float(RMD_L_9025_35T_PEAK_TORQUE_NM)}")

    used_geom_names: set[str] = set()
    for geom in root.findall(".//geom"):
        if geom.get("name"):
            used_geom_names.add(geom.get("name", ""))
    for geom in root.findall(".//geom[@type='mesh']"):
        if geom.get("name"):
            used_geom_names.discard(geom.get("name", ""))
        _set_visual_geom_defaults(geom, geom.get("mesh"), used_geom_names)

    _set_body_pos_z(root, "thigh_left", -0.162004)
    _set_body_pos_z(root, "thigh_right", -0.162006)
    _add_hoppertrex_collision_geoms(root)

    for geom in root.findall("./default//geom"):
        geom.attrib.pop("name", None)
        geom.attrib.pop("class", None)

    tree.write(mjcf_path, encoding="unicode")
    prettify_xml(mjcf_path)


def generate_mjcf(urdf_path: Path, mjcf_path: Path) -> None:
    import mujoco

    urdf_path = urdf_path.resolve()
    mjcf_path = mjcf_path.resolve()
    mesh_refs = find_urdf_mesh_references(urdf_path)
    import_urdf_path = write_mujoco_import_urdf(urdf_path)
    temporary_meshes: list[Path] = []
    cwd = os.getcwd()
    try:
        os.chdir(urdf_path.parent)
        for mesh_ref in mesh_refs:
            mesh_path = Path(mesh_ref)
            stripped_path = Path(mesh_path.name)
            source_path = Path(mesh_ref)
            if mesh_path == stripped_path or stripped_path.exists() or not source_path.exists():
                continue
            try:
                stripped_path.symlink_to(source_path)
            except OSError:
                shutil.copy2(source_path, stripped_path)
            temporary_meshes.append(stripped_path)

        model = mujoco.MjModel.from_xml_path(import_urdf_path.name)
        mujoco.mj_saveLastXML(mjcf_path.name, model)
    finally:
        for temporary_mesh in temporary_meshes:
            temporary_mesh.unlink(missing_ok=True)
        import_urdf_path.unlink(missing_ok=True)
        os.chdir(cwd)
    patch_mjcf(mjcf_path, urdf_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the configured Onshape assembly to URDF and MJCF."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "assets" / "hoppertrex",
        help="Directory where robot.urdf, robot.mjcf, and meshes/ are written.",
    )
    parser.add_argument(
        "--key-file",
        type=Path,
        default=Path("onshape.key"),
        help="Two-line Onshape API key file.",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("/tmp/hoppertrex_onshape_cache"),
        help="Exporter HTTP cache directory.",
    )
    parser.add_argument(
        "--onshape-url",
        default=DEFAULT_ONSHAPE_URL,
        help="Onshape workspace URL containing /documents/<id>/w/<id>/e/<id>.",
    )
    parser.add_argument("--robot-name", default="hoppertrex")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_onshape_env(args.key_file)
    patch_exporter_cache(args.cache)
    patch_exporter_revolute_dofs()

    target = parse_onshape_url(args.onshape_url)
    assembly_name = find_assembly_name(target)
    print(f"Using Onshape assembly {assembly_name!r}.")

    write_exporter_config(args.output, args.robot_name, assembly_name, target)
    run_urdf_exporter(args.output)

    urdf_path = organize_urdf_assets(args.output)
    simplify_large_stls(args.output / "meshes")
    apply_inertial_overrides(urdf_path)
    mjcf_path = args.output / "robot.xml"
    generate_mjcf(urdf_path, mjcf_path)

    print(f"Wrote {urdf_path}")
    print(f"Wrote {mjcf_path}")


if __name__ == "__main__":
    main()
