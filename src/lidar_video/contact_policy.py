"""Explicit, opt-in contact configuration for rigid compound-mesh scenes.

This is a versioned simulation policy, not estimated material physics. It changes
contact generation and activates existing finite torsional/rolling friction;
geometry, mass, friction coefficients and free joints are never modified here.
"""

import copy
import xml.etree.ElementTree as ET

POLICY_NAME = "legacy_mesh_support_v1"


def configure_mesh_contacts(root, policy=POLICY_NAME):
    """Configure explicit MJCF exports; reject unresolved defaults and overrides."""
    if policy != POLICY_NAME or root.tag != "mujoco":
        raise ValueError("Unknown rigid-mesh contact policy")
    if any(root.find(tag) is not None for tag in ["include", "default", "deformable", "contact/pair"]):
        raise ValueError("Contact policy requires explicit rigid geoms without defaults or pair overrides")
    if len(root.findall("option")) > 1:
        raise ValueError("Duplicate option elements")
    option = root.find("option")
    if option is not None and len(option.findall("flag")) > 1:
        raise ValueError("Duplicate contact flag elements")
    active = []
    for geom in root.findall(".//geom"):
        if int(geom.get("contype", "1")) or int(geom.get("conaffinity", "1")):
            if geom.get("type", "sphere") not in {
                "mesh",
                "box",
                "plane",
                "sphere",
                "capsule",
                "cylinder",
                "ellipsoid",
            }:
                raise ValueError("Only rigid collision geoms are supported")
            active.append(geom)
    if not active:
        raise ValueError("At least one active collision geom required")
    before = dict(
        option=copy.deepcopy(option.attrib) if option is not None else {},
        flags=copy.deepcopy(option.find("flag").attrib)
        if option is not None and option.find("flag") is not None
        else {},
        condim=[g.get("condim", "3") for g in active],
    )
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("cone", "pyramidal")
    option.set("impratio", "1")
    option.set("noslip_iterations", "3")
    option.set("noslip_tolerance", "1e-6")
    flags = option.find("flag")
    if flags is None:
        flags = ET.SubElement(option, "flag")
    flags.set("nativeccd", "disable")
    flags.set("multiccd", "enable")
    for geom in active:
        geom.set("condim", "6")
    return dict(
        policy=policy,
        before=before,
        option=copy.deepcopy(option.attrib),
        flags=copy.deepcopy(flags.attrib),
        active_collision_geoms=len(active),
        contact_dimension=6,
        geometry_mass_friction_coefficients_unchanged=True,
        measured_material_properties=False,
        scope="Explicit CPU contact policy; validate dynamics and force response separately",
    )
