import numpy as np
import pytest
import trimesh

from lidar_video.collision_mass import allocate_collision_mass


def test_density_allocation_matches_unequal_parts_and_preserves_total(tmp_path):
    paths = []
    for i, width in enumerate([0.01, 0.03]):
        path = tmp_path / f"{i}.obj"
        trimesh.creation.box([width, 0.01, 0.01]).export(path)
        paths.append(path)
    masses, report = allocate_collision_mass(paths, 0.08)
    np.testing.assert_allclose(masses, [0.02, 0.06])
    assert sum(masses) == pytest.approx(0.08)
    assert not report["density_measured"]


def test_open_mesh_uses_collision_hull_not_signed_triangle_volume(tmp_path):
    box = trimesh.creation.box([0.01, 0.01, 0.01])
    box.update_faces(np.arange(len(box.faces) - 2))
    path = tmp_path / "open.obj"
    box.export(path)
    masses, report = allocate_collision_mass([path], 0.08)
    np.testing.assert_allclose(masses, [0.08])
    assert report["convex_volumes_m3"][0] == pytest.approx(1e-6)


def test_bad_mass_empty_components_and_flat_mesh_reject(tmp_path):
    for value in [0, -1, float("nan"), True]:
        with pytest.raises(ValueError):
            allocate_collision_mass([], value)
    with pytest.raises(ValueError):
        allocate_collision_mass([], 0.08)
    flat = trimesh.Trimesh([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]], process=False)
    path = tmp_path / "flat.obj"
    flat.export(path)
    with pytest.raises(ValueError):
        allocate_collision_mass([path], 0.08)


def test_native_center_of_mass_matches_uniform_density(tmp_path):
    mujoco = pytest.importorskip("mujoco")
    paths = []
    for i, width in enumerate([0.01, 0.03]):
        mesh = trimesh.creation.box([width, 0.01, 0.01])
        mesh.apply_translation([i * 0.03, 0, 0])
        path = tmp_path / f"{i}.obj"
        mesh.export(path)
        paths.append(path)
    masses, _ = allocate_collision_mass(paths, 0.08)
    xml = (
        "<mujoco><asset>"
        + "".join(f'<mesh name="p{i}" file="{p}"/>' for i, p in enumerate(paths))
        + '</asset><worldbody><body name="object"><freejoint/>'
        + "".join(f'<geom type="mesh" mesh="p{i}" mass="{mass}"/>' for i, mass in enumerate(masses))
        + "</body></worldbody></mujoco>"
    )
    model = mujoco.MjModel.from_xml_string(xml)
    bid = model.body("object").id
    assert model.body_mass[bid] == pytest.approx(0.08)
    np.testing.assert_allclose(model.body_ipos[bid], [0.0225, 0, 0], atol=1e-8)
