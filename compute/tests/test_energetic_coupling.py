"""Mechanical identities and evolution of the energetic research closure."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tfenn_invcalib.energetic_coupling import EnergyClosure, MonotonicShear, plane_basis


def shear_model(evolving=True):
    return MonotonicShear(EnergyClosure(25000., .25, 2., 2.),
                          d0=.1, D=.2, overlap=.8, matrix_friction=60.,
                          plane_friction=2., resistance=.01,
                          evolving_localization=evolving)


def test_forces_are_derivatives_of_the_same_energy():
    model = EnergyClosure(24000., .23, 3.1, 1.7)
    strain = np.array([.001, -.0002, .0003, .0004, -.0003, .0002])
    cm = np.array([.0001, -.00003, .00004, .00009, .00003, -.00002])
    z = np.array([.00004, -.00008, .00003])
    normal = np.array([1., 2., 3.])
    d, D = .13, .21
    f = model.forces(strain, cm, z, d, D, normal)
    h = 1e-7
    for key, variable in [("stress", strain), ("matrix_force", cm), ("plane_force", z)]:
        difference = []
        for direction in np.eye(len(variable)):
            inputs_p = [strain.copy(), cm.copy(), z.copy()]
            inputs_m = [strain.copy(), cm.copy(), z.copy()]
            index = {"stress": 0, "matrix_force": 1, "plane_force": 2}[key]
            inputs_p[index] += h * direction
            inputs_m[index] -= h * direction
            difference.append((model.energy(*inputs_p, d, D, normal)
                               - model.energy(*inputs_m, d, D, normal)) / (2*h))
        np.testing.assert_allclose(f[key], np.array(difference) * (1 if key == "stress" else -1), rtol=2e-8, atol=1e-8)
    for key, index in [("matrix_release", 0), ("plane_release", 1)]:
        args_p, args_m = np.array([d, D]), np.array([d, D])
        args_p[index] += h
        args_m[index] -= h
        difference = -(model.energy(strain, cm, z, *args_p, normal)
                       - model.energy(strain, cm, z, *args_m, normal)) / (2*h)
        assert f[key] == pytest.approx(difference, rel=1e-7)
    assert f["plane_to_matrix_release"] > 0


@pytest.mark.parametrize("normal", [[0, 0, 1], [1, 2, 3]])
def test_condensation_recovers_sequential_compliance_without_double_counting(normal):
    model = EnergyClosure(30000., .28, 2.3, 1.4)
    d, D = .17, .26
    B = plane_basis(normal)
    s, _ = model.plane_compliance(d)
    A = np.column_stack([np.eye(6), B])
    storage = np.zeros((9, 9))
    storage[:6, :6] = model.Hc / d
    storage[6:, 6:] = np.diag(1/s) / D
    C = model.C0 - model.C0 @ A @ np.linalg.solve(A.T @ model.C0 @ A + storage, A.T @ model.C0)
    np.testing.assert_allclose(np.linalg.inv(C), model.condensed_compliance(d, D, normal), rtol=2e-13, atol=1e-16)
    assert np.linalg.eigvalsh(C)[0] > 0


def test_plane_damage_onset_can_precede_matrix_sliding():
    model = shear_model()
    onset = model.plane_driven_onset()
    previous = np.array([0., 0., model.d0])
    before = model.step(.99 * onset["strain"], previous)
    after = model.step(1.01 * onset["strain"], before)
    assert before[0] == 0 and after[0] == 0
    assert before[2] == model.d0
    assert after[2] > model.d0
    f = model.response(1.01 * onset["strain"], after)
    assert f["matrix_self_release"] == 0
    assert f["plane_to_matrix_release"] == pytest.approx(model.resistance, rel=1e-9)
    frozen = shear_model(evolving=False)
    state = frozen.step(1.01 * onset["strain"], previous)
    assert state[0] == 0 and state[2] == frozen.d0


@pytest.mark.parametrize("evolving", [False, True])
def test_incremental_solution_obeys_all_three_mechanism_conditions(evolving):
    model = shear_model(evolving)
    previous = np.array([0., 0., model.d0])
    for strain in np.linspace(0, .012, 81):
        state = model.step(strain, previous)
        f = model.response(strain, state)
        increments = state - previous
        assert np.all(increments >= -1e-14)
        excess = np.array([f["matrix_force"] - model.matrix_friction,
                           f["plane_force"] - model.plane_friction,
                           f["matrix_release"] - model.resistance])
        assert np.max(excess) < 1e-7
        np.testing.assert_allclose(excess * increments, 0, atol=1e-9)
        dissipation = (model.matrix_friction*increments[0]
                       + model.plane_friction*increments[1]
                       + model.resistance*increments[2])
        assert dissipation >= 0
        incremental_energy = f["energy"] + dissipation
        assert incremental_energy <= model.energy(strain, previous) + 1e-12
        previous = state
