"""Thermodynamic, contact and mixed-boundary checks for the porous extension."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tfenn_invcalib.energetic_coupling import VOLUME, EnergyClosure
from tfenn_invcalib.poromechanical_coupling import (
    TIElasticity, PorousEnergyClosure, mandel_rotation,
)


def model(normal=(.5, 0, np.sqrt(.75)), isotropic=False, plane_reference_damage=None,
          preparation=None):
    elastic = (TIElasticity(9000., 9000., .2, .2, 3750.) if isotropic else
               TIElasticity(9000., 4500., .2, .25, 3000.))
    if preparation is not None:
        elastic = elastic.at_confinement(preparation, 13., .66625)
    return PorousEnergyClosure(
        elastic, normal, grain_bulk=50000., fluid_bulk=2200., porosity=.15,
        matrix_slope=2., plane_normal_factor=5., plane_shear_factor=3.,
        compaction_onset=12., compaction_hardening=1500.,
        plane_reference_damage=plane_reference_damage)


def test_reciprocal_engineering_compliance_and_rotation():
    elastic = TIElasticity(9000., 4500., .2, .25, 3000.)
    S = elastic.local_compliance()
    assert -S[2, 0]/S[0, 0] == pytest.approx(.25)
    assert -S[0, 2]/S[2, 2] == pytest.approx(.125)
    assert S[3, 3] == pytest.approx(1/6000.)
    angle = .62
    R = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                  [-np.sin(angle), 0, np.cos(angle)]])
    Q = mandel_rotation(R)
    np.testing.assert_allclose(Q.T @ Q, np.eye(6), atol=4e-16)
    np.testing.assert_allclose(elastic.compliance(R[:, 2]), Q @ S @ Q.T, atol=1e-19)
    with pytest.raises(ValueError, match="positive definite"):
        TIElasticity(9000., 4500., .2, 2., 3000.).local_compliance()


@pytest.mark.parametrize("plane_reference", [None, .1])
@pytest.mark.parametrize("preparation", [None, 5., 13.])
def test_all_forces_follow_one_saturated_energy(plane_reference, preparation):
    m = model(plane_reference_damage=plane_reference, preparation=preparation)
    e = np.array([.001, -.0002, .0003, .0004, -.0003, .0002])
    cm = np.array([.0001, -.00003, .00004, .00009, .00003, -.00002])
    z = np.array([-.00004, -.00008, .00003])
    kappa, d, D, zeta = .0002, .13, .21, -.0001
    args = [e, cm, z, kappa, d, D, zeta]
    f = m.response(*args)
    names = ["total_stress", "matrix_force", "plane_force", "compaction_force",
             "matrix_release", "plane_release", "pressure"]
    h = 1e-7
    for index, name in enumerate(names):
        derivatives = []
        for direction in np.eye(np.size(args[index])):
            plus, minus = list(args), list(args)
            direction = direction if index < 3 else direction.item()
            plus[index] = args[index]+h*direction
            minus[index] = args[index]-h*direction
            derivatives.append((m.response(*plus)["energy"]-m.response(*minus)["energy"])/(2*h))
        sign = 1 if index in (0, 6) else -1
        np.testing.assert_allclose(np.ravel(f[name]), sign*np.array(derivatives), rtol=2e-7, atol=1e-8)
    np.testing.assert_allclose(f["plastic_stress"], f["total_stress"]-f["pressure"]*VOLUME)
    if plane_reference is None:
        assert f["plane_to_matrix_release"] > 0
    else:
        assert f["plane_to_matrix_release"] == 0


def test_preparation_updates_every_storage_dependency_and_preserves_exact_limit():
    low, reference = model(preparation=5.), model(preparation=13.)
    scale = .59
    np.testing.assert_allclose(low.S0, reference.S0/scale, rtol=1e-14, atol=1e-18)
    np.testing.assert_allclose(low.Hc, reference.Hc*scale, rtol=1e-14, atol=1e-11)
    for d in (.1, 2.):
        for actual, original in zip(low.plane_storage(d), reference.plane_storage(d)):
            np.testing.assert_allclose(actual, original*scale, rtol=1e-14, atol=1e-11)
    np.testing.assert_allclose(low.biot, VOLUME-scale*(VOLUME-reference.biot), atol=1e-15)
    assert 1/low.M-1/reference.M == pytest.approx(
        (low.biot-reference.biot) @ VOLUME/(3*50000.))
    elastic = TIElasticity(9000., 4500., .2, .25, 3000.)
    assert elastic.at_confinement(5., 13., 0.) == elastic
    assert elastic.at_confinement(13., 13., .66625) == elastic
    assert np.linalg.eigvalsh(low.S0).min() > 0
    assert low.M > 0
    with pytest.raises(ValueError, match="Prepared stiffness"):
        elastic.at_confinement(0., 13., 1.)


def test_projected_host_recovers_the_isotropic_penny_crack_slice():
    E, nu, slope = 9000., .2, 2.
    m = PorousEnergyClosure(TIElasticity(E, E, nu, nu, E/(2*(1+nu))), (1, 2, 3),
        grain_bulk=50000., fluid_bulk=2200., porosity=.15, matrix_slope=slope,
        plane_normal_factor=16*(1-nu**2)/3,
        plane_shear_factor=16*(1-nu)/(3*(2-nu)),
        compaction_onset=12., compaction_hardening=1500.)
    original = EnergyClosure(E, nu, slope, slope)
    for d in (.1, .6):
        expected, prime = original.plane_compliance(d)
        H, Hp = m.plane_storage(d)
        np.testing.assert_allclose(np.linalg.inv(H), np.diag(expected), atol=1e-18)
        np.testing.assert_allclose(Hp, np.diag(-prime/expected**2), atol=1e-12)


def test_inelastic_volume_enters_fluid_content_with_unit_coefficient():
    m = model()
    kappa = .001
    e = kappa*VOLUME/3
    f = m.response(e, np.zeros(6), np.zeros(3), kappa, .1, .2, 0.)
    assert f["pressure"] == pytest.approx(m.M*kappa)
    assert f["elastic_volume"] == pytest.approx(0.)
    assert m.fluid_content(e, np.zeros(6), np.zeros(3), kappa, 0.) == pytest.approx(-kappa)


def test_undrained_elastic_limit_matches_isotropic_biot_solution():
    m = model(isotropic=True)
    e = .0003
    result = m.normal_compaction_step(e, np.zeros(5), matrix_strain=np.zeros(6),
        tangential_modes=np.zeros(2), d=.1, D=.2, fluid_content=0.)
    Kd, G = 5000., 3750.
    alpha = 1-Kd/50000.
    Ku = Kd+alpha**2*m.M
    Eu = 9*Ku*G/(3*Ku+G)
    nuu = (3*Ku-2*G)/(2*(3*Ku+G))
    f = result["response"]
    assert result["compaction"] == 0.
    assert result["plane_modes"][0] == 0.
    assert f["total_stress"][0] == pytest.approx(Eu*e)
    assert f["pressure"] == pytest.approx(alpha*m.M*(1-2*nuu)*e)
    assert f["total_volume"] > 0


def test_drained_cap_has_the_closed_form_isotropic_solution():
    m = model(isotropic=True)
    axial, confinement = .015, 5.
    result = m.normal_compaction_step(axial, [confinement, confinement, 0, 0, 0],
        matrix_strain=np.zeros(6), tangential_modes=np.zeros(2), d=.1, D=.2, pressure=0.)
    E, nu, H, pc = 9000., .2, m.compaction_hardening, m.compaction_onset
    expected = (axial-(1-2*nu)*confinement/E-3*(pc-confinement)/E)/(1/3+3*H/E)
    assert result["compaction"] == pytest.approx(expected)
    assert result["response"]["total_stress"][0]-confinement == pytest.approx(3*(pc-confinement+H*expected))
    assert result["fluid_content"] < 0


def test_sealed_reference_preserves_the_consolidated_stress():
    m = model()
    confinement = 5.
    e0 = m.S0 @ (confinement*VOLUME)
    zeta0 = m.fluid_content(e0, np.zeros(6), np.zeros(3), 0., 0.)
    result = m.normal_compaction_step(e0[0], [confinement, confinement, 0, 0, 0],
        matrix_strain=np.zeros(6), tangential_modes=np.zeros(2), d=.1, D=.2,
        fluid_content=zeta0)
    np.testing.assert_allclose(result["strain"], e0, atol=1e-18)
    np.testing.assert_allclose(result["response"]["total_stress"], confinement*VOLUME, atol=1e-13)
    assert result["response"]["pressure"] == pytest.approx(0., abs=1e-13)
    assert result["compaction"] == 0.


@pytest.mark.parametrize("drained", [False, True])
def test_contact_cap_and_fluid_balance_over_load_unload(drained):
    m = model()
    prior = 0.
    for axial in np.r_[np.linspace(0, .025, 60), np.linspace(.025, .005, 30)]:
        bc = {"pressure": 0.} if drained else {"fluid_content": 0.}
        result = m.normal_compaction_step(axial, [5, 5, 0, 0, 0],
            matrix_strain=np.zeros(6), tangential_modes=np.zeros(2), d=.1, D=.2,
            previous_compaction=prior, **bc)
        f, z0, kappa = result["response"], result["plane_modes"][0], result["compaction"]
        r, cap = f["plane_force"][0], f["compaction_force"]-m.compaction_onset
        np.testing.assert_allclose(f["total_stress"][1:], [5, 5, 0, 0, 0], atol=1e-10)
        assert z0 <= 0 and r >= -1e-10
        assert r*z0 == pytest.approx(0, abs=1e-12)
        assert kappa >= prior and cap <= 1e-10
        assert (kappa-prior)*cap == pytest.approx(0, abs=1e-12)
        zeta = m.fluid_content(result["strain"], np.zeros(6), result["plane_modes"], kappa, f["pressure"])
        assert zeta == pytest.approx(result["fluid_content"], abs=1e-15)
        # This follows from constituent mass balance, independently of the
        # strain partition and of the anisotropic Biot tensor.
        bulk_volume = (np.mean(f["total_stress"][:3])/50000.
                       + .15*(1/2200.-1/50000.)*f["pressure"]-zeta)
        assert f["total_volume"] == pytest.approx(bulk_volume, abs=1e-15)
        assert f["compaction_force"]*(kappa-prior) >= -1e-12
        prior = kappa
    assert prior > 0


def test_plane_opens_in_tension_and_recloses_in_compression():
    m = model(normal=(1, 0, 0))
    args = dict(matrix_strain=np.zeros(6), tangential_modes=np.zeros(2), d=.1, D=.2, pressure=0.)
    opened = m.normal_compaction_step(-.001, np.zeros(5), **args)
    f = opened["response"]
    assert opened["plane_modes"][0] < 0
    assert f["plane_force"][0] == pytest.approx(0, abs=1e-12)
    H, _ = m.plane_storage(.1)
    assert opened["plane_modes"][0] == pytest.approx(.2*f["plastic_stress"][0]/H[0, 0])
    closed = m.normal_compaction_step(.001, np.zeros(5), **args)
    assert closed["plane_modes"][0] == 0.
    assert closed["response"]["plane_force"][0] > 0


def test_plane_damage_reciprocity_in_ti_host():
    m = model()
    e, cm, z = np.zeros(6), np.zeros(6), np.array([-.0001, .0002, -.0003])
    d, D, h = .12, .25, 1e-7
    def f(damage, modes):
        return m.response(e, cm, modes, 0., damage, D, 0.)
    dx_dd = (f(d+h, z)["plane_force"]-f(d-h, z)["plane_force"])/(2*h)
    dy_dz = [(f(d, z+h*v)["matrix_release"]-f(d, z-h*v)["matrix_release"])/(2*h)
             for v in np.eye(3)]
    np.testing.assert_allclose(dx_dd, dy_dz, rtol=2e-8, atol=1e-8)
