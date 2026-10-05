"""Mechanical checks of the combined porous friction/damage update."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tfenn_invcalib.energetic_coupling import VOLUME, KDEV
from tfenn_invcalib.poromechanical_coupling import TIElasticity, PorousEnergyClosure, mandel_rotation
from tfenn_invcalib.coupled_friction_damage import CoupledFrictionDamage, FrictionDamageLaw


def model(angle=30., **law_overrides):
    theta = np.deg2rad(angle)
    energy = PorousEnergyClosure(
        TIElasticity(9000., 4500., .2, .25, 3000.),
        [np.sin(theta), 0., np.cos(theta)],
        grain_bulk=50000., fluid_bulk=2200., porosity=.15,
        matrix_slope=2., plane_normal_factor=5., plane_shear_factor=3.,
        compaction_onset=16., compaction_hardening=1500.)
    law = dict(matrix_cohesion=5., matrix_friction=.6, matrix_dilatancy=.1,
               parallel_weight=.5, shear_weight=1., plane_friction=.25,
               damage_resistance=.015)
    law.update(law_overrides)
    return CoupledFrictionDamage(energy, FrictionDamageLaw(**law), .2)


def test_ti_metric_is_objective_and_recovers_von_mises():
    m = model(parallel_weight=1., shear_weight=1.)
    np.testing.assert_allclose(m.metric, 1.5*KDEV, atol=5e-16)
    m = model()
    force = np.array([18., 11., 6., 2., -1., 3.])
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    Q = mandel_rotation(rotation)
    normal = rotation @ np.array([.5, 0., np.sqrt(.75)])
    energy = PorousEnergyClosure(TIElasticity(9000., 4500., .2, .25, 3000.), normal,
        grain_bulk=50000., fluid_bulk=2200., porosity=.15,
        matrix_slope=2., plane_normal_factor=5., plane_shear_factor=3.,
        compaction_onset=16., compaction_hardening=1500.)
    rotated = CoupledFrictionDamage(energy, m.law, .2)
    before, after = m.matrix_surface(force), rotated.matrix_surface(Q @ force)
    assert before[0] == pytest.approx(after[0])
    np.testing.assert_allclose(Q @ before[1], after[1], atol=5e-16)
    assert m.matrix_surface(force+10*VOLUME)[2] == pytest.approx(before[2])


@pytest.mark.parametrize("angle, factor", [(0., np.sqrt(2.5)/2), (90., 1.)])
def test_intrinsic_matrix_strength_matches_analytical_axis_onset(angle, factor):
    m = model(angle, damage_resistance=1e4)
    confinement = 5.
    initial = m.consolidated_state(confinement, .1)
    qyield = (m.law.matrix_cohesion+m.law.matrix_friction*confinement)/(factor-m.law.matrix_friction/3)
    for multiplier in (.99, 1.01):
        trial_q = multiplier*qyield
        axial = initial["strain"][0]+m.energy.S0[0, 0]*trial_q
        result = m.step(axial, [confinement, confinement, 0., 0., 0.], initial, pressure=0.)
        assert np.linalg.norm(result["plane_modes"][1:]) < 1e-11
        if multiplier < 1:
            assert np.linalg.norm(result["matrix_strain"]) < 1e-10
        else:
            assert result["matrix_multiplier"] > 0
            assert result["matrix_yield"] == pytest.approx(0, abs=1e-6)


@pytest.mark.parametrize("drained", [False, True])
def test_locked_shear_recovers_the_exact_contact_compaction_subproblem(drained):
    m = model(90., matrix_cohesion=1e6, plane_friction=1e6, damage_resistance=1e6)
    initial = m.consolidated_state(13., .1)
    boundary = {"pressure": 0.} if drained else {"fluid_content": initial["fluid_content"]}
    axial = initial["strain"][0]+.012
    result = m.step(axial, [13., 13., 0., 0., 0.], initial, **boundary)
    exact = m.energy.normal_compaction_step(axial, [13., 13., 0., 0., 0.],
        matrix_strain=np.zeros(6), tangential_modes=np.zeros(2), d=.1, D=.2, **boundary)
    np.testing.assert_allclose(result["strain"], exact["strain"], atol=2e-11)
    assert result["compaction"] == pytest.approx(exact["compaction"], abs=2e-11)
    assert result["response"]["pressure"] == pytest.approx(exact["response"]["pressure"], abs=1e-7)


@pytest.mark.parametrize("drained", [False, True])
def test_combined_loading_satisfies_forces_mass_and_irreversibility(drained):
    m = model()
    initial = previous = m.consolidated_state(13., .1)
    boundary = {"pressure": 0.} if drained else {"fluid_content": initial["fluid_content"]}
    mechanisms = set()
    for axial in np.linspace(0, .015, 41):
        state = m.step(initial["strain"][0]+axial, [13., 13., 0., 0., 0.], previous, **boundary)
        f = state["response"]
        np.testing.assert_allclose(f["total_stress"][1:], [13., 13., 0., 0., 0.], atol=1e-9)
        assert state["d"] >= previous["d"]-1e-9
        assert state["compaction"] >= previous["compaction"]-1e-11
        assert state["plane_modes"][0] <= 1e-11
        assert f["plane_force"][0] >= -1e-6
        assert state["matrix_yield"] <= 1e-6 and state["plane_yield"] <= 1e-6
        assert f["matrix_release"] <= m.law.damage_resistance+1e-8
        assert f["compaction_force"] <= m.energy.compaction_onset+1e-6
        for name, value in state["dissipation_increments"].items():
            assert value >= -1e-8, (name, value)
            if value > 1e-8:
                mechanisms.add(name)
        q = f["total_stress"][0]-13.
        mass_volume = (q/(3*50000.)+.15*(1/2200.-1/50000.)*f["pressure"]
                       -(state["fluid_content"]-initial["fluid_content"]))
        assert VOLUME @ (state["strain"]-initial["strain"]) == pytest.approx(mass_volume, abs=1e-14)
        dc = state["matrix_strain"]-previous["matrix_strain"]
        if state["matrix_mean_force"] > 0:
            assert VOLUME @ dc == pytest.approx(-m.law.matrix_dilatancy*state["matrix_multiplier"], abs=1e-11)
        previous = state
    assert {"matrix_sliding", "plane_sliding", "damage"} <= mechanisms
    if drained:
        assert "compaction" in mechanisms


def test_reverse_slip_dissipates_while_matrix_damage_does_not_heal():
    m = model(matrix_cohesion=1e6)
    initial = previous = m.consolidated_state(13., .1)
    reversed_slip = False
    for axial in np.r_[np.linspace(0, .012, 21), np.linspace(.012, 0, 31)]:
        state = m.step(initial["strain"][0]+axial, [13., 13., 0., 0., 0.], previous, pressure=0.)
        if np.linalg.norm(state["plane_modes"][1:]) < np.linalg.norm(previous["plane_modes"][1:])-1e-8:
            reversed_slip = True
            assert state["dissipation_increments"]["plane_sliding"] > 0
        assert state["d"] >= previous["d"]-1e-8
        previous = state
    assert reversed_slip
    assert previous["d"] > initial["d"]
    np.testing.assert_allclose(previous["matrix_strain"], 0., atol=1e-10)


def test_open_plane_carries_no_friction_and_recloses():
    m = model(90., matrix_cohesion=1e6, damage_resistance=1e6)
    initial = m.consolidated_state(0., .1)
    opened = m.step(-.001, [0., 0., 0., 0., 1.], initial, pressure=0.)
    assert opened["plane_modes"][0] < 0
    np.testing.assert_allclose(opened["response"]["plane_force"], 0., atol=1e-6)
    closed = m.step(.001, np.zeros(5), opened, pressure=0.)
    assert closed["plane_modes"][0] == pytest.approx(0., abs=1e-11)
    assert closed["response"]["plane_force"][0] > 0


def test_constitutive_equations_decide_acceptance_independently_of_optimizer_flag(monkeypatch):
    import tfenn_invcalib.coupled_friction_damage as module
    original = module.least_squares
    m = model(90., matrix_cohesion=1e6, plane_friction=1e6, damage_resistance=1e6)
    initial = m.consolidated_state(13., .1)
    axial = initial["strain"][0]+.0001

    def stopped_at_root(*args, **kwargs):
        result = original(*args, **kwargs)
        result.success = False
        result.message = "The maximum number of function evaluations is exceeded."
        return result

    monkeypatch.setattr(module, "least_squares", stopped_at_root)
    state = m.step(axial, [13., 13., 0., 0., 0.], initial, pressure=0.)
    np.testing.assert_allclose(state["matrix_strain"], 0., atol=1e-11)
    expected_q = .0001/m.energy.S0[0, 0]
    assert state["response"]["total_stress"][0]-13. == pytest.approx(expected_q, abs=1e-7)

    def stopped_away_from_root(*args, **kwargs):
        result = original(*args, **kwargs)
        result.x[0] += 1.
        result.success = True
        return result

    monkeypatch.setattr(module, "least_squares", stopped_away_from_root)
    with pytest.raises(RuntimeError, match="scaled residual"):
        m.step(axial, [13., 13., 0., 0., 0.], initial, pressure=0.)


def test_pressure_reference_preserves_the_same_undrained_incremental_path():
    m = model()
    initial = [m.consolidated_state(13., .1, pressure=p) for p in (0., 5.)]
    previous = initial.copy()
    for axial in np.linspace(0., .012, 15):
        states = [m.step(s["strain"][0]+axial, [13.+p, 13.+p, 0., 0., 0.], prev,
                         fluid_content=s["fluid_content"])
                  for s, prev, p in zip(initial, previous, (0., 5.))]
        np.testing.assert_allclose(states[0]["response"]["plastic_stress"],
                                   states[1]["response"]["plastic_stress"], atol=2e-6)
        np.testing.assert_allclose(states[0]["strain"]-initial[0]["strain"],
                                   states[1]["strain"]-initial[1]["strain"], atol=2e-9)
        assert states[1]["response"]["pressure"]-states[0]["response"]["pressure"] == pytest.approx(5., abs=2e-6)
        previous = states


@pytest.mark.parametrize("drained", [False, True])
@pytest.mark.parametrize("fixed_storage", [False, True])
@pytest.mark.parametrize("evolving_metric", [False, True])
@pytest.mark.parametrize("mobilized_dilation", [False, True])
def test_coupled_jacobian_matches_force_perturbations(monkeypatch, drained, fixed_storage, evolving_metric, mobilized_dilation):
    import tfenn_invcalib.coupled_friction_damage as module
    original = module.least_squares
    m = model(damage_peak_density=.08 if evolving_metric else .8,
              parallel_residual_weight=1.2 if evolving_metric else None,
              damage_residual_fraction=.03, dilation_density_ratio=2.3,
              dilation_initial_density=.1 if mobilized_dilation else None,
              matrix_contraction=.8 if mobilized_dilation else 0.,
              friction_residual_fraction=.35 if evolving_metric else 1.,
              damage_shape=1.4, dilation_residual_fraction=.2)
    if fixed_storage:
        m.energy.plane_reference_damage = .1
    initial = m.consolidated_state(13., .1)

    def checked(fun, x0, *, jac, **kwargs):
        probe = x0+np.array([.5, -.1, -.2, .05, .03, .02, .2, -.1, -.02, .2, .3, .1])
        increments = 2e-6*np.eye(len(x0))
        numerical = np.column_stack([(fun(probe+h)-fun(probe-h))/(4e-6) for h in increments])
        np.testing.assert_allclose(jac(probe), numerical, rtol=3e-5, atol=2e-6)
        return original(fun, x0, jac=jac, **kwargs)

    monkeypatch.setattr(module, "least_squares", checked)
    boundary = {"pressure": 0.} if drained else {"fluid_content": initial["fluid_content"]}
    m.step(initial["strain"][0]+.003, [13., 13., 0., 0., 0.], initial, **boundary)


def test_positive_resistance_gives_softening_without_damage_healing():
    m = model(0., damage_resistance=.004, damage_peak_density=.5, damage_residual_fraction=.02,
              damage_shape=1.4, dilation_residual_fraction=.2)
    law = m.law
    assert law.resistance(.5)[0] == pytest.approx(.004)
    assert law.resistance(.5)[1] == pytest.approx(0.)
    assert law.resistance(.1)[1] > 0 and law.resistance(1.)[1] < 0
    initial = previous = m.consolidated_state(5., .1)
    loading = []
    for axial in np.r_[np.linspace(0., .015, 61), np.linspace(.015, .012, 10)]:
        state = m.step(initial["strain"][0]+axial, [5., 5., 0., 0., 0.], previous, pressure=0.)
        assert state["d"] >= previous["d"]-1e-8
        assert state["damage_resistance"] > 0
        assert state["response"]["matrix_release"] <= state["damage_resistance"]+1e-8
        assert state["dissipation_increments"]["damage"] >= -1e-9
        assert min(state["dissipation_increments"].values()) >= -1e-8
        if state["matrix_mean_force"] > 0:
            volume = VOLUME @ (state["matrix_strain"]-previous["matrix_strain"])
            assert volume == pytest.approx(-law.dilatancy(state["d"])[0]*state["matrix_multiplier"], abs=1e-10)
        loading.append(state["response"]["total_stress"][0]-5.)
        previous = state
    assert previous["d"] > law.damage_peak_density
    assert max(loading[:61]) > loading[60]+1.


def test_softening_fold_reaches_a_consistent_later_branch():
    energy = PorousEnergyClosure(TIElasticity(15000., 4500., .25, .38, 3500.), [1., 0., 0.],
        grain_bulk=50000., fluid_bulk=2200., porosity=.15, matrix_slope=1.,
        plane_normal_factor=1., plane_shear_factor=1., compaction_onset=30., compaction_hardening=1500.)
    law = FrictionDamageLaw(4., .8, .12, .75, 1., .25, .012, damage_peak_density=.6,
                            damage_residual_fraction=.03)
    m = CoupledFrictionDamage(energy, law, .6)
    initial = previous = m.consolidated_state(5., .05)
    jumped = False
    for axial in np.linspace(0., .012, 81):
        state = m.step(initial["strain"][0]+axial, [5., 5., 0., 0., 0.], previous, pressure=0.)
        if state["d"] > 2*previous["d"]:
            jumped = True
            assert state["response"]["total_stress"][0] < previous["response"]["total_stress"][0]
        assert state["matrix_yield"] <= 1e-6
        assert state["plane_yield"] <= 1e-6
        assert state["response"]["matrix_release"] <= state["damage_resistance"]+1e-8
        assert state["d"] >= previous["d"]-1e-8
        assert min(state["dissipation_increments"].values()) >= -1e-8
        previous = state
    assert jumped


@pytest.mark.parametrize("mobilized_dilation", [False, True])
def test_density_rescaling_preserves_evolving_material_response(mobilized_dilation):
    from dataclasses import replace
    m = model(damage_peak_density=.8, damage_residual_fraction=.03,
              damage_shape=1.4, dilation_residual_fraction=.2,
              parallel_residual_weight=1.2, dilation_density_ratio=2.3,
              friction_residual_fraction=.4)
    if mobilized_dilation:
        m.law = replace(m.law, dilation_initial_density=.1, matrix_contraction=.8)
    energy = PorousEnergyClosure(TIElasticity(9000., 4500., .2, .25, 3000.), [.5, 0., np.sqrt(.75)],
        grain_bulk=50000., fluid_bulk=2200., porosity=.15, matrix_slope=1.,
        plane_normal_factor=5., plane_shear_factor=3., compaction_onset=16., compaction_hardening=1500.)
    rescaled = CoupledFrictionDamage(energy, replace(m.law, damage_resistance=.0075,
        damage_peak_density=1.6, dilation_initial_density=.2 if mobilized_dilation else None), .2)
    initial = [m.consolidated_state(13., .1), rescaled.consolidated_state(13., .2)]
    previous = initial.copy()
    for axial in np.linspace(0., .015, 31):
        states = [current.step(start["strain"][0]+axial, [13., 13., 0., 0., 0.], old,
                               fluid_content=start["fluid_content"])
                  for current, start, old in zip((m, rescaled), initial, previous)]
        np.testing.assert_allclose(states[0]["response"]["total_stress"], states[1]["response"]["total_stress"], atol=2e-6)
        np.testing.assert_allclose(states[0]["strain"], states[1]["strain"], atol=2e-9)
        assert states[1]["d"] == pytest.approx(2*states[0]["d"], rel=1e-6)
        previous = states


@pytest.mark.parametrize("mobilized_dilation", [False, True])
def test_damage_evolves_strength_direction_and_flow_without_negative_dissipation(mobilized_dilation):
    m = model(0., damage_peak_density=.5, damage_resistance=.004,
              damage_residual_fraction=.02, parallel_residual_weight=1.2,
              dilation_density_ratio=2.3, dilation_residual_fraction=.2,
              friction_residual_fraction=.25)
    if mobilized_dilation:
        from dataclasses import replace
        m.law = replace(m.law, dilation_initial_density=.1)
    for density in (.1, .5, 2., 20.):
        metric, derivative = m.matrix_metric(density)
        weight = m.law.parallel_metric_weight(density)[0]
        assert .5 <= weight <= 1.2
        # Pure parallel triaxial force gives the exact instantaneous axis factor.
        force = np.array([1., 0., 0., 0., 0., 0.])
        assert m.matrix_surface(force, density)[2] == pytest.approx(np.sqrt(1+3*weight)/2)
        np.testing.assert_allclose(metric @ VOLUME, 0., atol=5e-16)
        assert np.linalg.eigvalsh(metric)[1] > 0
        if density != .5:
            h = density*1e-5
            numerical = (m.matrix_metric(density+h)[0]-m.matrix_metric(density-h)[0])/(2*h)
            np.testing.assert_allclose(derivative, numerical, atol=2e-10)
    initial = previous = m.consolidated_state(5., .1)
    for axial in np.linspace(0., .015, 61):
        state = m.step(initial["strain"][0]+axial, [5., 5., 0., 0., 0.], previous, pressure=0.)
        assert state["d"] >= previous["d"]-1e-8
        assert min(state["dissipation_increments"].values()) >= -1e-8
        beta = m.law.dilatancy(state["d"])[0]
        mean = max(0., state["matrix_mean_force"])
        mu = m.law.friction(state["d"])[0]
        assert beta <= mu
        expected = state["matrix_multiplier"]*(m.law.matrix_cohesion+(mu-beta)*mean)
        assert state["dissipation_increments"]["matrix_sliding"] == pytest.approx(expected, abs=1e-8)
        previous = state
    assert previous["d"] > m.law.damage_peak_density


@pytest.mark.parametrize("mobilized", [False, True])
def test_friction_and_flow_tangents_preserve_the_instantaneous_dissipation_bound(mobilized):
    law = model(matrix_friction=1.5, matrix_dilatancy=1.4, damage_peak_density=.5,
                friction_residual_fraction=.4, dilation_residual_fraction=.7,
                dilation_initial_density=.1 if mobilized else None,
                matrix_contraction=.8 if mobilized else 0.).law
    assert law.friction(.1) == (1.5, 0.)
    assert law.friction(.5) == (1.5, 0.)
    for density in (.2, .49, .51, 1., 5., 100.):
        mu, mu_prime = law.friction(density)
        beta, beta_prime = law.dilatancy(density)
        assert .6 <= mu <= 1.5
        assert beta <= mu
        h = density*1e-5
        assert mu_prime == pytest.approx((law.friction(density+h)[0]
                 -law.friction(density-h)[0])/(2*h), rel=1e-7, abs=1e-10)
        assert beta_prime == pytest.approx((law.dilatancy(density+h)[0]
                 -law.dilatancy(density-h)[0])/(2*h), rel=1e-7, abs=1e-10)


@pytest.mark.parametrize("fraction, peak", [(-.1, .5), (1.1, .5), (.5, None)])
def test_friction_evolution_requires_its_admissible_fraction_and_density(fraction, peak):
    with pytest.raises(ValueError, match="friction evolution"):
        model(friction_residual_fraction=fraction, damage_peak_density=peak)


def test_dilation_mobilizes_before_decaying_with_its_analytic_tangent():
    law = model(damage_peak_density=.5, dilation_density_ratio=2.,
                dilation_initial_density=.1, dilation_residual_fraction=.2).law
    assert law.dilatancy(.1)[0] == pytest.approx(.2*law.matrix_dilatancy)
    assert law.dilatancy(1.1)[0] == pytest.approx(law.matrix_dilatancy)
    assert law.dilatancy(1.1)[1] == pytest.approx(0.)
    assert law.dilatancy(.5)[1] > 0 and law.dilatancy(3.)[1] < 0
    for density in (.2, .5, 1.1, 3., 50.):
        value, derivative = law.dilatancy(density)
        assert 0 <= value <= law.matrix_friction
        h = density*1e-5
        numerical = (law.dilatancy(density+h)[0]-law.dilatancy(density-h)[0])/(2*h)
        assert derivative == pytest.approx(numerical, rel=1e-7, abs=1e-11)


@pytest.mark.parametrize("drained", [False, True])
def test_shear_contraction_transitions_to_dilation_with_positive_dissipation(drained):
    m = model(90., matrix_dilatancy=.5, matrix_contraction=.5,
              damage_peak_density=.5, damage_resistance=.004,
              dilation_density_ratio=.6, dilation_initial_density=.1,
              dilation_residual_fraction=0., damage_residual_fraction=.02)
    m.energy.compaction_onset = 100.
    law = m.law
    for density in (.11, .2, .5, 2.):
        value, derivative = law.dilatancy(density)
        h = density*1e-5
        numerical = (law.dilatancy(density+h)[0]-law.dilatancy(density-h)[0])/(2*h)
        assert derivative == pytest.approx(numerical, rel=1e-7)
        assert value <= law.matrix_friction
    initial = previous = m.consolidated_state(5., .1)
    boundary = {"pressure": 0.} if drained else {"fluid_content": initial["fluid_content"]}
    volumes = []
    for axial in np.linspace(0., .015, 81):
        state = m.step(initial["strain"][0]+axial, [5., 5., 0., 0., 0.], previous, **boundary)
        volume = VOLUME @ (state["matrix_strain"]-previous["matrix_strain"])
        beta = law.dilatancy(state["d"])[0]
        assert volume == pytest.approx(-beta*state["matrix_multiplier"], abs=1e-10)
        expected = state["matrix_multiplier"]*(law.matrix_cohesion
                   +(law.matrix_friction-beta)*state["matrix_mean_force"])
        assert state["dissipation_increments"]["matrix_sliding"] == pytest.approx(expected, abs=1e-8)
        assert min(state["dissipation_increments"].values()) >= -1e-8
        assert state["compaction"] == pytest.approx(0., abs=1e-10)
        volumes.append(volume)
        previous = state
    assert max(volumes) > 1e-6 and min(volumes) < -1e-6
