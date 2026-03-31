import copy
import logging
from typing import Literal, Sequence, cast

import numpy as np

import porepy as pp
from porepy.applications.test_utils.models import add_mixin
from porepy.compositional.materials import FractureDamageSolidConstants
from porepy.examples import fracture_damage as damage_examples
from porepy.models import fracture_damage as damage_models
from porepy.numerics.nonlinear.line_search import ConstraintLineSearchNonlinearSolver
from boundary_conditions import (
    BoundaryConditionsMassDirSomeSides,
    BoundaryConditionsEnergyDirSomeSides,
    FlowDirPartOfEastWestOnFractures,
    FlowDirSidesWestEast,
    SubsurfaceBoundaryConditions,
    Mechanical,
)
from parameter_study_data_saving import ParameterStudyDataSaving
from porepy.applications.md_grids.model_geometries import (
    CubeDomainOrthogonalFractures,
    SquareDomainOrthogonalFractures,
)
from initial_condition import InitialCondition, InitialConditionFromReference
from model import (
    FractureDamageMomentumBalance,
    override_methods,
    DarcyFluxDiscretizationMixin,
)

damage_types = {
    "dilation": damage_examples.DilationDamageMomentumBalance,
    "friction": damage_examples.FrictionDamageMomentumBalance,
}

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

ND = 3
INITIALIZATION_TIME = 100 * pp.YEAR


class Model(
    # pp.momentum_balance.TpsaMomentumBalanceMixin,
    Mechanical,
    ParameterStudyDataSaving,
    # SubsurfaceBoundaryConditions,
    FlowDirPartOfEastWestOnFractures,
    BoundaryConditionsMassDirSomeSides,
    BoundaryConditionsEnergyDirSomeSides,
    FlowDirSidesWestEast,  # Defers to OnFractures
    InitialConditionFromReference,
    (CubeDomainOrthogonalFractures if ND == 3 else SquareDomainOrthogonalFractures),
    # pp.applications.md_grids.model_geometries.SquareDomainOrthogonalFractures,
    # pp.constitutive_laws.GravityForce,
    DarcyFluxDiscretizationMixin,
    FractureDamageMomentumBalance,
    # pp.SinglePhaseFlow,
    # pp.MomentumBalance,
):
    def overburden_stress(self, coords: np.ndarray) -> np.ndarray:
        val = self.units.convert_units(1e7, "Pa")
        return (
            np.full(coords.shape, val) * self.background_stress_weights()[:, np.newaxis]
        )

    def intersection_permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Permeability of the intersections.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Cell-wise permeability operator.

        """
        size = sum(g.num_cells for g in subdomains)
        k = pp.ad.DenseArray(np.full(size, self.units.convert_units(1e-6, "m^2")))
        return self.isotropic_second_order_tensor(subdomains, k)

    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        """Boundary condition type for mechanics.

        Dirichlet boundary conditions are defined on the north and south boundaries.

        Parameters:
            sd: Subdomain for which to define boundary conditions.

        Returns:
            bc: Boundary condition object.

        """
        domain_sides = self.domain_boundary_sides(sd)
        bc = pp.BoundaryConditionVectorial(
            sd, domain_sides.north + domain_sides.south, "dir"
        )
        bc.internal_to_dirichlet(sd)
        return bc


def folder_name(
    isotropic: bool, damages: Sequence[str], num_cells: int, model_name: str
) -> str:
    """Return the output folder name for a given parameter combination.

    Parameters:
        isotropic: Whether isotropic damage length is used.
        damages: Damage types active in this run (e.g. ``["dilation"]``).
        num_cells: Number of cells in the discretization.
        model_name: Name of the model class.

    Returns:
        Relative folder path, e.g. ``"case3/isotropic_True_damages_dilation"``.
    """
    if len(damages) == 0:
        damage_str = "none"
    else:
        damage_str = "_".join(damages)
    return f"case3/isotropic_{isotropic}/model_{model_name}/damages_{damage_str}_num_cells_{num_cells}"


damage_combinations = [
    (True, []),
    (True, ["dilation"]),
    (True, ["friction"]),
    (True, ["dilation", "friction"]),
    # (False, ["dilation"]),
    # (False, ["friction"]),
    # (False, ["dilation", "friction"]),
]
cell_sizes = [
    # 8,
    # 16,
    32,
]


class PoroModel(Model, pp.Poromechanics): ...


class ThermoModel(Model, pp.Thermoporomechanics): ...


models = {
    # "poro": PoroModel,
    "thermo": ThermoModel,
}


def setup(
    isotropic: bool,
    damages: Sequence[str],
    num_time_steps: int,
    model: tuple[str, type],
    num_cells=8,
):
    """Construct a model for a time-dependent run using shared setup logic.

    Parameters:
        isotropic: If True, use isotropic damage length; otherwise anisotropic.
    dim: Spatial dimension of the bulk domain (2 or 3).
    damages: Iterable of damage types to include (subset of {"dilation", "friction"}).
    num_time_steps: Number of time steps to simulate.
        Number of time instants passed to the TimeManager (N); this yields N-1
        forward steps. For a single forward step, pass 2.

    Returns:
        A configured model instance ready to be run with pp.run_time_dependent_model.
    """
    params_local = copy.deepcopy(damage_examples.model_params)

    # Choose damage length (isotropic vs anisotropic) and exact solution
    if isotropic:
        model_class = add_mixin(damage_models.IsotropicFractureDamageLength, model[1])
    else:
        model_class = add_mixin(damage_models.AnisotropicFractureDamageLength, model[1])

    # Add requested damage equations and variables.
    for name in damages:
        model_class = add_mixin(damage_types[name], model_class)

    def sort_criterion(cls, domain):
        return domain.dim == cls.nd

    methods = ["darcy_flux"]
    dofs = [("faces", 1)]

    methods.append("fourier_flux")
    dofs.append(("faces", 1))
    if isinstance(model_class, DarcyFluxDiscretizationMixin):
        for method, dof in zip(methods, dofs):
            override_methods(model_class, method, dof, sort_criterion)
    solid_params = pp.solid_values.granite.copy()
    solid_params.update(
        {
            "dilation_angle": 0.1,  # [rad]
            "fracture_normal_stiffness": 1.1e8,  # [Pa m^-1]
            "maximum_elastic_fracture_opening": 0e-6,  # [m]
            "normal_permeability": 5.0e-8,  # [m^2]
            "permeability": 1e-16,  # [m^2]
            "fracture_gap": 0,  # [m]
            "residual_aperture": 3e-3,  # [m]
            "friction_coefficient": 0.5,
            "uniaxial_compressive_strength": 2e8,
            "characteristic_fracture_roughness": 1e-2,
            "initial_friction_damage": 0.50,
            "initial_dilation_damage": 0.50,
        }
    )
    period = 1 * pp.DAY
    dt_init = INITIALIZATION_TIME / 2 - 1000
    schedule = np.arange(0, num_time_steps - 1) * period + INITIALIZATION_TIME
    # Insert a short initial time step to capture the initial response to loading and trigger
    schedule = np.insert(schedule, 0, 0)
    inlet_pressure = np.full(num_time_steps + 1, 0e5)  # [Pa]
    inlet_temperature = np.full(num_time_steps + 1, 0.0)  # [K]
    # inlet_pressure[3::2] = 1e6  # Oscillating pressure to trigger damage evolution.

    # for i in range(3, (num_time_steps + 1), 2):
    #     inlet_pressure[i] = 10 + 2.5 * (i - 1)
    start = 2
    inlet_pressure[start::] = 4e6 + 5e5 * (np.arange(start, num_time_steps + 1) - start)
    if isinstance(model_class, pp.Thermoporomechanics):
        inlet_temperature[start:] -= 10.0
    north_displacement = np.zeros((ND, num_time_steps + 1))
    north_displacement[2, :] = 2e-1
    # north_displacement[2, :] = 1e-1
    north_displacement[1, :] = -2e-1
    size = 1e3
    background_stress = np.ones(ND)
    # background_stress[0] = 0.6
    # background_stress[2] = 1.35
    params_local.update(
        {
            "domain_size": size,
            "meshing_arguments": {
                "cell_size": 1 / num_cells * size,
                "cell_size_fracture": 1 / 12 * size,
            },
            # First fracture vertical, second horizontal. Index 1 picks the second.
            "num_fracture_points": 8,
            "background_stress_weights": background_stress,
            # First endpoints discarded for indices = [1], second endpoints used as x
            # coordinates for the horizontal fracture.
            # "fracture_endpoints": [
            #     np.array([0.2 * size, 0.8 * size]),
            #     np.array([0.2 * size, 0.8 * size]),
            # ],
            "time_manager": pp.TimeManager(
                schedule,
                dt_init,
                False,
                dt_min_max=(0.1, 50 * pp.YEAR),
                iter_relax_factors=(0.6, 2.0),
                iter_optimal_range=(5, 15),
                recomp_factor=0.1,
            ),
            # "time_manager": pp.TimeManager(schedule, dt, True),
            # Trim displacement BCs to requested dimension
            "adaptive_indicator_scaling": True,
            "constraint_violation_tolerance": 1e-2,
            # "grid_type": "simplex",
            "grid_type": "cartesian",
            # "units": pp.Units(kg=1e3),
            "north_displacements": north_displacement,
            "inlet_pressure": inlet_pressure,
            "inlet_temperature": inlet_temperature,
            "fracture_indices": [1],  # Use both fractures.
            "reference_variable_values": pp.ReferenceVariableValues(
                pressure=0, temperature=0.0
            ),
        }
    )
    params_local.pop("times_to_export", None)  # Use default export times.
    params_local.pop("north_stress", None)  # Disable stress BC for this test.
    params_local["folder_name"] = folder_name(
        isotropic,
        damages,
        num_cells,
        model[0],
    )

    # Reverse the increase in shear modulus used in the test cases.
    params_local["material_constants"] = {
        "solid": FractureDamageSolidConstants(**solid_params),  # type: ignore[arg-type]
        "fluid": pp.FluidComponent(**pp.fluid_values.water),  # type: ignore[arg-type]
        "numerical": pp.NumericalConstants(characteristic_displacement=1e-3),
    }
    solver_params = {
        "nl_max_iterations": 20,  # Hard nonlinear problems - expect slow convergence
        # "nl_convergence_tol_res": 1e-1,  # Bad scaling - expect slow convergence
        # "nl_convergence_tol": 1e-2,
        # "nl_divergence_tol": 1e12,
        "nl_convergence_res_atol": 1e-1,  # Bad scaling - expect slow convergence
        "nl_convergence_inc_atol": 1e-2,
        "nl_divergence_res_atol": 1e12,
        "nonlinear_solver": ConstraintLineSearchNonlinearSolver,
        "local_line_search": True,
        # "global_line_search": True,
    }

    return model_class, params_local, solver_params


if __name__ == "__main__":
    # dim x number of time steps displacement values on the north boundary
    for model_name, model_class in models.items():
        for num_cells in cell_sizes:
            for isotropic, damages in damage_combinations:
                cls, model_params, solver_params = setup(
                    isotropic=isotropic,
                    damages=damages,
                    num_time_steps=8,
                    num_cells=num_cells,
                    model=(model_name, model_class),
                )
                m = cls(model_params)
                pp.run_time_dependent_model(m, solver_params)
                m.save_results_to_file(
                    folder_name(isotropic, damages, num_cells, model_name) + "/results",
                    fmt="csv",
                )
