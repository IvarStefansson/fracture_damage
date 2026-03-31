import numpy as np
import porepy as pp
from typing import cast


class IterationExportingPP:
    def initialize_data_saving(self):
        """Initialize iteration exporter."""
        super().initialize_data_saving()
        appendix = (
            "_uchar_dynamic"
            if isinstance(
                self, pp.constitutive_laws.CharacteristicDisplacementFromTraction
            )
            else ""
        )
        self.iteration_exporter = pp.Exporter(
            self.mdg,
            file_name=self.params["file_name"] + "_iterations",
            folder_name=self.params["folder_name"] + appendix,
        )

    def data_to_export(self):
        """Returns data for iteration exporting.

        Returns:
            Any type compatible with data argument of pp.Exporter().write_vtu().

        """
        # First primary variables.
        data = super().data_to_export()

        # Then derived variables on fractures.
        sds = self.mdg.subdomains(dim=self.nd - 1)
        scalar_offsets = np.cumsum([0] + [sd.num_cells for sd in sds])
        method_names = {
            "aperture": "m",
            "dilation_damage": "-",
            "friction_damage_coefficient": "-",
            "friction_damage": "-",
            "dilation_damage_coefficient": "-",
            "contact_mechanics_open_state_characteristic": "-",
            "friction_bound": "Pa",
            "damage_length": "m",
            "displacement_jump": "m",
            "_common_factor_damage_coefficients": "-",
        }
        values = {}
        for name, units in method_names.items():
            if not hasattr(self, name):
                continue
            if name in data:
                continue
            if name == "damage_length":
                op, _ = self.damage_length(sds, time_step_index=0)
                vals = op.value(self.equation_system)
            else:
                vals = cast(
                    np.ndarray,
                    getattr(self, name)(sds).value(self.equation_system),
                )
            values[name] = self.units.convert_units(vals, f"{units}", to_si=True)

            for id, sd in enumerate(sds):
                offset = (
                    scalar_offsets
                    if name not in ["displacement_jump"]
                    else scalar_offsets * self.nd
                )
                data.append(
                    (
                        sd,
                        name,
                        values[name][offset[id] : offset[id + 1]],
                    )
                )

        return data

    def save_data_iteration(self):
        """Export current solution to vtu files.

        This method is typically called by after_nonlinear_iteration.

        Having a separate exporter for iterations avoids distinguishing between iterations
        and time steps in the regular exporter's history (used for export_pvd).

        """
        # To make sure the nonlinear iteration index does not interfere with the
        # time part, we multiply the latter by the next power of ten above the
        # maximum number of nonlinear iterations. Default value set to 10 in
        # accordance with the default value used in NewtonSolver
        n = self.params.get("max_iterations", 10)
        p = round(np.log10(n))
        r = 10**p
        if r <= n:
            r = 10 ** (p + 1)
        self.iteration_exporter.write_vtu(
            self.data_to_export(),
            time_dependent=True,
            time_step=self.nonlinear_solver_statistics.num_iterations
            + r * self.time_manager.time_index,
        )

    def after_nonlinear_iteration(self, solution_vector: np.ndarray) -> None:
        """Integrate iteration export into simulation workflow.

        Order of operations is important, super call distributes the solution to
        iterate subdictionary.

        """
        super().after_nonlinear_iteration(solution_vector)
        # self.save_data_iteration()
        # self.iteration_exporter.write_pvd()


class IterationExporting:
    def data_to_export(self):
        """Define the data to export to vtu.

        Returns:
            list: List of tuples containing the subdomain, variable name,
            and values to export.

        """
        data = super().data_to_export()
        for sd in self.mdg.subdomains(dim=self.nd - 1):
            vals = self.evaluate_and_scale([sd], "displacement_jump", "m")
            data.append((sd, "displacement_jump", vals))
        return data

    def initialize_data_saving(self):
        """Initialize iteration exporter."""
        super().initialize_data_saving()
        # Setting export_constants_separately to False facilitates operations such as
        # filtering by dimension in ParaView and is done here for illustrative purposes.
        self.iteration_exporter = pp.Exporter(
            self.mdg,
            file_name=self.params["file_name"] + "_iterations",
            folder_name=self.params["folder_name"],
            export_constants_separately=False,
        )

    def data_to_export_iteration(self):
        """Returns data for iteration exporting.

        Returns:
            Any type compatible with data argument of pp.Exporter().write_vtu().

        """
        # The following is a slightly modified copy of the method
        # data_to_export() from DataSavingMixin.
        data = []
        variables = self.equation_system.variables
        for var in variables:
            # Note that we use iterate_index=0 to get the current solution, whereas
            # the regular exporter uses time_step_index=0.
            scaled_values = self.equation_system.get_variable_values(
                variables=[var], iterate_index=0
            )
            units = var.tags["si_units"]
            values = self.units.convert_units(scaled_values, units, to_si=True)
            data.append((var.domain, var.name, values))
        for sd in self.mdg.subdomains(dim=self.nd - 1):
            vals = self.evaluate_and_scale([sd], "displacement_jump", "m")
            data.append((sd, "displacement_jump", vals))
        return data

    def save_data_iteration(self):
        """Export current solution to vtu files.

        This method is typically called by after_nonlinear_iteration.

        Having a separate exporter for iterations avoids distinguishing between iterations
        and time steps in the regular exporter's history (used for export_pvd).

        """
        # To make sure the nonlinear iteration index does not interfere with the
        # time part, we multiply the latter by the next power of ten above the
        # maximum number of nonlinear iterations. Default value set to 10 in
        # accordance with the default value used in NewtonSolver
        n = self.params.get("max_iterations", 10)
        p = round(np.log10(n))
        r = 10**p
        if r <= n:
            r = 10 ** (p + 1)
        self.iteration_exporter.write_vtu(
            self.data_to_export_iteration(),
            time_dependent=True,
            time_step=self.nonlinear_solver_statistics.num_iterations
            + r * self.time_manager.time_index,
        )

    def after_nonlinear_iteration(self, solution_vector: np.ndarray) -> None:
        """Integrate iteration export into simulation workflow.

        Order of operations is important, super call distributes the solution to
        iterate subdictionary.

        """
        super().after_nonlinear_iteration(solution_vector)
        self.save_data_iteration()
        self.iteration_exporter.write_pvd()
