import numpy as np
import porepy as pp


class InitialConditionFromReference(pp.PorePyModel):
    def ic_values_pressure(self, sd: pp.Grid) -> np.ndarray:
        return np.full(sd.num_cells, self.reference_variable_values.pressure)

    def ic_values_temperature(self, sd: pp.Grid) -> np.ndarray:
        return np.full(sd.num_cells, self.reference_variable_values.temperature)


class InitialCondition(pp.PorePyModel):
    def ic_values_pressure(self, sd: pp.Grid) -> np.ndarray:
        return self.pressure_from_depth(sd.cell_centers)

    def ic_values_displacement(self, sd: pp.Grid) -> np.ndarray:
        depth = self.displacement_from_depth(sd.cell_centers).ravel("F")
        return depth  # + coords

    def ic_values_temperature(self, sd: pp.Grid) -> np.ndarray:
        return self.temperature_from_depth(sd.cell_centers)

    def pressure_from_depth(self, coords) -> np.ndarray:
        p_top = self.reference_variable_values.pressure

        # Hydrostatic pressure at the top of the domain
        rho = self.fluid.reference_component.density
        g = self.units.convert_units(pp.GRAVITY_ACCELERATION, "m*s^-2")

        # Hydrostatic pressure
        p = p_top + rho * g * self.depth(coords)
        return p

    def temperature_from_depth(self, coords: np.ndarray) -> np.ndarray:
        """Depth-dependent temperature.

        Parameters:
            coords: Coordinates.

        Returns:
            Array with temperature values.

        """
        # Thermal gradient set to 50 K/km For the moment, we use a constant gradient.
        gradient = self.units.convert_units(5e-2, "K*m^-1")
        temperature = 293  # Surface temperature in K
        return temperature + gradient * self.depth(coords)

    def displacement_from_depth(self, coords: np.ndarray) -> np.ndarray:
        """Depth-dependent displacement.

        The x and y displacements are proportional to the distance from the center of
        the domain in the xy-plane and to depth. The z displacement is proportional to
        the depth only.

        Parameters:
            coords ``shape=(3, num_cells)``: Coordinates.

        Returns:
            Array with displacement values. Shape (nd, num_cells).

        """

        u_from_depth = self.overburden_stress(coords) / self.bulk_modulus(None)._value

        # compute from center of domain in the xy-plane
        box = self.domain.bounding_box
        center = np.array([box["xmin"] + box["xmax"], box["ymin"] + box["ymax"]]) / 2
        # Normalise by box size in the xy-plane

        distances = (center[:, np.newaxis] - coords[:2, :]) / np.array(
            [box["xmax"] - box["xmin"], box["ymax"] - box["ymin"]]
        )[:, np.newaxis]
        # Pad with one for the z-coordinate
        distances = np.vstack([distances, np.ones_like(distances[0])])
        # Compute product of gradient, distances and depth
        return u_from_depth * distances
