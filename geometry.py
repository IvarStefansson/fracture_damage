import porepy as pp
import numpy as np


class SingleEllipticFracture:
    """Create a mixed-dimensional grid for a square domain with a single elliptic
    fracture.

    To be used as a mixin taking precedence over
    :class:`~porepy.models.geometry.ModelGeometry`.

    """

    @property
    def domain_size(self) -> pp.number:
        """Return the side length of the square domain.

        The domain size is controlled by the parameter ``domain_size`` in the model
        parameter dictionary.

        """
        # Scale by length unit.
        return self.units.convert_units(self.params.get("domain_size", 1.0), "m")

    def set_fractures(self) -> None:
        """Assigns a single elliptic fracture to the domain.

        The fracture is defined in
        :meth:`porepy.applications.md_grids.fracture_sets.single_elliptic_fracture`, see
        that method for a further description.

        """
        s = self.domain_size
        # Specify the fracture center
        center = np.ones(3) * s / 2
        # The minor and major axis
        major_axis = 0.25 * s
        minor_axis = 0.25 * s

        # Rotate the major axis around the center.
        major_axis_angle = 0  # np.pi / 6

        # So far, the fracture is located in the xy-plane. To define the incline,
        # specify the strike angle and the dip angle. Note that the dip rotation is
        # carried out after the major_axis rotation (recall rotations are
        # non-commutative).
        strike_angle = np.pi / 2
        dip_angle = -np.pi / 4

        # Finally, the number of points used to approximate the ellipsis.
        # This is the only optional parameter; if not specified, 16 points will be used.
        num_points = self.params.get("num_fracture_points", 16)
        f_3 = pp.EllipticFracture(
            center,
            major_axis,
            minor_axis,
            major_axis_angle,
            strike_angle,
            dip_angle,
            # num_points=num_points,
        )
        self._fractures = [f_3]

    def set_domain(self) -> None:
        """Set the square domain.

        To control the size of the domain, the parameter ``domain_size`` can be passed
        in the model parameter dictionary.

        """
        self._domain = pp.applications.md_grids.domains.nd_cube_domain(
            3, self.domain_size
        )

    def set_well_network(self) -> None:
        """Assign well network class."""
        wells = []
        num_wells = self.params.get("num_wells", 2)
        for i in range(num_wells):
            name = f"well_{i}"
            j = i + 4
            k = 9
            pts = (
                np.array(
                    [
                        [j / k, j / k, 1 / 4],
                        [j / k, j / k, 1],
                    ]
                )
                * self.domain_size
            )
            well = pp.Well(pts.T, tags={"well_name": name})
            wells.append(well)

        self.well_network = pp.WellNetwork3d(
            domain=self._domain,
            wells=wells,
            parameters={"mesh_size": 0.1 * self.domain_size},
        )
        self.injection_well_names = ["well_0"]
        self.production_well_names = ["well_1"]

    def _parent_well(self, sd: pp.Grid) -> pp.Well:
        """Get the parent well of a well subdomain.

        Parameters:
            sd: Subdomain for which to get the parent well.

        Returns:
            The parent well of the subdomain.

        """
        if sd.dim == 1 and "parent_well_index" in sd.tags:
            return self.well_network.wells[sd.tags["parent_well_index"]]
        return None

    def depth(self, coords: np.ndarray) -> np.ndarray:
        """Depth from the surface.

        Parameters:
            coords: Coordinates.

        Returns:
            Array with depth values.

        """
        return 1e3 + self.domain_size - coords[2]


class SubsurfaceCuboidDomain:
    """Mixin class for cuboid subsurface domains.

    Provides method for setting domain, defining its side lengths and depth calculation.
    The resulting domain extends from surface to bottom in negative z direction. The
    depth calculation can be extended by adding an offset representing the depth of the
    top boundary if needed.

    """

    domain: pp.Domain
    """Model domain."""
    nd: int
    """Number of spatial dimensions."""
    params: dict
    """Model parameters."""
    units: pp.Units
    """Model units."""

    def domain_sizes(self) -> NDArray[np.float64]:
        """Return the size of the domain in each of the three coordinate directions."""
        # Hard-coded to 3 instead of self.nd since nd is not necessarily set when this
        # method is (first) called. Justified since this is a *cuboid* domain.
        return self.units.convert_units(
            self.params.get("domain_sizes", np.ones(3, dtype=float)), "m"
        )

    def set_domain(self) -> None:
        """Set the cubic domain."""
        x_size, y_size, z_size = self.domain_sizes()
        box = {
            "xmin": 0.0,
            "xmax": x_size,
            "ymin": 0.0,
            "ymax": y_size,
            "zmin": -z_size,
            "zmax": 0.0,
        }
        self._domain = pp.Domain(box)


class TwoWells3d(SubsurfaceCuboidDomain):
    """A mixin adding two wells to a 3d model.

    By default, one straight vertical well and one kinked well are added to a cubic
    domain. The domain size and well mesh size can be controlled by the parameters
    ``domain_sizes`` and ``well_mesh_size``, respectively.

    A sketch of the setup in the x-z plane is provided in the comments of the method
    :meth:`set_well_network`.
    """

    params: dict
    """Model parameters."""
    units: pp.Units
    """Model units."""

    @property
    def well_names(self) -> list[str]:
        """Return the names of the two wells.

        By default, the names are "injection_well" and "production_well". In this class,
        these names are used to tag the wells when creating them. If used e.g. for
        setting boundary conditions or source terms, the user should ensure consistency
        with these names, and may override this property to provide custom names or
        switch the roles of the wells.

        """
        return ["injection_well", "production_well"]

    def set_well_network(self) -> None:
        """Set the two wells.

        See below comment for a sketch of the setup.
        """
        # TODO: Revert to kinked wells once well geometry processing is reworked. The
        # sketch below is kept for reference.

        # With constant y coordinates for both wells, the projection in the x-z plane
        # looks roughly as follows, using double lines to indicate the domain
        # boundaries:
        #               w1      w2
        #     ==============================
        #     ||        |        |         ||
        #     ||        |        |         ||
        #     ||        |        |         ||
        #     ||        |        \         ||
        #     ||        |         \        ||
        #     ||        |          \       ||
        #     ||                           ||
        #     ==============================

        # Side lengths of the domain:
        dx, dy, dz = self.domain_sizes()
        # One straight vertical well at (0.4dx, 0.4dy) extending from z=0 to z=-0.8dz.
        well_1 = pp.Well(
            np.array([[0.4 * dx, 0.4 * dx], [0.4 * dy, 0.4 * dy], [0, -0.8 * dz]]),
            tags={"well_name": self.well_names[0]},
        )
        # One well at (0.6dx, 0.6dy).
        well_2 = pp.Well(
            np.array(
                [
                    [0.6 * dx, 0.6 * dx],
                    [0.6 * dy, 0.6 * dy],
                    [0, -0.8 * dz],
                ]
            ),
            tags={"well_name": self.well_names[1]},
        )
        self._wells = [well_1, well_2]

        mesh_size = self.params.get("well_mesh_size", {"mesh_size": 0.1 * dz})
        self.well_network = pp.WellNetwork3d(
            domain=self._domain, wells=self._wells, parameters=mesh_size
        )


class TwoEllipticFractures3d(SubsurfaceCuboidDomain):
    """A mixin adding two elliptic fractures to a 3d model.

    The fractures are defined by their centers, major and minor axes, strike and dip
    angles, and major axis angles. The parameters can be controlled by passing a
    dictionary ``fracture_params`` to the model parameter dictionary. See the property
    :meth:`fracture_params` for details on the available parameters and their default
    values.

    If extending to more than two fractures, the user should override all properties
    defining fracture parameters to return arrays of size (at least) self.num_fractures.
    The case num_fractures < (size of arrays) is allowed, in which case only the first
    num_fractures entries are used.

    """

    params: dict
    """Model parameters."""
    units: pp.Units
    """Model units."""

    def fracture_params(self) -> dict:
        """Return fracture parameters with defaults.

        The available parameters are:
            - num_fractures: Number of fractures (default 2)
            - fracture_major_axes: Major axes of the fractures (default [0.2, 0.2])
            - fracture_minor_axes: Minor axes of the fractures (default equal to
                major axes, whether explicitly provided or not)
            - strike_angles: Strike angles of the fractures (default [pi/4, pi/4])
            - dip_angles: Dip angles of the fractures (default [pi/2, pi/2])
            - major_axis_angles: Major axis angles of the fractures (default [0.0, 0.0])

        The fracture axes are scaled by the minimum of the domain sizes. For adjusting
        the fracture centers, the user should override the property
        :meth:`fracture_centers`.

        Returns:
            A dictionary with fracture parameters.

        """
        default_params = {
            "num_fractures": 2,
            "fracture_major_axes": np.array([0.2, 0.2]),
            "strike_angles": np.array([np.pi / 4, np.pi / 4]),
            "dip_angles": np.array([np.pi / 2, np.pi / 2]),
            "major_axis_angles": np.array([0.0, 0.0]),
        }
        user_params = self.params.get("fracture_params", {})
        default_params.update(user_params)
        if "fracture_minor_axes" not in default_params:
            default_params["fracture_minor_axes"] = default_params[
                "fracture_major_axes"
            ]
        return default_params

    @property
    def fracture_minor_axes(self) -> np.ndarray:
        params = self.fracture_params()
        # Scale minor axes by the minimum domain size.
        size = min(self.domain_sizes())
        return params["fracture_minor_axes"] * size

    @property
    def fracture_major_axes(self) -> np.ndarray:
        params = self.fracture_params()
        # Scale major axes by the minimum domain size.
        size = min(self.domain_sizes())
        return params["fracture_major_axes"] * size

    @property
    def fracture_centers(self) -> tuple[np.ndarray, np.ndarray]:
        dx, dy, dz = self.domain_sizes()
        center_1 = np.array([0.4 * dx, 0.4 * dy, -0.6 * dz])
        center_2 = np.array([0.6 * dx, 0.6 * dy, -0.6 * dz])
        return center_1, center_2

    def set_fractures(self):
        """Set the elliptic fractures as defined in the fracture parameters and the
        fracture centers method."""
        self._fractures = []
        params = self.fracture_params()
        for i in range(params["num_fractures"]):
            f = pp.create_elliptic_fracture(
                center=self.fracture_centers[i],
                num_points=self.params.get("num_fracture_points", 10),
                strike_angle=params["strike_angles"][i],
                dip_angle=params["dip_angles"][i],
                major_axis=self.fracture_major_axes[i],
                minor_axis=self.fracture_minor_axes[i],
                major_axis_angle=params["major_axis_angles"][i],
            )
            self._fractures.append(f)


class Case5Wells:
    """Wells positioned to intersect two orthogonal vertical fractures.

    The wells are positioned to intersect the fractures at their edges, at a specified
    depth. The positioning accounts for fracture rotation via strike_angle parameters.
    """

    params: dict
    """Model parameters."""
    units: pp.Units
    """Model units."""
    _domain: pp.Domain
    """Model domain."""

    @property
    def well_names(self) -> list[str]:
        """Return the names of the two wells."""
        return ["injection_well", "production_well"]

    def set_well_network(self) -> None:
        """Create two inclined wells intersecting the two fractures.

        The injection well intersects the first fracture, and the production well
        intersects the second fracture. Both wells extend from the surface (z=0) to
        the z-coordinate of their respective fracture centers.

        The wells are inclined to cross through the fractures rather than lying in
        the fracture plane. At depth, they intersect the fracture at a distance of
        fracture_size/2 from the center, along the fracture's strike direction.
        At the surface, they are offset perpendicular to the fracture.
        """
        # Get domain dimensions
        dx, dy, dz = self.domain_sizes()

        # Get fracture centers
        frac_center_0, frac_center_1 = self.fracture_centers

        # Get fracture parameters
        frac_params = self.fracture_params()
        fracture_size = self.fracture_major_axes[0]  # Assumes circular fractures
        half_size = fracture_size / 2

        # Get strike angles (with rotation)
        strike_angles = frac_params["strike_angles"]
        theta_0 = strike_angles[0]  # First fracture
        theta_1 = strike_angles[1]  # Second fracture (should be theta_0 + π/2)

        # Well inclination offset (distance from intersection point at surface)
        # Default to 20% of domain size in horizontal direction
        well_offset = half_size  # self.params.get("well_offset", 0.2 * min(dx, dy))

        # Compute well positions
        # NOTE: PorePy's create_elliptic_fracture uses strike_angle measured from the
        # x-axis (east), not from the y-axis (north) as in standard geological convention.
        # We follow PorePy's convention here for consistency.
        # For vertical fracture with strike_angle θ measured counter-clockwise from x-axis:
        #   - Strike direction (horizontal): [cos(θ), sin(θ)]
        #   - Normal direction (horizontal): [-sin(θ), cos(θ)]
        # Intersection point: frac_center + (-half_size) * [cos(θ), sin(θ), 0]
        # Well top at surface: intersection_point + [well_offset * (-sin(θ)), well_offset * cos(θ), frac_center[2]]

        # Injection well intersects first fracture
        # Intersection point at depth in horizontal plane
        inj_int_x = frac_center_0[0] + (-half_size) * np.cos(theta_0)
        inj_int_y = frac_center_0[1] + (-half_size) * np.sin(theta_0)
        inj_int_z = frac_center_0[2]  # Intersection at fracture's z-coordinate

        # Well top at surface (offset perpendicular to fracture)
        inj_top_x = inj_int_x + well_offset * (-np.sin(theta_0))
        inj_top_y = inj_int_y + well_offset * np.cos(theta_0)

        injection_well = pp.Well(
            np.array(
                [
                    [inj_top_x, inj_int_x],
                    [inj_top_y, inj_int_y],
                    [0, inj_int_z],
                ]
            ),
            tags={"well_name": self.well_names[0]},
        )

        # Production well intersects second fracture
        # Intersection point at depth in horizontal plane
        prod_int_x = frac_center_1[0] + (-half_size) * np.cos(theta_1)
        prod_int_y = frac_center_1[1] + (-half_size) * np.sin(theta_1)
        prod_int_z = frac_center_1[2]  # Intersection at fracture's z-coordinate

        # Well top at surface (offset perpendicular to fracture)
        prod_top_x = prod_int_x + well_offset * (-np.sin(theta_1))
        prod_top_y = prod_int_y + well_offset * np.cos(theta_1)

        production_well = pp.Well(
            np.array(
                [
                    [prod_top_x, prod_int_x],
                    [prod_top_y, prod_int_y],
                    [0, prod_int_z],
                ]
            ),
            tags={"well_name": self.well_names[1]},
        )

        self._wells = [injection_well, production_well]

        # Create well network
        mesh_size = self.params.get("well_mesh_size", {"mesh_size": 0.1 * dz})
        self.well_network = pp.WellNetwork3d(
            domain=self._domain, wells=self._wells, parameters=mesh_size
        )
        self.injection_well_names = [self.well_names[0]]
        self.production_well_names = [self.well_names[1]]

    def _parent_well(self, sd: pp.Grid) -> pp.Well:
        """Get the parent well of a well subdomain.

        Parameters:
            sd: Subdomain for which to get the parent well.

        Returns:
            The parent well of the subdomain.

        """
        if sd.dim == 1 and "parent_well_index" in sd.tags:
            return self.well_network.wells[sd.tags["parent_well_index"]]
        return None
