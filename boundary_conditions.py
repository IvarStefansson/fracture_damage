from collections.abc import Callable
import numpy as np
import porepy as pp


class FlowDirSidesWestEast:
    """Mixin class to provide a method for getting the sides of the domain where
    Dirichlet conditions are defined.

    The method `dir_sides` can be used by boundary condition methods to determine
    which sides of the domain have Dirichlet conditions.

    """

    def in_sides(self, g: pp.Grid | pp.BoundaryGrid) -> np.ndarray:
        """Get the sides of the domain where Neumann conditions are defined.

        Parameters:
            g: Grid for which to get the sides.

        Returns:
            Array of booleans indicating which sides of the domain have Neumann
                conditions.

        """
        domain_sides = self.domain_boundary_sides(g)
        return domain_sides.east

    def out_sides(self, g: pp.Grid | pp.BoundaryGrid) -> np.ndarray:
        """Get the sides of the domain where Neumann conditions are defined.

        Parameters:
            g: Grid for which to get the sides.

        Returns:
            Array of booleans indicating which sides of the domain have Neumann
                conditions.

        """
        domain_sides = self.domain_boundary_sides(g)
        return domain_sides.west

    def dir_sides(self, g: pp.Grid | pp.BoundaryGrid) -> np.ndarray:
        """Get the sides of the domain where Dirichlet conditions are defined.

        Parameters:
            g: Grid for which to get the sides.

        Returns:
            Array of booleans indicating which sides of the domain have Dirichlet
                conditions.

        """
        return self.in_sides(g) + self.out_sides(g)


class FlowDirPartOfEastWestOnFractures:
    restrict_dim = 2

    def inlet_limits(self, g: pp.Grid | pp.BoundaryGrid) -> tuple[float, float]:
        """Get the z limits of the domain.

        Parameters:
            g: Grid for which to get the z limits.

        Returns:
            Tuple with the minimum and maximum z coordinates of the domain.

        """
        return 0.375 * self.domain_size, 0.625 * self.domain_size

    def out_sides(self, g: pp.Grid | pp.BoundaryGrid) -> np.ndarray:
        """Get the sides of the domain where Neumann conditions are defined.

        Parameters:
            g: Grid for which to get the sides.

        Returns:
            Array of booleans indicating which sides of the domain have Neumann
                conditions.

        """
        where = self.domain_boundary_sides(g).east
        if (isinstance(g, pp.BoundaryGrid) and g.dim < self.nd - 1) or g.dim < self.nd:
            if self.nd == 3:
                coords = (
                    g.cell_centers[self.restrict_dim]
                    if isinstance(g, pp.BoundaryGrid)
                    else g.face_centers[self.restrict_dim]
                )
                z_mask = np.logical_and(
                    coords < self.inlet_limits(g)[1],
                    coords > self.inlet_limits(g)[0],
                )
                where = where * z_mask
            return where
        return np.zeros_like(where, dtype=bool)

    def in_sides(self, g: pp.Grid | pp.BoundaryGrid) -> np.ndarray:
        """Get the sides of the domain where Neumann conditions are defined.

        Parameters:
            g: Grid for which to get the sides.

        Returns:
            Array of booleans indicating which sides of the domain have Neumann
                conditions.

        """
        where = self.domain_boundary_sides(g).west
        if (isinstance(g, pp.BoundaryGrid) and g.dim < self.nd - 1) or (
            isinstance(g, pp.Grid) and g.dim < self.nd
        ):
            coords = (
                g.cell_centers[self.restrict_dim]
                if isinstance(g, pp.BoundaryGrid)
                else g.face_centers[self.restrict_dim]
            )
            if self.nd == 3:
                where = self.domain_boundary_sides(g).west
                z_mask = np.logical_and(
                    coords < self.inlet_limits(g)[1],
                    coords > self.inlet_limits(g)[0],
                )
                where = where * z_mask
            return where
        return np.zeros_like(where, dtype=bool)


class BoundaryConditionsEnergyDirSomeSides:  # type: ignore[misc]
    """Boundary conditions for the thermal problem.

    Dirichlet boundary conditions are defined on the east and west boundaries. Some
    of the default values may be changed directly through attributes of the class.

    The domain can be 2d or 3d.

    Usage: tests for models defining equations for any subset of the thermoporomechanics
    problem.

    """

    dir_sides: Callable[[pp.Grid | pp.BoundaryGrid], np.ndarray]

    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Boundary condition type for the Fourier heat flux.

        Dirichlet boundary conditions are defined on the north and south boundaries.

        Parameters:
            sd: Subdomain for which to define boundary conditions.

        Returns:
            bc: Boundary condition object.

        """
        return pp.BoundaryCondition(sd, self.dir_sides(sd), "dir")

    def bc_type_enthalpy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Boundary condition type for the enthalpy.

        Dirichlet boundary conditions are defined on the north and south boundaries.

        Parameters:
            sd: Subdomain for which to define boundary conditions.

        Returns:
            bc: Boundary condition object.

        """
        return pp.BoundaryCondition(sd, self.dir_sides(sd), "dir")

    def bc_values_temperature(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Boundary condition values for the temperature.

        Dirichlet boundary conditions are defined on the east and west boundaries,
        with a constant value equal to the fluid's reference temperature (which will be 0
        by default).

        Parameters:
            bg: Boundary grid for which to define boundary conditions.

        Returns:
            Boundary condition values array.

        """
        values = np.ones(bg.num_cells) * self.reference_variable_values.temperature
        schedule = self.time_manager.schedule
        ind = np.searchsorted(schedule, self.time_manager.time, side="left")
        T_in = self.units.convert_units(self.params["inlet_temperature"], "K")[ind]
        values[self.in_sides(bg)] = T_in
        return values

    def bc_values_displacement(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Boundary values for the mechanics problem as a numpy array.

        Values for the north boundary are retrieved from the parameter dictionary passed
        on model initialization. The values are time dependent and are retrieved from
        the parameter dictionary using the key "north_displacements" and indexed in the
        second dimension by the current time index.

        Parameters:
            bg: Boundary grid for which boundary values are to be returned.

        Returns:
            Array of boundary values, with one value for each dimension of the
                domain, for each face in the subdomain.

        """
        sides = self.domain_boundary_sides(bg)
        values = np.zeros((self.nd, bg.num_cells))
        if bg.dim < self.nd - 1:
            # No displacement is implemented on grids of co-dimension >= 2.
            return values.ravel("F")

        # Find relevant time index.
        t = self.time_manager.time
        schedule = self.time_manager.schedule
        ind = np.searchsorted(schedule, t, side="left")
        # Wrap as array for convert_units. Thus, the passed values can be scalar or
        # list. Then tile for correct broadcasting below.
        u_north = self.params["north_displacements"][:, ind]
        u_n = np.tile(u_north, (bg.num_cells, 1)).T
        values[:, sides.north] = self.units.convert_units(u_n, "m")[:, sides.north]
        return values.ravel("F")


class BoundaryConditionsMassDirSomeSides:  # type: ignore[misc]
    """Boundary conditions for the mass balance problem.

    Dirichlet boundary conditions are defined on the east and west boundaries. Some
    of the default values may be changed directly through attributes of the class.

    The domain can be 2d or 3d.

    Usage: tests for models defining equations for any subset of the thermoporomechanics
    problem.

    """

    dir_sides: Callable[[pp.Grid | pp.BoundaryGrid], np.ndarray]

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Boundary condition type for the mass balance equation.

        Dirichlet boundary conditions are defined on the east and west boundaries.

        Parameters:
            sd: Subdomain for which to define boundary conditions.

        Returns:
            bc: Boundary condition object.

        """
        return pp.BoundaryCondition(sd, self.dir_sides(sd), "dir")

    def bc_type_fluid_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        return pp.BoundaryCondition(sd, self.dir_sides(sd), "dir")

    def bc_values_pressure(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Boundary condition values for the pressure.



        Parameters:
            bg: Boundary grid for which to define boundary conditions.

        Returns:
            Boundary condition values array.
        """
        values = np.ones(bg.num_cells) * self.reference_variable_values.pressure
        schedule = self.time_manager.schedule
        ind = np.searchsorted(schedule, self.time_manager.time, side="left")
        p_in = self.units.convert_units(self.params["inlet_pressure"], "Pa")[ind]
        values[self.in_sides(bg)] = p_in
        return values


class BoundaryConditionsMassDirWestEast(
    pp.model_boundary_conditions.BoundaryConditionsMassDirWestEast
):
    def bc_values_pressure(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Boundary condition values for Darcy flux.

        Dirichlet boundary conditions are defined on the west and east boundaries,
        with a constant value equal to the fluid's reference pressure (which will be 0
        by default).

        Parameters:
            bg: Boundary grid for which to define boundary conditions.

        Returns:
            Boundary condition values array.

        """
        domain_sides = self.domain_boundary_sides(bg)
        values = np.zeros(bg.num_cells)
        values[domain_sides.west] = 1e2  # self.reference_variable_values.pressure
        return values


class Mechanical:
    def background_stress_weights(self) -> np.ndarray:
        """Weights for the background stress along each coordinate axis.

        Note: The weights are intended for cases when the principal stress directions
        are aligned with the coordinate axes.

        Returns:
            Array with weights for the background stress directions.
        """
        return self.params.get("background_stress_weights", np.array([0.7, 1.2, 1.0]))

    def bc_values_stress(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Stress values.

        Parameters:
            boundary_grid: Boundary grid for which boundary values are to be returned.

        Returns:
            Array of boundary values, with one value for each dimension of the
                problem, for each face in the subdomain.

        """
        # SHmin = [0.62:0.63] * Sv
        # SHmin = 38.85 MPa at 2338m depth
        # SHmax = [1.2 : 1.9] * Sv
        # For now, assume SHmax is aligned with the north-south direction, corresponding
        # to the y-axis.
        values = np.zeros((3, boundary_grid.num_cells))

        domain_sides = self.domain_boundary_sides(boundary_grid)
        overburden_stress = self.overburden_stress(boundary_grid.cell_centers)
        # The sign of the stress depends on the side of the domain according to the
        # direction of the normal vector.
        sides = [["west", "east"], ["south", "north"]]
        if self.nd == 3:
            sides.append(["bottom", "top"])
        for i, sides in enumerate(sides):
            for side, sign in zip(sides, [1, -1]):
                ind = getattr(domain_sides, side)
                if np.any(ind):
                    values[i, ind] = (
                        overburden_stress[i, ind]
                        * sign
                        * boundary_grid.cell_volumes[ind]
                        # * time_ramp
                    )

        return values.ravel("F")


class RotatedBackgroundStress:
    """Apply rotated background stress on boundaries.

    This class allows for the application of background stress where the principal
    stress axes are not aligned with the coordinate axes. The stress tensor is
    constructed from principal stress values and rotation angles, then applied
    to boundaries based on face normals.

    The rotation can be specified either by:
    - Euler angles (ZXZ convention for 3D, single angle for 2D)
    - A direct rotation matrix

    Attributes set via params:
        principal_stresses: Array of principal stress magnitudes [s1, s2, s3] in 3D
                           or [s1, s2] in 2D, where s1 >= s2 >= s3.
        rotation_angles: Euler angles [phi, theta, psi] in radians for 3D (ZXZ convention),
                        or single angle for 2D rotation.
        rotation_matrix: Alternative to rotation_angles - directly specify the rotation
                        matrix (3x3 for 3D, 2x2 for 2D).

    Example usage in params dict:
        params = {
            "principal_stresses": np.array([1.2, 1.0, 0.7]) * base_stress,
            "rotation_angles": np.array([np.pi/6, 0, 0]),  # 30° rotation about z-axis
        }
    """

    def principal_stresses(self) -> np.ndarray:
        """Get the principal stress magnitudes.

        Returns:
            Array of principal stress magnitudes. For 3D: [s1, s2, s3] where s1 >= s2 >= s3.
            For 2D: [s1, s2].
        """
        default_stresses = (
            np.array([1.2, 1.0, 0.7]) if self.nd == 3 else np.array([1.2, 0.7])
        )
        return self.params.get("principal_stresses", default_stresses)

    def rotation_angles(self) -> np.ndarray:
        """Get the rotation angles for the stress tensor.

        Returns:
            Euler angles [phi, theta, psi] in radians (ZXZ convention) for 3D,
            or single angle for 2D rotation about z-axis.
        """
        default_angles = np.zeros(3) if self.nd == 3 else np.array([0.0])
        return self.params.get("rotation_angles", default_angles)

    def rotation_matrix_from_euler_angles(self, angles: np.ndarray) -> np.ndarray:
        """Compute rotation matrix from Euler angles.

        For 3D, uses ZXZ convention: R = Rz(phi) * Rx(theta) * Rz(psi)
        For 2D, uses single rotation angle about z-axis.

        Parameters:
            angles: Euler angles [phi, theta, psi] for 3D or [angle] for 2D.

        Returns:
            Rotation matrix (3x3 for 3D, 2x2 for 2D).
        """
        if self.nd == 3:
            phi, theta, psi = angles

            # Rotation about z-axis
            Rz_phi = np.array(
                [
                    [np.cos(phi), -np.sin(phi), 0],
                    [np.sin(phi), np.cos(phi), 0],
                    [0, 0, 1],
                ]
            )

            # Rotation about x-axis
            Rx_theta = np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(theta), -np.sin(theta)],
                    [0, np.sin(theta), np.cos(theta)],
                ]
            )

            # Rotation about z-axis
            Rz_psi = np.array(
                [
                    [np.cos(psi), -np.sin(psi), 0],
                    [np.sin(psi), np.cos(psi), 0],
                    [0, 0, 1],
                ]
            )

            return Rz_phi @ Rx_theta @ Rz_psi
        else:
            # 2D rotation
            angle = angles[0]
            return np.array(
                [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
            )

    def get_rotation_matrix(self) -> np.ndarray:
        """Get the rotation matrix for the stress tensor.

        First checks if a rotation matrix is directly specified in params,
        otherwise computes it from rotation angles.

        Returns:
            Rotation matrix (3x3 for 3D, 2x2 for 2D).
        """
        if "rotation_matrix" in self.params:
            return self.params["rotation_matrix"]
        else:
            angles = self.rotation_angles()
            return self.rotation_matrix_from_euler_angles(angles)

    def rotated_stress_tensor(self, coords: np.ndarray) -> np.ndarray:
        """Compute the full stress tensor in global coordinates with rotation.

        The stress tensor is constructed from principal stresses and rotation:
        sigma = R * S * R^T
        where S is the diagonal matrix of principal stresses and R is the rotation matrix.

        Parameters:
            coords: Coordinates at which to evaluate the stress (shape: (nd, n_points)).

        Returns:
            Stress tensor at each point. Shape: (nd, nd, n_points) where sigma[:, :, i]
            is the stress tensor at point i.
        """
        n_points = coords.shape[1]
        principal_stress_mag = self.principal_stresses()
        R = self.get_rotation_matrix()

        # Get the stress magnitude (can be depth-dependent)
        overburden = self.overburden_stress(coords)  # Shape: (nd, n_points)

        # Create stress tensor at each point
        stress_tensor = np.zeros((self.nd, self.nd, n_points))

        for i in range(n_points):
            # Diagonal matrix of principal stresses scaled by overburden
            S = np.diag(principal_stress_mag * overburden[:, i])

            # Rotate: sigma = R * S * R^T
            stress_tensor[:, :, i] = R @ S @ R.T

        return stress_tensor

    def overburden_stress(self, coords: np.ndarray) -> np.ndarray:
        """Compute overburden stress scaling for each principal direction.

        This method can be overridden to provide depth-dependent or other
        spatially varying stress magnitudes.

        Parameters:
            coords: Coordinates at which to evaluate stress (shape: (nd, n_points)).

        Returns:
            Stress scaling factors (shape: (nd, n_points)).
        """
        phi = self.solid.porosity
        rho = (
            phi * self.fluid.reference_component.density
            + (1.0 - phi) * self.solid.density
        )
        gradient = -rho * pp.GRAVITY_ACCELERATION
        depth = self.depth(coords)
        # Return uniform scaling for all directions (can be overridden)
        return np.ones((self.nd, coords.shape[1])) * gradient * depth

    def bc_values_stress(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Apply rotated stress tensor on boundaries.

        For each boundary face, computes the traction vector as t = sigma * n,
        where sigma is the rotated stress tensor and n is the outward normal.

        Parameters:
            boundary_grid: Boundary grid for which boundary values are to be returned.

        Returns:
            Array of boundary values (forces), with one value for each dimension
            for each face in the boundary grid. Shape: (nd * num_cells,).
        """
        # Get stress tensor at boundary cell centers
        stress_tensor = self.rotated_stress_tensor(boundary_grid.cell_centers)

        # Get outward normals for boundary faces from the parent grid
        # parent.face_normals has shape (nd, num_faces_parent)
        # Use projection to map parent faces to boundary cells
        parent_normals = boundary_grid.parent.face_normals  # (nd, num_faces_parent)
        normals = parent_normals @ boundary_grid._projections.T  # (nd, num_cells)
        signs = (
            boundary_grid.projection() @ boundary_grid.parent.cell_faces
        ).data  # (num_cells, )
        # Normalize to get unit normals
        normals = normals / np.linalg.norm(normals, axis=0)
        # Get outward direction by multiplying by the sign of the projection
        signed_normals = normals * signs

        # Compute traction: t = sigma * n
        # For each face i: t_i = stress_tensor[:, :, i] @ normal[:, i]
        tractions = np.zeros((self.nd, boundary_grid.num_cells))
        for i in range(boundary_grid.num_cells):
            tractions[:, i] = stress_tensor[:, :, i] @ signed_normals[:, i]

        # Multiply by face area to get forces
        tractions *= boundary_grid.cell_volumes

        return tractions.ravel("F")


class SubsurfaceBoundaryConditions(Mechanical):
    """Boundary conditions for the Coso geothermal reservoir model.

    We impose a displacement value scaling linearly with x and time on all boundaries
    except the top, where we impose a zero normal traction.
    """

    def overburden_stress(self, coords: np.ndarray) -> np.ndarray:
        phi = self.solid.porosity
        # rho = (
        #     phi * self.fluid.density([boundary_grid]).value(self.equation_system)
        #     + (1.0 - phi) * self.solid.density
        # )
        rho = (
            phi * self.fluid.reference_component.density
            + (1.0 - phi) * self.solid.density
        )
        gradient = (
            # np.repeat([0.7, 1.2, 1.0], boundary_grid.num_cells).reshape(3, -1)
            self.background_stress_weights() * rho * pp.GRAVITY_ACCELERATION
        )
        return gradient[:, np.newaxis] * self.depth(coords)

    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        """Boundary condition type for mechanics.

        Dirichlet boundary conditions are defined on all boundaries except the top.

        Parameters:
            sd: Subdomain for which to define boundary conditions.

        Returns:
            bc: Boundary condition object.

        """
        if sd.dim < self.nd:
            return super().bc_type_mechanics(sd)
        bc = pp.BoundaryConditionVectorial(sd, self.fixed_faces(sd), "dir")
        bc.internal_to_dirichlet(sd)
        return bc

    def fixed_faces(self, sd: pp.Grid) -> np.ndarray:
        """Get the faces where displacement is fixed.

        Parameters:
            sd: Subdomain for which to get the fixed faces.

        Returns:
            Array of booleans indicating which faces are fixed.

        """
        domain_sides = self.domain_boundary_sides(sd)
        dir_sides = domain_sides.south
        # Pick out the three faces that are closest to the mean coordinate of the bottom
        # side.
        coords = sd.face_centers[:, dir_sides]
        mean_coord = np.mean(coords, axis=1)
        # mean_coord[0] = 0
        distances = np.linalg.norm(coords - mean_coord[:, np.newaxis], axis=0)
        fixed_faces = np.zeros(sd.num_faces, dtype=bool)
        fixed_faces[np.nonzero(dir_sides)[0][np.argsort(distances)[:3]]] = True
        return fixed_faces

    # class __InitialConditions:
    # def bc_values_displacement(self, sd: pp.Grid) -> np.ndarray:
    #     # depth = self.displacement_from_depth(sd.cell_centers).ravel("F")
    #     return np.zeros(sd.num_faces * self.nd)


class WellBoundaryConditions:
    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Pressure boundary values.

        Parameters:
            boundary_grid: Boundary grid for which pressure values are to be returned.

        Returns:
            Array of pressure boundary values for each cell in the boundary grid.
        """
        if boundary_grid.dim == 2:
            return self.pressure_from_depth(boundary_grid.cell_centers)
        # Else, we are at the top of a well. The pressure is set to the well head
        # pressure.
        # Get the parent well of the subdomain.
        sd = boundary_grid.parent
        parent_well = self._parent_well(sd)
        vals = np.zeros(boundary_grid.num_cells)
        domain_sides = self.domain_boundary_sides(boundary_grid)
        if parent_well is None or not self.wells_active():
            # We are not at a well or not in the schedule yet.
            vals[domain_sides.top] = self.pressure_from_depth(
                boundary_grid.cell_centers
            )[domain_sides.top]
        else:
            # Get the well head pressure for the well.
            vals[domain_sides.top] = (
                self.units.convert_units(1e6, "Pa")
                + self.pressure_from_depth(boundary_grid.cell_centers)[domain_sides.top]
            )
        return vals

    def bc_values_temperature(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Temperature boundary values.

        Parameters:
            boundary_grid: Boundary grid for which temperature values are to be returned.

        Returns:
            Array of temperature boundary values for each cell in the boundary grid.
        """
        if boundary_grid.dim == 2:
            return self.temperature_from_depth(boundary_grid.cell_centers)
        # Get the parent well of the subdomain.
        sd = boundary_grid.parent
        parent_well = self._parent_well(sd)
        vals = np.zeros(boundary_grid.num_cells)
        ind = self.domain_boundary_sides(boundary_grid).top
        if (
            parent_well is None
            or not self.wells_active()
            or not self.is_injection_well(parent_well)
        ):
            # We are not at a well or not in the schedule yet.
            vals[ind] = self.temperature_from_depth(boundary_grid.cell_centers)[ind]
        else:
            # Get the temperature for the well.
            vals[ind] = self.units.convert_units(300, "K")  # Placeholder value
        return vals

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Boundary condition type for Darcy flux.

        For the 3d domain, Dirichlet conditions are defined on all boundaries. For
        injection wells, Neumann conditions are imposed. Dirichlet conditions are
        set on production wells.

        Parameters:
            sd: Subdomain for which to define boundary conditions.

        Returns:
            bc: Boundary condition object.

        """
        return self._bc_type_diffusion(sd)

    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Boundary condition type for Fourier flux.

        For the 3d domain, Dirichlet conditions are defined on all boundaries. For
        injection wells, Neumann conditions are imposed. Dirichlet conditions are
        set on production wells.

        Parameters:
            sd: Subdomain for which to define boundary conditions.

        Returns:
            bc: Boundary condition object.

        """
        domain_sides = self.domain_boundary_sides(sd)
        parent_well = self._parent_well(sd)
        if (
            self.is_injection_well(parent_well) or self.is_production_well(parent_well)
        ) and not self.wells_active():
            return pp.BoundaryCondition(sd, domain_sides.top, "neu")

        return pp.BoundaryCondition(sd, domain_sides.all_bf, "dir")

    def _bc_type_diffusion(self, sd: pp.Grid) -> pp.BoundaryCondition:
        domain_sides = self.domain_boundary_sides(sd)
        parent_well = self._parent_well(sd)
        if self.is_injection_well(parent_well):
            # This is an injection well subdomain. It always has Neumann conditions,
            # albeit with different values depending on whether we are in the schedule
            # or not.
            neumann_sides = domain_sides.top
            return pp.BoundaryCondition(sd, neumann_sides, "neu")
        if self.is_production_well(parent_well) and not self.wells_active():
            # This is a production well subdomain, but we are not in the schedule
            # yet. Set (zero) Neumann conditions on the top boundary. If we are in the
            # schedule, we will set Dirichlet conditions on the top boundary.
            neumann_sides = domain_sides.top
            bc = pp.BoundaryCondition(sd, neumann_sides, "neu")
            return bc

        dirichlet_sides = domain_sides.all_bf
        bc = pp.BoundaryCondition(sd, dirichlet_sides, "dir")
        return bc

    def bc_values_darcy_flux(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Darcy flux boundary values.

        Parameters:
            boundary_grid: Boundary grid for which Darcy flux values are to be returned.

        Returns:
            Array of Darcy flux boundary values for each cell in the grid.
        """
        # Only need to set nonzero values for the injection wells.
        sd = boundary_grid.parent
        # Get the parent well of the subdomain.
        parent_well = self._parent_well(sd)
        vals = np.zeros(boundary_grid.num_cells)

        if self.is_injection_well(parent_well):
            domain_sides = self.domain_boundary_sides(boundary_grid)
            neumann_sides = domain_sides.top
            if not np.any(neumann_sides) or not self.wells_active():
                return vals

            mass_rate = self.units.convert_units(10, "kg*s^-1")  # kg/s
            if "injection_rate" in self.params:
                schedule = self.time_manager.schedule
                ind = np.searchsorted(schedule, self.time_manager.time, side="left")
                mass_rate = self.units.convert_units(
                    self.params["injection_rate"][ind], "kg*s^-1"
                )
                print(
                    f"Mass rate for injection well at time {self.time_manager.time:1.5e}: {mass_rate} kg/s"
                )
            # Get the Darcy flux values for the well. Divide by
            # Fluid density divided by viscosity [kg * m^(-3) * Pa^(-1) * s^(-1)]
            darcy_flux = mass_rate / self.equation_system.evaluate(
                self.advection_weight_mass_balance([boundary_grid])
            )
            # if self.well_protocol_index() < 2:
            #     # If we are before the first well protocol, we assume a zero flux.
            #     darcy_flux = np.zeros_like(darcy_flux)
            vals[neumann_sides] = -darcy_flux[neumann_sides]
        return vals

    def is_injection_well(self, well: pp.Well | None) -> bool:
        """Check if the well is an injection well.

        Parameters:
            well: Well object.

        Returns:
            True if the well is an injection well, False otherwise.
        """
        return well is not None and well.tags["well_name"] in self.injection_well_names

    def is_production_well(self, well: pp.Well | None) -> bool:
        """Check if the well is a production well.

        Parameters:
            well: Well object.

        Returns:
            True if the well is a production well, False otherwise.
        """
        return well is not None and well.tags["well_name"] in self.production_well_names

    def wells_active(self) -> bool:
        return self.time_manager.time > self.time_manager.schedule[1]
