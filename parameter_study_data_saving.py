"""Data saving utilities for parameter studies in case3.

This module provides classes for collecting and saving specific quantities at scheduled
times for comparison across parameter combinations.
"""

from dataclasses import dataclass
from typing import Callable, cast

import numpy as np
import porepy as pp
from porepy.viz.data_saving_model_mixin import FractureDeformationExporting


@dataclass
class ParameterStudySaveData:
    """Data container for parameter study results at a single time step.

    Attributes:
        time: Current simulation time [s].
        time_index: Time step index.
        inflow_rate: Total inflow rate at boundary [m^3/s or kg/s depending on model].
        outflow_rate: Total outflow rate at boundary [m^3/s or kg/s].
        max_friction_damage: Maximum friction damage value [-].
        max_dilation_damage: Maximum dilation damage value [-].
        average_friction_damage: Average friction damage over all fracture cells [-].
        average_dilation_damage: Average dilation damage over all fracture cells [-].
        average_slip_tendency: Average slip tendency over fracture cells [-].
        average_tangential_jump: Average magnitude of tangential displacement jump [m].
    """

    time: float
    time_index: int
    inflow_rate: float
    outflow_rate: float
    max_friction_damage: float
    max_dilation_damage: float
    average_friction_damage: float
    average_dilation_damage: float
    average_slip_tendency: float
    average_tangential_jump: float


class ParameterStudyDataSaving(pp.PorePyModel):
    """Mixin class for collecting data for parameter studies.

    This class collects boundary flow rates, damage metrics, slip tendency, and
    displacement jumps at scheduled time steps only (not intermediate steps).

    The collected data is stored in self.results as a list of ParameterStudySaveData
    objects, one per scheduled time step.

    Contract with other classes:
        - Must have darcy_flux method for computing flow rates
        - Must have friction_damage and dilation_damage methods
        - Must have displacement_jump and local_coordinates methods
        - Must have contact_traction and friction_coefficient methods
    """

    # Protocol requirements (methods from other mixins)
    darcy_flux: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    """Darcy flux operator."""
    friction_damage: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Friction damage operator."""
    dilation_damage: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Dilation damage operator."""
    displacement_jump: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Displacement jump operator."""
    local_coordinates: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Local coordinate system operator."""
    contact_traction: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Contact traction operator."""
    friction_coefficient: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Friction coefficient operator."""

    def collect_data(self) -> ParameterStudySaveData:
        """Collect parameter study data at the current (scheduled) time step.

        This method is called automatically by save_data_time_step() only at
        scheduled times (not intermediate adaptive steps).

        Returns:
            ParameterStudySaveData object containing all requested quantities.
        """

        # Get current time information
        time = self.time_manager.time
        time_index = self.time_manager.time_index

        # Get fracture subdomains (where damage is defined)
        fracture_sds = self.mdg.subdomains(dim=self.nd - 1)

        if len(fracture_sds) == 0:
            # No fractures: return placeholder data
            return ParameterStudySaveData(
                time=time,
                time_index=time_index,
                inflow_rate=0.0,
                outflow_rate=0.0,
                max_friction_damage=0.0,
                max_dilation_damage=0.0,
                average_friction_damage=0.0,
                average_dilation_damage=0.0,
                average_slip_tendency=0.0,
                average_tangential_jump=0.0,
            )

        # 1. Compute boundary flow rates
        inflow_rate, outflow_rate = self._compute_boundary_flow_rates()

        # 2. Compute damage metrics
        if hasattr(self, "friction_damage"):
            friction_vals = cast(
                np.ndarray,
                self.friction_damage(fracture_sds).value(self.equation_system),
            )
        else:
            friction_vals = np.full(sum(g.num_cells for g in fracture_sds), np.nan)

        if hasattr(self, "dilation_damage"):
            dilation_vals = cast(
                np.ndarray,
                self.dilation_damage(fracture_sds).value(self.equation_system),
            )
        else:
            dilation_vals = np.full(sum(g.num_cells for g in fracture_sds), np.nan)

        max_friction_damage = float(np.max(friction_vals))
        max_dilation_damage = float(np.max(dilation_vals))
        average_friction_damage = float(np.mean(friction_vals))
        average_dilation_damage = float(np.mean(dilation_vals))

        # 3. Compute average slip tendency
        average_slip_tendency = self._compute_average_slip_tendency(fracture_sds)

        # 4. Compute average tangential displacement jump magnitude
        average_tangential_jump = self._compute_average_tangential_jump(fracture_sds)

        return ParameterStudySaveData(
            time=time,
            time_index=time_index,
            inflow_rate=inflow_rate,
            outflow_rate=outflow_rate,
            max_friction_damage=max_friction_damage,
            max_dilation_damage=max_dilation_damage,
            average_friction_damage=average_friction_damage,
            average_dilation_damage=average_dilation_damage,
            average_slip_tendency=average_slip_tendency,
            average_tangential_jump=average_tangential_jump,
        )

    def _compute_boundary_flow_rates(self) -> tuple[float, float]:
        """Compute total inflow and outflow rates at domain boundaries.

        Returns:
            Tuple of (inflow_rate, outflow_rate) in [m^3/s] or [kg/s].
        """
        # Get highest dimensional subdomain (matrix/rock)

        # Get boundary grids
        sds = self.mdg.subdomains()

        if len(sds) == 0:
            return 0.0, 0.0

        # Compute darcy flux on boundaries
        flux_vals = cast(np.ndarray, self.darcy_flux(sds).value(self.equation_system))

        # Get face areas for each boundary grid
        total_inflow = 0.0
        total_outflow = 0.0

        offset = 0
        for sd in sds:
            num_faces = sd.num_faces  # Boundary grid cells = parent faces
            sd_flux = flux_vals[offset : offset + num_faces]
            # Get face areas from parent grid
            faces = self.domain_boundary_sides(sd).all_bf
            cells = sd.cell_faces.tocsr()[faces].indices
            cc_to_fc = sd.face_centers[:, faces] - sd.cell_centers[:, cells]
            outwards_normals = np.sum(sd.face_normals[:, faces] * cc_to_fc, axis=0) > 0
            # Get values corresponding to the faces (rows of the csc cell_faces)

            # Compute volumetric flux (flux * area)
            flux = sd_flux[faces]
            flux[~outwards_normals] *= -1  # Flip sign for outward normals

            # Separate inflow (negative flux) and outflow (positive flux)
            # Convention: positive flux = outward from domain
            total_inflow += float(np.sum(-flux[flux < 0]))
            total_outflow += float(np.sum(flux[flux > 0]))

            offset += num_faces

        return total_inflow, total_outflow

    def _compute_average_slip_tendency(self, fracture_sds: list[pp.Grid]) -> float:
        """Compute average slip tendency over all fracture cells.

        Args:
            fracture_sds: List of fracture subdomains.

        Returns:
            Average slip tendency (excluding NaN values where traction is zero).
        """
        # Get contact traction in local coordinates
        total_cells = sum(sd.num_cells for sd in fracture_sds)
        traction = cast(
            np.ndarray, self.contact_traction(fracture_sds).value(self.equation_system)
        )

        # Reshape to (nd, num_cells) with Fortran ordering, matching the convention
        # used in FractureDeformationExporting.
        traction = traction.reshape((self.nd, total_cells), order="F")
        friction = self.friction_coefficient(fracture_sds).value(self.equation_system)
        if hasattr(self, "friction_damage"):
            friction *= self.evaluate_and_scale(fracture_sds, "friction_damage", "")
        if isinstance(friction, float):
            friction = np.full(total_cells, friction)
        slip_tendency = FractureDeformationExporting.compute_slip_tendency(
            traction, friction
        )
        # nanmean already averages; exclude cells with zero normal traction (NaN).
        return float(np.nanmean(slip_tendency))

    def _compute_average_tangential_jump(self, fracture_sds: list[pp.Grid]) -> float:
        """Compute average magnitude of tangential displacement jump.

        Args:
            fracture_sds: List of fracture subdomains.

        Returns:
            Average tangential displacement jump magnitude [m].
        """
        # Get displacement jump in local coordinates
        # The displacement_jump method returns jump in global coords
        # We need to project to local tangential directions

        # Get local coordinate rotation matrix
        rotation = self.local_coordinates(fracture_sds)

        # Get displacement jump
        u_jump = self.displacement_jump(fracture_sds)

        # Project to local coordinates: u_local = R @ u_jump
        u_local = rotation @ u_jump

        # Evaluate
        u_local_vals = cast(np.ndarray, u_local.value(self.equation_system))

        # Reshape to (nd, num_cells)
        total_cells = sum(sd.num_cells for sd in fracture_sds)
        u_local_reshaped = u_local_vals.reshape((self.nd, total_cells))

        # Tangential components are first (nd-1) rows
        u_tangential = u_local_reshaped[:-1, :]

        # Compute magnitude of tangential jump for each cell
        tangential_magnitude = np.linalg.norm(u_tangential, axis=0)

        # Return average
        return float(np.mean(tangential_magnitude))

    def save_results_to_file(
        self,
        filepath: str,
        fmt: str = "npz",
    ):
        """Save collected results to file for later analysis.

        Args:
            filepath: Full path to the output file, without extension. The
                appropriate extension is appended automatically based on ``fmt``.
                The parent directory is created if it does not exist.
            fmt: File format. One of:
                - ``"npz"``  (default): NumPy binary archive.
                - ``"csv"``  : Comma-separated values, human-readable.
                - ``"txt"``  : Whitespace-separated values with a header line.
        """
        import csv
        from pathlib import Path

        if not hasattr(self, "results") or len(self.results) == 0:
            print("No results to save.")
            return

        fields = list(self.results[0].__dataclass_fields__)
        rows = [[getattr(r, f) for f in fields] for r in self.results]

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        if fmt == "npz":
            out = path.with_suffix(".npz")
            data = {f: np.array([row[i] for row in rows]) for i, f in enumerate(fields)}
            np.savez(out, **data)

        elif fmt == "csv":
            out = path.with_suffix(".csv")
            with open(out, "w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(fields)
                writer.writerows(rows)

        elif fmt == "txt":
            out = path.with_suffix(".txt")
            col_width = 24
            header = "".join(f.rjust(col_width) for f in fields)
            with open(out, "w") as fh:
                fh.write(header + "\n")
                for row in rows:
                    fh.write("".join(f"{v:{col_width}.6g}" for v in row) + "\n")

        else:
            raise ValueError(f"Unknown format {fmt!r}. Choose 'npz', 'csv', or 'txt'.")

        print(f"Saved {len(self.results)} time steps to {out}")
