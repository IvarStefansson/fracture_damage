# Fracture Damage Modelling — Reproducibility Archive

Data and code accompanying the paper:

> **[Title]** — [Authors], [Journal], [Year]. DOI: [paper DOI]

This archive contains a Docker image with all software pre-installed and the
simulation scripts needed to reproduce the results of Case 3 in the paper.

---

## Contents of this archive

| File | Description |
|------|-------------|
| `fracture-damage.tar.gz` | Docker image (load with `docker load`) |
| `README.md` | This file |

The Docker image contains two repositories at the exact commits used in the paper:

| Repository | Path inside container | Branch/commit |
|------------|-----------------------|---------------|
| `fracture_damage` | `/workspaces/fracture_damage` | `main` |
| `porepy` | `/workdir/porepy` | `fracture-damage-reformulation` |

---

## Requirements

- [Docker](https://docs.docker.com/get-docker/) (tested with Docker 24+)
- ~20 GB free disk space for the image
- ~10 GB additional space for simulation output

---

## Quickstart

### 1. Load the image

```bash
docker load < fracture-damage.tar.gz
docker images   # confirm "fracture-damage:zenodo" appears
```

### 2. Start the container

```bash
docker run -it --rm \
    -v "$(pwd)/results:/workspaces/fracture_damage/case3" \
    fracture-damage:zenodo \
    bash
```

The `-v` mount saves simulation output to a `results/` folder in your current
directory so it persists after the container exits. Omit it if you do not need
to keep the raw VTU/PVD files.

### 3. Run Case 3

Inside the container:

```bash
cd /workspaces/fracture_damage
bash run_case3.sh
```

This runs `case3.py` (all four damage combinations) and then generates all
plots. To monitor progress while the simulation runs, use the watch mode:

```bash
bash run_case3.sh --watch 120   # refresh plots every 120 s
```

### 4. Inspect results

Plots are written to `case3/plots/cell_size_32/thermo/`. The seven output
figures are described in the table below.

| File | Contents |
|------|----------|
| `inflow_rate.png` | Volumetric inflow rate vs time |
| `outflow_rate.png` | Volumetric outflow rate vs time |
| `flow_anomaly.png` | Flow rate change relative to no-damage baseline (%) |
| `damage_evolution.png` | Dilation and friction damage vs time (separate axes) |
| `damage_combined.png` | Both damage types on one axis |
| `slip_tendency.png` | Slip tendency phase portrait with time trajectory |
| `summary_bars.png` | Final-time summary across all combinations |

Raw time-series data are written to `results.csv` inside each run folder, e.g.
`case3/isotropic_True/model_thermo/damages_dilation_num_cells_32/results.csv`.

VTK output (`.pvd`/`.vtu`) can be opened in
[ParaView](https://www.paraview.org/) for 3-D visualisation.

---

## Simulation parameters (Case 3)

| Parameter | Value |
|-----------|-------|
| Domain | 1000 m × 1000 m × 1000 m cube |
| Fracture geometry | Two orthogonal fractures |
| Grid type | Cartesian |
| Cell size (bulk) | ~31 m (1000 m / 32) |
| Cell size (fracture) | ~83 m (1000 m / 12) |
| Physics | Thermo-poro-mechanics |
| Initialization period | 100 years |
| Simulation period | 8 daily time steps after initialization |
| Inlet pressure ramp | 4 MPa + 0.5 MPa per step |
| Inlet temperature offset | −10 K (relative to reference) |
| North boundary displacement (z) | +0.2 m (compression) |
| North boundary displacement (y) | −0.2 m |
| Solid material | Granite with damage parameters (see `case3.py`) |
| Fluid | Water |

Damage combinations run:

| Label | Dilation damage | Friction damage |
|-------|-----------------|-----------------|
| baseline | — | — |
| dilation | ✓ | — |
| friction | — | ✓ |
| both | ✓ | ✓ |

---

## Re-generating plots from existing results

If the simulation has already been run (or you have copied the `case3/`
output folder), you can regenerate only the plots:

```bash
cd /workspaces/fracture_damage
python plot_case3.py
```

Options:

```
--show          open interactive matplotlib windows
--hide-iso      omit "iso/aniso" prefix from legend labels
--no-normalize  show raw values in summary bars (no per-metric scaling)
--watch N       refresh plots every N seconds (useful while simulating)
```

---

## Source code

The simulation logic is split across the following files:

| File | Role |
|------|------|
| `case3.py` | Top-level script: parameter combinations, time schedule, solid parameters |
| `model.py` | Model base class and momentum-balance extensions |
| `boundary_conditions.py` | Pressure, temperature, and mechanics BCs |
| `initial_condition.py` | Initialisation from a reference steady state |
| `parameters.py` | Custom solid constants for damage models |
| `model_extensions.py` | Supplementary model mixins |
| `parameter_study_data_saving.py` | CSV export of scalar diagnostics |
| `plot_case3.py` | All plotting (7 figures) |
| `run_case3.sh` | Shell wrapper: simulate + plot (with optional watch mode) |

The damage constitutive laws live in the PorePy library:
`/workdir/porepy/src/porepy/models/fracture_damage.py` and
`/workdir/porepy/src/porepy/examples/fracture_damage.py`.

---

## License

The simulation code (`fracture_damage` repository) is released under the MIT
License. PorePy is released under the MIT License. See `LICENSE` files in
each repository for details.

---

## Citation

If you use this archive, please cite the paper:

```bibtex
@article{[key],
  author  = {[Authors]},
  title   = {[Title]},
  journal = {[Journal]},
  year    = {[Year]},
  doi     = {[DOI]},
}
```

This software archive:

```bibtex
@dataset{[key]_zenodo,
  author    = {[Authors]},
  title     = {Fracture Damage Modelling — Reproducibility Archive},
  year      = {[Year]},
  publisher = {Zenodo},
  doi       = {[Zenodo DOI]},
}
```
