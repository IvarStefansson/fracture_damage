# Reproducibility archive — Fracture damage simulations

This archive provides a Docker image and supporting files to reproduce the numerical
results of **Case 3** from the CouFrac extended abstract

> *Numerical simulation of shear-induced fracture wear in thermoporomechanical media*

Case 3 is a 3-D thermo-poromechanical simulation of a rock domain containing two
orthogonal fractures, subjecting to compressive overburden stress, a rising inlet fluid
pressure, and a temperature drawdown. It explores the coupled effect of dilation and
friction damage on fracture aperture, permeability, and stress redistribution over
multiple time steps.

---

## Contents of the archive

| File | Description |
|---|---|
| `README.md` | This file |
| `fracture-damage.tar.gz` | Pre-built Docker image (load and run directly) |

---

## Software versions

| Component | Version / commit |
|---|---|
| Python | 3.14.3 |
| PorePy | commit `3893457804ef1596daac0a42bd34de95c58fa37d` (branch `fracture-damage-reformulation`, <https://github.com/pmgbergen/porepy>) |
| Simulation scripts | commit `ee1daa7dc1f682d43edbace216ef1ff79ac74f79` (branch `main`, <https://github.com/IvarStefansson/fracture_damage>) |
| pypardiso | 0.4.7 |
| gmsh | 4.15.1 |
| numpy | 2.4.2 |
| scipy | 1.17.1 |

---

## Requirements

- [Docker](https://docs.docker.com/get-docker/) (tested with Docker Engine ≥ 24)
- ~6 GB free disk space for the image
- ~4 GB RAM (8 GB recommended for the 32-cell 3-D run)

---

## Quickstart — run Case 3

```bash
# 1. Load the pre-built image
docker load < fracture-damage.tar.gz

# 2. Run case3.py (results are written inside the container under case3/)
#    Map the output directory to your host to retrieve the CSV results
mkdir -p output
docker run --rm \
    -v "$(pwd)/output":/workdir/fracture_damage/case3 \
    fracture-damage:reproduce \
    python case3.py
```

The simulation runs four damage-model combinations sequentially
(no damage, dilation only, friction only, dilation + friction) at a 32-cell
Cartesian mesh over 8 time steps each.
Results are written as CSV files to
`output/isotropic_True/model_thermo/damages_<variant>_num_cells_32/results.csv`
on your host, where `<variant>` is one of `none`, `dilation`, `friction`, or `dilation_friction`.

### Interactive shell

```bash
docker run -it --rm \
    -v "$(pwd)/output":/workdir/fracture_damage/case3 \
    fracture-damage:reproduce bash
```

This opens a shell inside the container, where the simulation scripts are in
`/workdir/fracture_damage/` and the PorePy library is in `/workdir/porepy/`.

---

## License

The simulation scripts are released under the same license as the
[fracture_damage repository](https://github.com/IvarStefansson/fracture_damage).
PorePy is released under the MIT license; see the
[PorePy repository](https://github.com/pmgbergen/porepy) for details.
