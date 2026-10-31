# Code metadata

The code metadata table of a SoftwareX article, for `ruscuav-benchmark` 2.1.0. C2
links the repository and its 2.1.0 release (docs/RELEASING.md).

| Nr. | Code metadata description | |
|---|---|---|
| C1 | Current code version | 2.1.0 |
| C2 | Permanent link to code/repository used for this code version | https://github.com/betanddontcare/ruscuav-benchmark (release v2.1.0); https://doi.org/10.5281/zenodo.23825176 |
| C3 | Permanent link to reproducible capsule | none |
| C4 | Legal code license | MIT (code); CC BY 4.0 (data and documentation; `LICENSE_DATA_AND_ASSETS.md`) |
| C5 | Code versioning system used | git |
| C6 | Software code languages, tools and services used | Python; NumPy, SciPy, Matplotlib; Gymnasium; Stable-Baselines3 and PyTorch (PPO policies); OSMnx, GeoPandas, Shapely, Rasterio, PyProj (worlds from OpenStreetMap); pytest; GitHub Actions |
| C7 | Compilation requirements, operating environments and dependencies | No compilation (pure Python). Python 3.10–3.13; `pip install ".[env]"` in the repository, with the extras `rl` and `gis` for the policies and for building worlds (README.md); tested on Linux. `requirements-reproducibility.txt` pins the environment of the archived evaluations. |
| C8 | Link to developer documentation/manual | `docs/` in the repository (docs/README.md) |
| C9 | Support email for questions | igor.betkier@wat.edu.pl |

The archived evaluations, trained weights and paper-ready tables of the associated
manuscript are the RUSCUAV v2.0.0 archive on Zenodo,
https://doi.org/10.5281/zenodo.21331402 (the DOI of all versions of the record;
v1.0.0 is 10.5281/zenodo.21331403).
