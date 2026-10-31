# Code metadata

The code metadata table of a SoftwareX article (template version 6), for
`ruscuav-benchmark` 2.1.0. The left column is the template's wording. The 2.1.0 release
is archived on Zenodo, https://doi.org/10.5281/zenodo.23825176, which the article cites
as software (docs/RELEASING.md).

| Nr | Code metadata description | Metadata |
|---|---|---|
| C1 | Current code version | 2.1.0 |
| C2 | Permanent link to code/repository used for this code version | https://github.com/betanddontcare/ruscuav-benchmark |
| C3 | Legal code license | MIT (code); CC BY 4.0 (data, worlds and documentation; `LICENSE_DATA_AND_ASSETS.md`) |
| C4 | Code versioning system used | git |
| C5 | Software code languages, tools and services used | Python; NumPy, SciPy, Matplotlib; Gymnasium; Stable-Baselines3 and PyTorch (PPO policies); OSMnx, GeoPandas, Shapely, Rasterio, PyProj (worlds from OpenStreetMap); pytest; GitHub Actions |
| C6 | Compilation requirements, operating environments and dependencies | No compilation (pure Python). Python 3.10–3.13, tested on Linux. `pip install -e ".[env]"` in a checkout of the repository; the extras `rl` and `gis` add the PPO policies and the world builder. `requirements-reproducibility.txt` pins the environment of the archived evaluations. |
| C7 | If available, link to developer documentation/manual | https://github.com/betanddontcare/ruscuav-benchmark/blob/v2.1.0/docs/README.md |
| C8 | Support email for questions | igor.betkier@wat.edu.pl |

The associated manuscript is available as a preprint,
https://doi.org/10.13140/RG.2.2.24922.30400. Its archived evaluations, trained
weights and paper-ready tables are the RUSCUAV v2.0.0 archive on Zenodo,
https://doi.org/10.5281/zenodo.21331402 (the DOI of all versions of the record;
v1.0.0 is 10.5281/zenodo.21331403).
