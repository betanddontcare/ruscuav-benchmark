# Releasing a version

The steps of a release of the package, from 2.1.0 on. Version 2.1.0 was released on
2026-10-30 by these steps.

1. **Content.** Confirm that no third-party restricted data, trained weights or
   real-GIS layers with incompatible terms have been added. The processed real-GIS
   worlds (Vienna, Warsaw, Berlin) are never committed (OpenStreetMap, ODbL;
   docs/OSM_ATTRIBUTION.md).
2. **Repository.** Publish the repository and add its URL to `pyproject.toml`
   (`[project.urls]`, `Repository`), `CITATION.cff` (`repository-code`),
   `codemeta.json` (`codeRepository`) and docs/SOFTWAREX_CODE_METADATA.md (C2).
3. **Date.** Set the release date in `CHANGELOG.md` (`## 2.1.0 — YYYY-MM-DD`) and
   `CITATION.cff` (`date-released`).
4. **Checks.** The workflow `.github/workflows/tests.yml` passes on the commit to be
   released; `python -m build` and `twine check --strict dist/*` pass.
5. **Archive.** With the Zenodo GitHub integration enabled for the repository, tag the
   commit (`v2.1.0`) and publish a GitHub release: Zenodo archives it as software
   with the metadata of `.zenodo.json` and assigns it a DOI. (docs/ZENODO_UPLOAD.md
   describes the upload of the v2.0.0 dataset archive, a version of another record.)
6. **DOI.** Add the release's DOI to `CITATION.cff` (`doi`), `codemeta.json`
   (`identifier`), README.md ("Citation") and docs/SOFTWAREX_CODE_METADATA.md (C2).
7. **The processed worlds.** The processed Vienna world of the paper's evaluations and
   the Berlin world of the SoftwareX article's Example 2 are published apart, as one
   Zenodo dataset under the ODbL 1.0 with their attribution: the two worlds, the
   `city-example` protocol of each, and the OpenStreetMap extract and Overpass query
   the Berlin world was built from. Its DOI goes into README.md ("Repository and
   archive"), docs/OSM_ATTRIBUTION.md, docs/BUILDING_WORLDS.md,
   worlds/real_gis/README.md and examples/softwarex/README.md, and into the Data
   availability statement of the SoftwareX article. They are never committed here.
8. **PyPI (optional).** `twine upload dist/*`. Until then the package is installed from
   the repository (`pip install -e ".[env]"`; the messages about missing extras say so).
   The wheel does not hold the world of the canonical protocols
   (`worlds/synthetic/AOI_SYNTH_000/urban_world_layers.npz`, 14 MB), so after an install
   from PyPI they need `--world` or a checkout; bundling the world as package data
   would let them run without one.

The next version then starts as `## <version> (unreleased)` in `CHANGELOG.md`, with
the version set in `src/ruscuav_benchmark/__init__.py` (`__version__`, which
`pyproject.toml` reads), `VERSION`, `CITATION.cff`, `codemeta.json`, `.zenodo.json`
and docs/SOFTWAREX_CODE_METADATA.md (C1).
