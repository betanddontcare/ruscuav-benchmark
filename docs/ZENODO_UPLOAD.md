# Zenodo upload instructions for RUSCUAV v2.0.0

These steps uploaded the v2.0.0 archive (a dataset). From 2.1.0 on, versions of the
package are archived from GitHub releases with the metadata of `.zenodo.json`
(docs/RELEASING.md).

1. Open the existing record `10.5281/zenodo.21331403` and select **New version**.
2. Create the new version draft. The new version is a separate record with a new version-specific DOI linked to prior versions.
3. Upload `RUSCUAV-v2.0.0-zenodo.zip`.
4. Also upload the two PDFs separately so the manuscript can be selected as the default preview:
   - `RUSCUAV_manuscript_IEEE.pdf`
   - `RUSCUAV_supplementary_material.pdf`
5. Resource type: **Dataset**.
6. Version: **2.0.0**.
7. Creators, in order:
   - Igor Betkier, ORCID 0000-0002-5437-7669, corresponding author.
   - Mateusz Oszczypała, ORCID 0000-0002-1194-6913.
8. Add both licenses/rights:
   - MIT License — code and executable scripts.
   - Creative Commons Attribution 4.0 International — data, weights, results, figures, documentation and paper PDFs.
9. Visibility: public/open.
10. Funding: none.
11. Reserve the new DOI if it must be inserted into the manuscript before publication. Do not enter the previous version DOI as the DOI of the new upload.
12. Set the manuscript PDF as the default preview, save the draft, inspect Preview, and only then publish.

Zenodo supports multiple licenses for mixed-license uploads. The v2.0.0 archive holds its machine-readable metadata in `zenodo_metadata.json`, `CITATION.cff` and `codemeta.json`; this repository holds those of the package in `.zenodo.json`, `CITATION.cff` and `codemeta.json`.
