# RUSCUAV v2.0.0 dataset/model card

## Resource type

Mixed benchmark artifact: software, synthetic spatial data, trained PPO weights, planner and policy evaluation results, paper tables/figures and documentation. The Zenodo record is classified as a dataset because the principal preserved object is the benchmark and its result corpus.

## Intended uses

- reproduce manuscript-level tables and figures;
- benchmark new learned policies or planners under fixed semantic worlds and mission metrics;
- audit the distinction between completion and benchmark-defined acceptability;
- study training-seed, morphology-shift, dynamic-risk and information-regime effects.

## Out-of-scope uses

- regulatory approval or operational authorization;
- inference of actual population occupancy or casualty risk;
- claims of societal acceptance;
- high-fidelity aerodynamics or unrestricted 3-D trajectory optimization;
- real-flight validation.

## Data sources

Synthetic worlds are author-generated. Vienna/Warsaw experiments use OpenStreetMap-derived semantic data, but processed real-GIS NPZ files are not redistributed in this release. Scripts and metadata are provided for regeneration with appropriate attribution.

## Main limitations

- semantic exposure classes are proxies rather than measured crowd occupancy;
- dynamic hotspots are parameterized proxies;
- learned policies command horizontal acceleration while altitude is internally regulated;
- planners are 2.5-D fixed-altitude planners;
- budget values are benchmark parameters and partly depend on vehicle speed;
- Vienna/Warsaw are simulation AOIs, not physical flight trials.
