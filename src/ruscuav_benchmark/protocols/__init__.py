"""Evaluation protocols: what a method is flown on and how its results are scored.

A protocol fixes everything an evaluation has to share with the results it is
compared with: the world (by its layers checksum,
:meth:`~ruscuav_benchmark.world.UrbanWorldLayers.layers_sha256`), the missions
(episode seed, start and target), the regime, the dynamic risk, the
mission-acceptability budgets, the risk profile, the step length and the step
limit. Results of two methods are comparable, and can be paired mission by mission,
when they were flown under the same protocol.

The package has the protocols of the paper (:func:`list_protocols`):

- ``canonical-primary``: the primary (no-terminal) evaluation in the canonical
  synthetic world AOI_SYNTH_000, 200 missions from seed 5000, static acceptability;
- ``canonical-secondary``: the same missions with stage C operational stopping;
- ``canonical-dynamic``: the same missions with the reported dynamic risk field,
  dynamic acceptability (static budgets and at most 10 of cumulative dynamic risk);
- ``vienna-primary``: the primary evaluation in the processed Vienna footway-only
  world, 200 missions from seed 7000 (the world is not distributed, see
  docs/OSM_ATTRIBUTION.md).

:func:`make_protocol` records a protocol for any world, with the missions the
environment samples, so that other worlds can be benchmarked the same way. A
protocol is a JSON file (docs/PROTOCOLS.md).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from ..posthoc import DEFAULT_BUDGETS, DT_S
from ..scenarios import Episode
from ..world import UrbanWorldLayers, canonical_aoi_id

__all__ = [
    "FORMAT", "METHOD_FIELDS", "Mission", "Protocol", "list_protocols", "load_protocol", "make_protocol",
    "find_world", "protocol_world",
]

#: Format tag of protocol files.
FORMAT = "ruscuav-protocol-1"
_REGIMES = ("primary_no_terminal", "secondary_operational_stopping")
#: The repository root of a source checkout (worlds/ and scenarios/ live there).
_REPO = Path(__file__).resolve().parents[3]
#: Protocol paths of the protocols that come with the package start with this.
_PACKAGE_PATH = "<package>/"
#: The environment's mission-acceptability fields (``MissionConfig``) for the budgets.
_MISSION_FIELDS = (("cargo_min", "accept_cargo_min"), ("road_s", "accept_road_s"), ("ped_s", "accept_sidewalk_s"),
                   ("building_s", "accept_building_s"), ("sensitive_s", "accept_sensitive_s"))
#: Environment settings (``EnvConfig`` fields) that belong to the method: its
#: observation, reward, sensing and shields. Under a protocol every other setting is
#: the protocol's.
METHOD_FIELDS = ("observation", "urban_reward", "urban_risk_lambda", "green_bonus_lambda", "lidar", "shield",
                 "rewards")


@dataclass(frozen=True)
class Mission:
    """One mission: the episode seed (it also sets the environment's random state and
    the dynamic risk field) and the start and target positions (x, y, z; z absolute,
    in the height frame of the world's terrain)."""

    seed: int
    start: Tuple[float, float, float]
    target: Tuple[float, float, float]

    def episode(self) -> Episode:
        """``(seed, reset options)`` for the evaluation functions."""
        return int(self.seed), {"start": list(self.start), "target": list(self.target)}


@dataclass(frozen=True)
class Protocol:
    name: str
    description: str
    world_id: str
    world_layers_sha256: str
    missions: Tuple[Mission, ...]
    world_file: Optional[str] = None
    world_distributed: bool = False
    regime: str = "primary_no_terminal"
    dynamic_risk: Optional[Dict[str, Any]] = None
    acceptability: str = "static"
    budgets: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_BUDGETS))
    risk_profile: str = "balanced"
    dt_s: float = DT_S
    max_steps: int = 1000
    missions_source: Optional[Dict[str, Any]] = None
    path: Optional[str] = None

    def __post_init__(self) -> None:
        def refuse(problem: str) -> None:
            raise ValueError(f"protocol {self.name!r}: {problem}")

        if self.regime not in _REGIMES:
            refuse(f"regime {self.regime!r}; expected one of {list(_REGIMES)}")
        if self.acceptability not in ("static", "dynamic"):
            refuse("acceptability must be 'static' or 'dynamic'")
        if self.acceptability == "dynamic" and self.dynamic_risk is None:
            refuse("dynamic acceptability needs a dynamic risk configuration")
        missing = sorted(set(DEFAULT_BUDGETS) - set(self.budgets))
        if missing:
            refuse(f"budgets {missing} missing")
        unknown = sorted(set(self.budgets) - set(DEFAULT_BUDGETS))
        if unknown:
            refuse(f"unknown budgets {unknown}; the budgets are {list(DEFAULT_BUDGETS)}")
        budgets = {}
        for key in DEFAULT_BUDGETS:
            value = self.budgets[key]
            try:
                number = float(value)
            except (TypeError, ValueError):
                number = math.nan
            if isinstance(value, (bool, str)) or not math.isfinite(number) or number < 0:
                refuse(f"budget {key} must be a number of at least 0, not {value!r}")
            budgets[key] = number
        object.__setattr__(self, "budgets", budgets)
        if self.budgets["cargo_min"] > 1:
            refuse(f"budget cargo_min is a cargo health, at most 1, not {self.budgets['cargo_min']!r}")
        if self.dynamic_risk is not None:
            budget = self.dynamic_risk.get("dynamic_cumulative_risk_budget")
            if budget is None or float(budget) != float(self.budgets["dynamic"]):
                # the environment's dynamic budget (its verdict and the dynamic observation) is this field
                refuse(f"the dynamic risk's dynamic_cumulative_risk_budget ({budget!r}) must equal the dynamic "
                       f"budget ({self.budgets['dynamic']!r})")
        if not (isinstance(self.dt_s, (int, float)) and math.isfinite(self.dt_s) and self.dt_s > 0):
            refuse(f"dt_s must be positive, not {self.dt_s!r}")
        if int(self.max_steps) < 1:
            refuse(f"max_steps must be at least 1, not {self.max_steps!r}")
        if not self.missions:
            refuse("no missions")
        seeds = [m.seed for m in self.missions]
        if len(set(seeds)) < len(seeds):
            refuse("missions repeat a seed")

    # ------------------------------------------------------------------ views
    @property
    def dynamic(self) -> bool:
        """Whether mission acceptability includes the dynamic-risk budget."""
        return self.acceptability == "dynamic"

    @property
    def n_episodes(self) -> int:
        return len(self.missions)

    @property
    def seeds(self) -> List[int]:
        return [m.seed for m in self.missions]

    def first_missions(self, n_episodes: Optional[int] = None) -> Tuple[Mission, ...]:
        """All missions, or the first ``n_episodes`` (1 to the number of missions)."""
        if n_episodes is None:
            return self.missions
        if isinstance(n_episodes, bool) or int(n_episodes) != n_episodes or not 1 <= int(n_episodes) <= len(
                self.missions):
            raise ValueError(f"n_episodes={n_episodes!r}: protocol {self.name!r} has {len(self.missions)} missions; "
                             f"give 1 to {len(self.missions)}")
        return self.missions[: int(n_episodes)]

    def episodes(self, n_episodes: Optional[int] = None) -> List[Episode]:
        """The missions as ``(seed, reset options)``, all or the first ``n_episodes``."""
        return [m.episode() for m in self.first_missions(n_episodes)]

    def dynamic_risk_config(self) -> Any:
        """The dynamic risk as a :class:`~ruscuav_benchmark.env.DynamicRiskConfig`, or None."""
        if self.dynamic_risk is None:
            return None
        from ..env.dynamic_risk import DynamicRiskConfig  # needs the "env" extra

        return DynamicRiskConfig(**self.dynamic_risk)

    def mission_config(self) -> Any:
        """The environment's mission configuration (``MissionConfig``) with the protocol's
        acceptability budgets: the environment's own verdict and the budget-relative
        exposure of the semantic observations use them (needs the "env" extra)."""
        from ..env.config import MissionConfig

        return MissionConfig(**{name: float(self.budgets[key]) for key, name in _MISSION_FIELDS})

    def environment_config(self, method: str = "RISK+OBS-RL", dynamic_risk: bool = True, **method_settings: Any) -> Any:
        """The environment configuration (``EnvConfig``) of ``method`` (a preset of
        ``ruscuav_benchmark.env.METHOD_PRESETS``) under the protocol: its regime, risk
        profile, step length, step limit, budgets and, with ``dynamic_risk`` (for
        methods with dynamic risk), its dynamic risk. ``method_settings`` change the
        fields that belong to the method: ``observation``, ``urban_reward``,
        ``urban_risk_lambda``, ``green_bonus_lambda``, ``lidar``, ``shield`` and
        ``rewards``. Trajectories are scored in the configuration of ``RISK+OBS-RL``
        (needs the "env" extra)."""
        from dataclasses import replace

        from ..env import make_config
        from ..env.config import DynamicsConfig, EpisodeConfig

        other = sorted(set(method_settings) - set(METHOD_FIELDS))
        if other:
            raise ValueError(f"{', '.join(other)}: protocol {self.name!r} fixes these environment settings; a method "
                             f"can change {', '.join(METHOD_FIELDS)}")
        overrides: Dict[str, Any] = {
            "mission": self.mission_config(),
            "dynamics": replace(DynamicsConfig(), dt=float(self.dt_s)),
            "episode": replace(EpisodeConfig(), max_steps=int(self.max_steps)),
            **method_settings,
        }
        if dynamic_risk and self.dynamic_risk is not None:
            overrides["dynamic_risk"] = self.dynamic_risk_config()
        return make_config(method, regime=self.regime, risk_profile=self.risk_profile, **overrides)

    def check_environment(self, config: Any) -> None:
        """Refuse an environment configuration (``EnvConfig``) that would account
        episodes differently from the protocol: other vehicle dynamics (including the
        step length), cargo model, endings (including the step limit), mission
        settings and budgets, regime, risk profile or dynamic risk. The fields of
        :data:`METHOD_FIELDS` belong to the method and are not compared."""
        reference = self.environment_config()
        problems = [name for name in ("dynamics", "cargo", "episode", "mission")
                    if getattr(config, name) != getattr(reference, name)]
        if (config.stage, config.mode) != (reference.stage, reference.mode) or \
                config.resolved_terminate_on_urban_constraint != reference.resolved_terminate_on_urban_constraint \
                or config.resolved_hard_sensitive_zones != reference.resolved_hard_sensitive_zones:
            problems.append("regime")
        if config.risk_profile != reference.risk_profile:
            problems.append("risk_profile")
        if config.dynamic_risk is not None and config.dynamic_risk != reference.dynamic_risk:
            problems.append("dynamic_risk")
        if problems:
            raise ValueError(f"protocol {self.name!r} fixes the environment's {', '.join(problems)}, and this "
                             f"evaluation's environment differs (step {config.dynamics.dt} s and at most "
                             f"{config.episode.max_steps} steps; the protocol's: {self.dt_s} s and {self.max_steps})")

    def check_world(self, world: UrbanWorldLayers) -> None:
        """Refuse a world whose layers checksum is not the protocol's."""
        checksum = world.layers_sha256()
        if checksum != self.world_layers_sha256:
            raise ValueError(f"protocol {self.name!r} is defined for the world {self.world_id} with layers checksum "
                             f"{self.world_layers_sha256}; the given world ({world.aoi_id}) has {checksum}")

    def check_seeds(self, seeds: Iterable[Any], what: str = "the run") -> Dict[str, Any]:
        """Compare the seeds of a run with the protocol's missions: raises when the run
        has episodes of other seeds or repeats one; returns ``{"complete": bool,
        "missing": [...]}`` (a run of the first missions only is not complete)."""
        given = []
        for seed in seeds:
            if seed is None or isinstance(seed, bool) or (isinstance(seed, float) and not math.isfinite(seed)):
                raise ValueError(f"{what} has episodes without a seed: the missions of protocol {self.name!r} "
                                 "are identified by their seed")
            try:
                value = int(seed)
            except (TypeError, ValueError):
                raise ValueError(f"{what} has the seed {seed!r}; seeds are integers") from None
            if isinstance(seed, float) and value != seed:
                raise ValueError(f"{what} has the seed {seed!r}; seeds are integers")
            given.append(value)
        if len(set(given)) < len(given):
            raise ValueError(f"{what} repeats a mission seed")
        unknown = sorted(set(given) - set(self.seeds))
        if unknown:
            raise ValueError(f"{what} has episodes of seeds that are not missions of protocol {self.name!r}: "
                             f"{unknown[:5]}{' ...' if len(unknown) > 5 else ''}")
        missing = [s for s in self.seeds if s not in set(given)]
        return {"complete": not missing, "missing": missing}

    # -------------------------------------------------------------- settings
    def evaluation_settings(self, n_episodes: Optional[int] = None, dynamic_method: bool = False) -> Dict[str, Any]:
        """Keyword arguments of :func:`~ruscuav_benchmark.planners.evaluate_planner` and
        :func:`~ruscuav_benchmark.policies.evaluate_policy` for this protocol: the
        missions, the regime and the risk profile, and the dynamic risk for a method
        that models it (``dynamic_method``). A dynamic protocol needs one."""
        if self.dynamic and not dynamic_method:
            raise ValueError(f"protocol {self.name!r} has dynamic risk: fly a method with dynamic risk (the "
                             "DYN_RH_Astar planner, a DYN- policy preset, or a static-trained policy in DYN-RISK-RL)")
        settings: Dict[str, Any] = {"missions": self.episodes(n_episodes), "regime": self.regime,
                                    "risk_profile": self.risk_profile}
        if dynamic_method and self.dynamic_risk is not None:
            settings["dynamic_risk"] = self.dynamic_risk_config()
        return settings

    def apply(self, world: UrbanWorldLayers, *, dynamic_method: bool, n_episodes: Optional[int] = None,
              given: Optional[Mapping[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """For an evaluation function: (keyword arguments of :meth:`evaluation_settings`,
        the record of the protocol for the summary). Refuses a world with another
        layers checksum, settings in ``given`` that the protocol fixes (any value
        other than None), a static method in a dynamic protocol and a method with
        dynamic risk in a static one."""
        fixed = sorted(k for k, v in (given or {}).items() if v is not None)
        if fixed:
            raise ValueError(f"{', '.join(fixed)}: protocol {self.name!r} fixes these settings")
        self.check_world(world)
        if dynamic_method and not self.dynamic:
            raise ValueError(f"protocol {self.name!r} has no dynamic risk; use a dynamic protocol such as "
                             "canonical-dynamic for methods with dynamic risk")
        settings = self.evaluation_settings(n_episodes, dynamic_method=dynamic_method)
        n_run = len(settings["missions"])
        record = {"name": self.name, "digest": self.digest(), "missions": self.n_episodes, "missions_flown": n_run,
                  "complete": n_run == self.n_episodes, "acceptability": self.acceptability,
                  "budgets": dict(self.budgets)}
        return settings, record

    # ------------------------------------------------------------------- I/O
    def to_dict(self) -> Dict[str, Any]:
        return {
            "format": FORMAT,
            "name": self.name,
            "description": self.description,
            "world": {"id": self.world_id, "layers_sha256": self.world_layers_sha256, "file": self.world_file,
                      "distributed": bool(self.world_distributed)},
            "regime": self.regime,
            "dynamic_risk": self.dynamic_risk,
            "acceptability": self.acceptability,
            "budgets": dict(self.budgets),
            "risk_profile": self.risk_profile,
            "dt_s": self.dt_s,
            "max_steps": self.max_steps,
            "missions_source": self.missions_source,
            "missions": [{"seed": m.seed, "start": list(m.start), "target": list(m.target)} for m in self.missions],
        }

    def save(self, path: Union[str, Path]) -> Path:
        """Write the protocol as JSON, one mission per line."""
        data = self.to_dict()
        missions = data.pop("missions")
        head = json.dumps(data, indent=1)
        body = ",\n".join("  " + json.dumps(m) for m in missions)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(head[:-2] + ',\n "missions": [\n' + body + "\n ]\n}\n", encoding="utf-8")
        return path

    def digest(self) -> str:
        """SHA-256 of the protocol's content: everything but the description and the
        location of the world file (``world.file``). The world is identified by its
        layers checksum, so the same protocol recorded with its world at another path
        has the same digest."""
        content = {k: v for k, v in self.to_dict().items() if k != "description"}
        content["world"] = {k: v for k, v in content["world"].items() if k != "file"}
        return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], path: Optional[str] = None) -> "Protocol":
        if data.get("format") != FORMAT:
            raise ValueError(f"{path or 'protocol'}: format {data.get('format')!r}; expected {FORMAT!r}")
        world = data.get("world") or {}
        missions = tuple(Mission(int(m["seed"]), tuple(float(v) for v in m["start"]),  # type: ignore[arg-type]
                                 tuple(float(v) for v in m["target"])) for m in data.get("missions", []))
        for m in missions:
            if len(m.start) != 3 or len(m.target) != 3:
                raise ValueError(f"{path or 'protocol'}: mission {m.seed} needs start and target (x, y, z)")
        return cls(
            name=str(data["name"]), description=str(data.get("description", "")), world_id=str(world["id"]),
            world_layers_sha256=str(world["layers_sha256"]), missions=missions, world_file=world.get("file"),
            world_distributed=bool(world.get("distributed", False)),
            regime=str(data.get("regime", "primary_no_terminal")),
            dynamic_risk=data.get("dynamic_risk"), acceptability=str(data.get("acceptability", "static")),
            budgets={k: float(v) for k, v in (data.get("budgets") or DEFAULT_BUDGETS).items()},
            risk_profile=str(data.get("risk_profile", "balanced")), dt_s=float(data.get("dt_s", DT_S)),
            max_steps=int(data.get("max_steps", 1000)), missions_source=data.get("missions_source"), path=path)


def list_protocols() -> List[str]:
    """Names of the protocols that come with the package."""
    return sorted(p.name[:-5] for p in resources.files(__name__).iterdir() if p.name.endswith(".json"))


def load_protocol(protocol: Union[str, Path, Protocol]) -> Protocol:
    """A protocol by name (:func:`list_protocols`) or from a JSON file."""
    if isinstance(protocol, Protocol):
        return protocol
    text = str(protocol)
    if text in list_protocols():
        resource = resources.files(__name__) / f"{text}.json"
        return Protocol.from_dict(json.loads(resource.read_text(encoding="utf-8")), path=f"{_PACKAGE_PATH}{text}.json")
    path = Path(text)
    if not path.is_file():
        raise FileNotFoundError(f"no protocol {text!r}: not a file and not one of {list_protocols()}")
    return Protocol.from_dict(json.loads(path.read_text(encoding="utf-8")), path=str(path))


def find_world(protocol: Protocol, roots: Sequence[Union[str, Path]] = ()) -> Optional[Path]:
    """The protocol's world file (``world.file``). A relative path is looked for from
    ``roots``, the folder of the protocol file, the current directory and the
    repository the package was installed from, in this order (None if it is not
    found)."""
    if not protocol.world_file:
        return None
    folders = [Path(root) for root in roots]
    if protocol.path and not protocol.path.startswith(_PACKAGE_PATH):
        folders.append(Path(protocol.path).parent)
    for folder in [*folders, Path.cwd(), _REPO]:
        candidate = folder / protocol.world_file
        if candidate.is_file():
            return candidate
    return None


def protocol_world(protocol: Protocol, world: Union[str, Path, UrbanWorldLayers, None] = None,
                   check: bool = True) -> UrbanWorldLayers:
    """The world to evaluate the protocol in: ``world`` (a file or loaded layers) or
    the protocol's own world file; its layers checksum must be the protocol's."""
    if world is None:
        found = find_world(protocol)
        if found is None:
            where = f" ({protocol.world_file})" if protocol.world_file else ""
            raise FileNotFoundError(f"protocol {protocol.name!r} needs its world {protocol.world_id}{where}, layers "
                                    f"checksum {protocol.world_layers_sha256}: give it (--world)")
        world = found
    layers = world if isinstance(world, UrbanWorldLayers) else UrbanWorldLayers.load_npz(world)
    if check:
        protocol.check_world(layers)
    return layers


def make_protocol(world: Union[str, Path, UrbanWorldLayers], name: str, *, n_episodes: int = 200, seed0: int = 5000,
                  regime: str = "primary_no_terminal", dynamic: bool = False, dynamic_risk: Any = None,
                  budgets: Optional[Mapping[str, float]] = None, risk_profile: str = "balanced",
                  description: str = "", world_file: Optional[str] = None) -> Protocol:
    """A protocol for ``world`` with the missions ``RuscuavEnv.reset(seed)`` samples
    for ``n_episodes`` seeds from ``seed0`` (safe start and target, as in the paper's
    worlds). ``budgets`` replace the paper's acceptability budgets they name
    (``DEFAULT_BUDGETS``). With ``dynamic`` the protocol has the reported dynamic risk
    (or ``dynamic_risk``) and dynamic acceptability; the dynamic budget is then the
    one of ``budgets`` if they name it, otherwise the dynamic risk's. The world file
    is recorded as ``world_file``, by default the path ``world`` was given as (see
    :func:`find_world`). Needs the "env" extra."""
    from ..env import RuscuavEnv, make_config
    from ..env.dynamic_risk import REPORTED_DYNAMIC_CONFIG, DynamicRiskConfig

    if isinstance(n_episodes, bool) or int(n_episodes) != n_episodes or int(n_episodes) < 1:
        raise ValueError(f"n_episodes must be at least 1, got {n_episodes!r}")
    if isinstance(world, UrbanWorldLayers):
        layers = world
    else:
        layers = UrbanWorldLayers.load_npz(world)
        world_file = world_file if world_file is not None else Path(world).as_posix()
    chosen_budgets = {**DEFAULT_BUDGETS, **dict(budgets or {})}
    dyn = None
    if dynamic or dynamic_risk is not None:
        cfg = dynamic_risk if dynamic_risk is not None else REPORTED_DYNAMIC_CONFIG
        dyn = asdict(cfg if isinstance(cfg, DynamicRiskConfig) else DynamicRiskConfig(**dict(cfg)))
        if budgets is None or "dynamic" not in budgets:
            chosen_budgets["dynamic"] = float(dyn["dynamic_cumulative_risk_budget"])
        elif dynamic_risk is None or float(dyn["dynamic_cumulative_risk_budget"]) == float(budgets["dynamic"]):
            dyn["dynamic_cumulative_risk_budget"] = float(budgets["dynamic"])
        else:
            raise ValueError(f"the dynamic budget of budgets ({budgets['dynamic']}) and the "
                             "dynamic_cumulative_risk_budget of dynamic_risk "
                             f"({dyn['dynamic_cumulative_risk_budget']}) "
                             "differ; give one")
    config = make_config("RISK+OBS-RL", regime=regime, risk_profile=risk_profile)
    env = RuscuavEnv(layers, config)
    missions = []
    try:
        for seed in range(int(seed0), int(seed0) + int(n_episodes)):
            env.reset(seed=seed)
            start = tuple(float(v) for v in env.start_point)
            target = tuple(float(v) for v in env.target_point)
            missions.append(Mission(seed, start, target))  # type: ignore[arg-type]
    finally:
        env.close()
    return Protocol(
        name=name, description=description or f"{n_episodes} missions in {canonical_aoi_id(layers.aoi_id)} from seed "
                                              f"{seed0}, sampled by RuscuavEnv.reset",
        world_id=canonical_aoi_id(layers.aoi_id), world_layers_sha256=layers.layers_sha256(), missions=tuple(missions),
        world_file=world_file, world_distributed=False, regime=regime, dynamic_risk=dyn,
        acceptability="dynamic" if dyn is not None else "static", budgets=chosen_budgets, risk_profile=risk_profile,
        missions_source={"sampled_by": "RuscuavEnv.reset(seed)", "seed0": int(seed0), "n": int(n_episodes)})
