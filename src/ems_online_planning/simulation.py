from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


@dataclass(frozen=True)
class Node:
    name: str
    x: float
    y: float


@dataclass
class Incident:
    incident_id: int
    call_time: float
    scene: Node
    severity: str
    patient_type: str
    base_survival: float
    decay_rate: float
    scene_minutes: float


@dataclass
class Ambulance:
    amb_id: str
    current_node: Node
    level: str
    available_at: float = 0.0


@dataclass
class Hospital:
    hosp_id: str
    node: Node
    specialty: str
    quality: float
    remaining_capacity: int
    handover_minutes: float


@dataclass
class Crossing:
    barrier_x: float
    closed_windows: List[Tuple[float, float]]
    detour_penalty: float


@dataclass
class Decision:
    incident_id: int
    ambulance_id: str
    hospital_id: str
    policy: str
    response_mode: str
    transport_mode: str
    response_minutes: float
    transport_minutes: float
    handover_minutes: float
    wait_minutes: float
    total_care_minutes: float
    survival_score: float


@dataclass(frozen=True)
class ActionChoice:
    decision: Optional[Decision] = None
    hold_minutes: float = 0.0


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    n_incidents: int = 120
    n_ambulances: int = 8
    als_share: float = 0.5
    call_horizon_min: float = 720.0
    rural_bias: float = 0.65
    hospital_capacity_scale: float = 1.0
    handover_scale: float = 1.0
    closure_scale: float = 1.0
    detour_penalty: float = 28.0
    mass_casualty_events: int = 0
    mass_cluster_size: int = 5
    mass_cluster_spread_min: float = 8.0
    lookahead_depth: int = 3
    candidate_limit: int = 5
    strict_specialty_for_critical: bool = True
    prediction_horizon_min: float = 180.0
    forecast_incident_limit: int = 8
    mc_rollouts: int = 8
    mc_lookahead_depth: int = 4
    prediction_noise_minutes: float = 18.0
    prediction_noise_survival: float = 0.08
    allow_hold: bool = True
    hold_step_minutes: float = 10.0
    hold_max_steps: int = 2


AMB_LEVEL = {"BLS": 1, "ALS": 2}
SEVERITY_PROFILE = {
    "critical": {"required_level": 2, "default_scene": 18.0, "decay": 0.018},
    "urgent": {"required_level": 1, "default_scene": 14.0, "decay": 0.010},
    "moderate": {"required_level": 1, "default_scene": 10.0, "decay": 0.004},
}

OFFLINE_POLICIES = ("nearest", "fastest", "survival", "lookahead")
ONLINE_POLICIES = (
    "offline_oracle",
    "online_greedy",
    "online_mc_perfect",
    "online_mc_noisy",
    "online_mc_random",
)
DEFAULT_POLICIES = OFFLINE_POLICIES
MASS_EVENT_TIMES = (90.0, 190.0, 320.0, 480.0, 620.0)


def euclidean_minutes(origin: Node, destination: Node) -> float:
    return 1.8 * math.dist((origin.x, origin.y), (destination.x, destination.y)) + 4.0


def crosses_barrier(origin: Node, destination: Node, barrier_x: float) -> bool:
    return (origin.x < barrier_x < destination.x) or (destination.x < barrier_x < origin.x)


def waiting_time_at_crossing(arrival_minute: float, crossing: Crossing) -> float:
    for start, end in crossing.closed_windows:
        if start <= arrival_minute < end:
            return end - arrival_minute
    return 0.0


def leg_time(
    origin: Node,
    destination: Node,
    depart_minute: float,
    crossing: Crossing,
    mode: str,
) -> Tuple[float, float]:
    base_minutes = euclidean_minutes(origin, destination)
    if not crosses_barrier(origin, destination, crossing.barrier_x):
        return base_minutes, 0.0
    crossing_arrival = depart_minute + 0.5 * base_minutes
    wait_minutes = waiting_time_at_crossing(crossing_arrival, crossing)
    if mode == "wait":
        return base_minutes + wait_minutes, wait_minutes
    if mode == "detour":
        return base_minutes + crossing.detour_penalty, 0.0
    raise ValueError(f"Unknown route mode: {mode}")


def best_response_leg(
    origin: Node, destination: Node, depart_minute: float, crossing: Crossing
) -> Tuple[float, str, float]:
    wait_time, wait_delay = leg_time(origin, destination, depart_minute, crossing, "wait")
    detour_time, _ = leg_time(origin, destination, depart_minute, crossing, "detour")
    if wait_time <= detour_time:
        return wait_time, "wait", wait_delay
    return detour_time, "detour", 0.0


def severity_from_survival(base_survival: float) -> str:
    if base_survival < 0.55:
        return "critical"
    if base_survival < 0.80:
        return "urgent"
    return "moderate"


def scene_catalog(config: ScenarioConfig) -> List[Tuple[Node, float]]:
    return [
        (Node("VillageA", 74, 26), 0.20 + 0.20 * config.rural_bias),
        (Node("VillageB", 90, 31), 0.12 + 0.16 * config.rural_bias),
        (Node("FarmRoad", 68, 8), 0.14 + 0.16 * config.rural_bias),
        (Node("CrossTown", 58, 18), 0.14 + 0.12 * config.rural_bias),
        (Node("WestSuburb", 24, 30), 0.18 - 0.10 * config.rural_bias),
        (Node("TownCenter", 38, 22), 0.22 - 0.12 * config.rural_bias),
    ]


def sample_scene_node(rng: random.Random, config: ScenarioConfig) -> Node:
    scenes = scene_catalog(config)
    return rng.choices(
        [item[0] for item in scenes],
        weights=[max(weight, 0.02) for _, weight in scenes],
        k=1,
    )[0]


def build_profile_pool_from_patient_table(
    patient_table: pd.DataFrame,
    triss_col: str = "TRISS",
    patient_type_col: Optional[str] = None,
    sample_limit: Optional[int] = None,
) -> List[Dict[str, object]]:
    if triss_col not in patient_table.columns:
        raise ValueError(f"Column '{triss_col}' was not found.")
    table = patient_table.dropna(subset=[triss_col]).copy()
    table[triss_col] = table[triss_col].clip(0.02, 0.99)
    if sample_limit is not None and len(table) > sample_limit:
        table = table.sample(sample_limit, random_state=42)
    pool: List[Dict[str, object]] = []
    for _, row in table.iterrows():
        base_survival = float(row[triss_col])
        severity = severity_from_survival(base_survival)
        patient_type = (
            str(row[patient_type_col])
            if patient_type_col is not None and patient_type_col in table.columns
            else "trauma"
        )
        if patient_type not in {"trauma", "burn", "medical"}:
            patient_type = "medical"
        pool.append(
            {
                "base_survival": base_survival,
                "severity": severity,
                "patient_type": patient_type,
                "decay_rate": SEVERITY_PROFILE[severity]["decay"],
                "scene_minutes": SEVERITY_PROFILE[severity]["default_scene"],
            }
        )
    if not pool:
        raise ValueError("Profile pool is empty after filtering.")
    return pool


def sample_patient_profile(
    rng: random.Random, profile_pool: Optional[Sequence[Dict[str, object]]]
) -> Dict[str, object]:
    if profile_pool:
        profile = dict(rng.choice(profile_pool))
        profile["scene_minutes"] = max(6.0, float(profile["scene_minutes"]) + rng.uniform(-2.0, 2.0))
        return profile
    severity = rng.choices(["critical", "urgent", "moderate"], weights=[0.24, 0.46, 0.30], k=1)[0]
    patient_type = rng.choices(["trauma", "burn", "medical"], weights=[0.55, 0.12, 0.33], k=1)[0]
    base_survival = {"critical": 0.62, "urgent": 0.80, "moderate": 0.93}[severity]
    base_survival = float(max(0.02, min(0.99, base_survival + rng.uniform(-0.07, 0.07))))
    return {
        "base_survival": base_survival,
        "severity": severity,
        "patient_type": patient_type,
        "decay_rate": SEVERITY_PROFILE[severity]["decay"],
        "scene_minutes": max(6.0, SEVERITY_PROFILE[severity]["default_scene"] + rng.uniform(-2.0, 2.0)),
    }


def hospital_match_multiplier(
    patient_type: str,
    specialty: str,
    severity: str,
    strict_specialty_for_critical: bool,
) -> float:
    if patient_type == "medical":
        return 1.0 if specialty == "general" else 0.88
    if specialty == patient_type:
        return 1.18
    if specialty == "general":
        return 0.82 if severity == "critical" else 0.92
    if strict_specialty_for_critical and severity == "critical":
        return 0.30
    return 0.70


def hospital_is_feasible(
    incident: Incident, hospital: Hospital, strict_specialty_for_critical: bool
) -> bool:
    if hospital.remaining_capacity <= 0:
        return False
    if incident.patient_type == "medical" or incident.severity != "critical":
        return True
    if not strict_specialty_for_critical:
        return True
    return hospital.specialty in {incident.patient_type, "general"}


def candidate_decisions(
    incident: Incident,
    ambulances: List[Ambulance],
    hospitals: List[Hospital],
    crossing: Crossing,
    route_modes: Sequence[str],
    strict_specialty_for_critical: bool,
    decision_time: Optional[float] = None,
) -> List[Decision]:
    decisions: List[Decision] = []
    required_level = SEVERITY_PROFILE[incident.severity]["required_level"]
    for ambulance in ambulances:
        if AMB_LEVEL[ambulance.level] < required_level:
            continue
        dispatch_minute = max(incident.call_time, ambulance.available_at)
        if decision_time is not None:
            dispatch_minute = max(dispatch_minute, decision_time)
        response_minutes, response_mode, response_wait = best_response_leg(
            ambulance.current_node, incident.scene, dispatch_minute, crossing
        )
        scene_done = dispatch_minute + response_minutes + incident.scene_minutes
        for hospital in hospitals:
            if not hospital_is_feasible(incident, hospital, strict_specialty_for_critical):
                continue
            match_multiplier = hospital_match_multiplier(
                incident.patient_type,
                hospital.specialty,
                incident.severity,
                strict_specialty_for_critical,
            )
            for transport_mode in route_modes:
                transport_minutes, transport_wait = leg_time(
                    incident.scene, hospital.node, scene_done, crossing, transport_mode
                )
                total_care_minutes = (
                    dispatch_minute
                    - incident.call_time
                    + response_minutes
                    + incident.scene_minutes
                    + transport_minutes
                    + hospital.handover_minutes
                )
                survival_score = incident.base_survival * math.exp(
                    -incident.decay_rate * total_care_minutes
                )
                survival_score *= hospital.quality * match_multiplier
                survival_score = max(0.0, min(1.0, survival_score))
                decisions.append(
                    Decision(
                        incident.incident_id,
                        ambulance.amb_id,
                        hospital.hosp_id,
                        "",
                        response_mode,
                        transport_mode,
                        response_minutes,
                        transport_minutes,
                        hospital.handover_minutes,
                        response_wait + transport_wait,
                        total_care_minutes,
                        survival_score,
                    )
                )
    return decisions


def top_candidates(decisions: List[Decision], limit: int) -> List[Decision]:
    return sorted(
        decisions,
        key=lambda d: (d.survival_score, -d.total_care_minutes, -d.wait_minutes),
        reverse=True,
    )[:limit]


def apply_decision(
    decision: Decision,
    incident: Incident,
    ambulances: List[Ambulance],
    hospitals: List[Hospital],
    decision_time: Optional[float] = None,
) -> None:
    ambulance = next(a for a in ambulances if a.amb_id == decision.ambulance_id)
    hospital = next(h for h in hospitals if h.hosp_id == decision.hospital_id)
    dispatch_minute = max(incident.call_time, ambulance.available_at)
    if decision_time is not None:
        dispatch_minute = max(dispatch_minute, decision_time)
    service_end = (
        dispatch_minute
        + decision.response_minutes
        + incident.scene_minutes
        + decision.transport_minutes
        + decision.handover_minutes
    )
    ambulance.available_at = service_end
    ambulance.current_node = hospital.node
    hospital.remaining_capacity -= 1


def lookahead_value(
    incidents: Sequence[Incident],
    ambulances: List[Ambulance],
    hospitals: List[Hospital],
    crossing: Crossing,
    depth: int,
    candidate_limit: int,
    strict_specialty_for_critical: bool,
    decision_time: Optional[float] = None,
) -> float:
    if depth <= 0 or not incidents:
        return 0.0
    incident = incidents[0]
    decisions = candidate_decisions(
        incident,
        ambulances,
        hospitals,
        crossing,
        route_modes=("wait", "detour"),
        strict_specialty_for_critical=strict_specialty_for_critical,
        decision_time=decision_time,
    )
    if not decisions:
        return 0.0
    best = 0.0
    for decision in top_candidates(decisions, candidate_limit):
        next_ambulances = deepcopy(ambulances)
        next_hospitals = deepcopy(hospitals)
        apply_decision(decision, incident, next_ambulances, next_hospitals, decision_time)
        total = decision.survival_score + lookahead_value(
            incidents[1:],
            next_ambulances,
            next_hospitals,
            crossing,
            depth - 1,
            candidate_limit,
            strict_specialty_for_critical,
            decision_time,
        )
        best = max(best, total)
    return best


def choose_decision(
    policy: str,
    incident: Incident,
    future_incidents: Sequence[Incident],
    ambulances: List[Ambulance],
    hospitals: List[Hospital],
    crossing: Crossing,
    config: ScenarioConfig,
) -> Optional[Decision]:
    route_modes = ("wait",) if policy == "nearest" else ("wait", "detour")
    decisions = candidate_decisions(
        incident,
        ambulances,
        hospitals,
        crossing,
        route_modes=route_modes,
        strict_specialty_for_critical=config.strict_specialty_for_critical,
    )
    if not decisions:
        return None
    if policy == "nearest":
        best = min(decisions, key=lambda d: (d.response_minutes, d.total_care_minutes, -d.survival_score))
    elif policy == "fastest":
        best = min(decisions, key=lambda d: (d.total_care_minutes, -d.survival_score))
    elif policy == "survival":
        best = max(decisions, key=lambda d: (d.survival_score, -d.total_care_minutes))
    elif policy == "lookahead":
        best_decision: Optional[Decision] = None
        best_value = -1.0
        for decision in top_candidates(decisions, config.candidate_limit):
            next_ambulances = deepcopy(ambulances)
            next_hospitals = deepcopy(hospitals)
            apply_decision(decision, incident, next_ambulances, next_hospitals)
            future_value = lookahead_value(
                future_incidents,
                next_ambulances,
                next_hospitals,
                crossing,
                max(0, config.lookahead_depth - 1),
                config.candidate_limit,
                config.strict_specialty_for_critical,
            )
            total_value = decision.survival_score + future_value
            if total_value > best_value:
                best_value = total_value
                best_decision = decision
        best = best_decision
    else:
        raise ValueError(f"Unknown policy: {policy}")
    if best is not None:
        best.policy = policy
    return best


def policy_prediction_mode(policy: str) -> str:
    return {
        "online_mc_perfect": "perfect",
        "online_mc_noisy": "noisy",
        "online_mc_random": "random",
    }[policy]


def sample_random_future_incidents(
    config: ScenarioConfig,
    current_time: float,
    rng: random.Random,
    profile_pool: Optional[Sequence[Dict[str, object]]] = None,
) -> List[Incident]:
    horizon = min(config.prediction_horizon_min, max(0.0, config.call_horizon_min - current_time))
    if horizon <= 0:
        return []
    baseline_rate = config.n_incidents / max(config.call_horizon_min, 1.0)
    expected_count = baseline_rate * horizon
    for center in MASS_EVENT_TIMES[: config.mass_casualty_events]:
        if current_time <= center <= current_time + horizon:
            expected_count += 0.60 * config.mass_cluster_size
    sampled_count = int(round(rng.gauss(expected_count, math.sqrt(expected_count + 1.0))))
    sampled_count = max(0, min(config.forecast_incident_limit, sampled_count))
    cluster_centers = [
        center
        for center in MASS_EVENT_TIMES[: config.mass_casualty_events]
        if current_time <= center <= current_time + horizon
    ]
    incidents: List[Incident] = []
    for index in range(sampled_count):
        if cluster_centers and rng.random() < 0.45:
            center = rng.choice(cluster_centers)
            call_time = center + rng.uniform(-config.mass_cluster_spread_min, config.mass_cluster_spread_min)
        else:
            call_time = current_time + rng.uniform(0.0, horizon)
        profile = sample_patient_profile(rng, profile_pool)
        incidents.append(
            Incident(
                -(index + 1),
                max(current_time, min(config.call_horizon_min, call_time)),
                sample_scene_node(rng, config),
                str(profile["severity"]),
                str(profile["patient_type"]),
                float(profile["base_survival"]),
                float(profile["decay_rate"]),
                float(profile["scene_minutes"]),
            )
        )
    incidents.sort(key=lambda item: (item.call_time, item.incident_id))
    return incidents


def perturb_incident_prediction(
    incident: Incident, config: ScenarioConfig, current_time: float, rng: random.Random
) -> Incident:
    predicted_call_time = incident.call_time + rng.gauss(0.0, config.prediction_noise_minutes)
    predicted_call_time = max(current_time, min(config.call_horizon_min, predicted_call_time))
    predicted_survival = max(
        0.02, min(0.99, incident.base_survival + rng.gauss(0.0, config.prediction_noise_survival))
    )
    predicted_severity = severity_from_survival(predicted_survival)
    predicted_patient_type = incident.patient_type
    if rng.random() < 0.12:
        predicted_patient_type = rng.choice(["trauma", "burn", "medical"])
    predicted_scene = incident.scene
    if rng.random() < 0.18:
        predicted_scene = sample_scene_node(rng, config)
    return Incident(
        incident.incident_id,
        predicted_call_time,
        predicted_scene,
        predicted_severity,
        predicted_patient_type,
        predicted_survival,
        SEVERITY_PROFILE[predicted_severity]["decay"],
        max(6.0, incident.scene_minutes + rng.uniform(-2.5, 2.5)),
    )


def forecast_future_incidents(
    actual_future_incidents: Sequence[Incident],
    current_time: float,
    config: ScenarioConfig,
    prediction_mode: str,
    rng: random.Random,
    profile_pool: Optional[Sequence[Dict[str, object]]] = None,
) -> List[Incident]:
    if prediction_mode == "random":
        return sample_random_future_incidents(config, current_time, rng, profile_pool)
    window_end = min(config.call_horizon_min, current_time + config.prediction_horizon_min)
    visible_future = [
        deepcopy(incident)
        for incident in actual_future_incidents
        if incident.call_time <= window_end
    ][: config.forecast_incident_limit]
    if prediction_mode == "perfect":
        return visible_future
    if prediction_mode == "noisy":
        noisy = [
            perturb_incident_prediction(incident, config, current_time, rng)
            for incident in visible_future
        ]
        noisy.sort(key=lambda item: (item.call_time, item.incident_id))
        return noisy
    raise ValueError(f"Unknown prediction mode: {prediction_mode}")


def estimate_future_value(
    pending_incidents: Sequence[Incident],
    actual_future_incidents: Sequence[Incident],
    ambulances: List[Ambulance],
    hospitals: List[Hospital],
    crossing: Crossing,
    current_time: float,
    config: ScenarioConfig,
    prediction_mode: str,
    rng: random.Random,
    profile_pool: Optional[Sequence[Dict[str, object]]] = None,
) -> float:
    rollout_count = 1 if prediction_mode == "perfect" else config.mc_rollouts
    values: List[float] = []
    for _ in range(rollout_count):
        predicted_incidents = [deepcopy(incident) for incident in pending_incidents]
        predicted_incidents.extend(
            forecast_future_incidents(actual_future_incidents, current_time, config, prediction_mode, rng, profile_pool)
        )
        predicted_incidents.sort(key=lambda item: (item.call_time, item.incident_id))
        if not predicted_incidents:
            values.append(0.0)
            continue
        values.append(
            lookahead_value(
                predicted_incidents,
                deepcopy(ambulances),
                deepcopy(hospitals),
                crossing,
                config.mc_lookahead_depth,
                config.candidate_limit,
                config.strict_specialty_for_critical,
                current_time,
            )
        )
    return sum(values) / len(values) if values else 0.0


def estimate_oracle_future_value(
    pending_incidents: Sequence[Incident],
    actual_future_incidents: Sequence[Incident],
    ambulances: List[Ambulance],
    hospitals: List[Hospital],
    crossing: Crossing,
    current_time: float,
    config: ScenarioConfig,
) -> float:
    predicted_incidents = [deepcopy(incident) for incident in pending_incidents]
    predicted_incidents.extend(deepcopy(incident) for incident in actual_future_incidents)
    predicted_incidents.sort(key=lambda item: (item.call_time, item.incident_id))
    if not predicted_incidents:
        return 0.0
    oracle_depth = min(len(predicted_incidents), max(config.lookahead_depth, config.mc_lookahead_depth, 6))
    return lookahead_value(
        predicted_incidents,
        deepcopy(ambulances),
        deepcopy(hospitals),
        crossing,
        oracle_depth,
        config.candidate_limit,
        config.strict_specialty_for_critical,
        current_time,
    )


def hold_durations(config: ScenarioConfig, current_time: float) -> List[float]:
    return [
        step * config.hold_step_minutes
        for step in range(1, max(0, config.hold_max_steps) + 1)
        if current_time + step * config.hold_step_minutes <= config.call_horizon_min
    ]


def choose_online_decision(
    policy: str,
    pending_incidents: Sequence[Incident],
    actual_future_incidents: Sequence[Incident],
    current_time: float,
    ambulances: List[Ambulance],
    hospitals: List[Hospital],
    crossing: Crossing,
    config: ScenarioConfig,
    rng: random.Random,
    profile_pool: Optional[Sequence[Dict[str, object]]] = None,
) -> ActionChoice:
    available_ambulances = [ambulance for ambulance in ambulances if ambulance.available_at <= current_time]
    if not available_ambulances or not pending_incidents:
        return ActionChoice()
    decisions: List[Decision] = []
    for incident in pending_incidents:
        decisions.extend(
            candidate_decisions(
                incident,
                available_ambulances,
                hospitals,
                crossing,
                ("wait", "detour"),
                config.strict_specialty_for_critical,
                current_time,
            )
        )
    if not decisions:
        return ActionChoice()
    if policy == "online_greedy":
        best = max(decisions, key=lambda d: (d.survival_score, -d.total_care_minutes, -d.wait_minutes))
        best.policy = policy
        return ActionChoice(decision=best)
    if policy not in ONLINE_POLICIES:
        raise ValueError(f"Unknown online policy: {policy}")
    incident_lookup = {incident.incident_id: incident for incident in pending_incidents}
    best_decision: Optional[Decision] = None
    best_value = -1.0
    for decision in top_candidates(decisions, config.candidate_limit):
        next_ambulances = deepcopy(ambulances)
        next_hospitals = deepcopy(hospitals)
        selected_incident = deepcopy(incident_lookup[decision.incident_id])
        apply_decision(decision, selected_incident, next_ambulances, next_hospitals, current_time)
        remaining_pending = [
            deepcopy(incident) for incident in pending_incidents if incident.incident_id != decision.incident_id
        ]
        if policy == "offline_oracle":
            future_value = estimate_oracle_future_value(
                remaining_pending,
                actual_future_incidents,
                next_ambulances,
                next_hospitals,
                crossing,
                current_time,
                config,
            )
        else:
            future_value = estimate_future_value(
                remaining_pending,
                actual_future_incidents,
                next_ambulances,
                next_hospitals,
                crossing,
                current_time,
                config,
                policy_prediction_mode(policy),
                rng,
                profile_pool,
            )
        total_value = decision.survival_score + future_value
        if total_value > best_value:
            best_value = total_value
            best_decision = decision
    best_action = ActionChoice()
    if best_decision is not None:
        best_decision.policy = policy
        best_action = ActionChoice(decision=best_decision)
    if not config.allow_hold:
        return best_action
    for hold_minutes in hold_durations(config, current_time):
        hold_time = current_time + hold_minutes
        if policy == "offline_oracle":
            hold_value = estimate_oracle_future_value(
                pending_incidents, actual_future_incidents, ambulances, hospitals, crossing, hold_time, config
            )
        else:
            hold_value = estimate_future_value(
                pending_incidents,
                actual_future_incidents,
                ambulances,
                hospitals,
                crossing,
                hold_time,
                config,
                policy_prediction_mode(policy),
                rng,
                profile_pool,
            )
        if hold_value > best_value:
            best_value = hold_value
            best_action = ActionChoice(hold_minutes=hold_minutes)
    return best_action


def build_world(
    config: ScenarioConfig,
    seed: int,
    profile_pool: Optional[Sequence[Dict[str, object]]] = None,
) -> Tuple[List[Incident], List[Ambulance], List[Hospital], Crossing]:
    rng = random.Random(seed)
    base_nodes = [
        Node("WestBase", 10, 28),
        Node("SouthBase", 18, 8),
        Node("NorthBase", 28, 44),
        Node("EastBase", 84, 18),
    ]
    hospitals = [
        Hospital("H1", Node("GeneralWest", 20, 34), "general", 0.97, max(4, round_int(22 * config.hospital_capacity_scale)), 11.0 * config.handover_scale),
        Hospital("H2", Node("TraumaCenter", 14, 16), "trauma", 1.16, max(4, round_int(16 * config.hospital_capacity_scale)), 15.0 * config.handover_scale),
        Hospital("H3", Node("BurnCenter", 82, 10), "burn", 1.12, max(3, round_int(10 * config.hospital_capacity_scale)), 14.0 * config.handover_scale),
        Hospital("H4", Node("EastGeneral", 76, 34), "general", 0.93, max(4, round_int(18 * config.hospital_capacity_scale)), 12.0 * config.handover_scale),
        Hospital("H5", Node("RegionalGeneral", 42, 40), "general", 1.01, max(4, round_int(14 * config.hospital_capacity_scale)), 13.0 * config.handover_scale),
    ]
    n_als = max(1, round_int(config.n_ambulances * config.als_share))
    ambulances = [
        Ambulance(f"A{index + 1}", base_nodes[index % len(base_nodes)], "ALS" if index < n_als else "BLS")
        for index in range(config.n_ambulances)
    ]
    base_windows = [(70.0, 95.0), (180.0, 205.0), (315.0, 340.0), (470.0, 495.0), (610.0, 640.0)]
    crossing = Crossing(
        50.0,
        [(start, min(config.call_horizon_min, start + (end - start) * config.closure_scale)) for start, end in base_windows],
        config.detour_penalty,
    )
    n_mass = config.mass_casualty_events * config.mass_cluster_size
    n_regular = max(0, config.n_incidents - n_mass)
    call_times = [rng.uniform(0, config.call_horizon_min) for _ in range(n_regular)]
    for index in range(config.mass_casualty_events):
        center = MASS_EVENT_TIMES[index % len(MASS_EVENT_TIMES)]
        for _ in range(config.mass_cluster_size):
            call_time = center + rng.uniform(-config.mass_cluster_spread_min, config.mass_cluster_spread_min)
            call_times.append(max(0.0, min(config.call_horizon_min, call_time)))
    call_times = sorted(call_times[: config.n_incidents])
    scenes = scene_catalog(config)
    incidents: List[Incident] = []
    for incident_id, call_time in enumerate(call_times, start=1):
        profile = sample_patient_profile(rng, profile_pool)
        scene = rng.choices(
            [item[0] for item in scenes],
            weights=[max(weight, 0.02) for _, weight in scenes],
            k=1,
        )[0]
        incidents.append(
            Incident(
                incident_id,
                call_time,
                scene,
                str(profile["severity"]),
                str(profile["patient_type"]),
                float(profile["base_survival"]),
                float(profile["decay_rate"]),
                float(profile["scene_minutes"]),
            )
        )
    return incidents, ambulances, hospitals, crossing


def round_int(value: float) -> int:
    return int(round(value))


def result_row(
    policy: str,
    decisions: Sequence[Decision],
    total_incidents: int,
    extra_metrics: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    served = len(decisions)
    total_survival = sum(item.survival_score for item in decisions)
    row = {
        "policy": policy,
        "served": float(served),
        "missed": float(total_incidents - served),
        "service_rate": served / total_incidents if total_incidents else 0.0,
        "total_survival": total_survival,
        "mean_survival": total_survival / served if served else 0.0,
        "mean_response_min": sum(item.response_minutes for item in decisions) / served if served else 0.0,
        "mean_transport_min": sum(item.transport_minutes for item in decisions) / served if served else 0.0,
        "mean_handover_min": sum(item.handover_minutes for item in decisions) / served if served else 0.0,
        "mean_total_care_min": sum(item.total_care_minutes for item in decisions) / served if served else 0.0,
        "mean_crossing_wait_min": sum(item.wait_minutes for item in decisions) / served if served else 0.0,
        "detour_share": sum(1 for item in decisions if item.transport_mode == "detour") / served if served else 0.0,
        "mean_pending_queue": 0.0,
        "max_pending_queue": 0.0,
        "stalled_epochs": 0.0,
        "hold_count": 0.0,
        "hold_minutes_total": 0.0,
        "mean_hold_minutes": 0.0,
    }
    if extra_metrics:
        row.update(extra_metrics)
    return row


def run_offline_policy(
    policy: str,
    incidents: List[Incident],
    ambulances: List[Ambulance],
    hospitals: List[Hospital],
    crossing: Crossing,
    config: ScenarioConfig,
) -> Dict[str, float]:
    state_ambulances = deepcopy(ambulances)
    state_hospitals = deepcopy(hospitals)
    decisions: List[Decision] = []
    sorted_incidents = sorted(incidents, key=lambda item: item.call_time)
    for index, incident in enumerate(sorted_incidents):
        future_incidents = sorted_incidents[index + 1 : index + config.lookahead_depth]
        decision = choose_decision(policy, incident, future_incidents, state_ambulances, state_hospitals, crossing, config)
        if decision is None:
            continue
        apply_decision(decision, incident, state_ambulances, state_hospitals)
        decisions.append(decision)
    return result_row(policy, decisions, len(sorted_incidents))


def run_online_policy(
    policy: str,
    incidents: List[Incident],
    ambulances: List[Ambulance],
    hospitals: List[Hospital],
    crossing: Crossing,
    config: ScenarioConfig,
    seed: int = 0,
    profile_pool: Optional[Sequence[Dict[str, object]]] = None,
) -> Dict[str, float]:
    state_ambulances = deepcopy(ambulances)
    state_hospitals = deepcopy(hospitals)
    sorted_incidents = sorted(incidents, key=lambda item: item.call_time)
    pending: List[Incident] = []
    decisions: List[Decision] = []
    queue_snapshots: List[int] = []
    stalled_epochs = 0
    hold_count = 0
    hold_minutes_total = 0.0
    incident_index = 0
    current_time = 0.0
    rng = random.Random(seed * 997 + sum(ord(char) for char in policy))
    while True:
        while incident_index < len(sorted_incidents) and sorted_incidents[incident_index].call_time <= current_time:
            pending.append(sorted_incidents[incident_index])
            incident_index += 1
        available_now = [ambulance for ambulance in state_ambulances if ambulance.available_at <= current_time]
        if pending and available_now:
            queue_snapshots.append(len(pending))
            action = choose_online_decision(
                policy,
                pending,
                sorted_incidents[incident_index:],
                current_time,
                state_ambulances,
                state_hospitals,
                crossing,
                config,
                rng,
                profile_pool,
            )
            if action.decision is not None:
                decision = action.decision
                incident = next(item for item in pending if item.incident_id == decision.incident_id)
                apply_decision(decision, incident, state_ambulances, state_hospitals, current_time)
                pending = [item for item in pending if item.incident_id != decision.incident_id]
                decisions.append(decision)
                continue
            if action.hold_minutes > 0:
                hold_count += 1
                hold_minutes_total += action.hold_minutes
                current_time += action.hold_minutes
                continue
            stalled_epochs += 1
        next_call = sorted_incidents[incident_index].call_time if incident_index < len(sorted_incidents) else math.inf
        next_free = min(
            (ambulance.available_at for ambulance in state_ambulances if ambulance.available_at > current_time),
            default=math.inf,
        )
        if next_call == math.inf and next_free == math.inf:
            break
        current_time = min(next_call, next_free)
    return result_row(
        policy,
        decisions,
        len(sorted_incidents),
        {
            "mean_pending_queue": sum(queue_snapshots) / len(queue_snapshots) if queue_snapshots else 0.0,
            "max_pending_queue": float(max(queue_snapshots) if queue_snapshots else 0.0),
            "stalled_epochs": float(stalled_epochs),
            "hold_count": float(hold_count),
            "hold_minutes_total": hold_minutes_total,
            "mean_hold_minutes": hold_minutes_total / hold_count if hold_count else 0.0,
        },
    )


def run_policy(
    policy: str,
    incidents: List[Incident],
    ambulances: List[Ambulance],
    hospitals: List[Hospital],
    crossing: Crossing,
    config: ScenarioConfig,
    seed: int = 0,
    profile_pool: Optional[Sequence[Dict[str, object]]] = None,
) -> Dict[str, float]:
    if policy in OFFLINE_POLICIES:
        return run_offline_policy(policy, incidents, ambulances, hospitals, crossing, config)
    if policy in ONLINE_POLICIES:
        return run_online_policy(policy, incidents, ambulances, hospitals, crossing, config, seed, profile_pool)
    raise ValueError(f"Unknown policy: {policy}")


def run_experiment_grid(
    scenarios: Iterable[ScenarioConfig],
    seeds: Iterable[int],
    policies: Sequence[str] = DEFAULT_POLICIES,
    profile_pool: Optional[Sequence[Dict[str, object]]] = None,
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for scenario in scenarios:
        for seed in seeds:
            incidents, ambulances, hospitals, crossing = build_world(scenario, seed, profile_pool)
            for policy in policies:
                result = run_policy(policy, incidents, ambulances, hospitals, crossing, scenario, seed, profile_pool)
                result["scenario"] = scenario.name
                result["seed"] = seed
                result["n_incidents"] = scenario.n_incidents
                result["n_ambulances"] = scenario.n_ambulances
                rows.append(result)
    return pd.DataFrame(rows)


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby(["scenario", "policy"], as_index=False)
        .agg(
            mean_survival_mean=("mean_survival", "mean"),
            mean_survival_std=("mean_survival", "std"),
            total_survival_mean=("total_survival", "mean"),
            service_rate_mean=("service_rate", "mean"),
            care_minutes_mean=("mean_total_care_min", "mean"),
            wait_minutes_mean=("mean_crossing_wait_min", "mean"),
            detour_share_mean=("detour_share", "mean"),
            pending_queue_mean=("mean_pending_queue", "mean"),
            stalled_epochs_mean=("stalled_epochs", "mean"),
            hold_count_mean=("hold_count", "mean"),
            hold_minutes_total_mean=("hold_minutes_total", "mean"),
            mean_hold_minutes_mean=("mean_hold_minutes", "mean"),
        )
        .fillna(0.0)
        .sort_values(["scenario", "mean_survival_mean"], ascending=[True, False])
    )


def add_gap_vs_reference(
    summary: pd.DataFrame,
    reference_policy: str,
    metric_col: str = "mean_survival_mean",
) -> pd.DataFrame:
    reference = summary.loc[summary["policy"] == reference_policy, ["scenario", metric_col]].rename(
        columns={metric_col: "reference_metric"}
    )
    comparison = summary.merge(reference, on="scenario", how="left")
    comparison["gap_to_reference"] = comparison["reference_metric"] - comparison[metric_col]
    comparison["gap_pct_of_reference"] = comparison.apply(
        lambda row: 100.0 * row["gap_to_reference"] / row["reference_metric"]
        if row["reference_metric"] > 0
        else 0.0,
        axis=1,
    )
    return comparison.sort_values(["scenario", "gap_to_reference"])


def experiment_1_oracle_gap(
    scenarios: Iterable[ScenarioConfig],
    seeds: Iterable[int],
    profile_pool: Optional[Sequence[Dict[str, object]]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    results = run_experiment_grid(
        scenarios,
        seeds,
        policies=("offline_oracle", "online_greedy", "online_mc_perfect"),
        profile_pool=profile_pool,
    )
    return results, add_gap_vs_reference(summarize_results(results), "offline_oracle")


def experiment_2_uncertainty(
    scenarios: Sequence[ScenarioConfig],
    seeds: Iterable[int],
    profile_pool: Optional[Sequence[Dict[str, object]]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results = run_experiment_grid(
        scenarios,
        seeds,
        policies=("online_greedy", "online_mc_perfect"),
        profile_pool=profile_pool,
    )
    summary = summarize_results(results)
    greedy = summary.loc[
        summary["policy"] == "online_greedy",
        ["scenario", "mean_survival_mean", "service_rate_mean"],
    ].rename(
        columns={
            "mean_survival_mean": "greedy_survival",
            "service_rate_mean": "greedy_service_rate",
        }
    )
    planned = summary.loc[
        summary["policy"] == "online_mc_perfect",
        ["scenario", "mean_survival_mean", "service_rate_mean"],
    ].rename(
        columns={
            "mean_survival_mean": "planned_survival",
            "service_rate_mean": "planned_service_rate",
        }
    )
    comparison = greedy.merge(planned, on="scenario", how="inner")
    scenario_to_regime = {
        scenario.name: "mass_casualty_or_burst" if scenario.mass_casualty_events > 0 else "stable_or_regular"
        for scenario in scenarios
    }
    comparison["flow_regime"] = comparison["scenario"].map(scenario_to_regime)
    comparison["planning_gain"] = comparison["planned_survival"] - comparison["greedy_survival"]
    comparison["planning_gain_pct"] = comparison.apply(
        lambda row: 100.0 * row["planning_gain"] / row["greedy_survival"]
        if row["greedy_survival"] > 0
        else 0.0,
        axis=1,
    )
    regime_summary = (
        comparison.groupby("flow_regime", as_index=False)
        .agg(
            scenarios=("scenario", "count"),
            planning_gain_mean=("planning_gain", "mean"),
            planning_gain_pct_mean=("planning_gain_pct", "mean"),
            greedy_service_rate_mean=("greedy_service_rate", "mean"),
            planned_service_rate_mean=("planned_service_rate", "mean"),
        )
        .sort_values("planning_gain_mean", ascending=False)
    )
    return results, comparison.sort_values("planning_gain", ascending=False), regime_summary


def experiment_3_prediction_quality(
    scenarios: Iterable[ScenarioConfig],
    seeds: Iterable[int],
    profile_pool: Optional[Sequence[Dict[str, object]]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    results = run_experiment_grid(
        scenarios,
        seeds,
        policies=("online_mc_perfect", "online_mc_noisy", "online_mc_random"),
        profile_pool=profile_pool,
    )
    return results, add_gap_vs_reference(summarize_results(results), "online_mc_perfect")


def smoke_test_scenarios() -> List[ScenarioConfig]:
    return [
        ScenarioConfig(
            name="smoke_baseline",
            n_incidents=24,
            n_ambulances=6,
            rural_bias=0.55,
            lookahead_depth=2,
            candidate_limit=3,
            prediction_horizon_min=90.0,
            forecast_incident_limit=4,
            mc_rollouts=1,
            mc_lookahead_depth=2,
        )
    ]


def default_scenarios(include_stable_low_load: bool = True) -> List[ScenarioConfig]:
    scenarios = [
        ScenarioConfig(
            name="baseline",
            n_incidents=120,
            n_ambulances=8,
            rural_bias=0.55,
            hospital_capacity_scale=1.00,
            handover_scale=1.00,
            closure_scale=1.00,
            detour_penalty=28.0,
        ),
        ScenarioConfig(
            name="rural_crossing",
            n_incidents=120,
            n_ambulances=7,
            rural_bias=0.85,
            hospital_capacity_scale=0.95,
            handover_scale=1.05,
            closure_scale=1.70,
            detour_penalty=36.0,
        ),
        ScenarioConfig(
            name="overload",
            n_incidents=180,
            n_ambulances=7,
            rural_bias=0.70,
            hospital_capacity_scale=0.90,
            handover_scale=1.15,
            closure_scale=1.30,
            detour_penalty=32.0,
            mass_casualty_events=1,
            mass_cluster_size=6,
        ),
        ScenarioConfig(
            name="mass_casualty",
            n_incidents=200,
            n_ambulances=8,
            rural_bias=0.82,
            hospital_capacity_scale=0.88,
            handover_scale=1.20,
            closure_scale=1.60,
            detour_penalty=38.0,
            mass_casualty_events=3,
            mass_cluster_size=8,
            mass_cluster_spread_min=5.0,
        ),
    ]
    if include_stable_low_load:
        scenarios.append(
            ScenarioConfig(
                name="stable_low_load",
                n_incidents=72,
                n_ambulances=10,
                rural_bias=0.50,
                hospital_capacity_scale=1.20,
                handover_scale=0.90,
                closure_scale=0.75,
                detour_penalty=24.0,
                prediction_horizon_min=120.0,
                forecast_incident_limit=5,
            )
        )
    return scenarios


def save_core_figures(results: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_results(results)

    policies = ["offline_oracle", "online_greedy", "online_mc_perfect"]
    survival = summary[summary["policy"].isin(policies)].pivot(
        index="scenario", columns="policy", values="mean_survival_mean"
    )
    survival.plot(kind="bar", figsize=(10, 5))
    plt.ylabel("Mean survival score")
    plt.title("Mean survival by scenario and policy")
    plt.tight_layout()
    plt.savefig(out_dir / "mean_survival_by_policy.png", dpi=180)
    plt.close()

    if {"online_greedy", "online_mc_perfect"}.issubset(set(summary["policy"])):
        pivot = summary[summary["policy"].isin(["online_greedy", "online_mc_perfect"])].pivot(
            index="scenario", columns="policy", values="mean_survival_mean"
        )
        if {"online_greedy", "online_mc_perfect"}.issubset(pivot.columns):
            gain_pct = 100.0 * (pivot["online_mc_perfect"] - pivot["online_greedy"]) / pivot["online_greedy"]
            gain_pct.sort_values().plot(kind="barh", figsize=(8, 4), color="#2E74B5")
            plt.axvline(0, color="black", linewidth=0.8)
            plt.xlabel("Planning gain vs greedy, %")
            plt.title("Forecast-aware planning gain")
            plt.tight_layout()
            plt.savefig(out_dir / "planning_gain_vs_greedy.png", dpi=180)
            plt.close()
