"""Behaviour learning: state encoding, training examples, kNN policy, evaluation.

Policy = k-nearest-neighbour over reviewed (state -> action) examples. It is
deliberately simple and *explainable*: every proposal lists the neighbouring
examples it was based on. Corrections never change the active policy silently:
training creates a `candidate` model version; the user must deploy it.
"""
from __future__ import annotations

import json
import random
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .db import new_id, now_iso

FEATURES_PER_LABEL = 3          # count, mean x, mean y
EXTRA_FEATURES = 4              # player x/y, nearest-enemy dx/dy


class StateEncoder:
    def __init__(self, labels: list[str]):
        self.labels = list(labels)

    @property
    def dim(self) -> int:
        return FEATURES_PER_LABEL * len(self.labels) + EXTRA_FEATURES

    def encode(self, scene: dict) -> list[float]:
        v: list[float] = []
        counts = scene.get("counts", {})
        mean_pos = scene.get("mean_pos", {})
        for l in self.labels:
            c = float(counts.get(l, 0))
            mp = mean_pos.get(l, [0.0, 0.0])
            v += [min(c, 10.0) / 10.0, mp[0], mp[1]]
        p = scene.get("player") or {}
        ne = scene.get("nearest_enemy") or {}
        v += [p.get("cx", 0.0), p.get("cy", 0.0), ne.get("dx", 0.0), ne.get("dy", 0.0)]
        return v


def edit_similarity(a: list[str], b: list[str]) -> float:
    """1 - normalised Levenshtein distance between two action sequences."""
    if not a and not b:
        return 1.0
    n, m = len(a), len(b)
    d = list(range(m + 1))
    for i in range(1, n + 1):
        prev, d[0] = d[0], i
        for j in range(1, m + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return 1.0 - d[m] / max(n, m)


def outcome_similarity(a: dict | None, b: dict | None) -> float:
    if not a or not b:
        return 0.0
    keys = ["enemy_delta", "player_dx", "player_dy"]
    diffs = [abs(float(a.get(k, 0)) - float(b.get(k, 0))) for k in keys]
    return float(np.exp(-sum(diffs)))


def compare_examples(a: dict, b: dict) -> dict:
    """Compare two situations: state distance, action-sequence and outcome similarity."""
    va, vb = np.array(a["state_features"]), np.array(b["state_features"])
    dist = float(np.linalg.norm(va - vb) / np.sqrt(len(va)))
    seq = edit_similarity(a.get("context_actions", []), b.get("context_actions", []))
    return {"state_distance": dist, "sequence_similarity": seq,
            "outcome_similarity": outcome_similarity(a.get("outcome"), b.get("outcome")),
            "same_action": a["action"]["control"] == b["action"]["control"]}


# --------------------------------------------------------------------------
class OutcomeTracker:
    """Resolves the `outcome` of an example by comparing the scene N seconds later."""

    def __init__(self, db, window_sec):
        self.db = db
        self.window = window_sec
        self.pending: list[tuple] = []        # (example_id, ts, scene)
        self._lock = threading.Lock()

    def add(self, example_id, ts, scene):
        with self._lock:
            self.pending.append((example_id, ts, scene))

    def update(self, ts, scene):
        with self._lock:
            ready = [p for p in self.pending if ts - p[1] >= self.window]
            self.pending = [p for p in self.pending if ts - p[1] < self.window]
        for ex_id, _t, before in ready:
            b_p, a_p = before.get("player"), scene.get("player")
            outcome = {
                "enemy_delta": scene["counts"].get("enemy", 0) - before["counts"].get("enemy", 0),
                "item_delta": scene["counts"].get("item", 0) - before["counts"].get("item", 0),
                "player_dx": (a_p["cx"] - b_p["cx"]) if a_p and b_p else 0.0,
                "player_dy": (a_p["cy"] - b_p["cy"]) if a_p and b_p else 0.0,
                "scene_changed": before.get("scene_type") != scene.get("scene_type"),
            }
            try:
                self.db.update("training_examples", ex_id, {"outcome": outcome})
            except Exception:
                pass


# --------------------------------------------------------------------------
class PolicyModel:
    def __init__(self, data: dict):
        self.meta = data["meta"]
        self.labels = data["labels"]
        self.X = np.array(data["X"], dtype=float)
        self.mean = np.array(data["mean"], dtype=float)
        self.std = np.array(data["std"], dtype=float)
        self.controls = data["controls"]
        self.durations = data["durations"]
        self.contexts = data["contexts"]
        self.ids = data["ids"]
        self.k = int(data.get("k", 5))
        self.encoder = StateEncoder(self.labels)
        self.Z = (self.X - self.mean) / self.std if len(self.X) else self.X

    def predict(self, scene: dict, recent_actions: list[str], seq_weight=0.2) -> dict | None:
        if not len(self.X):
            return None
        q = (np.array(self.encoder.encode(scene)) - self.mean) / self.std
        d = np.linalg.norm(self.Z - q, axis=1) / np.sqrt(len(q))
        seq = np.array([1.0 - edit_similarity(recent_actions, c) for c in self.contexts])
        total = (1 - seq_weight) * d + seq_weight * seq
        idx = np.argsort(total)[: self.k]
        votes: dict[str, float] = defaultdict(float)
        for i in idx:
            votes[self.controls[i]] += float(np.exp(-total[i] * 3))
        best = max(votes, key=votes.get)
        share = votes[best] / (sum(votes.values()) or 1)
        win = [i for i in idx if self.controls[i] == best]
        sim = float(np.mean([np.exp(-total[i] * 3) for i in win]))
        dur = float(np.median([self.durations[i] for i in win]))
        return {
            "control": best, "duration": round(dur, 3), "confidence": round(share * sim, 3),
            "input_device": "mouse" if best.startswith("mouse_") else
                            "gamepad" if best.startswith("gamepad_") else "keyboard",
            "explanation": [{"example_id": self.ids[i], "control": self.controls[i],
                             "distance": round(float(total[i]), 3),
                             "state_distance": round(float(d[i]), 3),
                             "sequence_distance": round(float(seq[i]), 3)} for i in idx],
        }


def load_model(path: str) -> PolicyModel | None:
    try:
        return PolicyModel(json.loads(Path(path).read_text(encoding="utf-8")))
    except Exception:
        return None


def _usable(ex: dict) -> bool:
    return (ex.get("review_status") == "approved"
            and ex.get("kind", "demonstration") in ("demonstration", "correction")
            and ex.get("action", {}).get("control")
            and len(ex.get("state_features", [])) > 0)


def train_candidate(db, settings, log, profile: dict, k: int = 5) -> dict:
    """Train a candidate model on APPROVED examples; evaluate on held-out sessions."""
    labels = profile["detector"]["labels"]
    dim = StateEncoder(labels).dim
    exs = [e for e in db.find("training_examples", {"profile_id": profile["_id"]})
           if _usable(e) and len(e["state_features"]) == dim]
    if len(exs) < 10:
        raise ValueError(f"Need at least 10 approved examples (have {len(exs)}). "
                         "Record gameplay with 'Learn' on, then review/approve examples.")
    by_session = defaultdict(list)
    for e in exs:
        by_session[e.get("source_session", "?")].append(e)
    sessions = sorted(by_session)
    rng = random.Random(42)
    if len(sessions) >= 2:
        held = {sessions[-1]}
        test = [e for e in exs if e.get("source_session") in held]
        train = [e for e in exs if e.get("source_session") not in held]
        split = f"held-out session {sessions[-1][:8]}"
    else:
        shuffled = exs[:]
        rng.shuffle(shuffled)
        cut = max(1, int(len(shuffled) * 0.8))
        train, test = shuffled[:cut], shuffled[cut:]
        split = "random 80/20 (single session - optimistic estimate)"

    def build(rows, k_):
        X = np.array([r["state_features"] for r in rows], dtype=float)
        mean, std = X.mean(0), X.std(0)
        std[std < 1e-6] = 1.0
        return {"labels": labels, "k": k_, "X": X.tolist(), "mean": mean.tolist(),
                "std": std.tolist(), "controls": [r["action"]["control"] for r in rows],
                "durations": [float(r["action"].get("duration", 0.05)) for r in rows],
                "contexts": [r.get("context_actions", []) for r in rows],
                "ids": [r["_id"] for r in rows]}

    metrics = {"split": split, "n_train": len(train), "n_test": len(test)}
    if test:
        m = PolicyModel({"meta": {}, **build(train, k)})
        ok, fails, low = 0, Counter(), 0
        for r in test:
            # score directly from the stored feature vector
            q = (np.array(r["state_features"]) - m.mean) / m.std
            d = np.linalg.norm(m.Z - q, axis=1) / np.sqrt(len(q))
            seq = np.array([1.0 - edit_similarity(r.get("context_actions", []), c)
                            for c in m.contexts])
            tot = 0.8 * d + 0.2 * seq
            idx = np.argsort(tot)[:k]
            votes = defaultdict(float)
            for i in idx:
                votes[m.controls[i]] += float(np.exp(-tot[i] * 3))
            guess = max(votes, key=votes.get)
            truth = r["action"]["control"]
            if guess == truth:
                ok += 1
            else:
                fails[f"{truth}->{guess}"] += 1
            if votes[guess] / sum(votes.values()) < 0.5:
                low += 1
        metrics.update(top1_accuracy=round(ok / len(test), 3), low_confidence_cases=low,
                       failure_categories=dict(fails.most_common(8)))
    else:
        metrics["top1_accuracy"] = None

    final = build(exs, k)                         # final model uses ALL approved data
    version = 1 + max([m["version"] for m in db.find("models", {"profile_id": profile["_id"]})]
                      or [0])
    model_id = new_id()
    models_dir = settings.path("models_dir")
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / f"policy_{profile['_id'][:8]}_v{version}_{model_id[:6]}.json"
    meta = {"model_id": model_id, "version": version, "profile_id": profile["_id"]}
    path.write_text(json.dumps({"meta": meta, **final}), encoding="utf-8")
    doc = {"_id": model_id, "profile_id": profile["_id"], "version": version,
           "task": "action_policy_knn", "status": "candidate", "created": now_iso(),
           "training_metadata": {"n_examples": len(exs), "sessions": len(sessions), "k": k,
                                 "labels": labels},
           "evaluation_metrics": metrics, "file_ref": str(path)}
    db.insert("models", doc)
    log.info("learning", f"Candidate model v{version} trained (NOT deployed)",
             event_id="MODEL_TRAINED", metrics=metrics)
    return doc


def deploy_model(db, log, profile: dict, model_id: str) -> dict:
    """Explicit, user-approved deployment of a model version to a profile."""
    model = db.get("models", model_id)
    if not model or model["profile_id"] != profile["_id"]:
        raise ValueError("Model does not belong to this profile")
    for m in db.find("models", {"profile_id": profile["_id"], "status": "deployed"}):
        db.update("models", m["_id"], {"status": "retired"})
    db.update("models", model_id, {"status": "deployed", "deployed_at": now_iso()})
    db.update("profiles", profile["_id"],
              {"model_version": model["version"], "active_model_id": model_id,
               "updated": now_iso()})
    log.info("learning", f"Model v{model['version']} deployed to profile '{profile['name']}'",
             event_id="MODEL_DEPLOYED")
    return db.get("profiles", profile["_id"])
