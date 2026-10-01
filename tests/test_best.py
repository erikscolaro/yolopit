"""best.pt of the search: highest fitness within target * (1 + margin), latest epoch otherwise
(no model needed)."""
from types import SimpleNamespace

from yolopit.search import PITSearchTrainer, within_budget

INFO = {"target": {"ops": {"target_total": 100.0}, "params": {"target_total": 10.0}},
        "margin": 0.05}


def test_within_budget():
    assert within_budget({"ops": 105.0, "params": 10.5}, INFO) is True
    assert within_budget({"ops": 105.1, "params": 10.0}, INFO) is False
    assert within_budget({"ops": 90.0, "params": 11.0}, INFO) is False
    assert within_budget({"ops": 105.0, "params": 10.5}, {**INFO, "margin": 0.0}) is False
    assert within_budget({"ops": 1e9}, {"mode": "standard"}) is None


def run(epochs):
    """epochs: (fitness, ok) per epoch -> epochs saved to best.pt, and best_epoch at the end."""
    t = SimpleNamespace(best_within_fitness=None, best_epoch=None)
    saved = []
    for e, (fitness, ok) in enumerate(epochs):
        t.epoch, t.fitness = e, fitness
        if PITSearchTrainer._update_best(t, ok):
            saved.append(e)
    return saved, t.best_epoch


def test_best_follows_latest_until_within_budget():
    saved, best = run([(0.8, False), (0.7, False), (0.5, True), (0.6, True), (0.4, True),
                       (0.9, False)])
    assert saved == [0, 1, 2, 3] and best == 3


def test_best_is_latest_when_never_within_budget():
    saved, best = run([(0.8, False), (0.7, False), (0.6, False)])
    assert saved == [0, 1, 2] and best == 2


def test_best_without_fitness_is_latest_within_budget():
    saved, best = run([(None, False), (None, True), (None, True), (None, False)])
    assert saved == [0, 1, 2] and best == 2


def test_best_is_latest_without_targets():
    saved, best = run([(0.8, None), (0.5, None), (0.9, None)])
    assert saved == [0, 1, 2] and best == 2
