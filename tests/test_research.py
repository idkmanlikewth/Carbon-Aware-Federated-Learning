from dataclasses import replace
import itertools
import json
import numpy as np
import pytest
import torch
from research.agent import Agent, QNetwork, adapt_preference
from research.analysis import hypervolume, interval, paired_comparisons, pareto_mask
from research.config import Config
from research.data import load_data
from research.environment import Environment, normalized_reward
from research.run import run
from research.verify import verify


@pytest.fixture
def config():
    torch.set_num_threads(1)
    return Config(dataset="synthetic", clients=5, rounds=2, episodes=2, train_samples=50,
                  eval_samples=40, seeds=(1, 2), replay_batch=2, target_interval=1)


def test_hypervolume_against_inclusion_exclusion():
    points = np.array([[.7, -.2, -.5], [.5, -.1, -.3], [.8, -.7, -.2]])
    ref = np.array([0, -1, -1])
    expected = 0
    for n in range(1, 4):
        for subset in itertools.combinations(points, n):
            expected += (-1)**(n+1) * np.prod(np.min(subset, axis=0)-ref)
    assert hypervolume(points) == pytest.approx(expected)
    assert hypervolume([[.5, -.5, -.5]]) == pytest.approx(.125)
    assert hypervolume([[.5, -2, -.5]]) == 0
    assert hypervolume([]) == 0
    assert pareto_mask([[1, 2], [2, 2], [2, 2]]).tolist() == [False, True, True]
    with pytest.raises(ValueError):
        hypervolume([[float("nan"), 1, 2]])


def test_bounds_and_seed_validation(config):
    for update in ({"clients": 101}, {"seeds": (1, 1)}, {"alpha": float("nan")}, {"preferences": ((1., 1., 1., 1.),)}):
        with pytest.raises(ValueError):
            replace(config, **update)
    v = normalized_reward(.5, 999, 999, 1, config)
    np.testing.assert_array_equal(v, [1, -1, -1, -1])
    p = adapt_preference([.25]*4, 0., 1.)
    assert p.sum() == pytest.approx(1)
    assert p[0] > .25


def test_partition_and_hundred_clients(config):
    c = replace(config, clients=100, train_samples=200)
    data = load_data(c, 1)
    assert sum(len(y) for _, y in data.clients) == 200
    assert all(len(y) for _, y in data.clients)
    env = Environment(c, data, 7)
    assert env.observation().shape == (10,)
    _, _, done, info = env.step(2)
    assert set(info["selected"]) <= set(info["available"])
    assert len(info["selected"]) <= 20
    assert info["carbon_g"] > 0
    assert not done


def test_paired_traces_and_no_test_leakage(config):
    data = load_data(config, 1)
    a, b = Environment(config, data, 500), Environment(config, data, 500)
    a.step(0); b.step(2)
    np.testing.assert_array_equal(a.intensities, b.intensities)
    np.testing.assert_array_equal(a.available, b.available)
    # Changing held-out test labels must not affect training rewards or actions.
    altered = replace(data, test=(data.test[0], (data.test[1]+1) % 10))
    a, b = Environment(config, data, 500), Environment(config, altered, 500)
    for _ in range(config.rounds):
        _, reward_a, _, _ = a.step(5)
        _, reward_b, _, _ = b.step(5)
        np.testing.assert_array_equal(reward_a, reward_b)
    assert not np.array_equal(a.intensities[0], a.intensities[1])


def test_double_dqn_update_and_terminal_mask(config):
    agent = Agent(config, 3)
    p, s = np.array([.25]*4, dtype=np.float32), np.zeros(10, dtype=np.float32)
    before = [v.detach().clone() for v in agent.online.parameters()]
    for _ in range(3):
        agent.learn(s, p, 0, np.array([1., 0, 0, 0]), s, p, True)
    assert agent.steps == 2
    assert np.isfinite(agent.losses).all()
    assert any(not torch.equal(a, b) for a, b in zip(before, agent.online.parameters()))
    assert all(torch.equal(a, b) for a, b in zip(agent.target.parameters(), agent.online.parameters()))
    assert len(agent.replay) == 3


def test_preference_is_network_input():
    net = QNetwork()
    state = torch.zeros(10)
    assert not torch.equal(net(state, torch.tensor([1., 0, 0, 0])), net(state, torch.tensor([0., 0, 1., 0])))


def test_statistics_seed_unit():
    rows = [{"seed": s, "policy": p, "hypervolume": v} for s, a, b in [(1, .3, .1), (2, .4, .3), (3, .5, .1)] for p, v in [("adaptive", a), ("random", b)]]
    result = paired_comparisons(rows)[0]
    assert result["difference"]["n"] == 3
    assert result["p_holm"] >= result["p_raw"]
    assert interval([1])["ci95"] is None


def test_end_to_end_independent_and_reproducible(config, tmp_path):
    c = replace(config, policies=("adaptive", "random"))
    a, analysis = run(c, tmp_path/"one")
    b, _ = run(c, tmp_path/"two")
    assert a == b
    manifest = json.loads((tmp_path/"one"/"manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert len(manifest["training"]) == 2
    assert all(t["updates"] > 0 for t in manifest["training"])
    assert manifest["datasets"]["1"]["subset_sha256"] != manifest["datasets"]["2"]["subset_sha256"]
    assert analysis["summary"]["adaptive"]["n"] == 2
    assert len(manifest["artifact_sha256"]) >= 5
    assert verify(tmp_path/"one")["status"] == "verified"
    results_path = tmp_path/"one"/"results.json"
    original = results_path.read_bytes()
    results_path.write_bytes(original+b" ")
    with pytest.raises(ValueError, match="checksum"):
        verify(tmp_path/"one")
    with pytest.raises(FileExistsError):
        run(c, tmp_path/"one")


def test_real_dataset_missing_fails(config, tmp_path):
    c = replace(config, dataset="mnist", data_dir=str(tmp_path))
    with pytest.raises(RuntimeError, match="No synthetic substitution"):
        load_data(c, 1)
