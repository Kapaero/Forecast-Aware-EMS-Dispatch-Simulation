from ems_online_planning import run_experiment_grid, smoke_test_scenarios


def test_smoke_grid_runs():
    results = run_experiment_grid(
        smoke_test_scenarios(),
        seeds=range(1),
        policies=("online_greedy", "online_mc_perfect"),
    )
    assert len(results) == 2
    assert set(results["policy"]) == {"online_greedy", "online_mc_perfect"}
    assert (results["service_rate"] >= 0).all()
    assert (results["service_rate"] <= 1).all()
