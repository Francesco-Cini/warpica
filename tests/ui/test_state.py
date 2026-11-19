import pytest

from warpica.ui.core.state import Session, Stage


def test_a_new_session_has_nothing_and_only_the_metric_stage_is_open():
    session = Session()

    assert session.metric is None
    assert session.energy_tensor is None
    assert session.results == {}
    assert session.is_unlocked(Stage.METRIC)
    assert not session.is_unlocked(Stage.SOLVER)
    assert not session.is_unlocked(Stage.RESULTS)


def test_a_metric_opens_the_solver_stage():
    session = Session()

    session.set_metric({"name": "m"})

    assert session.is_unlocked(Stage.SOLVER)
    assert not session.is_unlocked(Stage.RESULTS)


def test_a_solved_tensor_opens_the_results_stage():
    session = Session()
    session.set_metric({"name": "m"})

    session.set_energy_tensor({"name": "t"})

    assert session.is_unlocked(Stage.RESULTS)


def test_a_tensor_needs_a_metric():
    with pytest.raises(ValueError, match="metric"):
        Session().set_energy_tensor({"name": "t"})


def test_a_result_needs_a_tensor():
    session = Session()
    session.set_metric({"name": "m"})

    with pytest.raises(ValueError, match="energy tensor"):
        session.add_result("Weak", 1)


def test_a_new_metric_discards_the_tensor_and_results():
    session = Session()
    session.set_metric({"name": "old"})
    session.set_energy_tensor({"name": "t"})
    session.add_result("Weak", 1)

    session.set_metric({"name": "new"})

    assert session.energy_tensor is None
    assert session.results == {}
    assert not session.is_unlocked(Stage.RESULTS)


def test_a_new_tensor_discards_the_results_but_keeps_the_metric():
    session = Session()
    session.set_metric({"name": "m"})
    session.set_energy_tensor({"name": "old"})
    session.add_result("Weak", 1)

    session.set_energy_tensor({"name": "new"})

    assert session.metric == {"name": "m"}
    assert session.results == {}


def test_a_result_with_the_same_name_replaces_the_old_one():
    session = Session()
    session.set_metric({"name": "m"})
    session.set_energy_tensor({"name": "t"})

    session.add_result("Weak", 1)
    session.add_result("Weak", 2)

    assert session.results == {"Weak": 2}


def test_every_change_notifies_the_subscribers():
    session = Session()
    calls = []
    session.subscribe(lambda: calls.append("changed"))

    session.set_metric({"name": "m"})
    session.set_energy_tensor({"name": "t"})
    session.add_result("Weak", 1)

    assert calls == ["changed"] * 3


def test_status_follows_the_workflow():
    session = Session()
    assert session.status(Stage.METRIC) == "not built"
    assert session.status(Stage.SOLVER) == "needs a metric"
    assert session.status(Stage.RESULTS) == "needs a solution"

    session.set_metric({"name": "m"})
    assert session.status(Stage.METRIC) == "built"
    assert session.status(Stage.SOLVER) == "not solved"

    session.set_energy_tensor({"name": "t"})
    assert session.status(Stage.SOLVER) == "solved"
    assert session.status(Stage.RESULTS) == "no results yet"

    session.add_result("Weak", 1)
    session.add_result("Null", 2)
    assert session.status(Stage.RESULTS) == "2 computed"
