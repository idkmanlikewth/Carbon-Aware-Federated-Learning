from pathlib import Path
import json
from streamlit.testing.v1 import AppTest


def test_dashboard_renders_and_switches_tabs():
    root = Path(__file__).resolve().parents[1]
    app = AppTest.from_file(str(root/"research"/"dashboard.py"), default_timeout=30).run()
    assert not app.exception
    assert app.title[0].value == "Carbon-Aware Federated Learning"
    assert app.warning
    completed = [p for p in (root/"evidence").glob("*/manifest.json") if json.loads(p.read_text())["status"] == "complete"]
    if completed:
        recorded = next(s for s in app.selectbox if s.label == "Recorded run")
        recorded.select(completed[0]).run()
        assert len(app.tabs) == 5
        assert len(app.dataframe) >= 2
        app.selectbox(key="pareto_seed").select(22).run()
        assert not app.exception


def test_dashboard_launches_actual_experiment():
    root = Path(__file__).resolve().parents[1]
    app = AppTest.from_file(str(root/"research"/"dashboard.py"), default_timeout=30).run()
    def widget(collection, label):
        return next(w for w in collection if w.label == label)
    widget(app.selectbox, "Dataset").select("synthetic")
    widget(app.slider, "Virtual clients").set_value(2)
    widget(app.number_input, "Federated rounds").set_value(2)
    widget(app.number_input, "DQN training episodes").set_value(2)
    widget(app.number_input, "Training subset").set_value(40)
    widget(app.number_input, "Validation + test subset").set_value(40)
    widget(app.text_input, "Independent seeds (space separated)").set_value("77")
    app.run()
    widget(app.button, "Run experiment").click().run()
    assert not app.exception
    assert app.session_state.process.wait(timeout=30) == 0
    path = Path(app.session_state.run_path)/"manifest.json"
    manifest = json.loads(path.read_text())
    assert manifest["status"] == "complete"
    assert manifest["config"]["dataset"] == "synthetic"
    assert manifest["config"]["clients"] == 2
    app.run()
    widget(app.selectbox, "Recorded run").select(path).run()
    assert len(app.tabs) == 5
    assert not app.exception
