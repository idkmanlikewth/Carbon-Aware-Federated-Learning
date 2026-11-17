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
