import io
import json
import threading
import time
from http.server import ThreadingHTTPServer
import urllib.request
import urllib.error
import zipfile
import pytest
import demo_server
from research import service


def test_research_http_bridge_runs_and_exports():
    server = ThreadingHTTPServer(('127.0.0.1', 0), demo_server.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    def get(path):
        with urllib.request.urlopen(base+path, timeout=20) as response:
            return response.read()
    def post(path, value):
        request = urllib.request.Request(base+path, data=json.dumps(value).encode(), headers={'Content-Type': 'application/json', 'Origin': base})
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.load(response)
    try:
        assert b'--bg:' in get('/replay')
        assert b'Preference-Conditioned DQN Research' in get('/research')
        assert isinstance(json.loads(get('/api/research/runs')), list)
        with pytest.raises(urllib.error.HTTPError) as error:
            post('/api/research/run', {'clients': 101})
        assert error.value.code == 400
        with pytest.raises(urllib.error.HTTPError):
            get('/api/research/result?id=evidence/../../config')
        job = post('/api/research/run', {'dataset': 'synthetic', 'clients': 2, 'rounds': 2, 'episodes': 2,
                                      'train_samples': 40, 'eval_samples': 40, 'seeds': [81], 'policies': ['adaptive', 'random']})
        assert job['status'] == 'running'
        # Startup state is valid even before heavy imports create the manifest.
        assert json.loads(get('/api/research/result?id='+job['id']))['manifest']['status'] in ('running', 'complete')
        deadline = time.monotonic()+30
        while service.PROCESS.poll() is None and time.monotonic() < deadline:
            time.sleep(.1)
        assert service.PROCESS.poll() == 0
        assert json.loads(get('/api/research/state'))['status'] == 'complete'
        detail = json.loads(get('/api/research/result?id='+job['id']))
        assert detail['manifest']['status'] == 'complete'
        assert len(detail['results']) == 6
        with zipfile.ZipFile(io.BytesIO(get('/api/research/export?id='+job['id']))) as archive:
            assert 'analysis.json' in archive.namelist()
            assert 'source_snapshot.zip' in archive.namelist()
        # A cancellation must persist a cancelled manifest, not a phantom running job.
        job = post('/api/research/run', {'dataset': 'synthetic', 'clients': 2, 'rounds': 50, 'episodes': 50,
                                      'train_samples': 40, 'eval_samples': 40, 'seeds': [82], 'policies': ['adaptive']})
        deadline = time.monotonic()+20
        while not (service.folder_for(job['id'])/'manifest.json').exists() and time.monotonic() < deadline:
            time.sleep(.1)
        post('/api/research/cancel', {})
        assert json.loads(get('/api/research/state'))['status'] == 'cancelled'
    finally:
        service.cancel()
        server.shutdown(); server.server_close(); thread.join(timeout=3)
