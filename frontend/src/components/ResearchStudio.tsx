import { useEffect, useState } from 'react';

type Run = { id: string; status: string; schema_version?: number; config: { dataset: string; clients: number; seeds: number[]; rounds: number } };
type Result = { seed: number; policy: string; preference_index: number; test_accuracy: number; carbon_g: number; latency_s: number; energy_wh: number; gini: number; validation_accuracy?: number; target_hit?: boolean; time_to_target_s?: number | null; carbon_to_target_g?: number | null; accuracy_constraint_violations?: number; carbon_budget_violations?: number; tail_latency_cvar?: number; tail_carbon_cvar?: number; inference_wall_s?: number; policy_training_wall_s?: number; final_accuracy_feasible?: boolean; interval_coverage?: number[] };
type Round = { seed: number; policy: string; preference_index: number; round: number; selected: number[]; available: number[]; intensity_g_kwh: number[]; validation_accuracy: number; carbon_g: number; latency_s: number; preference: number[] };
type Episode = {episode: number; validation_accuracy: number; validation_loss: number; return: number; dqn_loss_mean: number | null};
type Detail = {
  manifest: { status: string; schema_version?: number; device?: string; config: Run['config']; measurement_type: string; datasets: Record<string, { source: string }>; training: { updates: number; seed?: number; policy?: string; episodes?: Episode[] }[] };
  results?: Result[]; trajectories?: Round[]; log: string;
  analysis?: { summary: Record<string, { n: number; mean: number; ci95: number[] | null }>; paired_tests: { baseline: string; p_holm: number | null; difference: { mean: number } }[]; inference: string };
};
const POLICIES = ['adaptive', 'fixed', 'unconditioned', 'static_carbon', 'raw_reward', 'random', 'carbon', 'fair'];
const CARMA_POLICIES = ['fedavg', 'random', 'round_robin', 'fastest', 'carbon_greedy', 'weighted', 'dqn', 'pc_ddqn', 'oort_style', 'carma', 'no_uncertainty', 'no_risk', 'no_adaptation', 'no_constraint', 'no_conditioning', 'no_learned_selection'];
const COLORS = ['#C82D14', '#1D4ED8', '#0F766E', '#B45309', '#8752a1', '#57534A', '#679151', '#b86642'];
const color = (p: string) => COLORS[Math.max(0, [...CARMA_POLICIES, ...POLICIES].indexOf(p)) % COLORS.length];
async function request<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(`/api/research/${path}`, body === undefined ? {} : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || 'Research request failed');
  return value;
}

export function ResearchStudio({ offline }: { offline: boolean }) {
  const [engine, setEngine] = useState('carma');
  const [partition, setPartition] = useState('dirichlet');
  const [alpha, setAlpha] = useState(.5);
  const [target, setTarget] = useState(.9);
  const [carbonBudget, setCarbonBudget] = useState(20);
  const [campaign, setCampaign] = useState<{ status: string; workers?: number; jobs: {name: string; status: string}[] }>({status:'not_started', jobs:[]});
  const [runs, setRuns] = useState<Run[]>([]);
  const [selected, setSelected] = useState('');
  const [data, setData] = useState<Detail | null>(null);
  const [error, setError] = useState('');
  const [active, setActive] = useState<{ id: string | null; status: string }>({ id: null, status: 'idle' });
  const [dataset, setDataset] = useState('mnist');
  const [clients, setClients] = useState(10);
  const [rounds, setRounds] = useState(15);
  const [episodes, setEpisodes] = useState(8);
  const [samples, setSamples] = useState(6000);
  const [evalSamples, setEvalSamples] = useState(1000);
  const [seeds, setSeeds] = useState('101 102 103 104 105 106 107 108 109 110');
  const [download, setDownload] = useState(false);
  const [policies, setPolicies] = useState(['fedavg', 'random', 'pc_ddqn', 'carma']);
  const [policy, setPolicy] = useState('adaptive');
  const [seed, setSeed] = useState(11);
  const [pref, setPref] = useState(0);
  const [round, setRound] = useState(1);
  const [cost, setCost] = useState<'carbon_g' | 'latency_s'>('carbon_g');
  const [learningMetric, setLearningMetric] = useState<keyof Omit<Episode,'episode'>>('validation_accuracy');

  useEffect(() => {
    if (offline) return;
    let alive = true;
    async function refresh() {
      try {
        const [list, status, study] = await Promise.all([request<Run[]>('runs'), request<typeof active>('state'), request<typeof campaign>('campaign')]);
        if (!alive) return;
        setRuns(list); setActive(status); setCampaign(study);
        setSelected(previous => previous || list.find(r => r.status === 'complete' && r.config.dataset !== 'synthetic')?.id || list[0]?.id || '');
      } catch (e) { if (alive) setError(e instanceof Error ? e.message : 'Research service unavailable'); }
    }
    void refresh();
    const timer = window.setInterval(refresh, 2500);
    return () => { alive = false; window.clearInterval(timer); };
  }, [offline]);

  useEffect(() => {
    if (!selected || offline) return;
    let alive = true;
    let initialized = false;
    async function load() {
      try {
        const result = await request<Detail>(`result?id=${encodeURIComponent(selected)}`);
        if (!alive) return;
        setData(result);
        if (!initialized) {
          setSeed(result.manifest.config.seeds[0]); setRound(1); setPref(0);
          setPolicy(result.results?.[0]?.policy || 'adaptive'); initialized = true;
        }
      } catch (e) { if (alive) setError(e instanceof Error ? e.message : 'Unable to load run'); }
    }
    setData(null); void load();
    const timer = window.setInterval(load, 2500);
    return () => { alive = false; window.clearInterval(timer); };
  }, [selected, offline]);

  async function run() {
    setError('');
    try {
      const seedValues = seeds.trim().split(/\s+/).map(Number);
      if (!seedValues.length || seedValues.some(s => !Number.isInteger(s) || s < 0)) throw new Error('Enter whole-number seeds separated by spaces.');
      const job = await request<{ id: string; status: string }>('run', { engine, dataset, clients, rounds, episodes, train_samples: samples, eval_samples: evalSamples, seeds: seedValues, policies, download, ...(engine === 'carma' ? { partition, alpha, accuracy_target: target, carbon_budget_g: carbonBudget } : {}) });
      setActive(job);
      setTimeout(() => setSelected(job.id), 1000);
    } catch (e) { setError(e instanceof Error ? e.message : 'Experiment failed'); }
  }
  const rows = data?.results || [];
  useEffect(() => {
    if (data?.results?.length) setPolicy(previous => data.results!.some(r => r.policy === previous) ? previous : data.results![0].policy);
  }, [data]);
  const points = rows.filter(r => r.seed === seed);
  const trace = data?.trajectories?.find(r => r.seed === seed && r.policy === policy && r.preference_index === pref && r.round === round);
  const maxCost = Math.max(...points.map(r => r[cost]), 0.001) * 1.08;
  const pnames = [...new Set(rows.map(r => r.policy))];
  const prefs = [...new Set(rows.filter(r => r.policy === policy && r.seed === seed).map(r => r.preference_index))];
  const v2 = data?.manifest.schema_version === 2;
  const learning = data?.manifest.training.find(t => t.seed === seed && t.policy === policy);
  const learningValues = (learning?.episodes || []).map(e => e[learningMetric]).filter((n): n is number => n !== null);
  const learningMin = Math.min(...learningValues), learningMax = Math.max(...learningValues);
  const fmt = (n: number | null | undefined) => n == null ? 'not reached' : n.toFixed(3);

  return <div className="research-studio">
    <div className="panel">
      <div className="panel-head"><div className="head-inline"><h2>CARMA-FL Research</h2><span className="plate-tag num">CANDIDATE METHOD · INDEPENDENT SEEDS</span></div>
        <p className="panel-sub">Real classifier training on benchmark subsets. Energy, carbon intensity and infrastructure latency are simulated scenarios; no physical measurements are collected.</p></div>
      <div className="research-controls">
        <label>Research engine<select aria-label="Research engine" value={engine} onChange={e => { setEngine(e.target.value); setPolicies(e.target.value === 'carma' ? ['fedavg','random','pc_ddqn','carma'] : POLICIES); }}><option value="carma">CARMA-FL · convergence-aware</option><option value="legacy">Earlier DQN study · archived protocol</option></select></label>
        <label>Dataset<select aria-label="Research dataset" value={dataset} onChange={e => setDataset(e.target.value)}>{['mnist', 'fashion-mnist', 'cifar10', 'synthetic'].map(v => <option key={v}>{v}</option>)}</select></label>
        {([['Virtual clients', clients, setClients, 2, 100], ['Federated rounds', rounds, setRounds, 1, 200], ['DQN episodes', episodes, setEpisodes, 1, 500], ['Train samples', samples, setSamples, clients, 50000], ['Validation + test', evalSamples, setEvalSamples, 20, 10000]] as const).map(([label, value, setter, min, max]) => <label key={label}>{label}<input type="number" aria-label={label} value={value} min={min} max={max} onChange={e => setter(Number(e.target.value))} /></label>)}
        <label>Independent seeds<input aria-label="Research seeds" value={seeds} onChange={e => setSeeds(e.target.value)} /></label>
      </div>
      {engine === 'carma' && <div className="research-controls"><label>Partition<select value={partition} onChange={e => setPartition(e.target.value)}><option value="iid">IID</option><option value="dirichlet">Dirichlet non-IID</option></select></label><label>Alpha<input type="number" min="0.01" step="0.1" value={alpha} onChange={e => setAlpha(Number(e.target.value))} /></label><label>Validation accuracy constraint<input type="number" min="0.1" max="0.99" step="0.01" value={target} onChange={e => setTarget(Number(e.target.value))} /></label><label>Modeled carbon budget (g)<input type="number" min="0.01" value={carbonBudget} onChange={e => setCarbonBudget(Number(e.target.value))} /></label></div>}
      <div className="research-options">{(engine === 'carma' ? CARMA_POLICIES : POLICIES).map(p => <label key={p}><input type="checkbox" checked={policies.includes(p)} onChange={e => setPolicies(old => e.target.checked ? [...old, p] : old.filter(v => v !== p))} />{p.replaceAll('_', ' ')}</label>)}</div>
      <div className="research-actions"><label><input type="checkbox" checked={download} onChange={e => setDownload(e.target.checked)} /> Download official datasets if missing</label>
        <button className="btn-run" disabled={offline || active.status === 'running' || policies.length === 0} onClick={run}>▷ Run Research Experiment</button>
        {active.status === 'running' && <button className="btn-tune" onClick={async () => { try { await request('cancel', {}); setActive({ ...active, status: 'cancelled' }); } catch (e) { setError(String(e)); } }}>Stop experiment</button>}
        <span className="num">{active.status.toUpperCase()}</span></div>
      {error && <p className="error" role="alert">{error}</p>}
      {offline && <p className="panel-sub">Research execution requires the local Python server. Start START_DEMO.bat or run python demo_server.py.</p>}
      {campaign.jobs.length > 0 && <details><summary>GPU campaign: {campaign.jobs.filter(j => j.status === 'complete').length} / {campaign.jobs.length} configurations complete · {campaign.status}</summary><p className="panel-sub">Shared-GPU wall times depend on contention. Pending studies are not results.</p><div className="research-options">{campaign.jobs.map(j => <button className="btn-tune" key={j.name} disabled={j.status === 'pending'} onClick={() => setSelected(`carma_runs/ccgrid2027/${j.name}`)}>{j.name} · {j.status}</button>)}</div></details>}
    </div>
    <div className="panel">
      <div className="panel-head"><div className="head-inline"><h2>Recorded Experiments & Pareto Analysis</h2><span className="plate-tag num">AUDITABLE EVIDENCE</span></div></div>
      <div className="research-controls"><label>Recorded run<select aria-label="Recorded research run" value={selected} onChange={e => { setError(''); setSelected(e.target.value); }}>{runs.map(r => <option key={r.id} value={r.id}>{r.id} · {r.config.clients} clients · {r.status}</option>)}</select></label>
        <label>Seed<select aria-label="Research result seed" value={seed} onChange={e => setSeed(Number(e.target.value))}>{data?.manifest.config.seeds.map(s => <option key={s} value={s} disabled={data.manifest.status !== 'complete' && !rows.some(r => r.seed === s)}>{s}{data.manifest.status !== 'complete' && !rows.some(r => r.seed === s) ? ' (pending)' : ''}</option>)}</select></label>
        <label>Trade-off axis<select value={cost} onChange={e => setCost(e.target.value as typeof cost)}><option value="carbon_g">Modeled carbon (g)</option><option value="latency_s">Modeled latency (s)</option></select></label>
        {selected && data?.manifest.status === 'complete' && <a className="btn-tune" href={`/api/research/export?id=${encodeURIComponent(selected)}`}>↓ Export evidence ZIP</a>}</div>
      {data && <p className="panel-sub">{data.manifest.config.dataset} · {data.manifest.config.clients} virtual clients · {new Set(rows.map(r => r.seed)).size} / {data.manifest.config.seeds.length} seeds with completed evaluations · {data.manifest.status} · {data.manifest.device || 'CPU'} · {rows.length} completed policy evaluations {data.manifest.status !== 'complete' ? '(interim records; inference waits for completion)' : ''}</p>}
      {!runs.length && <p className="panel-sub">No recorded experiments yet. Run an experiment to populate this panel.</p>}
      {!!points.length && <svg className="research-chart" viewBox="0 0 720 290" role="img" aria-label="Research accuracy and cost trade-off">
        {[0, .25, .5, .75, 1].map(v => <g key={v}><line x1="60" x2="690" y1={240-v*200} y2={240-v*200} stroke="var(--line)" /><text x="48" y={244-v*200} textAnchor="end">{(v*100).toFixed(0)}%</text></g>)}
        {[0, .25, .5, .75, 1].map(v => <text key={v} x={60+v*630} y="260" textAnchor="middle">{(v*maxCost).toFixed(2)}</text>)}
        <text x="65" y="20">HELD-OUT TEST ACCURACY</text><text x="370" y="285" textAnchor="middle">{cost === 'carbon_g' ? 'MODELED CARBON (g CO₂e)' : 'MODELED LATENCY (s)'}</text>
        {points.map(r => <circle key={`${r.policy}-${r.preference_index}`} cx={60+r[cost]/maxCost*630} cy={240-r.test_accuracy*200} r="5" fill={r.final_accuracy_feasible === false ? 'none' : color(r.policy)} stroke={color(r.policy)}><title>{r.policy} · preference {r.preference_index} · {(r.test_accuracy*100).toFixed(2)}% · {r[cost].toFixed(4)} · feasible: {r.final_accuracy_feasible ?? 'legacy protocol'}</title></circle>)}
      </svg>}
      <div className="research-options">{pnames.map(p => <span key={p} style={{ color: color(p) }}>● {p}</span>)}</div>
      {v2 && <><p className="panel-sub">Hollow points fail the final validation-accuracy constraint and contribute zero feasible hypervolume. A lower carbon cost with failed accuracy is not counted as an improvement. Oort-style is a simplified inspired baseline; CARMA-FL novelty is unproven.</p>
        <div className="pm-table-wrap"><table className="pm-table"><thead><tr><th>Policy · seed {seed}</th><th>Test accuracy</th><th>Validation</th><th>Time to target s</th><th>Carbon to target g</th><th>Tail latency CVaR</th><th>Tail carbon CVaR</th><th>Accuracy violations</th><th>Carbon violations</th><th>Policy train s</th><th>Inference s</th><th>Gini</th></tr></thead><tbody>{points.map(r => <tr key={r.policy}><td>{r.policy}</td><td className="num">{(r.test_accuracy*100).toFixed(2)}%</td><td className="num">{((r.validation_accuracy || 0)*100).toFixed(2)}%</td><td className="num">{fmt(r.time_to_target_s)}</td><td className="num">{fmt(r.carbon_to_target_g)}</td><td className="num">{fmt(r.tail_latency_cvar)}</td><td className="num">{fmt(r.tail_carbon_cvar)}</td><td className="num">{r.accuracy_constraint_violations}</td><td className="num">{r.carbon_budget_violations}</td><td className="num">{fmt(r.policy_training_wall_s)}</td><td className="num">{fmt(r.inference_wall_s)}</td><td className="num">{fmt(r.gini)}</td></tr>)}</tbody></table></div></>}
      {data?.analysis && <><div className="pm-table-wrap"><table className="pm-table"><thead><tr><th>Policy / ablation</th><th>Independent seeds</th><th>Mean hypervolume</th><th>95% confidence interval</th></tr></thead><tbody>{Object.entries(data.analysis.summary).map(([p, s]) => <tr key={p}><td>{p}</td><td className="num">{s.n}</td><td className="num">{s.mean.toFixed(6)}</td><td className="num">{s.ci95 ? s.ci95.map(v => v.toFixed(6)).join(' — ') : 'Not estimated: one seed'}</td></tr>)}</tbody></table></div>
        <p className="panel-sub">{data.analysis.inference} {v2 ? 'Hypervolume uses normalized latency and carbon with fixed reference (−1, −1), subject to the validation-accuracy constraint.' : 'Hypervolume includes accuracy, normalized latency and normalized carbon, with fixed reference (0, −1, −1).'} The chart shows held-out test accuracy for context.</p>
        <details><summary>Paired statistical tests & provenance</summary><pre className="research-log">{JSON.stringify({ tests: data.analysis.paired_tests, manifest: data.manifest }, null, 2)}</pre></details></>}
      {data?.log && <details open={data.manifest.status !== 'complete'}><summary>Execution log</summary><pre className="research-log">{data.log}</pre></details>}
    </div>
    {!!rows.length && <div className="panel"><div className="panel-head"><div className="head-inline"><h2>Dynamic Client Participation & Round Replay</h2><span className="plate-tag num">VIRTUAL CLIENTS</span></div></div>
      <div className="research-controls"><label>Scheduling policy<select aria-label="Research replay policy" value={policy} onChange={e => setPolicy(e.target.value)}>{pnames.map(p => <option key={p}>{p}</option>)}</select></label>
        <label>Preference<select value={pref} onChange={e => setPref(Number(e.target.value))}>{prefs.map(p => <option key={p} value={p}>{['Accuracy-oriented', 'Balanced', 'Carbon-oriented'][p] || p}</option>)}</select></label>
        <label>Round {round} / {data?.manifest.config.rounds}<input aria-label="Research replay round" type="range" min="1" max={data?.manifest.config.rounds || 1} value={round} onChange={e => setRound(Number(e.target.value))} /></label></div>
      {v2 && learning && <details><summary>Policy learning curves · {learning.updates} optimizer updates</summary><select aria-label="Policy convergence metric" value={learningMetric} onChange={e => setLearningMetric(e.target.value as typeof learningMetric)}>{['validation_accuracy','validation_loss','return','dqn_loss_mean'].map(v => <option key={v}>{v}</option>)}</select>{learningValues.length > 0 && <svg className="research-chart" viewBox="0 0 720 180" role="img" aria-label="Policy training episode curve"><text x="30" y="20">{learningMetric}: {learningMin.toFixed(4)} — {learningMax.toFixed(4)}</text><polyline fill="none" stroke="var(--accent)" strokeWidth="2" points={learningValues.map((v,i) => `${30+i/Math.max(1,learningValues.length-1)*660},${150-(v-learningMin)/Math.max(.001,learningMax-learningMin)*110}`).join(' ')} /><text x="30" y="174">EPISODE 1</text><text x="610" y="174">EPISODE {learningValues.length}</text></svg>}<p className="panel-sub">Learning curves are diagnostics, not proof that the policy has converged. Each evaluation seed has its own freshly trained policy.</p></details>}
      {trace && <><div className="rkpi-grid">{[['Validation accuracy', `${(trace.validation_accuracy*100).toFixed(2)}%`], ['Modeled round latency', `${trace.latency_s.toFixed(3)} s`], ['Modeled round carbon', `${trace.carbon_g.toFixed(4)} g`], ['Selected / available', `${trace.selected.length} / ${trace.available.length}`]].map(([label, value]) => <div className="rkpi-box" key={label}><span>{label}</span><b className="num">{value}</b></div>)}</div>
        <div className="research-client-grid">{trace.intensity_g_kwh.map((ci, i) => <div key={i} className={`research-client ${trace.selected.includes(i) ? 'selected' : ''}`} style={{ opacity: trace.available.includes(i) ? 1 : .4 }}><strong>Client {i+1}</strong><span className="num">{ci.toFixed(0)} g/kWh</span><small>{trace.selected.includes(i) ? 'SELECTED' : trace.available.includes(i) ? 'STANDBY' : 'OFFLINE'}</small></div>)}</div>
        <p className="panel-sub">Observed preference {v2 ? '[latency, carbon, constraint urgency]' : '[accuracy, latency, carbon, fairness]'}: {trace.preference.map(v => v.toFixed(3)).join(' · ')}. All infrastructure values are modeled.</p></>}
    </div>}
  </div>;
}
