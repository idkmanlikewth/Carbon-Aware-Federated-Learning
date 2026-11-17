"""Meter-anchored scenario model with causal online uncertainty calibration.

Initial historical observations are SIMULATED prior samples, not physical probes.
Prediction intervals use split-conformal absolute relative residuals. Coverage is
empirically audited on selected evaluation clients, not guaranteed under shift.
"""
from collections import deque
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def cvar(values, alpha=.9):
    a=np.asarray(values,float)
    if not a.size or not np.isfinite(a).all() or not 0<alpha<1:
        raise ValueError('CVaR requires finite nonempty samples and alpha in (0,1)')
    # Fractional upper-tail mass handles discrete samples and ties exactly.
    tail=np.sort(a)[::-1]; mass=(1-alpha)*len(a); n=int(np.floor(mass))
    return float((tail[:n].sum()+(mass-n)*tail[min(n,len(tail)-1)])/mass)


class CostHistory:
    def __init__(self, histories, coverage=.9):
        # Histories shape: client x observation x [duration, Wh, CI, bandwidth].
        self.means=histories[:,:20].mean(1)
        self.residuals=[[deque(np.abs(histories[i,20:,j]-self.means[i,j]),maxlen=200) for j in range(4)] for i in range(len(histories))]
        self.ratios=[[deque(histories[i,:,j],maxlen=200) for j in range(4)] for i in range(len(histories))]
        self.coverage=coverage
        self.available_success=np.full(len(histories),19.)
        self.available_trials=np.full(len(histories),20.)

    def interval(self, nominal):
        mean=nominal*self.means
        q=np.zeros_like(mean)
        for i in range(len(mean)):
            for j in range(4):
                residual=np.asarray(self.residuals[i][j]); rank=min(len(residual),int(np.ceil((len(residual)+1)*self.coverage)))
                q[i,j]=np.sort(residual)[rank-1]*nominal[i,j]
        return mean,np.maximum(0,mean-q),mean+q

    def observe(self, nominal, actual, selected, availability):
        self.available_success += availability
        self.available_trials += 1
        for i in selected:
            for j in range(4):
                ratio=actual[i,j]/max(nominal[i,j],1e-12)
                self.residuals[i][j].append(abs(ratio-self.means[i,j]))
                self.ratios[i][j].append(ratio)
                self.means[i,j]=np.mean(self.ratios[i][j])

    def availability_probability(self):
        return (self.available_success+1)/(self.available_trials+2)


class Infrastructure:
    def __init__(self,c,data,seed,evaluation=False,model_bytes=100000):
        self.c=c; self.data=data; self.rng=np.random.default_rng(seed)
        cfg_path=Path(c.profiles)
        profile=json.loads(cfg_path.read_text(encoding='utf-8'))
        self.profile_provenance=profile
        self.profile_sha256=hashlib.sha256(cfg_path.read_bytes()).hexdigest()
        measurement=profile['actual_measurement']
        anchor=measurement['meter_energy_kwh']*1000/(measurement['meter_duration_minutes']/60)
        nodes=profile['nodes']; n=c.clients
        self.nodes=[nodes[i%len(nodes)] for i in range(n)]
        speed=np.array([p['relative_speed'] for p in self.nodes])
        power=np.array([p.get('active_power_w_measured',p['active_power_ratio_to_PC1']*anchor) for p in self.nodes])
        self.speed=1+c.heterogeneity*(speed-1)
        self.power=np.maximum(1,power.mean()+c.heterogeneity*(power-power.mean()))
        self.idle=np.minimum(self.power,np.array([p['idle_power_w_assumed'] for p in self.nodes]))
        self.network=np.array([p['network_seconds_assumed'] for p in self.nodes])
        self.model_bytes=model_bytes
        self.base_ci=np.array([p.get('grid_kgco2_per_kwh',.675)*1000 for p in self.nodes])
        self.phase=self.rng.uniform(0,2*np.pi,n)
        self.trace=None; self.trace_provenance=None
        if c.historical_trace:
            path=Path(c.historical_trace)
            rows=list(csv.DictReader(path.open(encoding='utf-8')))
            if len(rows)<2*(c.rounds+1) or any('carbon_g_kwh' not in r or 'timestamp' not in r for r in rows):
                raise ValueError('Historical trace requires timestamp/carbon_g_kwh and disjoint train/eval windows')
            values=np.array([float(r['carbon_g_kwh']) for r in rows])
            if not np.isfinite(values).all() or (values<0).any():raise ValueError('Invalid historical carbon trace')
            offset=len(rows)//2 if evaluation else 0
            self.trace=values[offset:offset+c.rounds+1]
            self.trace_provenance={'file':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'window_start':rows[offset]['timestamp'],'evaluation':evaluation}
        self.shift=c.scenario_shift if evaluation else 1.
        # Draw all exogenous outcomes first; policy cannot alter trace RNG draws.
        sigma=c.uncertainty*self.shift
        self.noise=self.rng.lognormal(-.5*sigma*sigma,sigma,size=(c.rounds+1,n,4))
        self.available=self.rng.random((c.rounds+1,n)) < max(.05,min(1,c.availability/(self.shift**.25)))
        self.available[:,0]=True
        history=self.rng.lognormal(-.5*c.uncertainty**2,c.uncertainty,size=(n,60,4))
        self.observation_provenance={'source':'SIMULATED_PRIOR_HISTORY','physical_measurements':False}
        if c.observation_history:
            path=Path(c.observation_history)
            rows=list(csv.DictReader(path.open(encoding='utf-8-sig')))
            fields=('duration_ratio','energy_ratio','ci_ratio','bandwidth_ratio')
            for i in range(n):
                selected=[r for r in rows if int(r['client_id'])==i]
                if len(selected)<60:raise ValueError('Observation history needs at least 60 preceding observations per client')
                values=np.array([[float(r[k]) for k in fields] for r in selected[:60]])
                if not np.isfinite(values).all() or (values<=0).any():raise ValueError('Invalid observation ratios')
                history[i]=values
            self.observation_provenance={'source':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                                         'physical_measurements':'USER-SUPPLIED: independently verify instrument and collection provenance'}
        self.history=CostHistory(history)
        self.history.available_success[:]=c.availability*20
        self.coverage_hits=np.zeros(4); self.coverage_count=0
        self.last_loss=np.full(n,2.3); self.last_seen=np.zeros(n)

    def nominal(self,r):
        n=self.c.clients
        sizes=np.array([len(y) for _,y in self.data.clients])
        # Training times are explicit scenario scaling, not measured CPU timings.
        duration=(2+sizes*self.c.local_epochs*.03)/self.speed
        bandwidth=np.maximum(.1,2*self.model_bytes*8/(np.maximum(self.network,.01)*1e6))
        ci=self.base_ci*(1+.3*np.sin(r*.35+self.phase)) if self.c.dynamic_carbon else self.base_ci.copy()
        if self.trace is not None:
            ci=self.trace[r]*self.base_ci/max(self.base_ci.mean(),1)
        return np.stack([duration,self.power*duration/3600,ci,bandwidth],axis=1)

    def predictions(self,r,uncertainty=True):
        nominal=self.nominal(r)
        if uncertainty:
            return self.history.interval(nominal)
        return nominal,nominal,nominal

    def cost(self,r,selected,inference_s=0.):
        nominal=self.nominal(r); actual=nominal*self.noise[r]
        mean,lo,hi=self.history.interval(nominal)
        ids=np.array(selected,dtype=int)
        # Network transfer time depends on uncertain bandwidth. Active energy
        # has independent modeled uncertainty around the anchored profile.
        transfer=2*self.model_bytes*8/(actual[:,3]*1e6)
        latency=float((actual[ids,0]+transfer[ids]).max()+.45+inference_s)
        wh=self.idle*latency/3600
        wh[ids]+=np.maximum(0,actual[ids,1]-self.idle[ids]*actual[ids,0]/3600)+self.power[ids]*transfer[ids]/3600
        coordinator_wh=30*latency/3600
        carbon=float(np.dot(wh,actual[:,2])/1000+coordinator_wh*400/1000)
        hits=(actual[ids]>=lo[ids])&(actual[ids]<=hi[ids])
        self.coverage_hits += hits.sum(0); self.coverage_count += len(ids)
        self.history.observe(nominal,actual,ids,self.available[r])
        return {'latency_s':latency,'energy_wh':float(wh.sum()+coordinator_wh),'carbon_g':carbon,
                'coverage_hits':hits.sum(0).tolist(),'coverage_n':len(ids), 'predicted_interval_width_mean':(hi[ids]-lo[ids]).mean(0).tolist(),
                'selected_intensity_g_kwh':actual[ids,2].tolist(),'observed_bandwidth_mbps':actual[ids,3].tolist()}

    def tail_estimates(self,r):
        # Empirical joint residual draws preserve observed marginal histories;
        # cross-client dependence is not identified and is an explicit limitation.
        nominal=self.nominal(r); n=self.c.clients
        duration=np.zeros(n); carbon=np.zeros(n)
        for i in range(n):
            ratios=np.array([list(self.history.ratios[i][j]) for j in range(4)]).T
            duration[i]=cvar(nominal[i,0]*ratios[:,0]+2*self.model_bytes*8/(nominal[i,3]*ratios[:,3]*1e6),self.c.risk_alpha)
            carbon[i]=cvar(nominal[i,1]*ratios[:,1]*nominal[i,2]*ratios[:,2]/1000,self.c.risk_alpha)
        return duration,carbon
