# Carbon history and replaceable measurement inputs

gb-2025-01-01.csv contains public Great Britain national half-hour carbon-intensity **actual estimates**, not device measurements. Raw API response is preserved separately. Source: https://api.carbonintensity.org.uk/intensity/2025-01-01T00:00Z/2025-01-08T00:00Z . Fields are timestamp and carbon_g_kwh. The API's inclusive boundary can include the interval starting before the requested start. No missing actual values are replaced with forecasts.

These observations drive a UK grid trace replay on virtual client profiles; they do not describe the user's location or an Indian physical experiment. The first half is used for training windows, the second for unseen evaluation. A federated round maps to one half-hour trace row as a scenario assumption; physical training time is not inferred from that mapping.

Original hardware descriptors and the user-reported 0.065 kWh / 190 min reference remain in config.json. PC1 anchor power is calculated as 20.53 W; reusing it as active FL power is a modeling assumption. Other profiles remain scenario ratios. No GPU power reading or new wall-meter observation has been collected.

Future profile JSON can be selected with Config.profiles. Nodes may supply active_power_w_measured alongside instrument provenance to override the ratio-derived active power. Supply positive speed, idle, network and grid values with explicit units and measurement scope. A measured active-power value alone does not calibrate training duration.

Optional Config.observation_history accepts a CSV with client_id (zero-based), duration_ratio, energy_ratio, ci_ratio, bandwidth_ratio: observed/nominal ratios from at least 60 preceding observations per client. The first 20 fit the mean and next 40 calibrate residual intervals. Preserve raw observations, nominal-model version, timestamps, workload, instrument scope and provenance beside that CSV. The framework hashes it but cannot authenticate whether a user-supplied record is a physical measurement. All later scheduler updates use only already-observed rounds. Without this file the initial history is explicitly simulated, not measured.
