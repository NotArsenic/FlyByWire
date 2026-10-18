# Use the active interpreter if the flybywire env is active, else `conda run`.
ifeq ($(CONDA_DEFAULT_ENV),flybywire)
PY ?= python
else
PY ?= conda run --no-capture-output -n flybywire python
endif

.PHONY: env check test smoke gate1 calibrate collect cache gate2 params lint lock

env:        ## create the conda env and install everything (scripts/setup_env.sh)
	bash scripts/setup_env.sh

check:      ## preflight / environment table
	$(PY) scripts/check_env.py

test:       ## pytest (GPU / flyvis tests skip when unavailable)
	$(PY) -m pytest -q

smoke:      ## end-to-end plumbing smoke run (NOT a result)
	$(PY) scripts/smoke_end_to_end.py

gate1:      ## flyvis state-carry equivalence + throughput gate
	$(PY) scripts/gate1_flyvis_spike.py

calibrate:  ## measure T4/T5 preferred directions -> data/flyvis/direction_map.json
	$(PY) scripts/calibrate_t4t5_directions.py

collect:    ## scripted-expert trajectories -> data/expert (300 episodes, ~7 min)
	$(PY) scripts/collect_expert.py

cache:      ## T4/T5 cache from data/expert -> data/cache (~3.5 GB)
	$(PY) scripts/build_cache.py

gate2:      ## ridge T4/T5 -> yaw rate on held-out trajectories -> runs/gate2/report.json
	$(PY) scripts/gate2_decode_yaw_rate.py

params:     ## effective trainable parameter counts
	$(PY) scripts/count_params.py

lint:
	$(PY) -m ruff check flybywire scripts tests

lock:
	$(PY) -m pip freeze --exclude-editable > requirements.lock.txt
