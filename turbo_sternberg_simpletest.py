# Turbo sternberg
# 
# 
import os
from psychopy import visual, event, core
from exputils import (SternbergExperiment, 
    generate_load, generate_loads)

window = visual.Window(monitor="testMonitor", units="deg")
exp = SternbergExperiment(window, "parameters.yaml")

# run one trial:
exp.show_trial(0)
core.wait(0.5)

# run more:
exp.run_trials(range(1,4))

pth = r'E:\Programy\EXP\TurboSternberg\data'
exp.save_data(pth)
