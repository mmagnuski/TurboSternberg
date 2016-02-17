# Turbo sternberg
# 
import os
from psychopy import visual, event, core, monitors
from exputils import (SternbergExperiment, 
    generate_load, generate_loads)

# monitorList = monitors.getAllMonitors()
# ind = 'BENQ-XL2411' in monitorList:
# monitorName = ['testMonitor', 'BENQ-XL2411'][ind]

participantDistance = 60
monitor = monitors.Monitor('BENQ-XL2411', width=53., 
	distance=participantDistance)
monitor.setSizePix([1920, 1080])

window = visual.Window(monitor='testMonitor', units="deg",
	fullscr=False, size=[1200, 800])

exp = SternbergExperiment(window, "parameters.yaml")
exp.get_subject_id()
exp.show_all_trials()
