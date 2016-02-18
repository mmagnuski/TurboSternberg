# -*- encoding: utf-8 -*-
# TurboSternberg procedure

import os
from psychopy import visual, event, core, monitors
from exputils import SternbergExperiment, Instructions

# how far is participant from the screen?
participantDistance = 60

# check correct monitor type
monitorList = monitors.getAllMonitors()
if 'BENQ-XL2411' in monitorList:
    monitor = monitors.Monitor('BENQ-XL2411', width=53., 
        distance=participantDistance)
    monitor.setSizePix([1920, 1080])
else:
    monitor = 'testMonitor'

# create temporary window
window = visual.Window(monitor=monitor, units="deg",
	fullscr=False, size=[1200,800])

exp = SternbergExperiment(window, "settings.yaml")
exp.get_subject_id()

window = visual.Window(monitor=monitor, units="deg", fullscr=True)
waitText = visual.TextStim(window, text=u'Proszę czekać...', height=2)
exp.set_window(window)
waitText.draw(); window.flip()

# set correct instruction pictures
instr_dir = os.path.join(os.getcwd(), 'instr')
instr = os.listdir(instr_dir)
if exp.resp_mapping['f']:
    del instr[2]
else:
    del instr[1]
instr = [os.path.join('instr', i) for i in instr]

# add examples to instructions
def example():
    exp.simple_trial([2,5,0,8], 5, True, get_resp=False)
    core.wait(0.5)

instr.insert(1, example)
instr = Instructions(window, instr)
instr.present(stop=4)

# training
for i in range(1, 16):
    exp.show_trial(i, feedback=True)
    if i > 1 and exp.df.loc[i, 'ifcorrect'] == 0:
        exp.show_keymap()
exp.create_trials()

instr.present(stop=5)

exp.show_all_trials()
instr.present()
