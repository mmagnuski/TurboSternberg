# -*- encoding: utf-8 -*-
# TurboSternberg procedure

import os
from psychopy import visual, event, core, monitors
from exputils import SternbergExperiment, Instructions


# TODOS:
# - [ ] get current dir from __file__ and not os.getcwd()
# - [ ] add randomization of true_key...


# how far is participant from the screen?
scr_dist = 60

def get_screen(scr_dist=scr_dist):
    # check correct monitor type
    monitorList = monitors.getAllMonitors()
    if 'BENQ-XL2411' in monitorList:
        monitor = monitors.Monitor('BENQ-XL2411', width=53.,
            distance=scr_dist)
        monitor.setSizePix([1920, 1080])
    else:
        monitor = 'testMonitor'
    return monitor


def run(window=None, subject_id=None, true_key='f',
        scr_dist=scr_dist):

    # set path to current file location
    file_path = os.path.join(*(__file__.split('\\')[:-1]))
    file_path = file_path.replace(':', ':\\')
    os.chdir(file_path)

    # create temporary window
    monitor = get_screen(scr_dist=scr_dist)
    temp_window = visual.Window(monitor=monitor, units="deg",
        fullscr=False, size=[1200,800])

    # use temp window to init sternberg exp
    exp = SternbergExperiment(temp_window, "settings.yaml")
    exp.set_subject_id(subject_id=subject_id)

    if window is None:
        window = visual.Window(monitor=monitor, units="deg", fullscr=True)
    waitText = visual.TextStim(window, text=u'Proszę czekać...', height=2)
    exp.set_window(window)
    waitText.draw(); window.flip()

    # hide mouse
    window.mouseVisible = False

    # at least for now:
    exp.set_resp(true_key=true_key)

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

    # main exp
    exp.show_all_trials()
    instr.present()


if __name__ == '__main__':
    run()
