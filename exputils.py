# -*- encoding: utf-8 -*-
from __future__ import division, print_function

import os
import six
import yaml
import random

import numpy as np
import pandas as pd

from psychopy import visual, event, core, gui

# TODOs:
# - [ ] add fixation cross
#
# consider:
# - [ ] what about breaks?
# - [ ] what about instructions?

class SternbergExperiment(object):

	def __init__(self, window, paramfile, frame_time=None):
		self.window = window
		if frame_time == None:
			self.frame_time = get_frame_time(window)
		else:
			self.frame_time = frame_time

		file_name = os.path.join(os.getcwd(), paramfile)
		with open(file_name, 'r') as f:
		    settings = yaml.load(f)

		self.loads = settings['loads']
		self.trials_per_load = settings['trials_per_load']

		self.resp_keys = settings['resp_keys']
		self.times = s2frames(settings['times'], self.frame_time)
		self.times['inter_trial'] = settings['times']['inter_trial']
		self.settings = settings

		rnd = random.sample([True, False], 1)[0]
		self.resp_mapping = {self.resp_keys[0]: rnd}
		self.resp_mapping.update({self.resp_keys[1]: not rnd})

		self.quitopt = settings['quit']
		if self.quitopt['enable']:
			self.resp_keys.append(self.quitopt['button'])

		# dataframe
		self.create_trials()

		self.clock = core.Clock()
		self.current_trial = 0

		self.subject = dict()
		self.subject['id'] = 'test_subject'
		self.create_stimuli()
		self.num_trials = self.df.shape[0]

	def create_trials(self):
		self.df = create_empty_df(len(self.loads) * self.trials_per_load)
		load_df = generate_loads(self.loads, self.trials_per_load)
		self.df.loc[:, load_df.columns] = load_df
		self.df = self.df.set_index('trial', drop=True)

	def create_stimuli(self):
		# create some stimuli
		self.fix = fix(self.window, **self.settings['fixation'])
		self.digits = [visual.TextStim(self.window, text=str(x),
			height=self.settings['digits']['height']) for x in range(10)]
		self.stim = dict()
		feedback_colors = (np.array([[0,147,68], [190, 30, 45]], dtype='float') / 255 - 0.5) * 2
		self.stim['feedback_correct'] = fix(self.window, height=self.settings['feedback_circle_radius'],
			color=feedback_colors[0,:])
		self.stim['feedback_incorrect'] = fix(self.window, height=self.settings['feedback_circle_radius'],
			color=feedback_colors[1,:])

	def set_window(self, window):
		self.window = window
		self.fix.win = window
		for d in self.digits:
			d.win = window
		for st in self.stim.values():
			st.win = window

	def get_random_time(self, time, key):
		if time == None:
			time = random.randint(*self.times[key])
		return time

	def show_all_trials(self):
			trials_without_break = 0
			self.show_keymap()
			for t in range(1, self.num_trials+1):
				self.show_trial(t)
				self.save_data()
				trials_without_break += 1
				if trials_without_break >= self.settings['break_every_trials']:
					trials_without_break = 0
					self.present_break()
					self.show_keymap()
			core.quit()

	def run_trials(self, trials):
		for t in trials:
			self.show_trial(t)
			self.window.flip()
			break_time = random.uniform(self.times['inter_trial'][0],
				self.times['inter_trial'][1]+0.0001)
			core.wait(round(break_time, 3))

	def show_trial(self, trial):
		digits = list(map(int, self.df.loc[trial, 'digits'].split()))
		probe = self.df.loc[trial, 'probe']
		corr_resp = self.df.loc[trial, 'isin']

		# random times
		fix_time, wait_time = [self.get_random_time(None, k)
			for k in ['fix', 'wait']]

		corr, rt = self.simple_trial(digits, probe, corr_resp,
			fix_time=fix_time, wait_time=wait_time)

		self.df.loc[trial, 'fixTime'] = fix_time
		self.df.loc[trial, 'waitTime'] = wait_time
		self.df.loc[trial,'ifcorrect'] = int(corr)
		self.df.loc[trial,'RT'] = rt

	def simple_trial(self, digits, probe, corr_resp,
		fix_time=None, wait_time=None):

		self.show_fix(fix_time=fix_time)
		self.show_digits(digits)
		self.check_quit()
		resp = self.wait_and_ask(probe, wait_time=wait_time)
		self.check_quit(key=resp)

		# check response
		if len(resp) > 0:
			key, rt = resp
			corr = self.resp_mapping[key] == corr_resp
		else:

	def show_feedback(self, corr):
		corr = int(corr)
		stims = ['feedback_incorrect', 'feedback_correct']
		fdb = [self.stim[s] for s in stims][corr]
		fdb.draw()
		self.window.flip()
		core.wait(0.7)

	def show_fix(self, fix_time=None):
		fix_time = self.get_random_time(fix_time, 'fix')
		if isinstance(self.fix, list):
			for t in range(fix_time):
				for el in self.fix:
					el.draw()
				self.window.flip()
		else:
			for t in range(fix_time):
				self.fix.draw()
				self.window.flip()

	def show_digits(self, show_digits):
		for d in show_digits:
			for t in range(self.times['digit']):
				self.digits[d].draw()
				self.window.flip()

			for t in range(self.times['inter']):
				self.window.flip()

	def wait_and_ask(self, ask_digit, wait_time=None):
		wait_time = self.get_random_time(wait_time, 'wait')

		if self.settings['fixation_during_wait']:
			for d in range(wait_time):
				self.fix.draw()
				self.window.flip()
		else:
			for d in range(wait_time):
				self.window.flip()

		ask_digit = self.digits[ask_digit]
		ask_digit.color = "yellow"
		ask_digit.draw()
		self.window.flip()
		self.clock.reset()

		resp = event.waitKeys(maxWait=self.times['response'],
							  keyList=self.resp_keys,
					  		  timeStamped=self.clock)
		resp = resp[0] if resp is not None else resp
		ask_digit.color = "white"
		return resp

	def check_quit(self, key=None):
		if self.quitopt['enable']:
			if key == None:
				key = event.getKeys()
			if key == None or len(key) == 0:
				return
			if isinstance(key[0], tuple):
				key = [k[0] for k in key]
			if isinstance(key, tuple):
				key, _ = key
			if self.quitopt['button'] in key:
				core.quit()

	def save_data(self):
		full_path = os.path.join('data', self.subject['id'])
		self.df.to_csv(full_path + '.csv')
		self.df.to_excel(full_path + '.xls')

	def present_break(self):
		text = self.settings['tekst_przerwy']
		text = text.replace('\\n', '\n')
		text = visual.TextStim(self.window, text=text)
		k = False
		while not k:
			text.draw()
			self.window.flip()
			k = event.getKeys()
			self.check_quit(key=k)

	def show_keymap(self):
		args = {'units': 'deg'}
		show_map = {k: bool_to_pl(v)
			for k, v in six.iteritems(self.resp_mapping)}
		text = u'Odpowiadasz klawiszami:\nf: {}\nj: {}'.format(
			show_map['f'], show_map['j'])
		stim = visual.TextStim(self.window, text=text, **args)
		stim.draw()
		self.window.flip()
		k = event.waitKeys()
		self.check_quit(key=k)

	def get_subject_id(self):
		myDlg = gui.Dlg(title="Subject Info", size = (800,600))
		myDlg.addText('Informacje o osobie badanej')
		myDlg.addField('ID:')
		myDlg.addField('wiek:', 30)
		myDlg.addField(u'płeć:', choices=[u'kobieta', u'mężczyzna'])
		myDlg.show()  # show dialog and wait for OK or Cancel

		if myDlg.OK:  # Ok was pressed
			self.subject['id'] = myDlg.data[0]
			self.subject['age'] = myDlg.data[1]
			self.subject['sex'] = myDlg.data[2]
		else:
			core.quit()


# stimuli
# -------
def fix(win, height=0.3, width=0.1, shape='circle', color=(0.5, 0.5, 0.5)):
	args = {'fillColor': color, 'lineColor': color,
		'interpolate': True, 'units': 'deg'}
	if shape == 'circle':
		fix_stim = visual.Circle(win, radius=height/2,
			edges=32, **args)
	else:
		h, w = (height/2, width/2)
		vert = np.array([[w, -h], [w, h], [-w, h], [-w, -h]])

		args.update(closeShape=True)
		fix_stim = [visual.ShapeStim(win, vertices=v, **args)
					for v in [vert, np.fliplr(vert)]]
	return fix_stim

def get_frame_time(win, frames=25):
	frame_rate = win.getActualFrameRate(nIdentical = frames)
	return 1.0 / frame_rate

def s2frames(time_in_seconds, frame_time):
	assert isinstance(time_in_seconds, dict)
	time_in_frames = dict()
	toframes = lambda x: int(round(x / frame_time))
	for k, v in six.iteritems(time_in_seconds):
		if isinstance(v, list):
			time_in_frames[k] = map(toframes, v)
		else:
			time_in_frames[k] = toframes(v)
	return time_in_frames


# creating trials
# ---------------
def create_empty_df(nrows):
	cols = ['trial', 'load', 'digits',
			'probe', 'isin', 'ifcorrect', 'RT']
	d = {'trial': list(range(1, nrows+1)),
		 'load': 3, 'digits': "1 2 3",
		 'probe': 2, 'isin': True,
		 'ifcorrect': 0, 'RT': 0.52361}
	df = pd.DataFrame(d)
	return df[cols]


def generate_load(load, numtrials):
	# all trials
	rpt = max(int(np.ceil(numtrials / load)), 2)
	rpt = rpt + rpt % 2 # must not be odd
	tri = rpt * load

	digits = np.array([random.sample(range(10), load)
					   for x in range(tri)])
	isin = np.tile([0, 1], [tri/2, 1]).transpose().ravel()
	probe_ind = np.tile(np.arange(load), [1, rpt]).ravel()

	probe = list()
	for i in range(tri):
		if isin[i]:
			probe.append(digits[i, probe_ind[i]])
		else:
			oth = other_digits(digits[i, :])
			probe.append(random.sample(oth, 1)[0])

	str_digits = list()
	for row in digits:
		str_digits.append(str(row)[1:-1])

	df = pd.DataFrame({'digits': str_digits, 'isin':isin, 'probe': probe})
	return shuffle_df(df, cutto=numtrials)


def generate_loads(loads, trials_per_load):
	alldf = pd.concat([generate_load(l, trials_per_load) for l in loads])
	alldf = alldf.reset_index(drop=True)
	return shuffle_df(alldf)


def shuffle_df(df, cutto=None):
	ind = np.array(df.index)
	np.random.shuffle(ind)
	cutto = len(df) if cutto is None else cutto
	df = df.iloc[ind[:cutto], :]
	df = df.reset_index(drop=True)
	return df


def other_digits(digits):
	return [x for x in range(10) if x not in digits]


def bool_to_pl(b):
	assert isinstance(b, bool)
	if b:
		return u'prawda'
	else:
		return u'fałsz'


class Instructions:
	def __init__(self, win, instrfiles):
		self.win = win
		self.nextpage   = 0
		self.navigation = {'left': 'prev', 'right': 'next',
			'space': 'next'}

		# get instructions from file:
		self.stop_at_page = len(self.pages)
		self.imagefiles = instrfiles
		self.images = []
		self.generate_images()

	def generate_images(self):
		self.images = [visual.ImageStim(self.win, image=f,
			units='pix', interpolate=True) for f in self.imagefiles]

	def present(self, start=None, stop=None):
		if not isinstance(start, int):
			start = self.nextpage
		if not isinstance(stop, int):
			stop = len(self.pages)

		# show pages:
		self.nextpage = start
		while self.nextpage < stop:
			# create page elements
			self.create_page()

			for it in self.pageitems:
				it.draw()
			self.win.flip()

			# wait for response
			k = event.waitKeys(keyList=self.navigation.keys())[0]
			action = self.navigation[k]

			# go next/prev according to the response
			if action == 'next':
				self.nextpage += 1
			else:
				self.nextpage = max(0, self.nextpage - 1)

	def create_page(self, page_num=None):
		if not isinstance(page_num, int):
			page_num = self.nextpage
		self.pageitems = [self.imagefiles[page_num]]