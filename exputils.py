from __future__ import division, print_function

import os
import six
import yaml
import random

import numpy as np
import pandas as pd

from psychopy import visual, event, core

# TODOs:
# - [ ] add fixation cross
#
# consider:
# - [ ] what about breaks?
# - [ ] what about instructions?

class SternbergExperiment(object):

	def __init__(self, window, paramfile, frame_time=None):
		self.window = window
		self.digits = [visual.TextStim(window, text=str(x), height=10)
				  for x in range(10)]
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

		rnd = random.sample([True, False], 1)
		self.resp_mapping = {self.resp_keys[0]: rnd}
		self.resp_mapping.update({self.resp_keys[1]: not rnd})

		self.quitopt = settings['quit']
		if self.quitopt['enable']:
			self.resp_keys.append(self.quitopt['button'])

		# dataframe
		self.df = create_empty_df(len(self.loads) * self.trials_per_load)
		load_df = generate_loads(self.loads, self.trials_per_load)
		self.df.loc[:, load_df.columns] = load_df

		self.clock = core.Clock()
		self.current_trial = 0

		self.subject_id = 'test_subject'

	def run_trials(self, trials):
		for t in trials:
			self.show_trial(t)
			break_time = random.uniform(self.times['inter_trial'][0],
				self.times['inter_trial'][1]+0.0001)
			core.wait(round(break_times, 3), 1))

	def show_trial(self, trial):
		digits = list(map(int, self.df.loc[trial, 'digits'].split()))
		probe = self.df.loc[trial, 'probe']
		corr_resp = self.df.loc[trial, 'isin']

		corr, rt = self.simple_trial(digits, probe, corr_resp)

		self.df.loc[trial,'ifcorrect'] = corr
		self.df.loc[trial,'RT'] = rt

	def simple_trial(self, digits, probe, corr_resp):
		self.show_digits(digits)
		self.check_quit()
		resp = self.wait_and_ask(probe)
		self.check_quit(key=resp)

		# check response
		if len(resp) > 0:
			key, rt = resp
			corr = self.resp_mapping[key] == corr_resp
		else:
			corr = False
			rt = np.nan
		return corr, rt

	def show_digits(self, show_digits):
		for d in show_digits:
			for t in range(self.times['digit']):
				self.digits[d].draw()
				self.window.flip()

			for t in range(self.times['inter']):
				self.window.flip()

	def wait_and_ask(self, ask_digit):
		for d in range(self.times['wait']):
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

	def save_data(self, path):
		full_path = os.path.join(path, self.subject_id)
		self.df.to_csv(full_path + '.csv')
		self.df.to_excel(full_path + '.xls')


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


def create_empty_df(nrows):
	cols = ['trial', 'load', 'digits',
			'probe', 'isin', 'ifcorrect', 'RT']
	d = {'trial': list(range(1, nrows+1)),
		 'load': 3, 'digits': "1 2 3",
		 'probe': 2, 'isin': True,
		 'ifcorrect': False, 'RT': 0.52361}
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