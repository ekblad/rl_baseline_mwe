
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from numba import jit
from gym import spaces

# based on example at: https://github.com/jdherman/ptreeopt/blob/master/examples/folsom/folsom.py

@jit(nopython=True)
def water_day(d):
	return np.where(d >= 274, d - 274, d + 91)

@jit(nopython=True)
def max_release(S):
	# rule from http://www.usbr.gov/mp/cvp//cvp-cas/docs/Draft_Findings/130814_tech_memo_flood_control_purpose_hydrology_methods_results.pdf
	storage =  np.array([90, 100, 400, 600, 975])
	# make the last one 130 for future runs
	release = 2.29568411 * 10**-5 * 86400 / 1000 * np.array([0, 35000, 40000, 115000, 130000])
	return np.interp(S, storage, release)

@jit(nopython=True)
def tocs(d):
	# d must be water-year date
	# TAF of flood capacity in upstream reservoirs. simplified version.
	# approximate values of the curve here:
	# http://www.hec.usace.army.mil/publications/ResearchDocuments/RD-48.pdf
	tp = np.array([0, 50, 151, 200, 243, 366])
	sp = np.array([975, 400, 400, 750, 975, 975])
	return np.interp(d, tp, sp)

@jit(nopython=True)
def volume_to_height(S):  # from HOBBES data
	sp = np.array([0, 48, 93, 142, 192, 240, 288, 386, 678, 977])
	ep = np.array([210, 305, 332, 351, 365, 376, 385, 401, 437, 466])
	return np.interp(S, sp, ep)

class FolsomEnv():
	"""
	Environment for the Folsom Reservoir model compatible with OpenAI gym, baselines RL implementations, and extensions (stable-baselines)
	Accepts image data as input, daily time step.
	"""
	# metadata = {'render.modes': ['console']} # for rendering observations/data as print output, image, video, etc.
	# reward_range = (-float('inf'), float('inf')) # not used 
	# constants
	cfs_to_taf = 2.29568411 * 10**-5 * 86400 / 1000
	taf_to_cfs = 1000 / 86400 * 43560

	def __init__(self, DATA_DIR, obs_dim = None, res_dim = None, forecast = False, stack = 0, inflow_stack = 0, epi_length = 1000, ens='r1i1p1',): 
		super(FolsomEnv,self).__init__()
		
		self.DATA_DIR = DATA_DIR
		self.ens = ens
		self.forecast = forecast
		self.obs_dim = obs_dim
		self.res_dim = res_dim
		self.stack = stack
		self.inflow_stack = inflow_stack
		self.cfs_to_taf = 2.29568411 * 10**-5 * 86400 / 1000
		self.taf_to_cfs = 1000 / 86400 * 43560
		self.epi_length = epi_length

		# initialize reservoir model
		self.K = 975  # capacity, TAF
		self.Q_max = self.K # inflow max bound, TAF

		self.turbine_elev = 134  # feet
		self.turbine_max_release = 8600  # cfs
		self.max_safe_release = 130000  # cfs
		self.use_tocs = False # for now
		self.fol_mm_to_taf = 0.00328084 * 1184 # 0.003 ft/mm * 1184 thousand-acres in Folsom watershed
		self.FLOWS_DIR = DATA_DIR / 'folsom' / 'data' # for this repo	
		self.Q_df = pd.read_csv(self.FLOWS_DIR  / '{}_fol_inflows.csv'.format(self.ens),header=0,index_col=0,parse_dates=[0],infer_datetime_format=True)*self.fol_mm_to_taf
		self.Q_df = self.Q_df * self.fol_mm_to_taf
		# Define action and observation space
		# They must be gym.spaces objects
		self.action_space = spaces.Box(low=0., high=self.max_safe_release*self.cfs_to_taf, shape=(1,),dtype=float) # instead of self.max_safe_release*self.cfs_to_taf
		self.reservoir_space = spaces.Box(low=np.array([0., 0.,]+[0. for i in range(self.inflow_stack)]), 
			high=np.array([self.K,365]+[self.K for i in range(self.inflow_stack)]),shape=self.res_dim,dtype=np.float32)		
		if self.obs_dim is not None:
			# Example for using image as input:			
			self.observation_space = spaces.Box(low=0, high=255,shape=self.obs_dim,dtype=np.uint8)
		else:
			self.observation_space = self.reservoir_space

	def reset(self, agent_type='planner', model='canesm2', ens='r1i1p1', epi_count=0, epi_start=0):
		"""
		Important: the observation must be a numpy array
		:return: (np.array) 
		"""

		self.agent_type = agent_type
		self.ens = ens
		self.model = model

		self.epi_count = epi_count
		self.t_start = epi_start
		self.MODEL_DIR = self.DATA_DIR / self.model
		# read in inflows:
		# self.FLOWS_DIR = DATA_DIR / (self.model + '_meteo') # in general case
		self.FLOWS_DIR = self.DATA_DIR / 'folsom' / 'data' # for this repo		
		self.Q_df = pd.read_csv(self.FLOWS_DIR  / '{}_fol_inflows.csv'.format(self.ens) ,header=0,index_col=0,parse_dates=[0],infer_datetime_format=True)*self.fol_mm_to_taf
		self.Q = self.Q_df['inflow'].values
		self.time_vector = self.Q_df.index.values
		self.dowy_vect = np.array([water_day(d) for d in self.Q_df.index.dayofyear])
		self.D = np.loadtxt(self.FLOWS_DIR / 'demand.txt')[self.dowy_vect] # in memory
		self.T = len(self.Q_df.index)

		# reinit. reservoir model
		self.runup_days = max(self.stack,self.inflow_stack)
		self.t = self.t_start # self.runup_days # reset to beginning of episode
		self.doy = (self.t+1) % 365
		self.dowy = self.dowy_vect[self.doy-1]
		self.S, self.R, self.target, self.shortage_cost, self.flood_cost, self.noise, self.rewards = [np.zeros(self.T) for _ in range(7)]
		self.R[self.t] = self.D[self.t]
		self.S[self.t] = 500.
		self.start_date = self.time_vector[self.t]

		# generate first observation:
		if self.agent_type == 'planner':
			if self.epi_count == 0:
				print('Training start date and ensemble member: ',self.start_date,self.ens)
				print('Observation : ',self.observation_space.shape)
				if self.forecast:
					print('Forecast: (add description later)')
			self.observation = np.array([self.S[self.t],float(self.doy),]+list(self.Q_future_numpy[self.t,:]))
		elif self.agent_type == 'baseline' or self.agent_type == 'spatial_climate':
			self.observation = np.array([self.S[self.t],float(self.doy),])
			self.data = xr.open_mfdataset(self.MODEL_DIR.rglob('*scaled*{}*.nc'.format(self.ens)),combine='by_coords',)
			self.data_vars = self.data.data_vars
			self.data = self.data.to_stacked_array('channel',sample_dims=['time','lat','lon']).values
			if self.epi_count == 0:			
				print('Observation (Time,Height,Width,Variable): ',self.observation.shape)
				print('Data variable dimension order: ',self.data_vars)
				assert len(self.Q_df) == self.data.shape[0], 'length of climate data != length of inflow record'
			self.observation = self.data[(self.t-self.stack):self.t,:,:,:]
		# elif self.agent_type == 'scalar_climate':
		elif self.agent_type == 'random':
			self.observation = np.array([self.S[self.t],float(self.doy),])

		return self.observation

	def step(self, action, noise):
		self.action = action[0]
		self.t += 1 # increment time
		self.doy = (self.t+1) % 365
		self.dowy = self.dowy_vect[self.doy-1]
		# if self.use_tocs:
		# 	self.target[self.t] = max(0.2 * (self.Q[self.t] + self.S[self.t - 1] - tocs(self.dowy[self.t])), self.target[self.t])
		# else:
		# self.target[self.t] = max(self.D[self.t],self.action)
		# self.target[self.t] = self.D[self.t] + self.action
		self.target[self.t] = self.action
		self.noise[self.t] = noise
		self.R[self.t] = min(self.target[self.t], self.S[self.t - 1] + self.Q[self.t-1])
		self.R[self.t] = min(self.R[self.t], max_release(self.S[self.t - 1]))
		# self.R[self.t] = max(self.action,0)
		# R[t] = np.clip(R[t], (1-k)*R[t], (1+k)*R[t]) # inertia --
		self.R[self.t] += max(self.S[self.t - 1] + self.Q[self.t] - self.R[self.t] - self.K, 0)  # spill
		self.S[self.t] = self.S[self.t - 1] + self.Q[self.t] - self.R[self.t]

		# squared deficit. Also penalize any total release over 100 TAF/day
		self.shortage_cost[self.t] = max(self.D[self.t] - self.R[self.t], 0)**2
		if self.R[self.t] > self.cfs_to_taf * self.max_safe_release:
			# flood penalty, high enough to be a constraint
			self.flood_cost[self.t] += 10**3 * (self.R[self.t] - self.cfs_to_taf * self.max_safe_release)
		self.reward = -(self.shortage_cost[self.t] + self.flood_cost[self.t])
		if self.agent_type == 'planner':
			self.observation = np.array([self.S[self.t],float(self.doy),]+list(self.Q_future_numpy[self.t,:]))
		elif self.agent_type == 'baseline' or self.agent_type == 'spatial_climate':
			self.observation = np.array([self.S[self.t],float(self.doy),])				
		# elif self.agent_type == 'scalar_climate':

		self.info = {'epi_done':False,'ens_done':False}
		if self.t-self.t_start == self.epi_length:
			if self.S[self.t] < 0.8*self.K:
				self.reward += -(0.8*self.K-self.S[self.t])**2
			self.info['epi_done'] = True
		if self.t == self.T - 1:
			self.info['ens_done'], self.info['epi_done'] = True, True
		self.rewards[self.t] = self.reward
		return self.observation, self.reward, self.info

	def render(self, mode='console'):
		if mode != 'console':
			raise NotImplementedError()
		# print some representation of the current env, ie. incurred penalties, objectives, reservoir levels, etc.
		
	def seed(self,seed):
		self.seed = seed # not implemented or needed

	def close(self):
		pass


