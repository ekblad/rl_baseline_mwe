
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from numba import jit
from gym import spaces
import matplotlib.pyplot as plt
import seaborn as sns

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

class Folsom():
	"""
	Environment for the Folsom Reservoir model compatible with OpenAI gym, baselines RL implementations, and extensions (stable-baselines)
	Accepts image data as input, daily time step.
	"""
	# metadata = {'render.modes': ['console']} # for rendering observations/data as print output, image, video, etc.
	# reward_range = (-float('inf'), float('inf')) # not used 
	# constants
	cfs_to_taf = 2.29568411 * 10**-5 * 86400 / 1000
	taf_to_cfs = 1000 / 86400 * 43560

	def __init__(self, DATA_DIR, obs_dim = None, res_dim = None, stack = 0, inflow_stack = 0, models = ['canesm2',], ensembles = ['r1i1p1',]): 
		super(Folsom,self).__init__()
		
		self.DATA_DIR = DATA_DIR
		self.models = models
		self.ensembles = ensembles
		self.ens = self.ensembles[0]
		self.model = self.models[0]
		self.obs_dim = obs_dim
		self.res_dim = res_dim
		self.stack = stack
		self.inflow_stack = inflow_stack
		self.cfs_to_taf = 2.29568411 * 10**-5 * 86400 / 1000
		self.taf_to_cfs = 1000 / 86400 * 43560

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

	def reset(self, agent):
		"""
		Important: the observation must be a numpy array
		:return: (np.array) 
		"""

		self.agent_type = agent.agent_type
		self.ens = agent.ens
		self.model = agent.model
		self.t_start = agent.epi_start
		self.epi_count = agent.epi_count
		self.epi_steps = agent.epi_steps
		self.MODEL_DIR = self.DATA_DIR / self.model
		# read in inflows:
		# self.FLOWS_DIR = DATA_DIR / (self.model + '_meteo') # in general case
		self.FLOWS_DIR = self.DATA_DIR / 'folsom' / 'data' # for this repo		
		self.Q_df = pd.read_csv(self.FLOWS_DIR  / '{}_fol_inflows.csv'.format(self.ens) ,header=0,index_col=0,parse_dates=[0],infer_datetime_format=True)*self.fol_mm_to_taf
		self.Q = self.Q_df['inflow'].values
		self.time_vector = self.Q_df.index.values
		self.dowy_vect = np.array([water_day(d) for d in self.Q_df.index.dayofyear])
		self.D_shift = np.loadtxt(self.FLOWS_DIR / 'demand.txt')
		self.D = self.D_shift[self.dowy_vect]
		self.T = len(self.Q_df.index)

		# reinit. reservoir model
		self.runup_days = max(self.stack,self.inflow_stack)
		self.t = self.t_start # self.runup_days # reset to beginning of episode
		self.doy = (self.t+1) % 365
		self.dowy = self.dowy_vect[self.doy-1]
		self.S, self.R, self.target, self.shortage_cost, self.flood_cost, self.storage_cost, self.noise, self.rewards = [np.zeros(self.T) for _ in range(8)]
		self.R[self.t] = self.D[self.t]
		self.S[self.t] = 500.
		self.start_date = self.time_vector[self.t]

		# generate first observation:
		if agent.agent_type == 'planner':
			if agent.epi_count == 0 and not agent.warmup:
				print('Training start date and ensemble member: ',self.start_date,self.ens)
				# print('Observation : ',self.observation_space.shape)
			self.observation = np.array([self.S[self.t],float(self.doy),]+list(self.Q_future_numpy[self.t,:]))
		elif agent.agent_type == 'baseline' or agent.agent_type == 'spatial_climate':
			self.observation = np.array([self.S[self.t],float(self.doy),])
			self.data = xr.open_mfdataset(self.MODEL_DIR.rglob('*scaled*{}*.nc'.format(self.ens)),combine='by_coords',)
			self.data_vars = self.data.data_vars
			self.data = self.data.to_stacked_array('channel',sample_dims=['time','lat','lon']).values
			if agent.epi_count == 0:			
				print('Observation (Time,Height,Width,Variable): ',self.observation.shape)
				print('Data variable dimension order: ',self.data_vars)
				assert len(self.Q_df) == self.data.shape[0], 'length of climate data != length of inflow record'
			self.observation = self.data[(self.t-self.stack):self.t,:,:,:]
		# elif self.agent_type == 'scalar_climate':
		elif agent.agent_type == 'random':
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
		self.shortage_cost[self.t] += max(self.D[self.t] - self.R[self.t], 0)**2
		if self.R[self.t] > self.cfs_to_taf * self.max_safe_release:
			# flood penalty, high enough to be a constraint
			self.flood_cost[self.t] += 10**3 * (self.R[self.t] - self.cfs_to_taf * self.max_safe_release)
		self.storage_cost[self.t] += max(200-self.S[self.t],0)
		self.reward = -(self.shortage_cost[self.t] + self.flood_cost[self.t] + self.storage_cost[self.t])
		if self.agent_type == 'planner':
			self.observation = np.array([self.S[self.t],float(self.doy),]+list(self.Q_future_numpy[self.t,:]))
		elif self.agent_type == 'baseline' or self.agent_type == 'spatial_climate':
			self.observation = np.array([self.S[self.t],float(self.doy),])				
		# elif self.agent_type == 'scalar_climate':
		self.info = {'epi_done':False,'ens_done':False}
		if self.t-self.t_start >= self.epi_steps:
			self.info['epi_done'] = True
		if self.t >= self.T - 1:
			self.info['ens_done'], self.info['epi_done'] = True, True
		self.rewards[self.t] = self.reward
		return self.observation, self.reward, self.info

	def render(self, agent = None, STOR_DIR = None, mode = ['figures']):
		# if mode != 'console':
		# 	raise NotImplementedError()
		if 'console' in mode:
			print('|| Ep: {} ||'.format('{:1.0f}'.format(self.epi_count)),
					't: {} ||'.format('{:5.0f}'.format(self.t)),
					'dowy: {} ||'.format('{:3.0f}'.format(self.dowy)),
					'R: {} ||'.format('{:7.0f}'.format(self.rewards[self.t])),
					'Ep. R: {} ||'.format('{:9.0f}'.format(np.sum(self.rewards[self.t_start:self.t]))),
					'Avg. R: {} ||'.format('{:4.0f}'.format(np.mean(self.rewards[self.t_start:self.t]))),
					'S: {} ||'.format('{:3.0f}'.format(self.S[self.t])),
					'A: {} ||'.format('{:3.0f}'.format(self.target[self.t])), 
					'Avg. A: {} ||'.format('{:3.1f}'.format(np.mean(self.target[self.t_start:self.t]))),	
					'I: {} ||'.format('{:3.0f}'.format(self.Q[self.t])),							
					'O: {} ||'.format('{:3.0f}'.format(self.R[self.t])),
					)
		if 'figures' in mode:
			# store most recent episode simulation results:
			epi_sim = {'time':self.Q_df.index[self.t_start:self.t],
						'inflow':self.Q_df['inflow'].iloc[self.t_start:self.t],
						'storage':self.S[self.t_start:self.t],
						'target':self.target[self.t_start:self.t],
						'noise':self.noise[self.t_start:self.t],
						'release':self.R[self.t_start:self.t],
						'shortage_cost':self.shortage_cost[self.t_start:self.t],
						'flood_cost':self.flood_cost[self.t_start:self.t],
						'storage_cost':self.storage_cost[self.t_start:self.t],
						'reward':self.rewards[self.t_start:self.t],}
			epi_sim = pd.DataFrame.from_dict(epi_sim,orient='columns')
			epi_sim.set_index('time')			
			epi_sim.to_csv(STOR_DIR / f'results_sim_{agent.epi_count:05}.csv')
			axes = epi_sim.plot(subplots=True,figsize=(8,12))
			axes[0].set_title('Planner Simulation - Episode {}'.format(self.epi_count))
			for ax in axes.flatten():
				ax.legend(frameon=False)
			plt.tight_layout()
			plt.savefig(STOR_DIR / f'results_sim_{agent.epi_count:05}.png',dpi=400)
			plt.close('all')

			avg_df = {'Episodic Rewards': agent.epi_reward_list,
						'Avg. Episodic Rewards': agent.epi_avg_reward_list,	
						'Average Rewards': agent.avg_reward_list,
						'Average Actions': agent.avg_action_list,
						}
			avg_df = pd.DataFrame.from_dict(avg_df)
			avg_df.to_csv(STOR_DIR / 'results_averages.csv')

			axes = avg_df.plot(subplots=True,figsize=(8,6))
			axes[0].set_title('Average and Episodic Results')
			axes[0].set_ylabel('Tot. Cost')
			axes[1].set_ylabel('Avg. Tot. Cost')			
			axes[2].set_ylabel('Avg. Cost')
			axes[3].set_ylabel('Release [TAF]')
			axes[3].set_xlabel('Episode')
			for ax in axes.flatten():
				ax.legend(frameon=False)
			plt.tight_layout()
			plt.savefig(STOR_DIR / 'results_averages.png', dpi=400)
			plt.close('all')

			fig,axes = plt.subplots(figsize=(7,5))
			axes.plot(range(1,agent.epi_count),agent.epi_avg_reward_list,label='Avg. Reward (last 40 40-yr episodes)',c = 'Blue')
			axes.axhline(-5783*40,label='40-yr Zero-Release Penalty',c='Red')
			axes.set_xlabel('Episode')
			axes.set_ylabel('Penalty')
			axes.set_title('Actor-Critic Convergence')
			plt.legend(frameon=False,loc='lower right', bbox_to_anchor=(1.,0.05))
			plt.tight_layout()
			plt.savefig(STOR_DIR / 'results_ep_reward.png', dpi=400)
			plt.close('all')

			epi_sim['day of year'] = [i.dayofyear for i in epi_sim.index]
			epi_sim['day of water year'] = water_day(epi_sim['day of year'].values)
			fig, axes = plt.subplots(4,1,figsize=(10, 10))
			axes[0].scatter(range(1,366),self.D_shift[:-1])
			axes[0].set_ylabel('daily demand (taf)')
			sns.kdeplot(ax=axes[1],x='day of water year',y='storage',fill=True,cut=0,thresh=0.1,levels=100,cmap="mako",data=epi_sim)
			sns.kdeplot(ax=axes[2],x='day of water year',y='target',fill=True,cut=0,thresh=0.1,levels=100,cmap="mako",data=epi_sim)
			sns.kdeplot(ax=axes[3],x='day of water year',y='release',fill=True,cut=0,thresh=0.1,levels=100,cmap="mako",data=epi_sim)
			fig.suptitle('seasonal results')
			plt.tight_layout()
			plt.savefig(STOR_DIR / f'results_seasonal_{agent.epi_count:05}.png', dpi=400)
			plt.close('all')
		
	def seed(self,seed):
		self.seed = seed # not implemented or needed

	def close(self):
		pass


