import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def water_day(d):
	return np.where(d >= 274, d - 274, d + 91)

dowy_vect = np.array([water_day(d) for d in range(1,366)])
D = np.loadtxt('demand.txt')[dowy_vect] # in memory

with open('results.txt') as f:
	lines = [line.rstrip('\n') for line in f]

avg_reward = {}
for line in lines:
	if line[0] == 'E':
		avg_reward[int(line.split()[2])] = float(line.split()[-1])
avg_reward = pd.DataFrame(avg_reward, index=[0]).T
print(avg_reward)
avg_reward = avg_reward.reset_index()
avg_reward.columns = ['Episode','Average Episodic Reward']

fig,ax = plt.subplots(figsize=(7,5))
ax.plot(avg_reward['Episode'],avg_reward['Average Episodic Reward'],label='Average 5-yr Episodic Reward',c = 'Blue')
# ax.axhline(-5783,label='1-yr Zero-Release Penalty',c='Red')
# ax.axhline(-28918,label='1-yr Zero-Release Penalty * 5',c='maroon')
ax.set_xlabel('Episode')
ax.set_ylabel('Penalty')
ax.set_title('Actor-Critic Results - Random Agent')
plt.legend(frameon=False,loc='lower right', bbox_to_anchor=(1.,0.05))
plt.tight_layout()
plt.savefig('avg_ep_reward_planner.png', dpi=400)
plt.close('all')
results = []
for line in lines:
	if line[0] == '|':
		result = {}		
		split = line.split()
		result['Episode'] = int(split[2])
		result['Time Step'] = int(split[5])
		result['Day of Water Year'] = int(split[8])
		result['Reward'] = int(split[11])
		result['Episodic Reward'] = int(split[15])
		result['Average Reward'] = int(split[19])
		result['Reservoir Storage'] = int(split[22])
		result['Reservoir Target Release'] = int(split[25])
		result['Average Target Release'] = int(split[29])
		result['Reservoir Inflow'] = int(split[32])
		result['Reservoir Real Release'] = int(split[35])
		results.append(result)
results = pd.DataFrame(results)

subset = results[['Episode','Episodic Reward','Average Reward','Average Target Release']]
subset = results[['Episode','Episodic Reward','Average Reward','Reservoir Storage','Average Target Release']]
subset = subset.groupby('Episode').mean()
# subset.index = subset['Episode']
fig,ax = plt.subplots(figsize=(12,8))
subset.plot(ax=ax,lw=1,subplots=True)
plt.tight_layout()
plt.savefig('avg_results_planner.png', dpi=400)
plt.close('all')
# fig,ax = plt.subplots(2,1)
# ax[0].scatter(x=results['Day of Water Year'].values[:-5000],y=results['Reservoir Storage'].values[:-5000])
# ax2 = fig.add_subplot(111, sharey=ax[0], frameon=False)
# ax2.scatter(x=range(1,366),y=D)
# ax2.yaxis.tick_right()
# ax2.yaxis.set_label_position("right")
# ax[1].scatter(x=results['Day of Water Year'].values[:-5000],y=results['Reservoir Target Release'].values[:-5000])
# ax3 = fig.add_subplot(111, sharey=ax[1], frameon=False)
# ax3.scatter(x=range(1,366),y=D)
# ax3.yaxis.tick_right()
# ax3.yaxis.set_label_position("right")
# plt.tight_layout()
# plt.savefig('seasonal_planner.png', dpi=400)

# subset = results.iloc[:-2500,:]
# f, ax = plt.subplots(figsize=(6.5, 6.5))
# sns.kdeplot(ax=ax,x='Day of Water Year',y='Reservoir Storage',data=subset)
# plt.show()

# subset = results.iloc[-5*365*500:,:]
# f, ax = plt.subplots(figsize=(6.5, 6.5))
# sns.histplot(ax=ax,x='Day of Water Year',y='Reservoir Storage',bins=10, pthresh=.1, cmap="mako",data=subset)
# plt.show()

# f, ax = plt.subplots(figsize=(6.5, 6.5))
# sns.histplot(ax=ax,x='Day of Water Year',y='Reservoir Target Release',bins=10, pthresh=.1, cmap="mako",data=subset)
# plt.show()

subset = results.iloc[-5*365*40:,:]
fig, axes = plt.subplots(4,1,figsize=(10, 10))
axes[0].scatter(range(1,366),D)
axes[0].set_ylabel('Daily Demand (TAF)')
sns.boxplot(ax=axes[1],x='Day of Water Year',y='Reservoir Storage',orient='v',data=subset)
sns.boxplot(ax=axes[2],x='Day of Water Year',y='Reservoir Target Release',orient='v',data=subset)
sns.boxplot(ax=axes[3],x='Day of Water Year',y='Reservoir Real Release',orient='v',data=subset)
fig.suptitle('Seasonal Results - Planner')
plt.tight_layout()
plt.savefig('seasonal_planner.png', dpi=400)

