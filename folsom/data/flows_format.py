import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

flows = pd.read_csv('r1i1p1_fol_inflows.csv',header=0,index_col=0,parse_dates=[0],infer_datetime_format=True)

def flows_format(Q_df):
	flows['inf_t+1'] = np.append(flows.inflow.values[1:],np.NaN)
	for i in range(2,6):
		flows['inf_t+{}'.format(i)] = np.append(flows['inf_t+{}'.format(i-1)].values[1:],np.NaN)
	for _,(i,j) in enumerate(zip(['5d','1m','3m','6m','1y','2y','3y','4y','5y'],[5,30,90,180,365,730,1095,1460,1825])):	
		flows['inf_{}_mean'.format(i)] = flows['inflow'].iloc[::-1].rolling(window='{}D'.format(j)).mean().iloc[::-1]
		flows['inf_{}_sum'.format(i)] = flows['inflow'].iloc[::-1].rolling(window='{}D'.format(j)).sum().iloc[::-1]
	Q_df = Q_df.ffill()
	return Q_df

flows = flows_format(flows)
flows.plot(subplots=True)
plt.show()