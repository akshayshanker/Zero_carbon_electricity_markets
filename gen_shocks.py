import numpy as np
import dill as pickle
from pathlib import Path


model_name = 'ERCOT_main_v_2'
TSN = int(1E7)
U = np.random.rand(TSN,1)

Path("{}/".format(model_name))\
										.mkdir(parents=True, exist_ok=True)

pickle.dump(U, open("{}/seed_u.pkl"\
						.format(model_name),"wb"))