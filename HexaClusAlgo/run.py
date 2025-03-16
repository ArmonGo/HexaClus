from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import numpy as np 
from cv import GridSearcher
from load_data import load_london, load_gdf, load_paris, load_newyork

parameters_grid = { 'Lr_ridge' : {"alpha": np.arange(0.1,1,0.1)},
                    'clustering' : {"resolutions": [5, 6, 7, 8, 9]}
                    }
save_path = './result/new_york/'
max_iter = 10000
patience = 150 
selected_area = "New York City, US"

df = load_newyork( split_rate=(0.7, 0.1, 0.2), scale =True, coords_only = False)
df = df.reset_index(drop=True)
(gdf_train, gdf_val, gdf_test), (df_train, df_val, df_test) = load_gdf(df)

searcher = GridSearcher(parameters_grid['clustering'], save_path = save_path)
m, rmse, preds, instances = searcher.cv_clustering( gdf_train, gdf_val, gdf_test, max_iter, patience, selected_area)


rg = Ridge
searcher = GridSearcher(parameters_grid['Lr_ridge'], save_path = save_path)
searcher.search(rg, df_train, df_val, df_test, 'Lr_ridge')

rg = RandomForestRegressor
searcher = GridSearcher(parameters_grid['RDF'], save_path = save_path)
searcher.search(rg, df_train, df_val, df_test, 'RDF')
