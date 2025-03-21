from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import numpy as np 
from cv import GridSearcher
from load_data import load_london, load_gdf, load_paris, load_newyork

parameters_grid = { 'Lr_ridge' : {"alpha": np.arange(0.1,1,0.1)},
                    'clustering' : {"resolutions": [5,6,7,8,9]},
                    'RDF' :  { "min_samples_split" : [2,3,5],
                                        "min_samples_leaf" : [3,5,10]}
                    }
citys = ['new_york', 'london', 'paris']
selected_areas = ["New York City, US", "London, UK", "Paris, FR"]
load_fs = [load_newyork, load_london, load_paris]
max_iter = 10000
patience = 300 
 # try to keep more polygons
thresholds = [5e-05, 5e-05, 0]

parameters_grid = { 'Lr_ridge' : {"alpha": np.arange(0.1,1,0.1)},
                    'clustering' : {"resolutions": [5,6,7,8,9]},
                    'RDF' :  { "min_samples_split" : [2,3,5],
                                        "min_samples_leaf" : [3,5,10]}
                    }

def main(parameters_grid, citys, selected_areas, max_iter, patience):
    for l in range(len(citys)):
        save_path =  './result/' + citys[l] + '/'
        l_f = load_fs[l]
        area = selected_areas[l]
        threshold = thresholds[l]
        # load data 
        df = l_f( split_rate=(0.7, 0.1, 0.2), scale =True, coords_only = False)
        df = df.reset_index(drop=True)
        (gdf_train, gdf_val, gdf_test), (df_train, df_val, df_test) = load_gdf(df)
        # begin search
        searcher = GridSearcher(parameters_grid['clustering'], save_path = save_path)
        m, rmse, preds, instances = searcher.cv_clustering( gdf_train, gdf_val, gdf_test, max_iter, patience, area, threshold)

        rg = Ridge
        searcher = GridSearcher(parameters_grid['Lr_ridge'], save_path = save_path)
        searcher.search(rg, df_train, df_val, df_test, 'Lr_ridge')




if __name__ == "__main__":
    main(parameters_grid, citys, selected_areas, max_iter, patience)
