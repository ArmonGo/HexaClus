from clustering import Clustering
import numpy as np
import copy
from sklearn.model_selection import ParameterGrid
import pickle 
from sklearn.metrics import mean_squared_error
        

class GridSearcher:
    def __init__(self, grid, save_path, score_f  = mean_squared_error):
        self.param_grid =  ParameterGrid(grid)
        self.score_f = score_f
        self.best_score = np.inf
        self.best_param = None
        self.best_model = None 
        self.save_path  = save_path

    def search(self, rg, df_train, df_val, df_test, rg_name):
        for param in self.param_grid:
            input_p = copy.deepcopy(param)
            regressor = rg() # instantiate the regressor 
            regressor.set_params(**input_p)
            regressor.fit(df_train.drop(columns=['geometry', 'label']), df_train['label'])
            # count score
            pred = regressor.predict(df_val.drop(columns=['geometry', 'label']))
            s = self.score_f(df_val['label'], pred)
            if s < self.best_score:
                self.best_score = s
                self.best_param = param
                self.best_model = copy.deepcopy(regressor)
        preds = self.best_model.predict(df_test.drop(columns=['geometry', 'label']))
        y = df_test['label']
        rmse = np.sqrt(np.mean((preds - y) ** 2))
        with open(self.save_path + rg_name + '_performance.pkl', 'wb') as file:
                pickle.dump((rmse, self.best_model, self.best_param), file)
        return rmse, self.best_model
    
    def cv_clustering(self, gdf_train, gdf_val, gdf_test, max_iter, patience, selected_area = 'London, UK'):
        for param in self.param_grid:
            r = param['resolutions']
            cl = Clustering(gdf_train.copy(), gdf_val.copy(),
                        save_path = self.save_path, selected_area = selected_area, 
                        resolution = r)
            cl.construct_clustering(max_iter = max_iter, patience= patience)
            b_m = cl.load_best_instance(self.save_path)
            if b_m.base_mse < self.best_score:
                self.best_score = b_m.base_mse
                self.best_param = r
                self.best_model = copy.deepcopy(b_m)
        rmse, preds, instances = self.best_model.predict(gdf_test)
        with open(self.save_path  + 'clustering_performance.pkl', 'wb') as file:
                pickle.dump((rmse, self.best_model,self.best_param, preds, instances), file)
        return self.best_model, rmse, preds, instances
