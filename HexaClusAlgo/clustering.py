from typing import Any, List, Optional
import numpy as np
from geopandas import GeoDataFrame
import geopandas as gpd
from shapely.geometry import Polygon
from shapely.ops import unary_union
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import cross_validate
from shapely.ops import unary_union
from polygon import SraiConstructor
import warnings
import pickle
import pandas as pd 
from sklearn.preprocessing import  MinMaxScaler
from collections import defaultdict
import copy 


def train_polygon_model(
    clustering: "Clustering",
    polygon_idxs: List[int]
) -> tuple[float, int, Any]:
    """
    Train a regression model over the instances of polygons.
    Measurements are aggregated according and concatenated.
    Returns the MSE, number of instances, and the trained model.
    """
    polygon_idxs = list(set(polygon_idxs)) # if it is the same one
    mse_scorer = make_scorer(lambda y, y_pred: mean_squared_error(y, y_pred))

    # Get instance labels and features of the polygons
    instance_labels = []
    instance_features = []
    for idx in polygon_idxs:
        instances = clustering.get_instances_in_polygon(idx)
        instance_features.append(instances.drop(columns=["geometry", "label"]).values)
        instance_labels.extend(instances["label"].values)
    instance_features = np.vstack(instance_features)

    # If there are no instances here, return 0 MSE
    if not len(instance_features):
        return 0, 1, None
    X = instance_features
    y = instance_labels
    # Train model
    model = Ridge(alpha=0.1)
    if len(X) > 10:
        scores = cross_validate(model, X, y, cv=5, scoring=mse_scorer)
        mse = np.mean(scores["test_score"])
        model.fit(X, y)
    else:
        model.fit(X, y)
        preds = model.predict(X)
        mse =  mean_squared_error(y_true, preds)
    return mse, len(X), model



class Clustering:
    """ Class to run a clustering run by merging a list of polygons given a list of instances and measurements.
    three pair_states for polygon pairs: 
        1 candidates for merge, 
        0 have been merged
    two polygon states
        1 active
        0 inactive """

    def __init__(self, instances: GeoDataFrame, 
                 val_instances: GeoDataFrame, 
                 test_instances : Optional[GeoDataFrame] = None, 
                 save_path : str = './algo/', 
                 measurements: Optional[List[GeoDataFrame]] = None, 
                 selected_area : str = None, 
                 resolution = 6):
        self.instances = instances
        self.measurements = measurements if measurements is not None else []
        self.val_instances = val_instances
        self.test_instances = test_instances
        self.save_path = save_path
        # initial all the settings
        self.polygons : List[Polygon] = []
        self.touching_pairs: List[tuple] = []
        self.polygon_states : List[int] = []
        self.length : List[int] = []
        self.mse : List[float] = []
        self.mse_diff : List[float] = []
        self.models : List[Any] = []
        self.instance_assignments : List[List[int]] = []
        self.val_instance_assignments : List[List[int]] = []
        self.polygon_neighbors =  defaultdict(set)
        self._constructor = SraiConstructor(selected_area = selected_area, 
                                            resolution = resolution, 
                                            encoder_sizes = [10, 5])
        self.geo_feats = []
        self.history  : List[List[int]] = []
    
    def aggregate_features_by_polygon(self, measurements, polygons):
        """
        Aggregates features from multiple GeoDataFrames into hexes
        """
        gdf_bins = GeoDataFrame(geomerty= polygons, 
                                data = {'polygon_ix' : list(range(len(polygons)))})
        all_feature_cols = set()
        for gdf in measurements:
            all_feature_cols.update(gdf.columns.difference(['geometry']))  # Exclude 'geometry'
        # Ensure each GeoDataFrame has the same columns (fill missing with NaN)
        standardized_gdfs = []
        for gdf in measurements:
            for col in all_feature_cols:
                if col not in gdf.columns:
                    gdf[col] = None  # Fill missing columns with NaN
            standardized_gdfs.append(gdf)
        gdf_combined = gpd.GeoDataFrame(pd.concat(standardized_gdfs, ignore_index=True), crs=measurements[0].crs)
        gdf_joined = gpd.sjoin(gdf_combined, gdf_bins, predicate='intersects', how='left')
        feature_cols = list(all_feature_cols)
        gdf_aggregated = gdf_joined.groupby(gdf_bins.index)[feature_cols].mean()
        gdf_final = gdf_bins.copy()
        gdf_final = gdf_final.merge(gdf_aggregated, left_index=True, right_index=True, how='left')
        gdf_final =  gdf_final.drop(columns = ['polygon_ix'])
        return gdf_final

    def clear_memory(self):
        self.touching_pairs.clear() 
        self.length.clear()  
        self.mse.clear()  
        self.models.clear()  
        self.instance_assignments.clear()
        self.val_instance_assignments.clear()
        self.mse_diff.clear()
        self.history.clear()
        self.polygon_neighbors = defaultdict(set)
    
    def filter_instances(self, instances):
        """make sure all the instances fall into the given boundary"""
        # check if inside the clustering area
        inside_ix = instances['geometry'].apply(lambda x: self.boundary.contains(x))
        if len(inside_ix) - sum(inside_ix) >0:
            warnings.warn(str(len(inside_ix) - sum(inside_ix)) + " instances are not all within given boundary!")
        instances = instances[inside_ix]
        return instances
    
    def append_geo_feast(self, instances):
        instances = gpd.sjoin(instances, self.geo_feats, predicate='intersects', how='left')
        instances = instances.drop(columns=[col for col in instances.columns if 'index' in str(col)])
        return instances
    
    def assign_instance_dict(self, instances, polygons):
        p_space = GeoDataFrame(geometry = polygons)
        joined = gpd.sjoin(p_space, instances, how="left", predicate="contains")
        # Create a nested list where each polygon gets the list of point indices
        instance_assignments = joined.groupby(joined.index).\
            apply(lambda x: x.index_right.dropna().astype(int).tolist())\
                .reindex(range(len(p_space)), fill_value=[]).tolist()
        return instance_assignments

    def initialize(self, polygons: List[Polygon], feats = Optional[List[GeoDataFrame]]) -> None:
        """Initialize the clustering solution with a list of polygons."""
        self.polygons = polygons
        # build the boundary 
        self.boundary = unary_union(self.polygons)
        # add feats to dictionary
        if feats is not None:
            self.geo_feats.append(feats)
        # fit the nr of measuresments 
        if len(self.measurements) > 0:
            gdf_measurements = self.aggregate_features_by_polygon(self.measurements, self.polygons)
            self.geo_feats.append(np.array(gdf_measurements.drop(columns = ['geometry'])))
        self.geo_feats = np.concatenate(self.geo_feats, axis=1)
        s = MinMaxScaler()
        self.geo_feats = s.fit_transform(self.geo_feats)
        self.geo_feats = GeoDataFrame(geometry = self.polygons, 
                                      data = {i : list(self.geo_feats[:, i]) for i in range(self.geo_feats.shape[1])})
        # initialize features
        self.instances = self.append_geo_feast(self.instances).reset_index(drop =True)
        self.val_instances = self.filter_instances(self.val_instances)
        self.val_instances = self.append_geo_feast(self.val_instances).reset_index(drop =True)
        if self.test_instances is not None:
            self.test_instances = self.filter_instances(self.test_instances)
            self.test_instances = self.append_geo_feast(self.test_instances)

        # initialize polygon settings
        self.clear_memory()
        self.instance_assignments  = self.assign_instance_dict(copy.deepcopy(self.instances), copy.deepcopy(self.polygons))
        self.val_instance_assignments  = self.assign_instance_dict(copy.deepcopy(self.val_instances), copy.deepcopy(self.polygons)) # make sure all polygons 
        assert len(self.instance_assignments) == len(self.val_instance_assignments)
        touching_pairs_tb = gpd.sjoin(GeoDataFrame(geometry = self.polygons), GeoDataFrame(geometry = self.polygons), predicate="touches")
        rest_touch_pairs = list(set(tuple(sorted((i, j))) for i, j in zip(touching_pairs_tb.index, touching_pairs_tb.index_right)))
        # add self touch 
        for p_ix in range(len(self.polygons)):
            mse, l, model = train_polygon_model(self, [p_ix])
            self.models.append(model)
            self.polygon_states.append(1) # all active 
            self.touching_pairs.append((p_ix, p_ix))                                                                                                                                                                                                                                                                                                                 
            self.mse.append(mse)
            self.length.append(l)
        for p in rest_touch_pairs:
            self.touching_pairs.append(p)
            mse, l, _ = train_polygon_model(self, [p[0], p[1]])
            self.mse.append(mse)
            self.length.append(l)
        # calculate the change mse
        for p in self.touching_pairs:
            self.mse_diff.append(self.get_mse_diff(p))
        # initiliase neighbor dict
        for key, value in self.touching_pairs:
            self.polygon_neighbors[key].add(value)
            self.polygon_neighbors[value].add(key)
        # Convert sets to sorted lists
        self.polygon_neighbors = {k: sorted(v) for k, v in self.polygon_neighbors.items()}


    def get_mse_diff(self, pair):
        if pair[0] == pair[1]:
            return -float('inf') # cannot merge itself 
        else:
            ix_c = self.touching_pairs.index(pair)
            ix_i = self.touching_pairs.index((pair[0], pair[0]))
            ix_j = self.touching_pairs.index((pair[1], pair[1]))
            mse_d = (self.length[ix_i] * self.mse[ix_i] + self.length[ix_j] *  self.mse[ix_j] ) / (self.length[ix_c] ) -  self.mse[ix_c]
            return mse_d

    def get_instances_in_polygon(self, polygon_idx: int, dict_instance =None, instances_used = None) -> GeoDataFrame:
        """Get all instances that lie within the specified polygon."""
        if dict_instance is None:
            dict_instance = self.instance_assignments
            instances_used = self.instances
        if polygon_idx >= len(self.polygons) or self.polygons[polygon_idx] is None:
            raise ValueError(f"No polygon with index {polygon_idx}")
        try:
            return instances_used.iloc[list(dict_instance[polygon_idx])]
        except:
            print('the expected index is', f'{polygon_idx}')
            print('dict_instance[polygon_idx]', f'{dict_instance[polygon_idx]}')
            #print(len(dict_instance.index), max(dict_instance.index))
            print(instances_used.iloc[list(dict_instance[polygon_idx])])
            

    def get_merge_polygon_pairs(self):
        ix = np.argmax(self.mse_diff).item()
        assert len(self.mse) == len(self.length)
        return self.touching_pairs[ix], ix 
    
    def drop_old_polygons(self, old_ix):
        neighbors = copy.deepcopy(self.polygon_neighbors[old_ix])
        for n in neighbors:
            pair = tuple(sorted([n, old_ix]))
            r_ix = self.touching_pairs.index(pair)
            self.touching_pairs.pop(r_ix)
            self.mse.pop(r_ix)
            self.mse_diff.pop(r_ix)
            self.length.pop(r_ix)
            self.polygon_neighbors[n].remove(old_ix)
        del self.polygon_neighbors[old_ix]
                         
    def merge_polygons(self):
        # save history first
        _, a_ix = self.get_active_polygons()
        self.history.append(a_ix)
        merge_p, _ = self.get_merge_polygon_pairs()
        if merge_p[0] == merge_p[1]:
            return False 
        # add the new polygons and related features 
        new_polygon_index = len(self.polygons)
        self.polygons.append(unary_union(([self.polygons[merge_p[0]], self.polygons[merge_p[1]]])))
        self.instance_assignments.append(self.instance_assignments[merge_p[0]] + self.instance_assignments[merge_p[1]])
        self.val_instance_assignments.append(self.val_instance_assignments[merge_p[0]] + self.val_instance_assignments[merge_p[1]])
        # change the polygon state
        self.polygon_states[merge_p[0]] = 0
        self.polygon_states[merge_p[1]] = 0
        self.polygon_states.append(1)
        
        ## append new touching pairs
        new_neighbors = copy.deepcopy(sorted(list(set(self.polygon_neighbors[merge_p[0]] + self.polygon_neighbors[merge_p[1]])
                                    - {merge_p[0], merge_p[1]}) # exlude old ones
                                    + [new_polygon_index])) # include new merged one
        # first add itself 
        self.touching_pairs.append((new_polygon_index, new_polygon_index))
        mse, l, model = train_polygon_model(self, [new_polygon_index, new_polygon_index])
        self.mse.append(mse)
        self.length.append(l)
        self.models.append(model)
        self.mse_diff.append(self.get_mse_diff((new_polygon_index, new_polygon_index)))
        self.polygon_neighbors[new_polygon_index] = sorted(new_neighbors)
        for j in new_neighbors[:-1]:
            self.touching_pairs.append((j, new_polygon_index))
            mse, l, model = train_polygon_model(self, [j, new_polygon_index])
            self.mse.append(mse)
            self.length.append(l)
            self.mse_diff.append(self.get_mse_diff((j, new_polygon_index)))
            self.polygon_neighbors[j].append(new_polygon_index)
        ## remove the old pairs related infomation
        self.drop_old_polygons(merge_p[0])
        self.drop_old_polygons(merge_p[1])
        return True
        
    def get_active_polygons(self):
        active_ixs = [i for i  in range(len(self.polygon_states)) if self.polygon_states[i] == 1]
        active_polygons = [self.polygons[i] for i in active_ixs]
        return active_polygons, active_ixs
    
    def save_best_instance(self, best_mse):
        with open(self.save_path + "best_model.pkl", "wb") as f:
            pickle.dump(self, f)
        print(f"Best model saved with MSE: {best_mse}")

    @staticmethod
    def load_best_instance(save_path):
        with open(save_path +"best_model.pkl", "rb") as f:
            return pickle.load(f)
        
    def construct_clustering(
        self,
        max_iter: int = 0,
        patience : int = 100, 
        restart: bool = True
    ) -> None:
        v_l = []
        t_l = []
        if not self._constructor:
            raise ValueError("No constructor initialized")
        if restart:
            p, f = self._constructor.construct(self.instances)
            self.initialize(p, f)
            self.base_mse = float("inf")
        # begin merge 
        merges = 0
        tol = 0
        print('merging begins...')
        while True:
            try:
                merge_r = self.merge_polygons()
                if not merge_r:
                    print('no available polygons...')
                    break
            except ValueError as e:
                print(e)
                break
            merges += 1
            val_mse, _, _ = self.validate()
            v_l.append(val_mse)
            if self.test_instances is not None:
                test_mse, _, _ = self.predict(instances=self.test_instances)
                t_l.append(test_mse)
            if val_mse < self.base_mse: # continue 
                self.base_mse = val_mse
                self.save_best_instance(best_mse=val_mse)
                tol = 0
            else:
                tol += 1 
            if max_iter > 0 and merges > max_iter:
                return v_l, t_l
            if tol >= patience: # stop and return the best model 
                print('run out of the patience!')
                return v_l, t_l

    def get_within_polygons_index(self,
                                  instance : GeoDataFrame) -> List[int]:
        poi = instance['geometry']
        active_polys, active_ixs = self.get_active_polygons()
        loc = list(map(lambda x:x.contains(poi), active_polys))
        model_ix = [ active_ixs[n] for n in range(len(active_ixs)) if loc[n] ==True] 
        if len(model_ix) == 0 :
            raise NotImplementedError('Not belong to any active polygon')
        if len(model_ix) > 1 :
            print('poi', poi)
            for b in range(len(loc)):
                if loc[b] == True:
                    print('active polygons', active_polys[b])
            raise NotImplementedError('Belong to more than 1 active polygon, ' + 'polygon nr : ' + str(len(model_ix)))
        return model_ix[0]
        
    def validate(self)-> None:
        _, active_ix = self.get_active_polygons()
        models = []
        X_list = []
        y_list = []
        for x in active_ix:
            if len(self.val_instance_assignments[x])>0:
                models.append(self.models[x])
                x_geo = self.get_instances_in_polygon(x, self.val_instance_assignments, self.val_instances)
                X_list.append(np.array(x_geo.drop(columns=["geometry", "label"])))
                y_list = y_list + list(x_geo['label'])
            else:
                pass
        preds_n = list(map(lambda a, b: b.predict(a), X_list, models))
        preds = np.array([item for sublist in preds_n for item in sublist])
        y_list = np.array(y_list)
        mse = np.mean((preds - y_list) ** 2)
        return mse, preds, self.val_instance_assignments
    
        
    def predict(
        self,
        instances
    ) -> None:
        # check if inside the clustering area
        # fliter data
        instances = self.filter_instances(instances)
        instances = self.append_geo_feast(instances)
        # check which polygon it belongs to 
        preds = []
        for i in range(len(instances)):
            m_ix = self.get_within_polygons_index(instances.iloc[i, :])
            # get features  
            instance_features = []
            instance_features.append(instances.drop(columns=["geometry", "label"]).iloc[i, :].values)
            instance_features = np.vstack(instance_features)
            X = instance_features
            preds.append(self.models[m_ix].predict(X).item()) 
        y = instances['label']
        rmse = np.sqrt(np.mean((preds - y) ** 2))
        print('predicted rmse:', rmse)
        return rmse, preds, instances
        
    
