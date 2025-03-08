import copy 
import pandas as pd 
import numpy as np
import copy 
from scipy import stats
from sklearn.preprocessing import  MinMaxScaler
from category_encoders.cat_boost import CatBoostEncoder
import kagglehub
import geopandas as gpd 
import shapely as shp


# load real dataset 
eps = 0.00001

def load_data_path(path, target_tb, format = 'csv'):
    path = kagglehub.dataset_download(path)
    if format =='csv':
        df = pd.read_csv(path + '/' +target_tb,encoding_errors='ignore')
    else:
        arrays = dict(np.load(path + '/' +target_tb))
        data = {k: [s.decode("utf-8") for s in v.tobytes().split(b"\x00")] if v.dtype == np.uint8 else v for k, v in arrays.items()}
        df = pd.DataFrame.from_dict(data)
    return df 


def scale_feats(df, scaler = MinMaxScaler, skip_coords = True, mask_col = 'split_type'):
    cols = df.columns
    if scaler is not None:
        s = scaler(feature_range = (0 + eps, 1 + eps ))
    else: # default
        s = MinMaxScaler()
    if skip_coords:
        cols = [col for col in df.columns if col not in ['lon', 'lat', mask_col]]
    if mask_col is not None:
        X = df[df[mask_col] == 0][cols]
    else:
        X = df[cols]
    s.fit(X)
    df.loc[:, cols] = s.transform(df[cols])
    return df 

def train_val_test_split(split_rate, length, shuffle = False, return_type = 'feats'):
    tr_r, val_r, te_r = split_rate
    assert tr_r + val_r + te_r == 1
    if shuffle:
        indices = np.random.permutation(length)
    else:
        indices = np.arange(length)
    ix_ls = [indices[:int(tr_r*length)], indices[int(tr_r*length):int((val_r + tr_r)*length)], indices[int((val_r + tr_r)*length):]]
    if return_type == 'index':
        mask_ls = []
        for i in range(3):
            mask = np.zeros(length, dtype=bool)
            mask[ix_ls[i]] = True
            mask_ls.append(mask)
        return mask_ls
    elif return_type == 'feats':
        split_type =  np.zeros(length, dtype=int)
        for i in range(3):
            split_type[ix_ls[i]] = i # 0-train, 1-val, 2-test
        return split_type
    
def load_london( split_rate=None, scale =True, coords_only = False):
    df_raw = load_data_path("jakewright/house-price-data", 'kaggle_london_house_price_data.csv')
    df = copy.deepcopy(df_raw[['bathrooms', 'bedrooms', 'floorAreaSqM', 'livingRooms',
    'tenure', 'propertyType', 'currentEnergyRating']])
    category_feats = ['tenure', 'propertyType']
    df["label"] = df_raw["history_price"]/ df_raw["floorAreaSqM"]
    df["lon"] = df_raw["longitude"] 
    df["lat"] = df_raw['latitude'] 
    d = {'A' : 7, 'B':6, 'C':5, 'D':4, 'E':3, 'F':2, 'G':1, np.nan:0}
    df['currentEnergyRating'] = df['currentEnergyRating'].map(d)
    df["history_date"] = pd.to_numeric(df_raw["history_date"].str.replace('-',''), errors='coerce')
    df = df[df['history_date'] >= 20230101] # not too old data 
    for i in df.columns:
        if i not in category_feats:
            df[i] = pd.to_numeric(df[i], errors='coerce')
    df = df.dropna() # delete nan first!
    df = df.drop_duplicates(subset=["lat", "lon"], keep="last") # drop duplicates
    df = df[(stats.zscore(df["label"])<5) & (stats.zscore(df["label"])>-2)]
    df = df.sort_values(by=['history_date']) # temporal split 
    df['x'] = df['lon'].copy()
    df['y'] = df['lat'].copy()
    if split_rate is not None:
        split_type = train_val_test_split(split_rate, length=len(df), shuffle = False, return_type = 'feats')
        df['split_type'] = split_type
        if len(category_feats)>0:
            encoder = CatBoostEncoder(cols = category_feats)
            encoder.fit(df[df['split_type'] == 0], df.loc[df['split_type'] == 0, 'label'])
            df = encoder.transform(df)
    if scale:
        scale_feats(df, scaler = MinMaxScaler, skip_coords = True, mask_col = 'split_type')
    df = df[['lon', 'lat'] + [col for col in df.columns if col not in ['lat', 'lon']]]
    if coords_only:
        return df[['lon', 'lat', 'x', 'y', 'label', 'split_type']]
    else:
        return df
    
def load_london( split_rate=None, scale =True, coords_only = False):
    df_raw = load_data_path("jakewright/house-price-data", 'kaggle_london_house_price_data.csv')
    df = copy.deepcopy(df_raw[['bathrooms', 'bedrooms', 'floorAreaSqM', 'livingRooms',
    'tenure', 'propertyType', 'currentEnergyRating']])
    category_feats = ['tenure', 'propertyType']
    df["label"] = df_raw["history_price"]/ df_raw["floorAreaSqM"]
    df["lon"] = df_raw["longitude"] 
    df["lat"] = df_raw['latitude'] 
    d = {'A' : 7, 'B':6, 'C':5, 'D':4, 'E':3, 'F':2, 'G':1, np.nan:0}
    df['currentEnergyRating'] = df['currentEnergyRating'].map(d)
    df["history_date"] = pd.to_numeric(df_raw["history_date"].str.replace('-',''), errors='coerce')
    df = df[df['history_date'] >= 20230101] # not too old data 
    for i in df.columns:
        if i not in category_feats:
            df[i] = pd.to_numeric(df[i], errors='coerce')
    df = df.dropna() # delete nan first!
    df = df.drop_duplicates(subset=["lat", "lon"], keep="last") # drop duplicates
    df = df[(stats.zscore(df["label"])<5) & (stats.zscore(df["label"])>-2)]
    df = df.sort_values(by=['history_date']) # temporal split 
    df['x'] = df['lon'].copy()
    df['y'] = df['lat'].copy()
    if split_rate is not None:
        split_type = train_val_test_split(split_rate, length=len(df), shuffle = False, return_type = 'feats')
        df['split_type'] = split_type
        if len(category_feats)>0:
            encoder = CatBoostEncoder(cols = category_feats)
            encoder.fit(df[df['split_type'] == 0], df.loc[df['split_type'] == 0, 'label'])
            df = encoder.transform(df)
    if scale:
        scale_feats(df, scaler = MinMaxScaler, skip_coords = True, mask_col = 'split_type')
    df = df[['lon', 'lat'] + [col for col in df.columns if col not in ['lat', 'lon']]]
    if coords_only:
        return df[['lon', 'lat', 'x', 'y', 'label', 'split_type']]
    else:
        return df

def load_newyork( split_rate=None, scale =True, coords_only = False):
    df_raw = load_data_path("nelgiriyewithana/new-york-housing-market", 'NY-House-Dataset.csv')
    df = copy.deepcopy(df_raw[['BEDS', 'BATH', 'PROPERTYSQFT', 'TYPE']])
    category_feats = ['TYPE']
    df["label"] = df_raw['PRICE']/df_raw['PROPERTYSQFT']
    df["lon"] = df_raw["LONGITUDE"] 
    df["lat"] = df_raw['LATITUDE']
    for i in df.columns:
        if i not in category_feats:
            df[i] = pd.to_numeric(df[i], errors='coerce')
    df = df.dropna() # delete nan first!
    df = df.drop_duplicates(subset=["lat", "lon"], keep="last") # drop duplicates
    df = df[(stats.zscore(df["label"])<5) & (stats.zscore(df["label"])>-2)]
    df = df.sample(frac=1).reset_index(drop=True) # random to shuffle the table because no temporal info
    df['x'] = df['lon'].copy()
    df['y'] = df['lat'].copy()
    if split_rate is not None:
        split_type = train_val_test_split(split_rate, length=len(df), shuffle = False, return_type = 'feats')
        df['split_type'] = split_type
        if len(category_feats)>0:
            encoder = CatBoostEncoder(cols = category_feats)
            encoder.fit(df[df['split_type'] == 0], df.loc[df['split_type'] == 0, 'label'])
            df = encoder.transform(df)
    if scale:
        scale_feats(df, scaler = MinMaxScaler, skip_coords = True, mask_col = 'split_type')
    df = df[['lon', 'lat'] + [col for col in df.columns if col not in ['lat', 'lon']]]
    if coords_only:
        return df[['lon', 'lat', 'x', 'y', 'label', 'split_type']]
    else:
        return df
    
def load_paris( split_rate=None, scale =True, coords_only = False):
    df_raw =  load_data_path("benoitfavier/immobilier-france", 'transactions.npz', format='npz')
    df_raw = df_raw[df_raw['ville'].str.startswith("PARIS ")]
    df_raw['date_transaction'] = df_raw['date_transaction'].dt.strftime('%Y-%m-%d')
    df_raw = df_raw[df_raw['date_transaction'] >= '2023-01-01']
    
    df = copy.deepcopy(df_raw[['date_transaction', 'type_batiment','n_pieces',
       'surface_habitable']])
    category_feats = ['type_batiment']
    df["label"] = df_raw['prix']/df_raw['surface_habitable']
    df["lon"] = df_raw["longitude"] 
    df["lat"] = df_raw['latitude']
    df["date_transaction"] = pd.to_numeric(df_raw["date_transaction"].str.replace('-',''), errors='coerce')
    for i in df.columns:
        if i not in category_feats:
            df[i] = pd.to_numeric(df[i], errors='coerce')
    df = df.dropna() # delete nan first!
    df = df.drop_duplicates(subset=["lat", "lon"], keep="last") # drop duplicates
    df = df[(stats.zscore(df["label"])<5) & (stats.zscore(df["label"])>-2)]
    df = df.sort_values(by=['date_transaction']) # temporal split 
    df['x'] = df['lon'].copy()
    df['y'] = df['lat'].copy()
    if split_rate is not None:
        split_type = train_val_test_split(split_rate, length=len(df), shuffle = False, return_type = 'feats')
        df['split_type'] = split_type
        if len(category_feats)>0:
            encoder = CatBoostEncoder(cols = category_feats)
            encoder.fit(df[df['split_type'] == 0], df.loc[df['split_type'] == 0, 'label'])
            df = encoder.transform(df)
    if scale:
        scale_feats(df, scaler = MinMaxScaler, skip_coords = True, mask_col = 'split_type')
    df = df[['lon', 'lat'] + [col for col in df.columns if col not in ['lat', 'lon']]]
    if coords_only:
        return df[['lon', 'lat', 'x', 'y', 'label', 'split_type']]
    else:
        return df
    
def load_gdf(df):
    feats = list(df.columns)
    feats.remove('lon')
    feats.remove('lat')
    d_dict = df[feats].to_dict('list')
    p = [shp.Point(i) for i in np.array(df[['lon', 'lat']]) ] 
    # convert to geopandas 
    gdf = gpd.GeoDataFrame(geometry=p, data=d_dict)
    gdf_train = gdf[gdf['split_type'] == 0].drop(columns= ['split_type', 'x', 'y'])
    gdf_val = gdf[gdf['split_type'] == 1].drop(columns= ['split_type', 'x', 'y'])
    gdf_test = gdf[gdf['split_type'] == 2].drop(columns= ['split_type', 'x', 'y'])

    df_train = gdf[gdf['split_type'] == 0].drop(columns= ['split_type'])
    df_val = gdf[gdf['split_type'] == 1].drop(columns= ['split_type'])
    df_test = gdf[gdf['split_type'] == 2].drop(columns= ['split_type'])
    return (gdf_train, gdf_val, gdf_test), (df_train, df_val, df_test)





