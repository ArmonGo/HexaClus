from typing import List, Optional
import numpy as np
from geopandas import GeoDataFrame
from shapely.geometry import Polygon
from shapely.geometry import MultiPoint
from srai.embedders import Hex2VecEmbedder
from srai.joiners import IntersectionJoiner
from srai.loaders import OSMOnlineLoader
from srai.neighbourhoods.h3_neighbourhood import H3Neighbourhood
from srai.regionalizers import H3Regionalizer, geocode_to_region_gdf
from shapely.ops import unary_union


class Constructor:
    """Base class for polygon creation strategies."""

    def construct(
        self,
        instances: Optional[GeoDataFrame] = None,
        remove_empty: bool = True,
    ) -> List[int]:
        polygons, feats = self._create_polygons(instances)
        print(f"Created {len(polygons)} polygons")
        polygons_retained = []
        feats_retained = []
        if remove_empty and instances is not None:
            instance_points = unary_union(instances.geometry)
            filtered = [(poly, feat) for poly, feat in zip(polygons, feats) if poly.intersects(instance_points)]
            polygons_retained, feats_retained = zip(*filtered) if filtered else ([], [])
        print(f"Retained {len(polygons_retained)} polygons")
        print(f"Retained {len(feats_retained)} feats")
        return list(polygons_retained), np.array(feats_retained)

    def _create_polygons(self, instances: Optional[GeoDataFrame]) -> List[Polygon]:
        """Create polygons based on the points."""
        raise NotImplementedError(
            "Constructor subclasses must implement _create_polygons()"
        )

class SraiConstructor(Constructor):
    """Creates polygons based on a SRAI regionalizer."""
    def __init__(self, 
                 selected_area  : str, 
                 resolution : int = 9, 
                 encoder_sizes  : Optional[list[int]] = [10, 5],
                 ):
        self.area = geocode_to_region_gdf(selected_area)
        self.loader = OSMOnlineLoader()
        self.joiner = IntersectionJoiner()
        self.regionalizer = H3Regionalizer(resolution=resolution) 
        self.embedder = Hex2VecEmbedder(encoder_sizes)
        
    def  _create_polygons(self, instances: Optional[GeoDataFrame], query = {
                                                        "leisure": "park",
                                                        "railway": "station",
                                                        "amenity": "school",
                                                        "amenity": "university",
                                                        "building": "supermarket",
                                                    }) -> List[Polygon]:
        # area
        area_limited = MultiPoint(list(instances.geometry)).convex_hull
        features_gdf = self.loader.load(area_limited, query)
        self.area.geometry = [area_limited]
        regions = self.regionalizer.transform(self.area)
        joint = self.joiner.transform(regions, features_gdf)
        # generate embeddings 
        
        neighbourhood = H3Neighbourhood(regions_gdf=regions)
        self.embedder.fit(regions, features_gdf, joint, neighbourhood, batch_size=128)
        embeddings = self.embedder.transform(regions, features_gdf, joint)
        return regions.geometry, np.array(embeddings)
            
