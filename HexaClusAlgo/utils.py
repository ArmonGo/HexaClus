from typing import List
import numpy as np
from geopandas import GeoDataFrame
from shapely.geometry import Polygon
from tqdm.auto import tqdm


def map_geodataframe_with_polygon_list(
    gdf: GeoDataFrame, polygons: List[Polygon]
) -> GeoDataFrame:
    # Create empty GeoDataFrame to store mapped points
    mapped_gdf = GeoDataFrame()

    # Iterate over the polygons and check for overlaps with the GeoDataFrame
    for polygon in tqdm(polygons):
        msk_overlaps = gdf.geometry.intersects(polygon)
        num_overlaps = np.sum(msk_overlaps)
        if not num_overlaps:
            continue
        # Add a point to the mapped GeoDataFrame using centroid of the polygon
        mapped_gdf = mapped_gdf.append(
            {
                "geometry": polygon.centroid,
                "value": num_overlaps,
                "value_relative": num_overlaps / len(gdf),
            },
            ignore_index=True,
        )

    return mapped_gdf
