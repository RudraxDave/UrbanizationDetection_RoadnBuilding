#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import rasterio
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt

from pprint import pprint
from rasterio.features import Window
from rasterio.windows import bounds
from shapely.geometry import MultiPolygon, box
from PIL import Image
from rasterio.features import Window
from subprocess import call
from IPython import display


# In[ ]:


get_ipython().system('head -n 30 buildings.geojson')


# In[ ]:


import geopandas as gpd
g = gpd.read_file('buildings.geojson')
g.head()


# In[ ]:


g[g['material'] == 'wood']


# In[ ]:


g.plot()


# In[ ]:


s = g['geometry'].loc[1]
s


# In[ ]:


print(s)


# In[ ]:


s.area


# In[ ]:


s.centroid


# In[ ]:


get_ipython().system('gdalinfo img.tif')


# In[ ]:


import rasterio
with rasterio.open('img.tif', 'r') as src:
    img = src.read()

img


# In[ ]:


import numpy as np
np.max(img)


# In[ ]:


import rasterio
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt

from pprint import pprint
from rasterio.features import Window
from rasterio.windows import bounds
from shapely.geometry import MultiPolygon, box
from PIL import Image
from rasterio.features import Window
from subprocess import call
from IPython import display
plt.imshow(img.T)


# In[ ]:


with rasterio.open('img.tif') as src:
    witdth = src.width
    hight = src.height
    p = src.profile.copy()


# In[ ]:


col_off = (src.width / 2) - (1000 / 2)
row_off = (src.height / 2) - (1000 / 2)


# In[ ]:


pprint(p['transform'])


# In[ ]:


win = Window(col_off=col_off, row_off=row_off, width=1000, height=1000)


# In[ ]:


x = rasterio.open('img.tif').window_transform(win)


# In[ ]:


x


# In[ ]:


with rasterio.open('img.tif') as src:
    f = src.read(window=win)
    p = src.profile.copy()
    p['width'] = win.width
    p['height'] = win.height
    p['transform'] = src.window_transform(win)


# In[ ]:


f.shape


# In[ ]:


with rasterio.open('sample.tif', 'w', **p) as dst:
    dst.write(f)


# In[ ]:


get_ipython().system('ls')


# In[ ]:


all_bldgs = gpd.read_file('buildings.geojson')


# In[ ]:


img_bounds = src.bounds


# In[ ]:


# left, bottom, right, top
l, b, r, t = img_bounds


# In[ ]:


from shapely.geometry import Polygon


# In[ ]:


from shapely.geometry import Polygon
img_bbox = Polygon([(l, b), (l, t), (r, t), (r, b)])
img_bbox


# In[ ]:


import geopandas as gpd
bbox_gdf = gpd.GeoDataFrame({'geometry': [img_bbox]}, crs = 32636)


# In[ ]:


bbox_gdf.plot()


# In[ ]:


all_bldgs


# In[ ]:


bldgs = gpd.overlay(all_bldgs, bbox_gdf, how='intersection')


# In[ ]:


import os
from shutil import rmtree

if os.path.isdir('buildings'):
    rmtree('buildings')
os.makedirs('buildings/true')
os.makedirs('buildings/false')


# In[ ]:


mp = MultiPolygon(bldgs['geometry'].values)


# In[ ]:


# specify the png image size (in pixels) 
png_size = 600


# In[ ]:


with rasterio.open('img.tif') as src:
    
    # gather width and height of input image
    width = src.width
    height = src.height
    
    # iterate over the image in a grid of 1200x1200 pixel squares
    for w in range(0, width, png_size):
        for h in range(0, height, png_size):

            # construct Window object using row/col and size
            win = Window(w, h, png_size, png_size)
            
            # find the corresponding spatial coordinates
            trans = src.window_transform(win)
            
            # read the window portion in as a numpy array
            a = src.read(window=win)
            
            # create shapely object that represents the bounds of the window
            p = src.profile.copy()
            p['width'] = win.width
            p['height'] = win.height
            p['transform'] = src.window_transform(win)
            with rasterio.open('/tmp/tmp.tif', 'w', **p) as dst:
                bnds = dst.bounds
            
            x = Polygon(box(*bnds))
            
            # check whether the window intersects with any buildings
            has_bldg = x.intersects(mp)
            
            if has_bldg is True:
                label = 'true'
            else:
                label = 'false'
                        
            # create a PIL image from the numpy array
            im = Image.fromarray(a[0:3].T)
            
            # save the image off as a png
            fp = f'buildings/{label}/{w}-{h}.png'
            im.save(fp)


# In[ ]:


display.Image(filename=f'buildings/false/{os.listdir("buildings/false")[1]}')


# In[ ]:


display.Image(filename=f'buildings/true/{os.listdir("buildings/true")[1]}')


# In[ ]:


# get a shapely polygon of a window's bounding box
def window_bbox(dataset_reader, win):
    p = dataset_reader.profile.copy()
    p['width'] = win.width
    p['height'] = win.height
    p['transform'] = dataset_reader.window_transform(win)
    with rasterio.open('/tmp/tmp.tif', 'w', **p) as dst:
        bnds = dst.bounds
            
    x = Polygon(box(*bnds))
    
    return x


# In[ ]:


with rasterio.open('img.tif', 'r') as src:
    win = Window(0, 0, 1000, 1000)
    a = src.read(window = win)
    win_bbox = window_bbox(src, win)
    trans = src.window_transform(win)


# In[ ]:


win_bldgs = gpd.clip(bldgs, win_bbox)


# In[ ]:


win_bldgs['condition_num'] = win_bldgs['condition'].apply(lambda x: 1 if x == 'poor' else 2)


# In[ ]:


win_bldgs


# In[ ]:


from rasterio.features import rasterize


# In[ ]:


def tuple_to_list(t):
    return list(map(tuple_to_list, t)) if isinstance(t, (list, tuple)) else t


# In[ ]:


def preformat_geom(geom):
    gj = geom.__geo_interface__
    gj['coordinates'] = tuple_to_list(gj['coordinates'])
    
    return gj


# In[ ]:


shapes = list(zip(win_bldgs['geometry'], win_bldgs['condition_num']))


# In[ ]:


shapes


# In[ ]:


a.shape


# In[ ]:


out_shape = (1000, 1000)


# In[ ]:


fill = 0


# In[ ]:


transform = trans


# In[ ]:


dtype = a.dtype
dtype


# In[ ]:


pixel_labs = rasterize(shapes=shapes, out_shape=out_shape, fill=fill, transform = transform, dtype = dtype)


# In[ ]:


pixel_labs.shape


# In[ ]:


# an example of the middle of the raster layer
pixel_labs[400:600, 400:600]


# In[ ]:


plt.imshow(pixel_labs)


# In[ ]:


plt.imshow(a[0:3].transpose([1, 2, 0]))


# In[ ]:


win_bldgs.to_file('rasterize-bldgs.geojson', driver='GeoJSON')


# In[ ]:


get_ipython().system('gdal_rasterize -a condition_num -a_srs EPSG:32636 -ts 1000 1000 rasterize-bldgs.geojson bldg-raster.tif')


# In[ ]:


with rasterio.open('bldg-raster.tif', 'r') as src:
    a = src.read()


# In[ ]:


plt.imshow(a[0])

