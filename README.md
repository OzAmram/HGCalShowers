## HGCalShowers

A repo to process CMS simulations of HGCal into a voxelized geometry in an h5 file format similar to
that of the CaloChallenge.
The HGCal geometry is quite complicated, making the conversion non-trivial.
The inputs to the method are the HGCal NTuples produced with the
[cms\_hgcal\_prod](https://github.com/DeGeSim/cms_hgcal_prod) tool,
as well as a geometry file produced with
[cms\_geo\_extractor](https://github.com/DeGeSim/cms_geo_extractor). 

One first then defines a geometry, which maps between DetID's and each voxel of
the h5, using the `define_geo.py` script.
Cells up to a specified distance away from the shower center are included in
the mapping, the default is 20 rings around the center. 
The geometry stores the DetID mapping as well as the spacial location of each
cell in the h5. 
Note that because there are varying cell sizes and orientations, voxels do
not have a consistent distribution between different layers.
The geometry creation also makes plots to visualize the locations of the cells.
The cells form a rough hexagonal grid, with irregularities on the regions that
border the different cell sizes. 
This geoemtry is then saved as pickled file, geometries for previous HGCal
productions have been saved in the `geoms/` folder. 

This geometry can then be used to convert the HGCal NTuples into the h5 file
format with the `convert_showers.py` script. 
The h5 files have the dimension (NShowers, NLayers, NVoxels). 
NVoxels is the max voxels in a layer, across all the layers, meaning many layers are zero
padded. 
These h5 files can then be used to train your preferred generative model.
Note that because of the irregular structure in each layer, using the geometry
file  which has the spacial location of each cell, as part of your preprocessing can be very useful.

