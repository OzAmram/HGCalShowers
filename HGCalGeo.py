from pathlib import Path
from typing import Optional
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm as LN
from math import ceil
import numpy as np
import pickle

#hex size = dist from center of edge to center of edge (2 times apothem)
#type = 0 200 micron thick Si, hexsize = 0.80079mm
#type = 1 200 micron thick Si, hexsize = 1.20118mm

class HGCalGeo:
    def __init__(self, nlayers, nrings, hexsize = 1.2):
        self.nrings = nrings
        self.nlayers = nlayers
        #self.max_cells = hex_num(nrings)
        self.max_cells = 3000

        #Width of each hexagon
        self.hexsize = 1.2

        self.shape = (self.nlayers, self.max_cells)

        #array of cells ids
        self.cell_map = np.zeros((nlayers, self.max_cells), np.int64)
        #relative x and y positions of each cell (origin = central cell)
        self.xmap = np.zeros((nlayers, self.max_cells), np.float32)
        self.ymap = np.zeros((nlayers, self.max_cells), np.float32)
        self.ring_map = np.zeros((nlayers, self.max_cells), np.float32) - 1
        self.type_map = np.zeros((nlayers, self.max_cells), np.float32) - 1

        #global x and y locations of center of central cell in each layer
        self.center_x = np.zeros((nlayers))
        self.center_y = np.zeros((nlayers))
        self.ncells = np.zeros((nlayers))

    def get_neighs_by_dist(self, dRs, cell_ids, iRing = 0, hex_size = 1.2):
        dR_cut = (dRs < (iRing + 0.5) * hex_size) &  (dRs > (iRing - 0.5) * hex_size)
        return cell_ids[dR_cut]


    def build_layer(self, ilay, center_id, cell_ids, cell_x, cell_y, neighbors, dRs = None,  cell_type = None, plot = False, manual_neighs = False, hex_size = 1.21, plot_dir = 'plots/'):

        center_cell_idx = np.nonzero(cell_ids == center_id)[0][0]
        print('Contructing GeoMap for Layer %i, center cell is %i' % (ilay, center_id))
        id_lookup_dict = dict()
        for i in range(len(cell_ids)):
            id_lookup_dict[cell_ids[i]] = i


        self.center_x[ilay], self.center_y[ilay] = cell_x[center_cell_idx], cell_y[center_cell_idx]
        print('center x,y', self.center_x[ilay], self.center_y[ilay])

        xs = [0.]
        ys = [0.]
        cs = [1]


        if(plot):
            fig = plt.figure(figsize=(12, 8))
            plt.title("Cells, Layer %i" % (ilay+1))
            plt.scatter( [0], [0], c='black', s = 20)
            colors = ['blue', 'green', 'purple', 'red', 'orange', 'yellow', 'magenta', 'turquoise'] * 10

        filled = set()
        seeds = []

        #starting cell
        filled.add(center_id)
        seeds.append((center_id, center_cell_idx))
        self.cell_map[ilay,0] = center_id

        self.xmap[ilay, 0] = 0.
        self.ymap[ilay, 0] = 0.
        self.ring_map[ilay, 0] = 0
        if(cell_type is not None): self.type_map[ilay, 0] = cell_type[center_cell_idx]

        cell_count = 1


        #Iteratively find all the neighbors
        for i in range(self.nrings):
            new_seeds = []
            thetas = []
            ring = []
            xs = []
            ys = []
            types = []
            ring_map =[]

            if(manual_neighs):
                neighs = self.get_neighs_by_dist(dRs, cell_ids, iRing = i+1, hex_size = hex_size)
            else:
                neighs = []
                for seed_id, seed_idx in seeds:
                    for j,neigh in enumerate(neighbors):
                        neighs.append(neigh[seed_idx])

            for nid in neighs:
                if(nid in filled or nid == 0): continue #already added

                n_idx =  id_lookup_dict.get(nid)
                if(n_idx is not None):
                    n_x, n_y = cell_x[n_idx] - self.center_x[ilay], cell_y[n_idx] - self.center_y[ilay]
                    #print(nid, n_x,n_y)

                    filled.add(nid)
                    new_seeds.append((nid, n_idx))
                    xs.append(n_x)
                    ys.append(n_y)
                    if(cell_type is not None): types.append(cell_type[n_idx])

                    #angle from 0 to 2pi, arctan2 has (y,x) convention for some reason
                    theta = np.arctan2(n_y, n_x) % (2. *np.pi)
                    thetas.append(theta)
                    ring.append(nid)
                    ring_map.append(i+1)
                    cs.append(theta)

            seeds = new_seeds


            #Order cells in each layer by theta
            #aka counterclockwise starting at 3 oclock
            order = np.argsort(thetas)
            sorted_ids = np.array(ring)[order].reshape(-1)
            sorted_xs = np.array(xs)[order].reshape(-1)
            sorted_ys = np.array(ys)[order].reshape(-1)
            sorted_types = np.array(types)[order].reshape(-1)
            n_ring_cells = len(thetas)

            #print(thetas)
            #rs = (sorted_xs**2 + sorted_ys**2)**0.5
            #print(sorted_xs)
            #print(sorted_ys)
            #print(sorted_types)

            self.cell_map[ilay, cell_count : cell_count + n_ring_cells] = sorted_ids
            self.xmap[ilay, cell_count : cell_count + n_ring_cells] = sorted_xs
            self.ymap[ilay, cell_count : cell_count + n_ring_cells] = sorted_ys
            self.type_map[ilay, cell_count : cell_count + n_ring_cells] = sorted_types
            self.ring_map[ilay, cell_count : cell_count + n_ring_cells] = np.array(ring_map)

            if(plot): plt.scatter( sorted_ys, -sorted_xs, c=colors[i], s= 20)

            cell_count += n_ring_cells


        print("Found %i cells" % cell_count)
        self.ncells[ilay] = cell_count
        if(plot):
            #thetas = np.arctan2(self.ymap[ilay], self.xmap[ilay]) % (2. * np.pi)
            vals = np.ones_like(self.xmap[ilay])
            #plot_shower_hex(self.xmap[ilay], self.ymap[ilay], vals, nrings = self.nrings, alpha = 0.5, cmap = 'Blues', fig = fig)
            fout = plot_dir + 'geo_lay%i.png' %(ilay+1)
            print("Saving %s" % fout)
            plt.savefig(fout)
    
    def construct_id_map(self):
        #dictionary for quick inverse lookup
        self.id_map = dict()
        for lay in range(self.nlayers):
            for idx in range(len(self.cell_map[lay])):
                cellid = self.cell_map[lay, idx]
                self.id_map[cellid] = (lay, idx)

    def voxelize(self, cellids, energies):
        shower = np.zeros(self.cell_map.shape)
        #Ugly for loop, this could be better optimized, 
        #vanilla np doesn't support vectorized dict lookup though
        for i,cellid in enumerate(cellids):
            idx = self.id_map.get(cellid)
            if(idx is not None): shower[idx] += energies[i]
        return shower


    def save(self, fout):
        f = open(fout, 'wb')
        pickle.dump(self, f)
        f.close()


def plot_shower_hex(xs, ys, Es, nrings = 8, hexwidth = 1.2091, fout = "", fig = None, alpha = 1.0, cmap = 'viridis', zscale = 'max', vmin = 1e-4, vmax = None, log_scale = True):
    """Plot a shower with proper hexagonal binning.
    x and y locations assumed to be relative to center of a central cell.
    Note that the HGCal geometry has 'horizonally' oriented hexagons (pointy edge of center cell is up)
    But matplotlib plots vertically oriented hexagons, so we rotate the shower 90 degrees clockwise before plotting"""

    #skip empty showers
    if(np.sum(Es) <= 0.): return None



    if (fig is None): fig = plt.figure(figsize=(12, 8))

    #set up hex binning
    nhex = nrings*2 + 1
    nhex +=2
    width = nhex * hexwidth / 2.0
    #mpl counts number of hex's in y strangely
    nyhex = int(np.ceil(nhex / np.sqrt(3)))
    ratio = nyhex / (nhex / np.sqrt(3))
    ywidth = width * ratio
    extent = (-width, width, -ywidth, ywidth)

    #Mpl makes it hard/impossible to fully specify hexbin locations
    #First make a dummy set of hexbins, and then compute an offset to correct the center location
    hexbins = plt.hexbin([0], [0], gridsize = (nhex, nyhex), extent=extent, alpha = 0.)
    all_offsets = np.array(hexbins.get_offsets())

    dists = (all_offsets[:,0]**2 + all_offsets[:,1]**2)**0.5
    closest = np.argmin(dists)
    #offset of central hexbin from origin 
    xoff, yoff = all_offsets[closest]

    xs = np.array(xs)
    ys = np.array(ys)

    #rotate 90 degrees to align hex orientation
    xs_rot,ys_rot = ys, -xs


    if(vmax is None): vmax = np.amax(Es)
    if(log_scale):
        norm = LN(vmin, vmax)
        vmin,vmax = None, None
    else:
        norm = None
    
    hexbins2 = plt.hexbin(xs_rot + xoff, ys_rot + yoff, C = Es, gridsize = (nhex, nyhex), extent=extent, edgecolor = 'black', 
            alpha = alpha, cmap = cmap, vmin = vmin, vmax = vmax, norm = norm, reduce_C_function = np.sum)

    #correct offset
    all_offsets2 = np.array(hexbins2.get_offsets())
    hexbins2.set_offsets(all_offsets2 - [xoff, yoff])
    plt.colorbar()
    plt.tight_layout()

    if(fout != ""): plt.savefig(fout)

    return hexbins2
