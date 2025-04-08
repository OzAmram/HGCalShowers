from Utils import *
from HGCalGeo import *

plot_dir = 'plots/'

geo_path = "/eos/cms/store/group/offcomp-sim/HGCal_Sim_Samples_2024/DetIdLUT.root"
data_path = "/eos/cms/store/group/offcomp-sim/HGCal_Sim_Samples_2024/SinglePion_E-1To1000_Eta-2_Phi-1p57_Z-321-CloseByParticleGun/Phase2Spring24DIGIRECOMiniAOD-noPU_AllTP_140X_mcRun4_realistic_v4-v1_tree/HGCal_TTrees/hgcal_tree_1.root"
fout = 'HGCal_geo_2024.pkl'

rf = uproot.open(geo_path)
geo = rf["analyzer/tree"].arrays()
do_plot = True

manual_neighs = True
nrings = 20

hex_size = 1.2091 #new
#hex_size = 1.2011 #william

keydf = ak.to_dataframe(geo[0])
keydf = keydf.set_index("globalid")

#branches = ['genPh_eta', 'genPh_phi']
branches = ['genPart_eta', 'genPart_phi']
array = readpath(Path(data_path), start = None, end = None, branches = branches )


cell_layer = (ak.to_numpy(geo["layerid"][0])).reshape(-1)
cell_id =  ak.to_numpy(geo["globalid"][0]).reshape(-1)
detector = ak.to_numpy(geo["detectorid"][0]).reshape(-1)
cell_x = ak.to_numpy(geo["x"][0]).reshape(-1)
cell_y = ak.to_numpy(geo["y"][0]).reshape(-1)
cell_type = ak.to_numpy(geo["celltype"][0]).reshape(-1)
cell_is_silicon = ak.to_numpy(geo["issilicon"][0]).reshape(-1)


neighs = [ak.to_numpy(geo['n%i'%i][0] ).reshape(-1) for i in range(8)]



gen_eta = ak.to_numpy(array[branches[0]])
gen_phi = ak.to_numpy(array[branches[1]])
gen_theta = 2* np.arctan(np.exp(-gen_eta[:,0]))

print("Gen eta, phi : %.2f, %.2f" % (gen_eta[0], gen_phi[0]))

#Restrict to EE
det_mask = detector == 8 
#det_mask = np.abs(detector) > 0
#print(np.amin(detector), np.amax(detector))

nlayers = np.amax(cell_layer)
print("%i layers" % nlayers)

Rlength = 15
xlength = Rlength/(2**0.5)
ylength = xlength




#Order of cells stored in each layer

geom = HGCalGeo(nlayers, nrings)



for lay in range(1, nlayers+1):

    #find avg gen particle location
    gen_r = np.tan(gen_theta) * layer_Z[lay-1]
    gen_x = np.mean(gen_r * np.cos(gen_phi[:,0]))
    gen_y = np.mean(gen_r * np.sin(gen_phi[:,0]))


    lay_mask = (cell_layer == lay)
    c_x, c_y = cell_x[det_mask & lay_mask], cell_y[det_mask & lay_mask]

    dR = ( (c_x - gen_x)**2 + (c_y - gen_y)**2)**0.5
    #restrict dR range to reduce search space
    dR_cut = dR < (nrings * hex_size * 2.0)
    center_cell = np.argmin(dR)


    mask = lay_mask & det_mask
    mask[mask] = dR_cut
    center_cell_id = cell_id[lay_mask & det_mask][center_cell]
    c0 = np.nonzero(cell_id == center_cell_id)[0]
    print("Center", center_cell_id, np.amin(dR))

    #plt.figure()
    #plt.scatter(cell_x[mask], cell_y[mask], s=20)
    #plt.savefig(plot_dir + "All_scatter.png")

    neigh_lay = [n[mask] for n in neighs]
    geom.build_layer(lay-1, center_cell_id, cell_id[mask], cell_x[mask], cell_y[mask], neigh_lay, cell_type = cell_type[mask], dRs = dR[dR_cut],
            plot = do_plot, plot_dir = plot_dir, manual_neighs = manual_neighs, hex_size = hex_size)

geom.construct_id_map()
geom.save(fout)

