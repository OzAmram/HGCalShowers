from HGCalGeo import *
from Utils import *


data_path = "/eos/cms/store/group/offcomp-sim/HGCal_Sim_Samples_2024/SinglePion_E-1To1000_Eta-2_Phi-1p57_Z-321-CloseByParticleGun/Phase2Spring24DIGIRECOMiniAOD-noPU_AllTP_140X_mcRun4_realistic_v4-v1_tree/HGCal_TTrees/hgcal_tree_1.root"
f_geo = 'HGCal_geo_2024.pkl'
plot_dir = 'plots/showers/'

start = None
end = None
branches = ['simHit_detId', 'simHit_layer', 'simHit_x', 'simHit_y', 'simHit_E', 'genPart_eta', 'genPart_phi', 'genPart_E']
array = readpath(Path(data_path), start = start, end = end, branches = branches)

f_geo = open(f_geo, 'rb')
geo = pickle.load(f_geo)
print(geo.nrings)

detids = array['simHit_detId']
layers = array['simHit_layer']
sim_x = array['simHit_x']
sim_y = array['simHit_y']
sim_E = array['simHit_E']
gen_eta = array['genPart_eta']
gen_phi = array['genPart_phi']
gen_E = array['genPart_E']
print(len(gen_E))


gen_theta = 2* np.arctan(np.exp(-gen_eta[:,0]))
nlayers = geo.nlayers
nrings = geo.nrings

n_showers = 3

for i in range(n_showers):
    shower_voxelized = geo.voxelize(ak.to_numpy(detids[i]), ak.to_numpy(sim_E[i]))
    total_shower_E = np.sum(shower_voxelized)
    total_sim_E = np.sum(sim_E[i])
    print("Total energy: direct %.3f vox %.3f gen %.3f" % (total_sim_E, total_shower_E, gen_E[i][0]))
    lay_sum_direct = 0.
    lay_sum_vox = 0.
    for lay in range(1,nlayers+1):
        mask = (layers[i] == lay)
        cells_in_lay = detids[i][mask]

        dx = ak.to_numpy(sim_x[i][mask]) - geo.center_x[lay-1]
        dy = ak.to_numpy(sim_y[i][mask]) - geo.center_y[lay-1]

        Es = ak.to_numpy(sim_E[i][mask])

        dist_mask = (np.abs(dx) < nrings*1.2 ) & (np.abs(dy) < nrings * 1.2)

        lay_E = np.sum(sim_E[i][mask])
        lay_E_inside = np.sum(sim_E[i][mask][dist_mask])

        hex_direct = plot_shower_hex(dx, dy, Es, nrings = nrings )

        gen_r = np.tan(gen_theta[i]) * layer_Z[lay-1]
        gen_x = gen_r * np.cos(gen_phi[i,0])
        gen_y = gen_r * np.sin(gen_phi[i,0])

        dx_gen = gen_x - geo.center_x[lay-1]
        dy_gen = gen_y - geo.center_y[lay-1]

        plt.scatter(dy_gen, -dx_gen, marker = 'x', c = 'black', s=30)
        plt.savefig(plot_dir + "shower%i_lay%i_direct.png" % (i, lay))


        hex_vox = plot_shower_hex(geo.xmap[lay-1], geo.ymap[lay-1], shower_voxelized[lay-1], nrings = nrings)
        plt.scatter(dy_gen, -dx_gen, marker = 'x', c = 'black', s=30)
        plt.savefig(plot_dir + "shower%i_lay%i_voxelized.png" %(i,lay))

        if(hex_direct is not None and hex_vox is not None): 
            print(lay, np.sum(hex_direct.get_array()), np.sum(hex_vox.get_array()))
            lay_sum_direct += np.sum(hex_direct.get_array())
            lay_sum_vox += np.sum(hex_vox.get_array())
    print(lay_sum_direct, lay_sum_vox)


#Do avg across all showers
print(len(sim_x))
for lay in range(1,nlayers+1):
    mask = (layers == lay)

    dx = ak.flatten(sim_x[mask]) - geo.center_x[lay-1]
    dy = ak.flatten(sim_y[mask]) - geo.center_y[lay-1]

    Es = ak.flatten(sim_E[mask])

    hex_direct = plot_shower_hex(dx, dy, Es, nrings = nrings)
    plt.savefig(plot_dir + "avg_shower_lay%i_direct.png" % (lay))
