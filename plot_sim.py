from Utils import * 

# %%
#start = 0
#end = 1000
start = end = None
#data_path = "/uscms/home/oamram/nobackup/CMSSW_14_1_0_pre5/src/test_samples/hgcal_tree.root"
data_path = "/eos/cms/store/group/offcomp-sim/HGCal_Sim_Samples_2024/SinglePion_E-1To1000_Eta-2_Phi-1p57_Z-321-CloseByParticleGun/Phase2Spring24DIGIRECOMiniAOD-noPU_AllTP_140X_mcRun4_realistic_v4-v1_tree/HGCal_TTrees/hgcal_tree_1.root"
array = readpath(Path(data_path), start = start, end = end)

#geo_path = "/uscms/home/oamram/nobackup/CMSSW_14_1_0_pre5/src/DetIdLUT_noGapFix.root"
geo_path = "/eos/cms/store/group/offcomp-sim/HGCal_Sim_Samples_2024/DetIdLUT.root"
rf = uproot.open(geo_path)
geo = rf["analyzer/tree"].arrays()

cell_layer = ak.to_numpy(geo["layerid"])
cell_id =  ak.to_numpy(geo["globalid"])
detector = ak.to_numpy(geo["detectorid"])
cell_x = ak.to_numpy(geo["x"])
cell_y = ak.to_numpy(geo["y"])


detids = array['simHit_detId']
layers = array['simHit_layer']
sim_x = array['simHit_x']
sim_y = array['simHit_y']
sim_z = array['simHit_z']
sim_E = array['simHit_E']
sim_eta = array['simHit_etaFixed']
sim_phi = array['simHit_phi']
gen_eta = ak.to_regular(array['genPart_eta'])
gen_phi = ak.to_regular(array['genPart_phi'])
gen_E = ak.to_regular(array['genPart_E'])
eta_jacobi = np.abs(2* np.exp(-gen_eta)/ (1 + np.exp(-2*gen_eta)))

n_sim = ak.num(sim_eta, axis=1)


dphi = sim_phi -gen_phi
deta = sim_eta - gen_eta

#print(deta, deta[0], deta.type)

print("detector", np.amin(detector), np.amax(detector))
print("cell layer", np.amin(cell_layer), np.amax(cell_layer))
print("CE cell layer", np.amin(cell_layer[detector==8]), np.amax(cell_layer[detector==8]))
print("CH cell layer", np.amin(cell_layer[detector==9]), np.amax(cell_layer[detector==9]))
print("Scin cell layer", np.amin(cell_layer[detector==10]), np.amax(cell_layer[detector==10]))


sim_r = np.sqrt(sim_x**2 + sim_y**2)**(0.5)
deta_mm = deta * eta_jacobi * (sim_r**2 + sim_z**2)**(0.5)
dphi_mm = dphi * sim_r
dR = (deta_mm**2 + dphi_mm**2)**(0.5)
print(dR)


gen_theta = 2* np.arctan(np.exp(-gen_eta[0,0]))
print(gen_eta[0,0], gen_theta, gen_phi[0,0])

nlayers = 28
gen_xs = []
gen_ys = []
center_cells = []
for i in range(nlayers):
    gen_r = np.tan(gen_theta) * layer_Z[i]
    gen_x = gen_r * np.cos(gen_phi[0,0])
    gen_y = gen_r * np.sin(gen_phi[0,0])

    gen_xs.append(gen_x)
    gen_ys.append(gen_y)

    det_mask = detector == 8 
    lay_mask = (cell_layer == (i+1))
    c_x, c_y = cell_x[det_mask & lay_mask], cell_y[det_mask & lay_mask]

    dR = ( (c_x - gen_x)**2 + (c_y - gen_y)**2)**0.5
    center_cell = np.argmin(dR)
    center_cells.append(cell_id[det_mask & lay_mask][center_cell])


dR_max1 = 10.
dR_max2 = 15.
eps = 1e-6

E_tot = ak.sum(sim_E, axis = 1)


#E_frac1 = ak.mean(ak.sum(sim_E[(dR < dR_max1)], axis = 1) / (E_tot + eps))
#E_frac2 = ak.mean(ak.sum(sim_E[(dR < dR_max2)], axis = 1) / (E_tot + eps))

#print(f"TOTAL: Frac. E within {dR_max1} : {E_frac1}, Frac. E within {dR_max2} : {E_frac2}")

E_lay_frac = []
cells_count = []

nShowers = len(gen_eta)

dmax = dR_max2



for lay in range(1,nlayers+1):
    mask = (layers == lay)
    cells_in_lay = detids[layers ==lay]
    cells_count.append(ak.mean(ak.num(cells_in_lay, axis =1)))

    sim_E_lay = sim_E[layers == lay]
    dx = sim_x[layers == lay] - gen_xs[lay-1]
    dy = sim_y[layers == lay] - gen_ys[lay-1]

    dR_v2 = (dx**2 + dy**2)**(0.5)

    E_lay = ak.sum(sim_E_lay, axis = 1)
    E_lay_frac.append(ak.mean(E_lay))
    #dR_mean = ak.mean(dR_v2)
    #E_frac1 = ak.mean(ak.sum(sim_E_lay[(dR_v2 < dR_max1)], axis = 1) / (E_lay + eps))
    #E_frac2 = ak.mean(ak.sum(sim_E_lay[(dR_v2 < dR_max2)], axis = 1) / (E_lay + eps))
    #print(f"Layer {lay}, mean hits {cells_count}, Lay E Frac {E_lay_frac}, dR Avg. {dR_mean}, Frac. E within {dR_max1} : {E_frac1}, Frac. E within {dR_max2} : {E_frac2}")

    
    fig = plt.figure(figsize=(10, 8))
    hb = plt.hist(ak.num(cells_in_lay), 30)
    plt.xlabel("Num. Sim Hits, Layer %i" % lay)
    plt.tight_layout()
    plt.savefig(f"plots/nSimHit_lay%i.png" % lay)

    center_cell_mask = (detids[layers == lay] == center_cells[lay-1])

    fig = plt.figure(figsize=(10, 8))
    hb = plt.hist(ak.num(cells_in_lay[center_cell_mask]), 30)
    plt.xlabel("Num. Sim Hits, Central Cell, Layer %i" % lay)
    plt.tight_layout()
    plt.savefig(f"plots/nSimHit_centerCell_lay%i.png" % lay)


    fig = plt.figure(figsize=(10, 8))
    hb = plt.hexbin(ak.flatten(dx), ak.flatten(dy), gridsize=24, cmap='inferno', extent = (-dmax, dmax, -dmax, dmax))
    plt.xlim(-dR_max2, dR_max2)
    plt.ylim(-dR_max2, dR_max2)
    cb = fig.colorbar(hb, label='counts')
    plt.tight_layout()
    plt.savefig(f"plots/hits_lay%i.png" % lay)

    fig = plt.figure(figsize=(10, 12))
    hb = plt.hexbin(ak.flatten(dx), ak.flatten(dy), ak.flatten(sim_E_lay), gridsize=24, cmap='inferno', extent = (-dmax, dmax, -dmax, dmax))
    plt.xlim(-dR_max2, dR_max2)
    plt.ylim(-dR_max2, dR_max2)
    cb = fig.colorbar(hb, label='Energy')
    plt.tight_layout()
    plt.savefig(f"plots/energy_lay%i.png" % lay)


fig = plt.figure(figsize=(10, 8))
cb = plt.plot(np.arange(nlayers), E_lay_frac)
plt.ylabel("Energy")
plt.xlabel("Layer")
plt.tight_layout()
plt.savefig(f"plots/energy_per_lay.png")
