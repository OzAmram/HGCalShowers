from HGCalGeo import *
from Utils import *
import h5py


data_dir = "/nashome/o/oamram/HGCal/hgcal_photons_fixed_angle_william/"
plot_dir = 'plots/showers/'
#start = 0
#end = 300
start = end = None
branches = ['simHit_detid', 'simHit_layer', 'simHit_x', 'simHit_y', 'simHit_E', 'genPh_eta', 'genPh_phi', 'genPh_E']
array = readpath(Path(data_dir + "ntupleTree_1.root"), start = start, end = end, branches = branches)

fout = "HGCal_showers.h5"

f_geo = open('geom.pkl', 'rb')
geo = pickle.load(f_geo)

detids = array['simHit_detid']
layers = array['simHit_layer']
sim_x = array['simHit_x']
sim_y = array['simHit_y']
sim_E = array['simHit_E']
gen_eta = ak.to_regular(array['genPh_eta'])
gen_phi = ak.to_regular(array['genPh_phi'])
gen_E = ak.to_regular(array['genPh_E'])
num_showers = len(gen_E)

print("Creating %i showers" % num_showers)

showers = np.zeros((num_showers, geo.nlayers, geo.max_cells), dtype=np.float32)
gen_info = np.zeros((num_showers,3), dtype=np.float32)
gen_info[:,0] = ak.flatten(gen_E)
gen_info[:,1] = ak.flatten(gen_eta)
gen_info[:,2] = ak.flatten(gen_phi)


for i in range(num_showers):
    if(i%500 == 0):print("Constructing shower %i" % i)
    showers[i] = geo.voxelize(ak.to_numpy(detids[i]), ak.to_numpy(sim_E[i]))


print("Outputing to %s" % fout)
with h5py.File(fout, "w") as f:
    f.create_dataset("showers", data = showers, compression = 'gzip')
    f.create_dataset("gen_info", data = gen_info, compression = 'gzip')


