# %%
from pathlib import Path
from typing import Optional
from matplotlib import pyplot as plt
from math import ceil
import numpy as np

import awkward as ak
import uproot

# gen particle
gen_branches=[
'pileup_n',
 'rho',
 'genPh_n',
 'genPh_E',
 'genPh_px',
 'genPh_py',
 'genPh_pz',
 'genPh_pT',
 'genPh_eta',
 'genPh_phi',
]

# simhits
simhit_branches = [
    "simHit_n",
    "simHit_E",
    "simHit_x",
    "simHit_y",
    "simHit_z",
    "simHit_eta",
    "simHit_phi",
    "simHit_layer",
    "simHit_zside",
    "simHit_detid",
]

# rechits
rechit_branches = [
    "recHit_n",
    "recHit_E",
    "recHit_x",
    "recHit_y",
    "recHit_z",
    "recHit_eta",
    "recHit_phi",
    "recHit_layer",
    "recHit_zside",
    "recHit_iCell1",
    "recHit_iCell2",
]

branches = gen_branches + simhit_branches + rechit_branches


rootprefix = "treeMaker/tree"
det_id_dict = {
    v: k
    for k, v in dict(
        Tracker=1,
        Muon=2,
        Ecal=3,
        Hcal=4,
        Calo=5,
        Forward=6,
        VeryForward=7,
        HGCalEE=8,
        HGCalHSi=9,
        HGCalHSc=10,
        HGCalTrigger=11,
    ).items()
}
detectors = [8, 9, 10]
det_labels = [det_id_dict[d] for d in detectors]


class HistCollector:
    def __init__(self) -> None:
        self.arrlist = []

    def save_arr(self, name: str, arr):
        self.arrlist.append([name, arr])

    def plot(self):
        plt.cla()
        plt.clf()
        n_arrs = len(self.arrlist)
        fig, axs = plt.subplots(ceil(n_arrs / 3), 3, figsize=(6 * 3, 4 * n_arrs // 3))

        for iarr in range(n_arrs):
            arr = self.arrlist[iarr][1]
            title = self.arrlist[iarr][0]
            axes: plt.Axes = axs[iarr // 3, iarr % 3]
            # bins
            if title.endswith("layer"):
                bins = np.arange(0, 30, 1)
            elif title[-2:] in ["_x", "_y", "_z"]:
                low = np.quantile(arr, 0.01)
                high = np.quantile(arr, 0.99)
                bins = np.arange(low, high, (high - low)/100.)
                axes.set_yscale("log")
            elif title.endswith("Hit_detector"):
                bins = np.arange(8, 10.5, 0.5)
            else:
                bins = 100

            # Hists
            if len(arr) == 3:
                axes.hist(arr, bins=bins, label=det_labels, stacked=False)
                axes.set_yscale("log")
                axes.legend()
            elif title.endswith("Hit_detector"):
                axes.hist(arr, bins=bins)
                axes.set_yscale("log")
            else:
                axes.hist(arr, bins=bins)
            
            #Scale
            if title.endswith("Hit_E") or title.endswith("Hit_eta"):
                axes.set_yscale("log")
            axes.set_title(title)

        fig.tight_layout()
        self.arrlist = []
        return fig


hist_collector = HistCollector()


# %%
def readpath(
    fn: Path,
    start: Optional[int] = None,
    end: Optional[int] = None,
) -> ak.highlevel.Array:
    with uproot.open(fn) as rfile:
        roottree = rfile[rootprefix]
        if start is end is None:
            array = roottree.arrays(
                branches,
                library="ak",
            )
        elif isinstance(start, int) and isinstance(end, int):
            array = roottree.arrays(
                branches.values(),
                entry_start=start,
                entry_stop=end,
                library="ak",
            )
        else:
            raise ValueError()
        # Some cells don't have correct eta value -> recompute
        new_eta= np.arctanh(array['simHit_z'] / np.sqrt((array['simHit_x']**2 + array['simHit_y']**2 + array['simHit_z']**2)))
        array['simHit_etaFixed'] = new_eta
        #new_eta_gen= np.arctanh(array['genPh_pz'] / np.sqrt((array['genPh_px']**2 + array['genPh_py']**2 + array['genPh_pz']**2)))
        #array['genPh_myEta'] = new_eta_gen
        #z_mask = ak.all(array["simHit_z"]>200,1)
        #return array[z_mask ]
        return array


# %%
data_dir = "/nashome/o/oamram/HGCal/hgcal_photons_fixed_angle_william/"
array = readpath(Path(data_dir + "ntupleTree_1.root"))

# GEN
for var_name in gen_branches:
    try:
        marginal = ak.to_numpy(array[var_name])
        hist_collector.save_arr(var_name,marginal)
    except ValueError:
        print(f"Could not convert {var_name}")
hist_collector.plot().savefig("plots/forward_gen.png")


# SIM 
for var_name in simhit_branches + ['simHit_etaFixed']:
    try:
        if var_name=="simHit_n":
            marginal = ak.to_numpy(array[var_name])
        if('detid' in var_name):
            arr = ak.to_numpy(array[var_name])
            detid_unique = np.unique(arr, axis = 1)
            hist_collector.save_arr('unique_detids',marginal)
            marginal = np.flatten(arr)
        else:
            marginal = ak.to_numpy(ak.flatten(array[var_name]))
        hist_collector.save_arr(var_name,marginal)
        print(marginal)
    except ValueError:
        print(f"Could not convert {var_name}")

hist_collector.plot().savefig("plots/forward_sim.png")

