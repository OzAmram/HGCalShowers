# %%
from pathlib import Path
from typing import Optional

import awkward as ak
import uproot

rootprefix = "treeMaker/tree"
braches = {"id": "simHit_detid", "energy": "genPh_E", "hit_energy": "simHit_E"}


# %%
def readpath(
    fn: Path,
    start: Optional[int] = None,
    end: Optional[int] = None,
) -> ak.highlevel.Array:
    with uproot.open(fn) as rfile:
        roottree = rfile[rootprefix]
        if start is end is None:
            return roottree.arrays(
                list(braches.values()),
                library="ak",
            )
        elif isinstance(start, int) and isinstance(end, int):
            return roottree.arrays(
                list(braches.values()),
                entry_start=start,
                entry_stop=end,
                library="ak",
            )
        else:
            raise ValueError()


# %%
#data_dir = "~/nashome/HGCal/hgcal_photons_fixed_angle_william/"
data_dir = "/nashome/o/oamram/HGCal/hgcal_photons_fixed_angle_william/"
array = readpath(Path(data_dir + "ntupleTree_1.root"))
