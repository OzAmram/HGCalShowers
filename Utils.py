# %%
from pathlib import Path
from typing import Optional
from matplotlib import pyplot as plt
from math import ceil
import numpy as np
import pickle

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

#Z position of EE layers in centimeters

layer_Z = [
318.23610858494,
322.80441320201487,
324.9304390125258,
325.9931005063338,
328.0207643215954,
328.9872549851506,
331.0062672838031,
331.94573478881887,
333.98274167957027,
334.9171869123731,
336.93773775558725,
337.8973347631627,
339.9228116094987,
340.867221048759,
342.89269443987047,
343.83732173072275,
345.86275369581557,
346.80716767140086,
348.83279212911356,
349.7773189197943,
351.80277361910015,
352.7472448243897,
354.7727767201523,
355.71726777223705,
357.69951676230744,
358.68731884057974,
360.7128175397913,
361.65723907309723,
]

#hex width (Apothem) is 1.20118mm
#Sim hit lengthn units are mm

def hex_num(n):
    #Return total number of hexagon cells with n rings (ie index starting from 0)
    #https://en.wikipedia.org/wiki/Centered_hexagonal_number
    n +=1
    return 3*n**2 - 3 * n + 1

def ring_num(n):
    #return number of cells in a given ring
    return 6 * n if n > 0 else 1 


all_branches = gen_branches + simhit_branches
def readpath(
    fn: Path,
    start: Optional[int] = None,
    end: Optional[int] = None,
    branches : Optional[list] = None,
) -> ak.highlevel.Array:
    rootprefix = "treeMaker/tree"
    with uproot.open(fn) as rfile:
        if(branches is None): branches = all_branches
        roottree = rfile[rootprefix]
        if start is end is None:
            array = roottree.arrays(
                branches,
                library="ak",
            )
        elif isinstance(start, int) and isinstance(end, int):
            array = roottree.arrays(
                branches,
                entry_start=start,
                entry_stop=end,
                library="ak",
            )
        else:
            raise ValueError()
        # Some cells don't have correct eta value -> recompute
        if('simHit_z' in branches):
            new_eta= np.arctanh(array['simHit_z'] / np.sqrt((array['simHit_x']**2 + array['simHit_y']**2 + array['simHit_z']**2)))
            array['simHit_etaFixed'] = new_eta
        return array





