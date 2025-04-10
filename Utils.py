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
#'pileup_n',
#'rho',
 'genPart_n',
 'genPart_E',
 'genPart_px',
 'genPart_py',
 'genPart_pz',
 'genPart_pT',
 'genPart_eta',
 'genPart_phi',
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
    "simHit_detId",
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





