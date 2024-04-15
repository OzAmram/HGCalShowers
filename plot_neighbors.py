# %%
import yaml
import uproot
import os
import pickle
import numpy as np
import pandas as pd
import awkward as ak

# %%
import matplotlib as mpl

# mpl.use("pgf")
import matplotlib.pyplot as plt
import seaborn as sns



# plt.rcParams.update({
#     "text.usetex": True,
#     "pgf.texsystem": "lualatex",
#     "pgf.rcfonts": False,
#     "font.family": "serif",
#     "font.serif": [],
#     "font.sans-serif": [],
#     "font.monospace": [],
#     # "figure.figsize": [default_width, default_width * default_ratsio],
#     "pgf.preamble": "\\usepackage{mymplsetup}"
# })
# plt.rcParams = plt.rcParamsDefault


plt.rcParams = plt.rcParamsDefault
mpl.rcParams.update({"font.size": 16})
# %%
rf = uproot.open(d_dir + "DetIdLUT.root")
arr = rf["analyzer/tree"].arrays()
keydf = ak.to_pandas(arr[0])
keydf = keydf.set_index("globalid")
detIDD = {8: "EE", 9: "HSi", 10: "HSc"}
keydf["detectorid"] = keydf["detectorid"].apply(lambda x: detIDD[x])

keydf.head()
# %%
# Debug code to see the if the arrays are filled correctly
index = [
    "globalid",
    "detectorid",
    "subdetid",
    "layerid",
    "waferortileid.first",
    "waferortileid.second",
    "cellid.first",
    "cellid.second",
    "x",
    "y",
    "celltype",
    "issilicon",
    "next",
    "previous",
    "nneighbors",
    "ngapneighbors",
    "n0",
    "n1",
    "n2",
    "n3",
    "n4",
    "n5",
    "n6",
    "n7",
]
for key in keydf.columns:
    foo = ak.to_pandas(arr[0][key])
    print(key, foo.shape)


# %%
#fngeopic = d_dir + "geometry.pickle"
#if os.path.isfile(fngeopic):
#    with open(fngeopic, "rb") as f:
#        geoD = pickle.load(f)
#else:
#    with open(d_dir + "geometry.yaml", "r") as f:
#        geoD = yaml.load(f, Loader = yaml.CLoader)
#    with open(fngeopic, "wb") as f:
#        pickle.dump(geoD, f)
#
#print("Loaded!")
# %%
plotall = True

# %%
# # Types of the detector cells
if plotall:
    sns.catplot(
        x="celltype",
        data=keydf,
        kind="count",
        hue="detectorid",
    )
    # plt.tight_layout()
    plt.savefig("plots/CellTypes.png")
# %%
# Number of neighbors
if plotall:
    ax = sns.catplot(
        x="layerid",
        data=keydf,
        kind="count",
        hue="detectorid",
        legend_out=True,
        height=4,
        aspect=2.5,
    )
    locs, labels = plt.xticks()
    plt.xticks(locs[::3], labels[::3])
    plt.savefig("plots/CellsPerLayer.png")
# %%
if plotall:
    plt.cla()
    plt.clf()
    plt.close()

    fig, axes = plt.subplots(4, 7, figsize=(35, 20), sharex=True, sharey=False)
    for i in range(len(axes)):
        for j in range(len(axes[0])):
            layerid = i * (len(axes[0]) - 1) + j + 1
            if layerid >= 23:
                continue
            dfsel = keydf[(keydf["layerid"] == layerid) & (keydf["detectorid"] == "EE")]
            print(f"pos {i} {j} => {layerid} ({len(dfsel)})")
            sns.histplot(
                x="nneighbors",
                data=dfsel,
                discrete=True,
                multiple="dodge",
                ax=axes[i][j],
            )
            axes[i][j].set_title(f"layer {layerid}")
            axes[i][j].set_yscale("log")
    fig.suptitle("Number of Neighbors per Cell in HgCalEE")

    plt.tight_layout()
    fig.subplots_adjust(top=0.95)
    plt.savefig("plots/neighborsHistEE.png")
# # %%

# Hadronic Neighbors
if plotall:
    plt.cla()
    plt.clf()
    plt.close()

    fig, axes = plt.subplots(4, 7, figsize=(35, 20), sharex=True, sharey=False)
    for i in range(len(axes)):
        for j in range(len(axes[0])):
            layerid = i * (len(axes[0]) - 1) + j + 1
            if layerid >= 23:
                continue
            dfsel = keydf[(keydf["layerid"] == layerid) & (keydf["detectorid"] != "EE")]
            print(f"pos {i} {j} => {layerid} ({len(dfsel)})")
            sns.histplot(
                x="nneighbors",
                data=dfsel,
                discrete=True,
                multiple="dodge",
                hue="detectorid",
                ax=axes[i][j],
            )
            axes[i][j].set_title(f"layer {layerid}")
            axes[i][j].set_yscale("log")
    fig.suptitle("Number of Neighbors per Cell in HSi and HSc")

    plt.tight_layout()
    fig.subplots_adjust(top=0.95)
    plt.savefig("plots/neighborsHistH.png")


# %%
# Added neighbors
if plotall:
    plt.cla()
    plt.clf()
    plt.close()

    fig, axes = plt.subplots(4, 7, figsize=(35, 20), sharex=True, sharey=True)
    for i in range(len(axes)):
        for j in range(len(axes[0])):
            layerid = i * (len(axes[0]) - 1) + j + 1
            if layerid >= 23:
                continue
            dfsel = keydf[
                (keydf["layerid"] == layerid)
                & (keydf["detectorid"] != "EE")
                & (keydf["ngapneighbors"] != 0)
            ]
            print(f"pos {i} {j} => {layerid} ({len(dfsel)})")
            sns.histplot(
                x="ngapneighbors",
                data=dfsel,
                discrete=True,
                multiple="dodge",
                hue="detectorid",
                ax=axes[i][j],
            )
            axes[i][j].set_title(f"layer {layerid}")
    fig.suptitle("Number of added Neighbors from the other subdetector in the HGCalH")

    plt.tight_layout()
    fig.subplots_adjust(top=0.95)
    plt.savefig("plots/gapneighbors.png")

# %%
# Hadronic Neighbors with gapfixing
if plotall:
    plt.cla()
    plt.clf()
    plt.close()

    fig, axes = plt.subplots(4, 7, figsize=(35, 20), sharex=True, sharey=False)
    for i in range(len(axes)):
        for j in range(len(axes[0])):
            layerid = i * (len(axes[0]) - 1) + j + 1
            if layerid >= 23:
                continue
            dfsel = keydf[(keydf["layerid"] == layerid) & (keydf["detectorid"] != "EE")]
            dfsel = dfsel.assign(
                totalneighbors=dfsel["nneighbors"] + dfsel["ngapneighbors"]
            )
            print(f"pos {i} {j} => {layerid} ({len(dfsel)})")
            sns.histplot(
                x="totalneighbors",
                data=dfsel,
                discrete=True,
                multiple="dodge",
                hue="detectorid",
                ax=axes[i][j],
            )
            axes[i][j].set_title(f"layer {layerid}")
            axes[i][j].set_yscale("log")
    fig.suptitle(
        "Number of Neighbors per Cell in HSi and HSc after fixing the gaps."
    )

    plt.tight_layout()
    fig.subplots_adjust(top=0.95)
    plt.savefig("plots/neighborsHistH_with_gapfixing.png")

# %%
# Cell Scatterplot
if plotall:
    plt.cla()
    plt.clf()
    plt.close()

    fig, axes = plt.subplots(
        4, 7, figsize=(35, 20), sharex=True, sharey=True, squeeze=True
    )
    for i in range(len(axes)):
        for j in range(len(axes[0])):
            layerid = i * (len(axes[0]) - 1) + j + 1
            if layerid >= 23:
                continue
            dfsel = keydf[(keydf["layerid"] == layerid) & (keydf["detectorid"] != "EE")]

            dfsel = pd.concat(
                [
                    dfsel[(keydf["detectorid"] == "HSi")],
                    dfsel[(keydf["detectorid"] == "HSc")],
                ]
            )
            print(f"pos {i} {j} => {layerid} ({len(dfsel)})")
            sns.scatterplot(
                x="x",
                y="y",
                data=dfsel,
                hue="detectorid",
                ax=axes[i][j],
            )
            axes[i][j].set_title(f"layer {layerid}")
    fig.suptitle("Cells Scatterplot")

    plt.tight_layout()
    fig.subplots_adjust(top=0.95)
    plt.savefig("plots/scatter.png")

# %%
def is_unidirectional(originid, targetid):
    target = keydf.loc[targetid]
    directed = originid not in target[["n" + str(e) for e in range(8)]].values
    return directed


# %%
# check for unidirectional connections
cellsDirectedS = set()
for originid, row in keydf.iterrows():
    if row.ngapneighbors == 0:
        continue
    for i in range(row.nneighbors, row.nneighbors + row.ngapneighbors):
        if(i >= 8): continue
        targetid = row["n" + str(i)]
        directed = is_unidirectional(originid, targetid)
        if directed:
            cellsDirectedS.add(originid)
            cellsDirectedS.add(targetid)

# %%
### Arrow plot
# Cell Scatterplot
if plotall:
    for layerid in range(9, 23):
        #  if layerid != 21:
        #      continue
        plt.cla()
        plt.clf()
        plt.close()

        fig = plt.figure(figsize=(10, 12))

        dfsel = keydf[(keydf["layerid"] == layerid) & (keydf["detectorid"] != "EE")]
        print(f"layer {layerid}")

        sns.scatterplot(
            x="x",
            y="y",
            data=dfsel,
            hue="detectorid",
            #  markers=["o", "o"],
            style="detectorid",
        )
        for originid, row in dfsel.iterrows():
            # if row.ngapneighbors == 0:
            #     continue
            # for i in range(row.nneighbors, row.nneighbors + row.ngapneighbors):
            if (
                originid % 10 != 0
                and row.ngapneighbors == 0
                and originid not in cellsDirectedS
            ):
                continue
            for i in range(row.nneighbors + row.ngapneighbors):
                if(i >= 8): continue
                targetid = row["n" + str(i)]

                if (
                    # only regular neighbors
                    i < row.nneighbors
                    # with gapneighbors
                    and row.ngapneighbors > 0
                    # that dont have a dicrected connection
                    and originid not in cellsDirectedS
                    and targetid not in cellsDirectedS
                    # every third cells
                    and originid % 3 != 0
                ):
                    continue

                dx = keydf.loc[targetid].x - row.x
                dy = keydf.loc[targetid].y - row.y
                if i >= row.nneighbors:
                    color = "blue"
                else:
                    color = "black"
                if is_unidirectional(originid, targetid):
                    color = "red"
                plt.arrow(
                    row.x,
                    row.y,
                    dx,
                    dy,
                    head_width=0.1,
                    color=color,
                    length_includes_head=True,
                )
                # plt.arrow(row.x, row.y, dx, dy, head_width=0.4, length_includes_head=True)

        plt.title(f"Connections added by gapfixing layer {layerid}")

        plt.tight_layout()
        fig.subplots_adjust(top=0.95)
        plt.savefig(f"plots/gaparrows-{layerid}.png")

# %%


# %%
def getnextcellsfreq(detector, layer, threshold=1):
    foo = [
        geoD[detector][layer][wafer][cell]["next"]
        for wafer in recurkeys(geoD[detector][layer])
        for cell in recurkeys(geoD[detector][layer][wafer])
    ]
    return [
        b for a, b in np.array(np.unique(foo, return_counts=True)).T if b > threshold
    ]


def countcells(detector, layer):
    foo = [
        1  # geoD[detector][layer][wafer][cell]
        for wafer in recurkeys(geoD[detector][layer])
        for cell in recurkeys(geoD[detector][layer][wafer])
    ]
    return len(foo)
