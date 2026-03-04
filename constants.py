import open3d as o3d
import torch
import numpy as np

OBJECT_CLICK_COLOR = [0.2, 0.81, 0.2] # colors between 0 and 1 for open3d
BACKGROUND_CLICK_COLOR = [0.81, 0.2, 0.2] # colors between 0 and 1 for open3d
# A unit tetrahedron (4 vertices, 4 triangular faces).
# You can scale it later per instance.
TETRA_VERTS = np.array([
    [ 1,  1,  1],
    [-1, -1,  1],
    [-1,  1, -1],
    [ 1, -1, -1],
], dtype=np.float32) * 0.01  # small base size

TETRA_FACES = np.array([
    [0, 2, 1],
    [0, 1, 3],
    [0, 3, 2],
    [1, 2, 3],
], dtype=np.int32)

FURNITURE_CLASS_IDS =  {0: "background",
                        1: "bathtub",
                        2: "bed",
                        3: "chair",
                        4: "desk",
                        5: "dresser",
                        6: "monitor",
                        7: "night_stand",
                        8: "sofa",
                        9: "table",
                        10: "toilet"}


FURNITURE_CLASS_COLORS = {
    0: [0.0, 0.0, 0.0],        # Background (black)
    1: [1.0, 0.2, 0.2],        # Bathtub (red family)
    2: [0.2, 1.0, 0.2],        # Bed (green family)
    3: [0.2, 0.2, 1.0],        # Chair (blue family)
    4: [1.0, 1.0, 0.2],        # Desk (yellow family)
    5: [1.0, 0.5, 1.0],        # Dresser (purple family)
    6: [1.0, 0.75, 0.75],      # Monitor (light red family)
    7: [0.5, 0.75, 0.75],      # Night Stand (light green family)
    8: [0.75, 0.75, 0.5],      # Sofa (light blue family)
    9: [0.75, 0.75, 0.75],     # Table (light yellow family)
   10: [0.5, 0.5, 0.75]        # Toilet (light purple family)
}

PRISM_CLASS_IDS = {
    0: "Sphere",
    1: "Prism",
    2: "Cone"
}

PRISM_CLASS_COLORS = {
    0: [1.0, 0, 0],              # Sphere
    1: [0, 1.0, 0],              # Rectangular Prism
    2: [0, 0, 1.0]               # Cone               
}


###* ScanNet Below
#! Note to self, I have changed it from the old version to one where the classId's are actually in order
#! It was stupid otherwise and causing too many annoying bugs
VALID_CLASS_IDS_20 = (
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    14,
    16,
    24,
    28,
    33,
    34,
    36,
    39,
)

CLASS_LABELS_20 = (
    "wall",
    "floor",
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "desk",
    "curtain",
    "refrigerator",
    "shower curtain",
    "toilet",
    "sink",
    "bathtub",
    "otherfurniture",
)

SCANNET_COLOR_MAP_20 = {
    0: (0.0, 0.0, 0.0),
    1: (174.0, 199.0, 232.0),
    2: (152.0, 223.0, 138.0),
    3: (31.0, 119.0, 180.0),
    4: (255.0, 187.0, 120.0),
    5: (188.0, 189.0, 34.0),
    6: (140.0, 86.0, 75.0),
    7: (255.0, 152.0, 150.0),
    8: (214.0, 39.0, 40.0),
    9: (197.0, 176.0, 213.0),
    10: (148.0, 103.0, 189.0),
    11: (196.0, 156.0, 148.0),
    12: (23.0, 190.0, 207.0),
    14: (247.0, 182.0, 210.0),
    15: (66.0, 188.0, 102.0),
    16: (219.0, 219.0, 141.0),
    17: (140.0, 57.0, 197.0),
    18: (202.0, 185.0, 52.0),
    19: (51.0, 176.0, 203.0),
    20: (200.0, 54.0, 131.0),
    21: (92.0, 193.0, 61.0),
    22: (78.0, 71.0, 183.0),
    23: (172.0, 114.0, 82.0),
    24: (255.0, 127.0, 14.0),
    25: (91.0, 163.0, 138.0),
    26: (153.0, 98.0, 156.0),
    27: (140.0, 153.0, 101.0),
    28: (158.0, 218.0, 229.0),
    29: (100.0, 125.0, 154.0),
    30: (178.0, 127.0, 135.0),
    32: (146.0, 111.0, 194.0),
    33: (44.0, 160.0, 44.0),
    34: (112.0, 128.0, 144.0),
    35: (96.0, 207.0, 209.0),
    36: (227.0, 119.0, 194.0),
    37: (213.0, 92.0, 176.0),
    38: (94.0, 106.0, 211.0),
    39: (82.0, 84.0, 163.0),
    40: (100.0, 85.0, 144.0),
}


SCANNET_CLASS_IDS = {
    0: "wall",
    1: "floor",
    2: "cabinet",
    3: "bed",
    4: "chair",
    5: "sofa",
    6: "table",
    7: "door",
    8: "window",
    9: "bookshelf",
    10: "picture",
    11: "counter",
    12: "desk",
    13: "curtain",
    14: "refrigerator",
    15: "shower curtain",
    16:"toilet",
    17: "sink",
    18:"bathtub",
    19: "otherfurniture",
}

SCANNET_CLASS_COLORS = {
    -1: (0,0,0),
    0:  (0.75, 0.32, 0.30),  # softened red
    1:  (0.32, 0.65, 0.34),  # balanced green
    2:  (0.30, 0.48, 0.78),  # softened blue
    3:  (0.85, 0.55, 0.25),  # warm orange
    4:  (0.55, 0.35, 0.75),  # violet
    5:  (0.25, 0.70, 0.65),  # teal
    6:  (0.80, 0.40, 0.60),  # rose
    7:  (0.60, 0.40, 0.20),  # brown
    8:  (0.78, 0.75, 0.25),  # mustard
    9:  (0.25, 0.60, 0.45),  # sea green

    10: (0.80, 0.45, 0.20),  # burnt orange
    11: (0.35, 0.60, 0.85),  # sky blue
    12: (0.70, 0.25, 0.45),  # raspberry
    13: (0.45, 0.75, 0.25),  # lime (controlled)
    14: (0.85, 0.35, 0.15),  # deep orange-red
    15: (0.40, 0.65, 0.90),  # cool blue
    16: (0.30, 0.80, 0.55),  # aqua green
    17: (0.65, 0.45, 0.75),  # lavender
    18: (0.85, 0.50, 0.70),  # magenta
    19: (0.35, 0.35, 0.75),  # indigo
}

ORIGINAL_WILDSCENES_CLASS_IDS = {
    0: "bush",
    1: "dirt",
    2: "fence",
    3: "grass",
    4: "gravel",
    5: "log",
    6: "mud",
    7: "other-object",
    8: "other-terrain",
    9: "rock",
    10: "structure",
    11: "tree-foliage",
    12: "tree-trunk",
}
# Combine dirt and mud
WILDSCENES_CLASS_IDS = {
    0: "bush",
    1: "dirt",
    2: "fence",
    3: "grass",
    4: "gravel",
    5: "log",
    6: "other-object",
    7: "other-terrain",
    8: "rock",
    9: "structure",
    10: "tree-foliage",
    11: "tree-trunk",
}

WILDSCENES_CLASS_COLORS = {
    0: [34.0, 139.0, 34.0],        # bush
    1: [139.0, 69.0, 19.0],        # dirt
    2: [160.0, 82.0, 45.0],        # fence
    3: [124.0, 252.0, 0.0],        # grass
    4: [190.0, 190.0, 190.0],      # gravel
    5: [139.0, 69.0, 19.0],        # log
    # 6: [160.0, 82.0, 45.0],        # mud
    6: [255.0, 0.0, 255.0],        # other-object
    7: [255.0, 255.0, 0.0],        # other-terrain
    8: [128.0, 128.0, 128.0],      # rock
    9: [70.0, 130.0, 180.0],      # structure
    10: [34.0, 139.0, 34.0],       # tree-foliage
    11: [139.0, 69.0, 19.0],       # tree-trunk
}
