import numpy as np
import skfuzzy as fz
import matplotlib.pyplot as plt

# assuming singleton inputs

# membership functions for Distance
#NOTE assumption_1: range and step in units of meters
distance = {
    "near": None,
    "far": None,
    "very_far": None,
    "range": [0, 10],
    "step": 0.5
}
#NOTE assumption_1: range and step in units of degrees
angle = {
    "small": None,
    "medium": None,
    "large": None,
    "range": [0, 90],
    "step": 1
}
#NOTE assumption_1: range and step in units of meters/second
speed = {
    "slow": None,
    "medium": None,
    "fast": None,
    "max": None,
    "range": [0, 5], #
    "step": 0.2
}
#NOTE assumption_1: range and step in units of degrees/second
turn = {
    "mild": None,
    "sharp": None,
    "very_sharp": None,
    "range": [0, 90],
    "step": 1
}


# pseudo-code logic RULES
# Rule 1: if obstacle in front of you, turn
# Rule 2: if no obstacle move forward
# Rule 3: if turning lower speed
# Rule 4: if cruising increase speed

#                                       TURNING             SPEED
# if distance near and angle large      -> turn mild        -> fast
# if distance near and angle medium     -> turn sharp       -> medium
# if distance near and angle small      -> turn very sharp  -> slow

# if distance medium and angle large    -> turn mild        -> fast
# if distance medium and angle medium   -> turn sharp       -> medium
# if distance medium and angle small    -> turn sharp       -> medium

# if distance large and angle large     -> turn mild        -> max
# if distance large and angle medium    -> turn mild        -> fast
# if distance large and angle small     -> turn mild        -> medium
