import numpy as np
import matplotlib.pyplot as plt

# ASSUMPTION 0: units are in meters, m/s, and degrees
# ASSUMPTION 1: control frequency is 1ms
# ASSUMPTION 2: start our speed and steering at 0
# ASSUMPTION 3: the environment is boundless (no walls)
# ASSUMPTION 4: we have one obstacle to avoid at a time
# ASSUMPTION 5: once we pass an obstacle (|ang| > 90) the next one is spawned
# ASSUMPTION 6: the obstacle spawns at a random distance from the robot between 1-11m
# ASSUMPTION 7: obstacles spawn at a random angle between 0-90
# ASSUMPTION 8: running sim for 10 seconds
# ASSUMPTION 9: assuming singleton input values

dt = 0.001
runtime = 10 # seconds
steps = runtime / dt

speed = 0
steer = 0
pos = [0, 0]
positions = []
positions.append(pos)
obstacles = []

def gen_random_obstacle(pos, steer):
    # select a random angle between -90 to 90
    ang = np.random.choice(np.arange(-90, 90, 1), size=1)
    # choose a random dist between 1 and 11
    dist = np.random.choice(np.arange(1, 11, 0.5), size=1)
    # offset our angle to be relative to our steering angle
    ang += steer
    dx = np.sin(steer) * dist
    dy = np.cos(steer) * dist
    obstacle = [pos[0] + dx, pos[1] + dy]

    return obstacle

def get_input(obstacle, pos, steer):
    dist = np.linalg.norm(obstacle-pos)
    #TODO get angle between pos and obstacle, given pos is at steer angle
    ang = 
    return dist, ang

def next_pos(steer, speed, pos, dt):
    step_size = speed * dt
    dx = np.sin(steer) * step_size
    dy = np.cos(steer) * step_size
    pos[0] += dx
    pos[1] += dy
    return pos

angle_sign = 1
for ii in steps:
    if ii == 0:
        obstacle = gen_random_obstacle(pos=pos, steer=steer)
        obstacles.append(obstacle)

    # calculate our distance and angle to target to simulate our sensors
    dist, ang = get_input(obstacle, pos, steer)

    # our fuzzy system takes positive inputs, so store the sign here to restore it later
    angle_sign = np.sign(ang)

    if abs(ang) > 90:
        obstacle = gen_random_obstacle(pos=pos, steer=steer)
        obstacles.append(obstacle)
        dist, ang = get_input(obstacle, pos, steer)

    # run a step of our sim
    ctrl_sim.input['distance'] = dist
    ctrl_sim.input['angle'] = abs(ang)
    ctrl_sim.compute()

    # retrieve our control outputs
    speed = ctrl_sim.output['speed']
    steer = angle_sign * ctrl_sim.output['steer']

    # calculte our updated position
    pos = next_pos(steer=steer, speed=speed, pos=pos, dt=dt)
    positions.append(pos)

obstacles = np.asarray(obstacles)
positions = np.asarray(positions)

plt.figure()
plt.scatter(obstacles[:, 0], obstacles[:, 1], label='obstacles')
plt.scatter(positions[:, 0], positions[:, 1], label='positions')
plt.legend()
plt.show()
