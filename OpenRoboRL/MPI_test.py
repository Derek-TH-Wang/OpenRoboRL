import mpi4py.MPI as MPI
import numpy as np
import pybullet as p
import pybullet_data
import time
import math

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

flag = [False]*comm_size
local_flag = False

if comm_rank == 0:
    cnt = 0
    physicsClient = p.connect(p.GUI_SERVER)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
    p.setRealTimeSimulation(1)
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    init_position = [0, 0, 0.5]
    init_orn = [0, 0, 0, 1]
    # quadruped = p.loadURDF("/home/derek/RL/algorithm/OpenRoboRL/OpenRoboRL/robots/mini_cheetah.urdf",
    #                         init_position, init_orn, useFixedBase=False)
    quadruped = p.loadURDF("mini_cheetah/mini_cheetah.urdf", init_position, init_orn, useFixedBase=False)
else:
    init_position = [0, 0+comm_rank, 0.5]
    init_orn = [0, 0, 0, 1]
    time.sleep(comm_rank)
    print("rank ", comm_rank, " connect to the pybullet shared memory server ...")
    physicsClient = p.connect(p.SHARED_MEMORY)
    # quadruped = p.loadURDF("/home/derek/RL/algorithm/OpenRoboRL/OpenRoboRL/robots/mini_cheetah.urdf",
    #                         init_position, init_orn, useFixedBase=False)
    quadruped = p.loadURDF("mini_cheetah/mini_cheetah.urdf", init_position, init_orn, useFixedBase=False)
    print("rank ", comm_rank, " DONE!")


local_flag = comm.scatter(flag, root=0)
if p.isConnected():
    local_flag = True
flag = comm.gather(local_flag,root=0)
print(flag)
start = comm.scatter(flag, root=0)
print(start)
if start:
    time.sleep(comm_rank*0.1)
    startPos = [0.02, -0.78, 1.74, 0.02, -0.78,
                1.74, -0.02, -0.78, 1.74, -0.02, -0.78, 1.74]
    footFR_index = 3
    footFL_index = 7
    footHR_index = 11
    footHL_index = 15
    for i in range(0, footFR_index):
        p.setJointMotorControl2(quadruped, i, p.POSITION_CONTROL, startPos[i])  # , force = 20)
    for i in range(footFR_index + 1, footFL_index):
        p.setJointMotorControl2(quadruped, i, p.POSITION_CONTROL, startPos[i-1])  # , force = 20)
    for i in range(footFL_index + 1, footHR_index):
        p.setJointMotorControl2(quadruped, i, p.POSITION_CONTROL, startPos[i-2])  # , force = 20)
    for i in range(footHR_index + 1, footHL_index):
        p.setJointMotorControl2(quadruped, i, p.POSITION_CONTROL, startPos[i - 3])  # , force = 20)

    t = 0
    while 1:
        time.sleep(0.5)
        t += 1
        startPos = [0.02, -0.78+0.1*math.sin(t), 1.74+0.1*math.sin(t), 0.02, -0.78+0.1*math.sin(t),
                    1.74+0.1*math.sin(t), -0.02, -0.78+0.1*math.sin(t), 1.74+0.1*math.sin(t), -0.02, 
                    -0.78+0.1*math.sin(t), 1.74+0.1*math.sin(t)]
        for i in range(0, footFR_index):
            p.setJointMotorControl2(quadruped, i, p.POSITION_CONTROL, startPos[i])  # , force = 20)
        for i in range(footFR_index + 1, footFL_index):
            p.setJointMotorControl2(quadruped, i, p.POSITION_CONTROL, startPos[i-1])  # , force = 20)
        for i in range(footFL_index + 1, footHR_index):
            p.setJointMotorControl2(quadruped, i, p.POSITION_CONTROL, startPos[i-2])  # , force = 20)
        for i in range(footHR_index + 1, footHL_index):
            p.setJointMotorControl2(quadruped, i, p.POSITION_CONTROL, startPos[i - 3])  # , force = 20)
