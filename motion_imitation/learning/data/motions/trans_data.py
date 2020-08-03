import numpy
import math
import json
import tf


def quaternion_to_rotation_matrix(quat): # wxyz
    q = quat.copy()
    n = numpy.dot(q, q)
    if n < numpy.finfo(q.dtype).eps:
        return numpy.identity(4)
    q = q * numpy.sqrt(2.0 / n)
    q = numpy.outer(q, q)
    rot_matrix = numpy.array(
        [[1.0 - q[2, 2] - q[3, 3], q[1, 2] + q[3, 0], q[1, 3] - q[2, 0], 0.0],
        [q[1, 2] - q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] + q[1, 0], 0.0],
        [q[1, 3] + q[2, 0], q[2, 3] - q[1, 0], 1.0 - q[1, 1] - q[2, 2], 0.0],
        [0.0, 0.0, 0.0, 1.0]],
        dtype=q.dtype)
    return rot_matrix


def checkdomain(D):
    if D > 1 or D < -1:
        print("____OUT OF DOMAIN____")
        if D > 1: 
            D = 0.99999999999999
            return D
        elif D < -1:
            D = -0.99999999999999
            return D
    else:
        return D


def IK(coord , coxa , femur , tibia, left_right):
    D = (coord[1]**2+(-coord[2])**2-coxa**2+(-coord[0])**2-femur**2-tibia**2)/(2*tibia*femur)  #siempre <1
    D = checkdomain(D)
    gamma = numpy.arctan2(-numpy.sqrt(1-D**2),D)
    tetta = -numpy.arctan2(coord[2],coord[1])-numpy.arctan2(numpy.sqrt(coord[1]**2+(-coord[2])**2-coxa**2),-left_right*coxa)
    if tetta > math.pi:
        tetta -= math.pi*2
    elif tetta < -math.pi:
        tetta += math.pi*2
    alpha = numpy.arctan2(-coord[0],numpy.sqrt(coord[1]**2+(-coord[2])**2-coxa**2))-numpy.arctan2(tibia*numpy.sin(gamma),femur+tibia*numpy.cos(gamma))
    angles = numpy.array([-tetta, -alpha, -gamma])
    return angles


def FK(angle, coxa , femur , tibia, left_right):
    sideSign = left_right
    s1 = numpy.sin(angle[0])
    s2 = numpy.sin(angle[1])
    s3 = numpy.sin(angle[2])
    c1 = numpy.cos(angle[0])
    c2 = numpy.cos(angle[1])
    c3 = numpy.cos(angle[2])

    c23 = c2 * c3 - s2 * s3
    s23 = s2 * c3 + c2 * s3
    p0 = tibia * s23 + femur * s2
    p1 = coxa * (sideSign) * c1 + tibia * (s1 * c23) + femur * c2 * s1
    p2 = coxa * (sideSign) * s1 - tibia * (c1 * c23) - femur * c1 * c2
    p = numpy.array([p0, p1, p2])
    return p


if __name__ == '__main__':
    cnt = 0
    coxa = 0.0671
    femur = 0.206
    tibia = 0.185

    with open("/home/derek/RL/algorithm/motion_imitation/motion_imitation/data/motions/dog_trot.txt", "r") as f:
      motion_json = json.load(f)
      dogtrot = numpy.array(motion_json["Frames"])

    inv = numpy.array([0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    pos = dogtrot[:, 0:3].copy()
    quat = dogtrot[:,3:7].copy()
    for i in range(1):#len(dogtrot)):
        rotm = quaternion_to_rotation_matrix(quat[i])
        print(quat[i])
        print(rotm)
        # quat[i,0] = dogtrot[i,6].copy()
        # quat[i,3] = dogtrot[i,3].copy()
        print(quat[i])
        print(tf.transformations.quaternion_matrix(quat[i]))
        # res = inv * rotm


    while 1:
        cnt += 1
        angle = numpy.random.random(3)*3.14/2
        angle[1] = -angle[1]
        pos_l = FK(angle, coxa , femur , tibia, 1)
        pos_r = FK(angle, coxa , femur , tibia, -1)
        ang_l = IK(pos_l, coxa, femur, tibia, -1)
        ang_r = IK(pos_r, coxa, femur, tibia, 1)
        if max((ang_l-angle)) > 0.0001 or max((ang_r-angle)) > 0.0001:
            print(angle)
            print((ang_l-angle))
            print((ang_r-angle))
        # print(cnt)