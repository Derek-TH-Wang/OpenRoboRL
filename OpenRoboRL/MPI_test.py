import mpi4py.MPI as MPI
import numpy as np

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

if comm_rank == 0:
    data = np.random.rand(comm_size, 2)
    print( data )
else:
    data = None

local_data = comm.scatter( data, root=0 )
local_data = -local_data # 对数据进行处理
print( "rank %d got data and finished dealing." % comm_rank )
print( local_data )

# all_sum = comm.reduce(local_data, root=0, op=MPI.SUM)
# if comm_rank == 0:
#     print( "sum is : %f", all_sum )
all_sum = np.zeros_like(local_data)
comm.Allreduce(local_data, all_sum, op=MPI.SUM)
print( "sum is : %f", all_sum )
print(comm.Get_size())