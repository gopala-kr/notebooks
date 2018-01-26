
# Threads
adepted from: 
https://www.olcf.ornl.gov/support/system-user-guides/accelerated-computing-guide/#2661

http://cuda-programming.blogspot.co.il/2013/01/vector-addition-in-cuda-cuda-cc-program.html

CUDA threads are logically divided into 1,2, or 3 dimensional groups referred to as thread blocks. 
Threads within a block can cooperate though access to low latency shared memory as well as synchronization capabilities. 
All threads in a block must reside on the same streaming multiprocessor and share the limited resources, additionally a 1024 
threads per block upper limit is imposed.

The thread blocks of a given kernel are partitioned into a 1,2, or 3 dimensional logical grouping referred to as a grid. 
The grid encompasses all threads required to execute a given kernel. There is no cooperation between blocks in a grid, 
as such blocks must be able to be executed independently. When a kernel is launched the number of threads per threadblock 
and the number of threadblocks is specified, this in turn defines the total number of cuda threads launched.

Each thread has access to it's integer position within it's own block as well as the integer position of the thread's 
enclosing block within the grid, as displayed below. In general the thread uses this position information to read/write 
to/from device global memory. In this fashion although each thread is executing the same kernel code each thread has it's 
own data to operate on.

(from stack overflow) 
# Calculations
#### Hardware
If a GPU device has, for example, 4 multiprocessing units, and they can run 768 threads each: then at a given moment no more than 4*768 threads will be really running in parallel (if you planned more threads, they will be waiting their turn).

#### Software

threads are organized in blocks. A block is executed by a multiprocessing unit. The threads of a block can be indentified (indexed) using 1Dimension(x), 2Dimensions (x,y) or 3Dim indexes (x,y,z) but in any case xyz <= 768 for our example (other restrictions apply to x,y,z, see the guide and your device capability).

Obviously, if you need more than those 4*768 threads you need more than 4 blocks. Blocks may be also indexed 1D, 2D or 3D. There is a queue of blocks waiting to enter the GPU (because, in our example, the GPU has 4 multiprocessors and only 4 blocks are being executed simultaneously).

#### Now a simple case: processing a 512x512 image

Suppose we want one thread to process one pixel (i,j).

We can use blocks of 64 threads each. Then we need 512*512/64 = 4096 blocks (so to have 512x512 threads = 4096*64)

It's common to organize (to make indexing the image easier) the threads in 2D blocks having blockDim = 8 x 8 (the 64 threads per block). I prefer to call it threadsPerBlock.

dim3 threadsPerBlock(8, 8);  // 64 threads
and 2D gridDim = 64 x 64 blocks (the 4096 blocks needed). I prefer to call it numBlocks.

dim3 numBlocks(imageWidth/threadsPerBlock.x,  /* for instance 512/8 = 64*/
              imageHeight/threadsPerBlock.y); 
The kernel is launched like this:

myKernel <<<numBlocks,threadsPerBlock>>>( /* params for the kernel function */ );       
Finally: there will be something like "a queue of 4096 blocks", where a block is waiting to be assigned one of the multiprocessors of the GPU to get its 64 threads executed.

In the kernel the pixel (i,j) to be processed by a thread is calculated this way:

uint i = (blockIdx.x * blockDim.x) + threadIdx.x;
uint j = (blockIdx.y * blockDim.y) + threadIdx.y;


http://on-demand.gputechconf.com/gtc-express/2011/presentations/StreamsAndConcurrencyWebinar.pdf
