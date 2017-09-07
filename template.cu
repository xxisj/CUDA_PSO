/*
 * Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

#include <cuda_runtime.h>

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <time.h>

//宏定义

#define CHECK_BANK_CONFLICTS 0
#if CHECK_BANK_CONFLICTS
#define FS(i, j) CUT_BANK_CHECKER(((float*)&Fs[0][0]), (BLOCK_SIZE * i + j))
#define PS(i, j) CUT_BANK_CHECKER(((float*)&Ps[0][0]), (BLOCK_SIZE * i + j))
#define US(i, j) CUT_BANK_CHECKER(((float*)&Us[0][0]), (BLOCK_SIZE * i + j))
#define DS(i, j) CUT_BANK_CHECKER(((float*)&Ds[0][0]), (BLOCK_SIZE * i + j))
#define RS(i, j) CUT_BANK_CHECKER(((float*)&Rs[0][0]), (BLOCK_SIZE * i + j))
#else
#define PS(i, j)  Ps[i][j]
#define US(i, j)  Us[i][j]
#define DS(i, j)  Ds[i][j]
#define FS(i, j)  Fs[i][j]
#define RS(i, j)  Rs[i][j]
#endif

#define BLOCK_SIZE 10
#define MAX_ITER 5000            //迭代次数
#define NUM_PARTICLES 10000    //粒子数目
#define WIDTH 100
#define HIGHT 100              //WIDTH*HIGHT=NUM_PARTICLES
#define DIMENSION  50          //维数
#define RAND    1000
#define PI  3.1415
#define RUN_NO 1                 //设置重复运行PSO的次数
#define CENTER 0

//用来计算粒子的适应度值
__global__ void Fitness_Kernel(float* dev_Position, float* dev_Fitness,unsigned int choice)
{
    //块的坐标
    unsigned int bx = blockIdx.x;
    unsigned int by = blockIdx.y;

    //线程坐标
    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;

    //二维粒子数组的坐标
    unsigned int x=bx*blockDim.x+tx;
    unsigned int y=by*blockDim.y+ty;

    float P[DIMENSION];
    for(unsigned int i=0;i<DIMENSION;i++)
    {
        P[i]=0;
        //将所有粒子中第i维的数据存储到PS(ty,tx)中。
         __shared__ float Ps[BLOCK_SIZE][BLOCK_SIZE];
         PS(ty,tx)=dev_Position[i*NUM_PARTICLES+y*WIDTH+x];
        __syncthreads();
        P[i]=PS(ty,tx);
    }

    float tmp=0;
    float tmp1=0;
    float tmp2=1;

    for(unsigned int i=0;i<DIMENSION;i++)
    {
        switch(choice)
        {
            case 1:   //shifted sphere
                tmp+=(P[i]-CENTER)*(P[i]-CENTER);
                __syncthreads();
                break;

            case 2:   //Rastrigin
                tmp+=(P[i]-CENTER)*(P[i]-CENTER)+10-10*cos(2*PI*(P[i]-CENTER));
                break;

            case 3:   //Griewangk
                tmp1+=(P[i]-CENTER)*(P[i]-CENTER)/4000;
                tmp2*=cos((P[i]-CENTER)/sqrt((float)i+1));
                break;

            case 4:  //Rosenbrock
                if(i==DIMENSION-1)
                    break;
                tmp+=100*(P[i]*P[i]-P[i+1])*(P[i]*P[i]-P[i+1]);
                __syncthreads();
                tmp+=(P[i]-1)*(P[i]-1);
                __syncthreads();
                break;

            default:
                break;
        }
    }
    if (choice==3){
        dev_Fitness[y*WIDTH+x]=tmp1-tmp2+1;  //f3
    }
    else{
        dev_Fitness[y*WIDTH+x]=tmp;
    }

    __syncthreads();
}
__global__ void Update_pbFitness_Kernel(float* dev_Fitness, float* dev_pbest_Fitness,float* dev_Position,float*dev_pbest_Position)
{
    //块的坐标
    unsigned int bx = blockIdx.x;
    unsigned int by = blockIdx.y;

    //线程坐标
    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;

    //二维粒子数组的坐标
    unsigned int x=bx*blockDim.x+tx;
    unsigned int y=by*blockDim.y+ty;

    //把dev_Fitness上的数据转移到shared memory上来。
    __shared__  float Fs[BLOCK_SIZE][BLOCK_SIZE];
    FS(ty,tx)=dev_Fitness[y*WIDTH+x];
    __syncthreads();

    if(dev_pbest_Fitness[y*WIDTH+x]>FS(ty,tx))
    {
        dev_pbest_Fitness[y*WIDTH+x]=FS(ty,tx);
        __syncthreads();

        for(unsigned int k=0;k<DIMENSION;k++)
        {
            dev_pbest_Position[k*NUM_PARTICLES+y*WIDTH+x]=dev_Position[k*NUM_PARTICLES+y*WIDTH+x];
        }
         __syncthreads();
    } //end if
}
//得到全局最优适应度值和全局最优位置。
__global__ void get_Best_Kernel(float* dev_pbest_Position,float* dev_gbest_Position,float* dev_pbest_Fitness,float* gbest_Fitness)
{
     //块的坐标
    unsigned int bx = blockIdx.x;
    unsigned int by = blockIdx.y;

    //线程坐标
    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;

    //二维粒子数组的坐标
    unsigned int x=bx*blockDim.x+tx;
    unsigned int y=by*blockDim.y+ty;

    unsigned int pos=0;
    unsigned int Tag=0;

    __shared__ float Ds[BLOCK_SIZE][BLOCK_SIZE];
    DS(ty,tx)=dev_pbest_Fitness[y*WIDTH+x];
    __syncthreads();

    if(gbest_Fitness[y*WIDTH+x]>DS(ty,tx))
    {
         gbest_Fitness[y*WIDTH+x]=DS(ty,tx);
         __syncthreads();

        pos=y*WIDTH+x;
        Tag=1;
    }

    unsigned int idx1=(NUM_PARTICLES+y*WIDTH+x-1)%NUM_PARTICLES;
    __shared__ float Fs[BLOCK_SIZE][BLOCK_SIZE];
    FS(ty,tx)=dev_pbest_Fitness[idx1];
    __syncthreads();

    if(gbest_Fitness[y*WIDTH+x]>FS(ty,tx))
    {
         gbest_Fitness[y*WIDTH+x]=FS(ty,tx);
         __syncthreads();

         pos=(NUM_PARTICLES+y*WIDTH+x-1)%NUM_PARTICLES;
         Tag=1;
    }

    unsigned int idx2=(y*WIDTH+x+1)%NUM_PARTICLES;
    __shared__ float Rs[BLOCK_SIZE][BLOCK_SIZE];
    RS(ty,tx)=dev_pbest_Fitness[idx2];
    __syncthreads();

    if(gbest_Fitness[y*WIDTH+x]>RS(ty,tx))
    {
        gbest_Fitness[y*WIDTH+x]=RS(ty,tx);
         __syncthreads();

        pos=(y*WIDTH+x+1)%NUM_PARTICLES;
        Tag=1;
    }

    if(Tag)                   //表明左右近邻当前出现新的最优值。
    {
        for(int k=0;k<DIMENSION;k++)
        {
            dev_gbest_Position[y*WIDTH+x+k*NUM_PARTICLES]=dev_pbest_Position[pos+k*NUM_PARTICLES];
        }
    }
}
__global__ void update_VP_kernel(float* dev_Position,float* dev_Velocity,float* dev_pbest_Position,float* dev_gbest_Position,float* dev_Rand,unsigned int R1,unsigned int R2,float w,float MAX)
{
    //块的坐标
    unsigned int bx = blockIdx.x;
    unsigned int by = blockIdx.y;

    //线程坐标
    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;

    //二维粒子数组的坐标
    unsigned int x=bx*blockDim.x+tx;
    unsigned int y=by*blockDim.y+ty;

    //按维的顺序更新。
    for(unsigned int i=0;i<DIMENSION;i++)
    {
        __shared__ float Us[BLOCK_SIZE][BLOCK_SIZE];      //取粒子的速度信息。
        __shared__ float Ps[BLOCK_SIZE][BLOCK_SIZE];      //取粒子历史最优值位置信息。
        __shared__ float Ds[BLOCK_SIZE][BLOCK_SIZE];      //取种群当前位置信息
        __shared__ float Rs[BLOCK_SIZE][BLOCK_SIZE];      //取随机数组信息
        __shared__ float Fs[BLOCK_SIZE][BLOCK_SIZE];

        //下面从globle memory中读取数据。
        US(ty,tx)=dev_Velocity[i*NUM_PARTICLES+y*WIDTH+x];
        __syncthreads();

        PS(ty,tx)=dev_pbest_Position[i*NUM_PARTICLES+y*WIDTH+x];
        __syncthreads();

        DS(ty,tx)=dev_Position[i*NUM_PARTICLES+y*WIDTH+x];
        __syncthreads();

        FS(ty,tx)=dev_gbest_Position[i*NUM_PARTICLES+y*WIDTH+x];
        __syncthreads();

        //第二项
        RS(ty,tx)=dev_Rand[R1+i*NUM_PARTICLES+y*WIDTH+x];   //取一次随机数
        __syncthreads();
        US(ty,tx)+=(PS(ty,tx)-DS(ty,tx))*RS(ty,tx);
        __syncthreads();

        //第三项
        RS(ty,tx)=dev_Rand[R2+i*NUM_PARTICLES+y*WIDTH+x];   //再取一次随机数
        __syncthreads();
        US(ty,tx)+=(FS(ty,tx)-DS(ty,tx))*RS(ty,tx);
        __syncthreads();

        US(ty,tx)=w*US(ty,tx);
        __syncthreads();

        //限定速度的范围。
        if(US(ty,tx)>MAX)
        {
            US(ty,tx)=MAX;
        }
        __syncthreads();

        if(US(ty,tx)<-MAX)
        {
            US(ty,tx)=-MAX;
        }
        __syncthreads();

        //更新粒子位置信息。
        dev_Position[i*NUM_PARTICLES+y*WIDTH+x]+=US(ty,tx);
        __syncthreads();

        //更新粒子的速度信息
        dev_Velocity[i*NUM_PARTICLES+y*WIDTH+x]=US(ty,tx);
        __syncthreads();

        //限定位置的范围。
        unsigned int idx=i*NUM_PARTICLES+y*WIDTH+x;
        if(dev_Position[idx]>MAX)
        {
            dev_Position[idx]=MAX;
        }
        __syncthreads();

        if(dev_Position[idx]<-MAX)
        {
            dev_Position[idx]=-MAX;
        }
        __syncthreads();
    }
}

////////////////////////////////////////////////////////////////////////////////

void randomInit(float*, float*,float*,float*,float*,float*,float*,unsigned int,float,unsigned int );

extern "C"
void CPU_PSO(float* Position, float* Velocity,float* pbest_Position,float* Fitness,float* pbest_Fitness,float* gbest_Position,float* gbest_Fitness,unsigned int choice);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
    char fncnm[20]="";
    //printf("\n说明：本程序分别在CPU上和GPU上运行粒子群优化算法（PSO）来优化4个实参数函数：Sphere，Rastrigin和Griewangk以及Rosenbrock。\n 四个函数最优值均为0，最后比较CPU_PSO和GPU_PSO的优化精度及所用时间之比。\n 由于粒子数目较多，CPU上的PSO运行较慢，整个程序运行完成耗时较长（大概10几分钟），请耐心等待。\n 为了缩短运行时间，这里只演示对Rastrigin和Griewangk函数的优化情况\n\n\n");
    
    //变量的声明
    float Inf=10000000000.0f;
    for (unsigned choice=2;choice<=3;choice++){  //选择函数,可以选择从函数1到4，顺序执行
    printf("\n*******************************************************************************\n");
	switch(choice)
	{
		case 1:
			strcpy_s(fncnm,"Sphere");
			printf("CPU 和 GPU 分别用PSO优化Sphere函数：维数50,粒子数10000，迭代5000次。  \n");
			printf("*******************************************************************************\n\n");
			break;

		case 2:
			strcpy_s(fncnm,"Rastrigin");
			printf("CPU 和 GPU 分别用PSO优化Rastrigin函数：维数50,粒子数10000，迭代5000次。\n");
			printf("*******************************************************************************\n\n");
			break;

		case 3:
			strcpy_s(fncnm,"Griewangk");
			printf("CPU 和 GPU 分别用PSO优化Griewangk函数：维数50,粒子数10000，迭代5000次。\n");
			printf("*******************************************************************************\n\n");
			break;

		case 4:
			strcpy_s(fncnm,"Rosenbrock");
			printf("CPU 和 GPU 分别用PSO优化Rosenbrock函数：维数50,粒子数10000，迭代5000次。\n");
			printf("*******************************************************************************\n\n");
			break;

		default:
			break;
	}
    
    //位于CPU上的变量空间申请。 
    unsigned int size1=sizeof(float)*NUM_PARTICLES*DIMENSION;
    float* Position=(float*)malloc(size1);                        //粒子的位置。
    float* Velocity=(float*)malloc(size1);                        //粒子的速度。
    float* pbest_Position=(float*)malloc(size1);            //每个粒子的历史最优位置。 
    float* gbest_Position=(float*)malloc(size1);            //CPU上的当前种群全局最佳位置。采用lbest方式。
    
    unsigned int size2=sizeof(float)*NUM_PARTICLES;
	float*   Fitness=(float*)malloc(size2);                 //每个粒子的适应度值
	float*   pbest_Fitness=(float*)malloc(size2);           //每个粒子的历史最优的适应度值
	float*   gbest=(float*)malloc(size2);
	
    float t1=0;   //计时变量
    float t2=0;
	
	float cpu_best[RUN_NO]; //记录cpu最优值
	float gpu_best[RUN_NO]; //记录gpu最优值
	
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                                                                           //CPU上运行SPSO//
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////  
   time_t startTime = time (NULL);


     //运行CPU上的SPSO,共RUN_NO次。 
    for(unsigned int run=0;run<RUN_NO;run++)
	 { 
		printf("CPU_PSO开始第%i次优化函数%s： \n\n",run+1,fncnm);

		CPU_PSO(Position,Velocity,pbest_Position,Fitness,pbest_Fitness,gbest_Position,gbest,choice);    //CPU上运行。
	    float T1=Inf;
		for(int i=0;i<NUM_PARTICLES;i++)
	    {
		  if(gbest[i]<T1)
			 T1=gbest[i];
	    }
	    cpu_best[run]=T1;     
	 }
	 //得到几次运行的平均值
	 float mean1=0;
	 for (int k=0;k<RUN_NO;k++)
		mean1+=cpu_best[k];
	 mean1=mean1/RUN_NO;
	 
     time_t endTime = time (NULL);
     t1=endTime-startTime;
     
     printf("\nCPU上的PSO优化函数%s的结果为： %6.12f，所用时间为 %f(s)。\n \n",fncnm,mean1,t1);

    
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                                                                           //GPU上运行SPSO//
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////  
    
    float* Rand_Num;
	float c1=2.05;
	float c2=2.05;
	unsigned int size3=sizeof(float)*RAND*RAND;
	Rand_Num=(float*)malloc(size3);
	
    //位于GPU上的变量声明。
	float* dev_Position;                     
	float* dev_Velocity;
	float* dev_pbest_Position;
	float* dev_Fitness;
	float* dev_pbest_Fitness;
	float* dev_gbest_Position;                                           
	float* gbest_Fitness; 
	float* dev_Rand;  
	
	//GPU空间的申请。
    (cudaMalloc((void**)&dev_Position, size1));
    (cudaMalloc((void**)&dev_Velocity, size1));
    (cudaMalloc((void**)&dev_pbest_Position, size1));
    (cudaMalloc((void**)&dev_gbest_Position,size1));
	
    (cudaMalloc((void**)&dev_Fitness, size2));
    (cudaMalloc((void**)&dev_pbest_Fitness, size2));
    (cudaMalloc((void**)&gbest_Fitness,size2));
	
    (cudaMalloc((void**)&dev_Rand,size3));
   
    startTime = time (NULL);

    
    //核函数计算配置参数的定义。
	dim3  grid( WIDTH/BLOCK_SIZE, HIGHT/BLOCK_SIZE, 1);
	dim3  threads( BLOCK_SIZE, BLOCK_SIZE, 1); 
	
	
    //运行GPU上的SPSO，共RUN_NO次。 
    for(unsigned int run=0;run<RUN_NO;run++)
    { 
        printf("GPU_PSO开始第%i次优化函数%s：\n\n",run+1,fncnm); 
		
        //初始化   
		unsigned int num=NUM_PARTICLES*DIMENSION;
		randomInit(Position, Velocity,pbest_Position,Fitness,pbest_Fitness,gbest_Position,gbest,choice,Inf,num);

	    //随机数组的产生。为了避免CPU与GPU之间频繁地传送随机数，生成随机数组，传入GPU。
	   for(unsigned int k=0;k<RAND*RAND;k++) 
		{
			Rand_Num[k]=c1*rand()/(float)(RAND_MAX+1);
		} 
		//将CPU上的初始数据传送到GPU上。
        (cudaMemcpy(dev_Position,Position, size1, cudaMemcpyHostToDevice));
        (cudaMemcpy(dev_Velocity, Velocity, size1, cudaMemcpyHostToDevice));
		
        //(cudaMemcpy(dev_pbest_Position, pbest_Position, size1, cudaMemcpyHostToDevice));
		
        (cudaMemcpy(dev_Fitness,Fitness, size2, cudaMemcpyHostToDevice));
        (cudaMemcpy(dev_pbest_Fitness,pbest_Fitness, size2, cudaMemcpyHostToDevice));
        (cudaMemcpy(gbest_Fitness,gbest,size2, cudaMemcpyHostToDevice));
		
        (cudaMemcpy(dev_Rand,Rand_Num,size3, cudaMemcpyHostToDevice));
		
		//数据已经转移到GPU上。下面进行循环运算。 
		
		float fi=c1+c2;
	    float w=2/abs(2-fi-sqrt(fi*fi-4*fi));
	    
	    float MAX=0;
		if(choice==2 || choice==4)
			MAX=10;
		if(choice==3)
			MAX=600;
		if(choice==1)
			MAX=100;
	    
		//开始循环///////////////////////////////////////////////////////////////////////////////////////////////////////////////
		for(unsigned int iter=0;iter<MAX_ITER;iter++)
		{ 
				if((iter+1)%1000==0)
					printf("GPU上第 %i 次迭代完成 \n",iter+1);
				 
				// 计算每个粒子的适应度值。
				Fitness_Kernel<<< grid, threads>>>(dev_Position,dev_Fitness,choice); 
				
				//更新粒子当前历史最优适应度值及其位置。
				Update_pbFitness_Kernel<<<grid,threads>>>(dev_Fitness,dev_pbest_Fitness,dev_Position,dev_pbest_Position);
			    
				//得到全局最优的适应度值，并保存其位置。 
				get_Best_Kernel<<<grid,threads>>>(dev_pbest_Position,dev_gbest_Position,dev_pbest_Fitness,gbest_Fitness);
			    
				//限定随机数产生的范围，确保不会越界。
				unsigned int Dom=RAND*RAND-DIMENSION*NUM_PARTICLES;
				unsigned int R1=floor(((float)rand()/(RAND_MAX+1))*Dom); 
				unsigned int R2=floor(((float)rand()/(RAND_MAX+1))*Dom); 
				        
				//更新粒子的速度以及位置。
				update_VP_kernel<<<grid,threads>>>(dev_Position,dev_Velocity,dev_pbest_Position,dev_gbest_Position,dev_Rand,R1,R2,w,MAX);		
		}//end iter
		
		//将结果数据copy至CPU
        (cudaMemcpy(gbest,gbest_Fitness, size2, cudaMemcpyDeviceToHost));
		float T2=Inf;
		for(int i=0;i<NUM_PARTICLES;i++)
		{
			if(gbest[i]<T2)
				T2=gbest[i];
		}
	    gpu_best[run]=T2;
	}//end run
    
    float mean2=0;
	for (int k=0;k<RUN_NO;k++)
		mean2+=gpu_best[k];
	mean2=mean2/RUN_NO;
    
    endTime = time (NULL);
    t2=endTime-startTime;

    printf("GPU上的PSO优化函数%s的结果为： %6.12f，所用时间为 %f(s)。\n \n",fncnm,mean2,t2);
    
    printf("############################################################################\n");
    printf("####在函数%s上，加速比（GPU_PSO比CPU_PSO运行快的倍速）为：%2.4f ####\n",fncnm,t1/t2);
    printf("############################################################################\n\n");
    	 
    //释放存储空间
    free(Rand_Num);
	free(Position);
	free(Velocity); 
	free(pbest_Position);
	free(Fitness);
	free(pbest_Fitness); 
	free(gbest_Position);
	free(gbest);
    (cudaFree(dev_Position));
    (cudaFree(dev_Velocity));
    (cudaFree(dev_pbest_Position));
    (cudaFree(dev_Fitness));
    (cudaFree(dev_pbest_Fitness));
    (cudaFree(dev_gbest_Position));
    (cudaFree(dev_Rand));
    (cudaFree(gbest_Fitness));
	 
	}//end choice
}
void randomInit(float*Position, float*Velocity,float*pbest_Position,float*Fitness,float*pbest_Fitness,float*gbest_Position,float*gbest,unsigned int choice,float Inf,unsigned int num)
{
		float left_bound;
		float right_bound;	
		switch(choice)
		{
			case 1:
				left_bound=50;                                //初始化下边界
				right_bound=100;                              //初始化上边界
				break;
				
			case 2:
				left_bound=5;                                //初始化下边界
				right_bound=10;                              //初始化上边界
				break;
				
			case 3:
				left_bound=300;                                //初始化下边界
				right_bound=600;                              //初始化上边界
				break;
				
			case 4:
				left_bound=5;                                //初始化下边界
				right_bound=10;                              //初始化上边界
				break;
				
			default:
				break;
		}
		for(unsigned int i=0;i<num;i++)
		{
			//left_bound和right_bound之间
			Position[i]=left_bound+(float)rand()/(RAND_MAX+1)*(right_bound-left_bound);   
			Velocity[i]=left_bound+(float)rand()/(RAND_MAX+1)*(right_bound-left_bound);
			if(rand()<RAND_MAX/2)
				Position[i]=-Position[i];
			if(rand()<RAND_MAX/2)
				Velocity[i]=-Velocity[i];
			pbest_Position[i]=Position[i];                       //第一代粒子最优位置就是初始位置。
			gbest_Position[i]=Position[i];
		}
		for(unsigned int j=0;j<NUM_PARTICLES;j++)
		{
			Fitness[j]=Inf;
			pbest_Fitness[j]=Inf;
			gbest[j]=Inf;
		}
}
