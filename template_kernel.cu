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

#ifndef _TEMPLATE_KERNEL_H_
#define _TEMPLATE_KERNEL_H_

#include <stdio.h>

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
#endif // #ifndef _TEMPLATE_KERNEL_H_
