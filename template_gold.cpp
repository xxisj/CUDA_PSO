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

////////////////////////////////////////////////////////////////////////////////
// export C interface
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

extern "C" 
void CPU_PSO(float* Position, float* Velocity,float* pbest_Position,float* Fitness,float* pbest_Fitness,float* gbest_Position,float* gbest_Fitness,unsigned int choice);

void
CPU_PSO(float* Position, float* Velocity,float* pbest_Position,float* Fitness,float* pbest_Fitness,float* gbest_Position,float* gbest_Fitness,unsigned int choice) 
{
	//初始化。
	int NUM_PARTICLES=10000;
	int DIMENSION=50;
	int MAX_ITER=5000;
	float CENTER=0;

	float Inf=10000000000.0f;  //用于初始化的无穷大数

	float MAX;                 //Maximum velocity and position
	
	float left_bound;          //初始化左边界和右边界
	float right_bound;

	switch(choice)
	{
		case 1:
			MAX=100;
			left_bound=50;
			right_bound=100;
			break;

		case 2:
			MAX=10;
			left_bound=5;
			right_bound=10;
			break;

		case 3:
			MAX=600;
			left_bound=300;
			right_bound=600;
			break;

		case 4:
			MAX=10;
			left_bound=5;
			right_bound=10;
			break;

		default:
			break;
	} 	
	//初始化
	int num=NUM_PARTICLES*DIMENSION;
	for(int i=0;i<num;i++)
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
	for(int i=0;i<NUM_PARTICLES;i++)
	{
		Fitness[i]=Inf;
	    pbest_Fitness[i]=Inf;
		gbest_Fitness[i]=Inf;
	}
    //PSO 参数设定
	float c1=2.05f;
	float c2=2.05f;
	float fi=c1+c2;
	float w=(float)2/abs(2-fi-sqrt(fi*fi-4*fi));

	float pi=3.1415f;
	//进入循环。
	for(int iter=0;iter<MAX_ITER;iter++)
	{
		if((iter+1)%1000==0)
			printf("CPU上第 %i 次迭代完成\n",iter+1);
		//计算粒子的适应度值。
		for(int i=0;i<NUM_PARTICLES;i++)
		{
			float tmp=0;
			float tmp1=0;
			float tmp2=1;

			for(int j=0;j<DIMENSION;j++)
			{ 
				float x=Position[i+j*NUM_PARTICLES];
				float y;
				switch(choice)
				{
				case 1:
					tmp+=(x-CENTER)*(x-CENTER);          //shifted sphere函数 CENTER=0则没有移位
					break;
				case 2:
					tmp+=(x-CENTER)*(x-CENTER)+10-10*cos(2*pi*(x-CENTER));
					break;

				case 3:
					tmp1+=(x-CENTER)*(x-CENTER)/4000;
					tmp2*=cos((x-CENTER)/sqrt((float)i+1));
					break;

				case 4:
					if(j==DIMENSION-1)
						break;
					y=Position[i+(j+1)*NUM_PARTICLES];
					tmp+=100*(x*x-y)*(x*x-y);
					tmp+=(x-1)*(x-1);
					break;

				default:
					break;
				}//end switch    
			}//end j
			if (choice==3){
				Fitness[i]=tmp1-tmp2+1; //f3
			}
			else{
				Fitness[i]=tmp;
			}
		}//end i

		//更新粒子历史最优信息。
		for(int i=0;i<NUM_PARTICLES;i++)
		{
			if(pbest_Fitness[i]>Fitness[i])
			{
				pbest_Fitness[i]=Fitness[i];
				for(int j=0;j<DIMENSION;j++)
				{
					pbest_Position[i+j*NUM_PARTICLES]=Position[i+j*NUM_PARTICLES];
				}
			}
		}

	  //获取全局最优的粒子适应度值及其位置。
	  for(int i=0;i<NUM_PARTICLES;i++)
      {
		  int pos=0; 
		  int Tag=0; 
		  int l=(NUM_PARTICLES+ i -1)%NUM_PARTICLES;
		  int r=(i+1)%NUM_PARTICLES;
		  if(pbest_Fitness[l]<gbest_Fitness[i])
		  {
				gbest_Fitness[i]=pbest_Fitness[l];
				pos=l;
				Tag=1;
		  }
		  if(pbest_Fitness[i]<gbest_Fitness[i])
		  {
				gbest_Fitness[i]=pbest_Fitness[i];
				pos=i;
				Tag=1;
		  }
		  if(pbest_Fitness[r]<gbest_Fitness[i])
		  {
				gbest_Fitness[i]=pbest_Fitness[r];
				pos=r;
				Tag=1;
		  }
	      if(Tag)                   //表明左右近邻当前出现新的最优值。
		  {
			  for(int k=0;k<DIMENSION;k++)
			  {
				 gbest_Position[i+k*NUM_PARTICLES]=pbest_Position[pos+k*NUM_PARTICLES];
			  }
		  }
      }

    //更新速度。
	 int idx=0;
	 for(int i=0;i<DIMENSION;i++)
	 {
		 for(int j=0;j<NUM_PARTICLES;j++)
		 {
			 float r1=(float)rand()/(RAND_MAX+1);
             float r2=(float)rand()/(RAND_MAX+1);
			 idx=j+i*NUM_PARTICLES;

			 Velocity[idx]=w*(Velocity[idx]+c1*r1*(pbest_Position[idx]-Position[idx])+c2*r2*(gbest_Position[idx]-Position[idx]));

			 if(Velocity[idx]>MAX)
				 Velocity[idx]=MAX;
			 if(Velocity[idx]<-MAX)
				 Velocity[idx]=-MAX;

			 Position[idx]+=Velocity[idx];

			 if(Position[idx]>MAX)
				 Position[idx]=MAX;
			 if(Position[idx]< -MAX)
				 Position[idx]=-MAX;
		 } 
	 }
	}//end iter
}

