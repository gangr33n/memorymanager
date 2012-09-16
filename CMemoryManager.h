/**
 * @file
 * @author Wayne Madden <s3197676@student.rmit.edu.au>
 * @version 0.2
 *
 * @section LICENSE
 * Free to re-use and reference from within code as long as the original owner
 * is referenced as per GNU standards
 *
 * @section DESCRIPTION
 * This is an early version of the Memory Manager Superclass that I am
 * developing for my final year Design project at RMIT University
 */

#ifndef CMEMORYMANAGER
#define CMEMORYMANAGER

#include <iostream>
#include <cstring>
#include <cstdlib>

#include <cuda_runtime_api.h>

#include "MemoryManager.h"

#define FALSE 0
#define TRUE 1

#define FAILURE 0
#define SUCCESS 1

class CMemoryManager
{
   /*methods*/
   public:
      unsigned int getXDimension();
      unsigned int getYDimension();
      unsigned int getNumBuckets();
      unsigned int getItemSize();
      unsigned char* getMemory();
	  float getUtilisation();
      virtual unsigned int get(void*, unsigned int, unsigned int) = 0;
      virtual unsigned int set(unsigned int, unsigned int, void*,
                                                            unsigned int) = 0;
	  void copy(MemoryManager*);

   /*variables*/
   private:
      unsigned int xdimension;
      unsigned int ydimension;
      unsigned int itemSize;
      unsigned char* memory;
	  float utilisation;
};

#endif
