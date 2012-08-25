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

#ifndef MEMORYMANAGER
#define MEMORYMANAGER

#include <iostream>
#include <cstring>
#include <cstdlib>

#include <cuda_runtime_api.h>

#define FALSE 0
#define TRUE 1

#define FAILURE 0
#define SUCCESS 1

class MemoryManager
{
   /*constructor*/
   public:
      ~MemoryManager();
      
   /*methods*/
   public:
      unsigned int getDimension();
      unsigned int getNumBuckets();
      void setDimension(unsigned int);
      unsigned int getItemSize();
      void setItemSize(unsigned int);
      unsigned char* getMemory();
      unsigned int setMemory();
      virtual unsigned int get(void*, unsigned int, unsigned int) = 0;
      virtual unsigned int set(unsigned int, unsigned int, void*, unsigned int) = 0;
      virtual unsigned int del(unsigned int, unsigned int) = 0;
      virtual void cleanup() = 0;

   /*variables*/
   private:
      unsigned int dimension;
      unsigned int itemSize;
      unsigned char* memory;
};

#endif