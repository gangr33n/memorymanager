/**
 * @file
 * @author Wayne Madden <s3197676@student.rmit.edu.au>
 * @version 0.1
 *
 * @section LICENSE
 * Free to re-use and reference from within code as long as the original owner
 * is referenced as per GNU standards
 *
 * @section DESCRIPTION
 * This is an early version of the Fixed Length Memory manager that I am
 * developing for my final year Design project at RMIT University
 */

#ifndef FIXEDSIZEMEMORYMANAGER
#define FIXEDSIZEMEMORYMANAGER

#include "MemoryManager.h"

class FixedSizeMemoryManager : public MemoryManager
{
   /*constructor*/
   public:
      FixedSizeMemoryManager(unsigned int, unsigned int);
      ~FixedSizeMemoryManager();
   
   /*methods*/
   public:
      void* get(unsigned int, unsigned int);
      unsigned int set(unsigned int, unsigned int, void*, unsigned int);
      unsigned int del(unsigned int, unsigned int);
      void cleanup();

   /*variables*/
   private:
      
};

#endif