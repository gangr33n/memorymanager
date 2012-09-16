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
 * This is an early version of the Fixed Length Memory manager that I am
 * developing for my final year Design project at RMIT University
 */

#ifndef CFIXEDSIZEMEMORYMANAGER
#define CFIXEDSIZEMEMORYMANAGER

#include "CMemoryManager.h"

class CFixedSizeMemoryManager : public CMemoryManager
{
   /*methods*/
   public:
      unsigned int get(void*, unsigned int, unsigned int);
      unsigned int set(unsigned int, unsigned int, void*, unsigned int);

   /*variables*/
   private:
      
};

#endif
