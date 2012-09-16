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
 * This is an early version of the variable length Memory manager that I am
 * developing for my final year Design project at RMIT University
 */
 
#ifndef CVARIABLESIZEMEMORYMANAGER
#define CVARIABLESIZEMEMORYMANAGER

#include "CMemoryManager.h"
#include "VariableSizeMemoryManager.h"
#include "Bitmap.h"

class CVariableSizeMemoryManager : public CMemoryManager
{
   /*methods*/
   public:
      unsigned int get(void*, unsigned int, unsigned int);
      unsigned int set(unsigned int, unsigned int, void*, unsigned int);
	  virtual void copyReferences(VariableSizeMemoryManager*);

   /*variables*/
   private:
      VariableSizeBucket* references; /*REFERS TO DEVICE MEMORY*/
};

#endif
