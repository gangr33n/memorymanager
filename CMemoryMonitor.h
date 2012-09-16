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
 
#ifndef CMEMORYMONITOR
#define CMEMORYMONITOR

#include "CMemoryManager.h"
#include "MemoryMonitor.h"
#include "Bitmap.h"

class CMemoryMonitor : public CMemoryManager
{
   /*constructor*/
   public:
      ~CMemoryMonitor();
   
   /*methods*/
   public:
      unsigned int get(void*, unsigned int, unsigned int);
      unsigned int set(unsigned int, unsigned int, void*, unsigned int);
	  virtual void copyReferences(MemoryMonitor*);

   /*variables*/
   private:
      MonitorBucket* references; /*REFERS TO DEVICE MEMORY*/
};

#endif
