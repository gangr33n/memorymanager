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
 * This is an early version of the variable length Memory Monitor program that I
 * am developing for my final year Design project at RMIT University
 */

#ifndef MEMORYMONITOR
#define MEMORYMONITOR

#include "MemoryManager.h"
#include "Bitmap.h"

typedef struct monitorbucket
{
   unsigned int size;
   unsigned char* firstByte;
} MonitorBucket;

class MemoryMonitor : public MemoryManager
{
   /*constructor*/
   public:
      MemoryMonitor(unsigned int, unsigned int);
      ~MemoryMonitor();
   
   /*methods*/
   public:
      unsigned int get(void*, unsigned int, unsigned int);
      unsigned int set(unsigned int, unsigned int, void*, unsigned int);
      unsigned int del(unsigned int, unsigned int);
      void cleanup();

   /*variables*/
   private:
      MonitorBucket* references;
      unsigned int nextFree;
      Bitmap* bitmap;
};

#endif
