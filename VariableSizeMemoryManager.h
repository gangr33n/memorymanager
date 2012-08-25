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
 
#ifndef VARIABLESIZEMEMORYMANAGER
#define VARIABLESIZEMEMORYMANAGER

#include "MemoryManager.h"
#include "Bitmap.h"

typedef struct variablesizebucket
{
   unsigned int size;
   unsigned char* firstByte;
} VariableSizeBucket;

class VariableSizeMemoryManager : public MemoryManager
{
   /*constructor*/
   public:
      VariableSizeMemoryManager(unsigned int, unsigned int);
      ~VariableSizeMemoryManager();
   
   /*methods*/
   public:
      unsigned int get(void*, unsigned int, unsigned int);
      unsigned int set(unsigned int, unsigned int, void*, unsigned int);
      unsigned int del(unsigned int, unsigned int);
      void cleanup();

   /*variables*/
   private:
      VariableSizeBucket* references; /*REFERS TO DEVICE MEMORY*/
      unsigned int nextFree;
      Bitmap* bitmap;
};

#endif