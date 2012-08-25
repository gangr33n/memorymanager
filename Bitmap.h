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
 * This is an early version of the Bitmap that I am developing for my final year
 * Design project at RMIT University
 */

#ifndef BITMAP
#define BITMAP

#include <stdint.h>
#include <cstdlib>
#include <iostream>

#define BYTE_SIZE 8

#define EMPTY 0
#define USED 1

class Bitmap
{
   /*constructor*/
   public:
      Bitmap(unsigned int);
      ~Bitmap();

   /*methods*/
   public:
      unsigned int getBit(unsigned int);
      void setBits(unsigned int, unsigned int);
      void clearBits(unsigned int, unsigned int);

   /*variables*/
   private:
      uint8_t* map;
      unsigned int bits;
};

#endif
