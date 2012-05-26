#include "Bitmap.h"

/**
 * Constructor for the Bitmap object, assigns memory and initialises
 *
 * @param bits Number of bits in the map
 */
Bitmap::Bitmap(unsigned int bits)
{
   unsigned int i;
   
   this->bits = bits;
   
   map = new uint8_t[bits];
   if (map == NULL)
   {
      std::cout << "Unable to allocate " << bits/BYTE_SIZE
                                          << " bytes of memory! Exiting...\n";
      exit(EXIT_FAILURE);
   }

   for (i = 0; i < bits/BYTE_SIZE; i++)
       map[i] = 0;
}

/**
 * Destructur for the bitmap object, clears the current map
 */
Bitmap::~Bitmap()
{
   delete[] map;
}

/**
 * Returns the state of a selected bit in the map
 *
 * @param n The bit number to be retrieved
 */
unsigned int Bitmap::getBit(unsigned int n)
{
   uint8_t bit;

   bit = map[n/BYTE_SIZE] & (1 << n%BYTE_SIZE);
   if (bit == EMPTY)
      return EMPTY;
   else
      return USED;
}

/**
 * Sets the selected bits of the map
 *
 * @param n The first bit to be set
 * @param c The number of bits to be set
 */
void Bitmap::setBits(unsigned int n, unsigned int c)
{
   unsigned int i;

   for (i = n; i < n+c; i++)
      map[i/BYTE_SIZE] |= (1 << i%BYTE_SIZE);
}

/**
 * Clears the selected bits of the map
 *
 * @param n The first bit to be cleared
 * @param c The number of bits to be cleared
 */
void Bitmap::clearBits(unsigned int n, unsigned int c)
{
   unsigned int i;

   for (i = n; i < n+c; i++)
      map[i/BYTE_SIZE] &= ~(1 << i%BYTE_SIZE);
}
