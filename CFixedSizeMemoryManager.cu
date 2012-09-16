#include "CFixedSizeMemoryManager.h"
/**
 * Returns a pointer to the bucket specified
 *
 * @return Result of operation, SUCCESS if successful, FAILURE if unsuccessful.
 * Process is unsuccessful when the bucket is unused
 * @param dest Location to store retreived data in
 * @param x The x coordinate of the cell to be retrieved
 * @param y The y coordinate of the cell to be retrieved
 */
unsigned int CFixedSizeMemoryManager::get(void* dest,
                                                unsigned int x, unsigned int y)
{
   memcpy(dest, getMemory()+(x*getXDimension()+y)*getItemSize(), getItemSize());
   return SUCCESS;
}

/**
 * Sets the value of an already allocated bucket.
 *
 * @return Result of operation, SUCCESS if successful, FAILURE if unsuccessful.
 * Process is unsuccessful when the target is not already allocated
 * @param x The x coordinate of the cell to be retrieved
 * @param y The y coordinate of the cell to be retrieved
 * @param data Pointer to the data that is to be stored
 * @param size Size of the data to be stored
 */
unsigned int CFixedSizeMemoryManager::set(unsigned int x,
                                 unsigned int y, void* data, unsigned int size)
{
   if (size > getItemSize())
      return FAILURE;

   memcpy(getMemory()+(x*getXDimension()+y)*getItemSize(), data, getItemSize());

   return SUCCESS;
}
