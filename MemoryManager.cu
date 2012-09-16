#include "MemoryManager.h"

/**
 * Deconstructor, deallocates dynamic memory for frame
 */
MemoryManager::~MemoryManager()
{
   if (memory != NULL)
      cudaFree(memory);
}

/**
 * Accessor for the x dimension of the matrix
 *
 * @return A copy of the xdimension variable
 */
unsigned int MemoryManager::getXDimension()
{
   return xdimension;
}

/**
 * Accessor for the y dimension of the matrix
 *
 * @return A copy of the ydimension variable
 */
unsigned int MemoryManager::getYDimension()
{
   return ydimension;
}

/**
 * Accessor for the number of elements in the matrix
 *
 * @return The number of elements in the matrix
 */
unsigned int MemoryManager::getNumBuckets()
{
   return xdimension * ydimension;
}

/**
 * Sets the x dimension of the memory manager
 *
 * @param newx The new x dimension for the manager
 */
void MemoryManager::setXDimension(unsigned int newx)
{
   this->xdimension = newx;
}

/**
 * Sets the y dimension of the memory manager
 *
 * @param newy The new y dimension for the manager
 */
void MemoryManager::setYDimension(unsigned int newy)
{
   this->ydimension = newy;
}

/**
 * Accessor for the itemSize variable
 *
 * @return A copy of the itemSize variable
 */
unsigned int MemoryManager::getItemSize()
{
   return itemSize;
}

/**
 * Sets the item size stored by the memory manager
 *
 * @param itemSize The new item size for the manager
 */
void MemoryManager::setItemSize(unsigned int itemSize)
{
   this->itemSize = itemSize;
}

/**
 * Accessor for the base memory address
 *
 * @return Location of the base memory
 */
unsigned char* MemoryManager::getMemory()
{
   return memory;
}

/**
 * Sets by means of allocation the memory space of the manager
 *
 * @return Result of the operation, 'FAILURE' or 'SUCCESS'
 */
unsigned int MemoryManager::setMemory()
{
   if (cudaMalloc((void**) &memory, getNumBuckets()*itemSize*utilisation)
                                                               != cudaSuccess)
      return FAILURE;
   return SUCCESS;
}

/**
 * Sets the utilisation percentage insatnce variable
 *
 * @param utilisation The new utilisation percentage
 */
void MemoryManager::setUtilisation(float utilisation)
{
   this->utilisation = utilisation;
}

/**
 * Returns a copy of the current utilisation percentage
 *
 * @return A copy of the current utilisation percentage
 */
float MemoryManager::getUtilisation()
{
   return utilisation;
}
