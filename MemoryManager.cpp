#include "MemoryManager.h"

/**
 * Deconstructor, deallocates dynamic memory for frame
 */
MemoryManager::~MemoryManager()
{
   if (memory != NULL)
      delete[] memory;
}

/**
 * Accessor for the numBuckets variable
 *
 * @return A copy of the numBuckets variable
 */
unsigned int MemoryManager::getDimension()
{
   return dimension;
}

/**
 * Accessor for the number of elements in the matrix
 *
 * @return The number of elements in the matrix
 */
unsigned int MemoryManager::getNumBuckets()
{
   return dimension * dimension;
}

/**
 * Sets the number of buckets used by the memory manager
 *
 * @numBuckets The new number of buckets for the manager
 */
void MemoryManager::setDimension(unsigned int dimension)
{
   this->dimension = dimension;
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
 * @numBuckets The new item size for the manager
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
 */
void MemoryManager::setMemory()
{
   memory = new unsigned char[getNumBuckets()*itemSize];
}
