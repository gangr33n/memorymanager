#include "FixedSizeMemoryManager.h"

/**
 * Constructor for the fixed length Memory Manager object, assigns total memory
 * frame
 *
 * @param numBuckets Number of items to be stored
 * @param maxItemSize Size to be used for each item by the Memory Manager
 */
FixedSizeMemoryManager::FixedSizeMemoryManager(unsigned int dimension,
                                                         unsigned int itemSize)
{
   unsigned int i, j, t;
   
   setItemSize(itemSize);
   setDimension(dimension);

   setMemory();
   if (getMemory() == NULL)
   {
      std::cout << "Unable to allocate memory for Manager! Exiting...\n";
      exit(EXIT_FAILURE);
   }

   t = 0;
   for (i = 0; i < getDimension(); i++)
   {
      for (j = 0; j < getDimension(); j++)
      {
         memcpy(getMemory()+(i*getDimension()+j)*getItemSize(), &t, getItemSize());
      }
   }
}

/**
 * Deconstructor for the fixed length Memory Manager object
 */
FixedSizeMemoryManager::~FixedSizeMemoryManager()
{
   
}

/**
 * Returns a pointer to the bucket specified
 *
 * @param index Index of bucket to be retrieved
 */
void* FixedSizeMemoryManager::get(unsigned int x, unsigned int y)
{
   return getMemory()+(x*getDimension()+y)*getItemSize();
}

/**
 * Sets the value of an already allocated bucket.
 *
 * @param index Result of operation, SUCCESS if successful, FAILURE if
 * unsuccessful. Process is unsuccessful when the target is not already
 * allocated
 */
unsigned int FixedSizeMemoryManager::set(unsigned int x, unsigned int y, void* data,
                                                            unsigned int size)
{
   if (size > getItemSize())
      return FAILURE;

   memcpy(getMemory()+(x*getDimension()+y)*getItemSize(), data, size);

   return SUCCESS;
}

/**
 * Deletes the contents of a selected bucket
 *
 */
unsigned int FixedSizeMemoryManager::del(unsigned int x, unsigned int y)
{
   unsigned int i;
   unsigned char c;

   c = 0;
   for (i = 0; i < getItemSize(); i++)
      memcpy(getMemory()+(x*getDimension()+y)*getItemSize()+i, &c, sizeof(char));
   
   return SUCCESS;
}

/**
 *
 */
void FixedSizeMemoryManager::cleanup()
{
   
}
