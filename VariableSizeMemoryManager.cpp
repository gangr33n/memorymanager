#include "VariableSizeMemoryManager.h"

/**
 * Constructor for the variable length Memory Manager object, assigns total memory
 * frame
 *
 * @param numBuckets Number of items to be stored
 * @param maxItemSize Maximum iize to be used for each item by the Memory
 * Manager
 */
VariableSizeMemoryManager::VariableSizeMemoryManager(unsigned int dimension,
                                                      unsigned int maxItemSize)
{
   unsigned int i;

   setItemSize(maxItemSize);
   setDimension(dimension);

   bitmap = new Bitmap(getNumBuckets()*getItemSize());
   references = new VariableSizeBucket[getNumBuckets()];
   setMemory();
   if (references == NULL || getMemory() == NULL || bitmap == NULL)
   {
      std::cout << "Unable to allocate memory for manager! Exiting...\n";
      exit(EXIT_FAILURE);
   }
   for (i = 0; i < getNumBuckets(); i++)
   {
      references[i].size = 0;
      references[i].firstByte = NULL;
   }
   nextFree = 0;
}

/**
 * Deconstructor for the fixed length Memory Manager object, frees the total
 * memory frame
 */
VariableSizeMemoryManager::~VariableSizeMemoryManager()
{
   cleanup();
}

/**
 * Returns a pointer to the bucket specified
 *
 * @param index Index of bucket to be retrieved
 */
void* VariableSizeMemoryManager::get(unsigned int x, unsigned int y)
{
   if (references[x*getDimension()+y].size == 0)
      return NULL;
   return references[x*getDimension()+y].firstByte;
}

/**
 * Sets the value of an already allocated bucket.
 *
 * @param index Result of operation, SUCCESS if successful, FAILURE if
 * unsuccessful. Process is unsuccessful when the target is not already
 * allocated
 */
unsigned int VariableSizeMemoryManager::set(unsigned int x, unsigned int y, void* data,
                                                            unsigned int size)
{
   unsigned int i, j, flag, temp;

   if (size > getItemSize())
      return FAILURE;

   for (i = nextFree; i != nextFree-1; i++)
   {
      flag = FALSE;
      for (j = 0; j < size; j++)
         if (bitmap->getBit(i+j) == USED)
         {
            flag = TRUE;
            break;
         }
      if (flag)
      {
         i += j;
         continue;
      }

      bitmap->setBits(i, size);
      references[x*getDimension()+y].size = size;
      references[x*getDimension()+y].firstByte = getMemory() + i;

      temp = nextFree;
      do
      {
         if (bitmap->getBit(nextFree) == EMPTY)
            break;
         nextFree++;
         if (nextFree == getNumBuckets())
            nextFree = 0;
      } while (temp != nextFree);

      memcpy(references[x*getDimension()+y].firstByte, data, size);
      return SUCCESS;
   }
   return FAILURE;
}

/**
 * Deletes an element and frees the memory
 *
 */
unsigned int VariableSizeMemoryManager::del(unsigned int x, unsigned int y)
{
   unsigned int bit;

   if (references[x*getDimension()+y].firstByte == NULL)
      return FAILURE;
   
   bit = references[x*getDimension()+y].firstByte - getMemory();
   bitmap->clearBits(bit, references[x*getDimension()+y].size);

   references[x*getDimension()+y].firstByte = NULL;
   references[x*getDimension()+y].size = 0;
   
   return SUCCESS;
}

/**
 *
 */
void VariableSizeMemoryManager::cleanup()
{
   delete[] references;
   delete bitmap;
}
