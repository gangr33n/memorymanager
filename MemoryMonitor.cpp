#include "MemoryMonitor.h"

/**
 * Constructor for the variable length Memory Monitor object
 *
 * 
 */
MemoryMonitor::MemoryMonitor(unsigned int dimension, unsigned int maxItemSize)
{
   unsigned int i;

   setItemSize(maxItemSize);
   setDimension(dimension);

   bitmap = new Bitmap(getNumBuckets()*getItemSize());
   references = new MonitorBucket[getNumBuckets()];
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
 * Deconstructor for the Memory Monitor
 */
MemoryMonitor::~MemoryMonitor()
{
   cleanup();
}

/**
 * Returns a pointer to the bucket specified
 *
 * @param index Index of bucket to be retrieved
 */
void* MemoryMonitor::get(unsigned int x, unsigned int y)
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
unsigned int MemoryMonitor::set(unsigned int x, unsigned int y, void* data,
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
unsigned int MemoryMonitor::del(unsigned int x, unsigned int y)
{
   unsigned int bit, i, j, k;

   if (references[x*getDimension()+y].firstByte == NULL)
      return FAILURE;

   k = 0;
   for (i = 0; i < getDimension(); i++)
   {
      for (j = 0; j < getDimension(); j++)
      {
         if (references[i*getDimension()+j].firstByte > references[x*getDimension()+y].firstByte && references[i*getDimension()+j].size == references[i*getDimension()+y].size)
            k = i*getDimension()+j;
      }
   }

   memcpy(references[x*getDimension()+y].firstByte, references[k].firstByte, references[k].size);
   bit = references[k].firstByte - getMemory();
   bitmap->clearBits(bit, references[k].size);

   references[x*getDimension()+y].firstByte = NULL;
   references[x*getDimension()+y].size = 0;

   return SUCCESS;
}

/**
 *
 */
void MemoryMonitor::cleanup()
{
   delete[] references;
   delete bitmap;
}
