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
   if (bitmap == NULL || references == NULL || setMemory() == FAILURE)
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
 * @return Result of operation, SUCCESS if successful, FAILURE if unsuccessful.
 * Process is unsuccessful when the bucket is unused
 * @param dest Location to store retreived data in
 * @param x The x coordinate of the cell to be retrieved
 * @param y The y coordinate of the cell to be retrieved
 */
unsigned int VariableSizeMemoryManager::get(void* dest, unsigned int x,
                                                               unsigned int y)
{
   if (references[x*getDimension()+y].firstByte == NULL)
      return FAILURE;
   cudaMemcpy(dest, references[x*getDimension()+y].firstByte, getItemSize(), cudaMemcpyDeviceToHost);
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
	  cudaMemcpy(references[x*getDimension()+y].firstByte, data, size, cudaMemcpyHostToDevice);

      temp = nextFree;
      do
      {
         if (bitmap->getBit(nextFree) == EMPTY)
            break;
         nextFree++;
         if (nextFree == getNumBuckets())
            nextFree = 0;
      } while (temp != nextFree);

      return SUCCESS;
   }
   return FAILURE;
}

/**
 * Deletes the contents of a selected bucket
 *
 * @return Result of operation, SUCCESS if successful, FAILURE if unsuccessful.
 * Process is unsuccessful when the target is not already allocated
 * @param x The x coordinate of the cell to be retrieved
 * @param y The y coordinate of the cell to be retrieved
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
 * Free any dynamic memory assigned to the manager
 */
void VariableSizeMemoryManager::cleanup()
{
   delete[] references;
   delete bitmap;
}
