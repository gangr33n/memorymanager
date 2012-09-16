#include "VariableSizeMemoryManager.h"

/**
 * Constructor for the variable length Memory Manager object, assigns total
 * memory frame
 *
 * @param x Number of rows to be stored
 * @param y Number of columns to be stored
 * @param maxItemSize Maximum iize to be used for each item by the Memory
 * Manager
 * @param utlisation Maxiumum percentage of matrix space that will be used
 */
VariableSizeMemoryManager::VariableSizeMemoryManager(unsigned int x,
                  unsigned int y, unsigned int maxItemSize, float utilisation)
{
   unsigned int i;

   setItemSize(maxItemSize);
   setXDimension(x);
   setYDimension(y);
   setUtilisation(utilisation);

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
   if (references[x*getXDimension()+y].firstByte == NULL)
      return FAILURE;
   cudaMemcpy(dest, references[x*getXDimension()+y].firstByte, getItemSize(),
                                                      cudaMemcpyDeviceToHost);
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
unsigned int VariableSizeMemoryManager::set(unsigned int x, unsigned int y,
                                                void* data, unsigned int size)
{
   unsigned int i, j, flag, temp;

   if (size > getItemSize())
      return FAILURE;

   /*if element is already set, update the value*/
   if (references[x*getXDimension()+y].firstByte != NULL)
   {
      if (references[x*getXDimension()+y].size == size)
	  {
         cudaMemcpy(references[x*getXDimension()+y].firstByte, data, size,
                                                      cudaMemcpyHostToDevice);
         return SUCCESS;
	  }
      else
         this->del(x, y);
   }

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
		 if (i >= getNumBuckets() * getItemSize())
			 i = 0;
         continue;
      }

      bitmap->setBits(i, size);
	  references[x*getXDimension()+y].size = size;
	  references[x*getXDimension()+y].firstByte = getMemory() + i;
	  cudaMemcpy(references[x*getXDimension()+y].firstByte, data, size,
                                                      cudaMemcpyHostToDevice);

      temp = nextFree;
      do
      {
         if (bitmap->getBit(nextFree) == EMPTY)
            break;
         nextFree++;
         if (nextFree == getNumBuckets() * getItemSize())
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

   if (references[x*getXDimension()+y].firstByte == NULL)
      return FAILURE;
   
   bit = references[x*getXDimension()+y].firstByte - getMemory();
   bitmap->clearBits(bit, references[x*getXDimension()+y].size);

   references[x*getXDimension()+y].firstByte = NULL;
   references[x*getXDimension()+y].size = 0;
   
   return SUCCESS;
}

/**
 * Returns a pointer to the references array
 *
 * @return A pointer to the references array
 */
VariableSizeBucket* VariableSizeMemoryManager::getReferences()
{
   return references;
}

/**
 * Free any dynamic memory assigned to the manager
 */
void VariableSizeMemoryManager::cleanup()
{
   delete[] references;
   delete bitmap;
}
