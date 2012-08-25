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
   setItemSize(itemSize);
   setDimension(dimension);

   if (setMemory() == FAILURE)
   {
      std::cout << "Unable to allocate memory for Manager! Exiting...\n";
      exit(EXIT_FAILURE);
   }

   cudaMemcpy(getMemory(), NULL, getNumBuckets()*itemSize,
      cudaMemcpyHostToDevice);
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
 * @return Result of operation, SUCCESS if successful, FAILURE if unsuccessful.
 * Process is unsuccessful when the bucket is unused
 * @param dest Location to store retreived data in
 * @param x The x coordinate of the cell to be retrieved
 * @param y The y coordinate of the cell to be retrieved
 */
unsigned int FixedSizeMemoryManager::get(void* dest, unsigned int x,
                                                               unsigned int y)
{
   cudaMemcpy(dest, getMemory()+(x*getDimension()+y)*getItemSize(),
      getItemSize(), cudaMemcpyDeviceToHost);
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
unsigned int FixedSizeMemoryManager::set(unsigned int x, unsigned int y,
	                             	              void* data, unsigned int size)
{
   if (size > getItemSize())
      return FAILURE;

   cudaMemcpy(getMemory()+(x*getDimension()+y)*getItemSize(), data,
                                       getItemSize(), cudaMemcpyHostToDevice);

   return SUCCESS;
}

/**
 * Deletes the contents of a selected bucket
 *
 * @return Result of operation, SUCCESS if successful, FAILURE if unsuccessful.
 * Process is unsuccessful when the target is not already allocated
 * @param x The x coordinate of the cell to be retrieved
 * @param y The y coordinate of the cell to be retrieved
 */
unsigned int FixedSizeMemoryManager::del(unsigned int x, unsigned int y)
{
   unsigned int i;
   unsigned char c;

   c = 0;
   for (i = 0; i < getItemSize(); i++)
      cudaMemcpy(getMemory()+(x*getDimension()+y)*getItemSize()+i, &c,
                                          sizeof(char), cudaMemcpyHostToDevice);
   
   return SUCCESS;
}

/**
 * Free any dynamic memory assigned to the manager
 */
void FixedSizeMemoryManager::cleanup()
{
   
}
