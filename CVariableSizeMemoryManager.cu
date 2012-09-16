#include "CVariableSizeMemoryManager.h"

/**
 * Returns a pointer to the bucket specified
 *
 * @return Result of operation, SUCCESS if successful, FAILURE if unsuccessful.
 * Process is unsuccessful when the bucket is unused
 * @param dest Location to store retreived data in
 * @param x The x coordinate of the cell to be retrieved
 * @param y The y coordinate of the cell to be retrieved
 */
unsigned int CVariableSizeMemoryManager::get(void* dest,
                                                unsigned int x, unsigned int y)
{
   if (references[x*getXDimension()+y].firstByte == NULL)
      return FAILURE;
   memcpy(dest, references[x*getXDimension()+y].firstByte, getItemSize());
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
unsigned int CVariableSizeMemoryManager::set(unsigned int x,
                                 unsigned int y, void* data, unsigned int size)
{
   if (size > getItemSize())
      return FAILURE;

   /*if element is already set, update the value*/
   if (references[x*getXDimension()+y].firstByte != NULL &&
                                 references[x*getXDimension()+y].size == size)
   {
         memcpy(references[x*getXDimension()+y].firstByte, data, size);
         return SUCCESS;
   }
   return FAILURE;
}

/**
 * Copy variables from MemoryManager, called from host to create device manager
 *
 * @param m Memorymanager object to be copied from
 */
void CVariableSizeMemoryManager::copyReferences(VariableSizeMemoryManager* m)
{
   VariableSizeBucket* v;

   v = m->getReferences();

   cudaMemcpy(&references, &v, sizeof(VariableSizeBucket)*m->getNumBuckets(),
		 cudaMemcpyHostToDevice);
}
