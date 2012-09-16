#include "CMemoryManager.h"

/**
 * Accessor for the x dimension of the matrix
 *
 * @return A copy of the xdimension variable
 */
unsigned int CMemoryManager::getXDimension()
{
   return xdimension;
}

/**
 * Accessor for the y dimension of the matrix
 *
 * @return A copy of the ydimension variable
 */
unsigned int CMemoryManager::getYDimension()
{
   return ydimension;
}

/**
 * Accessor for the number of elements in the matrix
 *
 * @return The number of elements in the matrix
 */
unsigned int CMemoryManager::getNumBuckets()
{
   return xdimension * ydimension;
}

/**
 * Accessor for the itemSize variable
 *
 * @return A copy of the itemSize variable
 */
unsigned int CMemoryManager::getItemSize()
{
   return itemSize;
}

/**
 * Accessor for the base memory address
 *
 * @return Location of the base memory
 */
unsigned char* CMemoryManager::getMemory()
{
   return memory;
}

/**
 * Returns a copy of the current utilisation percentage
 *
 * @return A copy of the current utilisation percentage
 */
float CMemoryManager::getUtilisation()
{
   return utilisation;
}

/**
 * Copy variables from MemoryManager, called from host to create device manager
 *
 * @param m Memorymanager object to be copied from
 */
void CMemoryManager::copy(MemoryManager* manager)
{
	unsigned int x, y, s;
	unsigned char* m;
	float u;

	x = manager->getXDimension();
	y = manager->getYDimension();
	s = manager->getItemSize();
	m = manager->getMemory();
	u = manager->getUtilisation();

	cudaMemcpy(&xdimension, &x, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(&ydimension, &y, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(&itemSize, &s, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(&memory, &m, sizeof(unsigned char*), cudaMemcpyHostToDevice);
	cudaMemcpy(&utilisation, &u, sizeof(float), cudaMemcpyHostToDevice);
}
