#include "Main.h"

/**
 * Main method for test harness program on Linux
 *
 * @argc Number of command line arguments should be 3 (constant NUM_ARGS)
 * @argv[] Array of command line input containing 0: the binary name, 1:Memory
 * Manager Type, 2: the size of the matrix and 3: the input matrix file
 */
int main(int argc, char* argv[])
{
   MemoryManager* manager;
   int i, j, matrixSide;
   float* temp;
   LARGE_INTEGER start, end, freq;
   FILE* fp;
   char line[LINE_LENGTH];
   char* token;
   float data;

   /*check for correct command line input*/
   if (argc != NUM_ARGS)
   {
      cerr << "Please select a memory manager by entering 'memorymanager" <<
	                    "{fixed|variable|monitor} {arraySize} {matrixFile}'\n";
      exit(EXIT_FAILURE);
   }
   matrixSide = atoi(argv[ARG_SIZE]);

   /*allocate space for temporary and pointer arrays*/
   temp = new float[matrixSide * matrixSide];
   if (temp == NULL)
   {
      cerr << "Unable to allocate memory! Press enter to continue...";
      exit(EXIT_FAILURE);
   }
   
   /*necesary for microsecond accuracy*/
   QueryPerformanceFrequency(&freq);

   /*1. initialise memory*/
   QueryPerformanceCounter(&start);
   if (strcmp(argv[ARG_TYPE], FIXED) == 0)
      manager = new FixedSizeMemoryManager(matrixSide, matrixSide,
                                                               sizeof(float));
   else if (strcmp(argv[ARG_TYPE], VARIABLE) == 0)
      manager = new VariableSizeMemoryManager(matrixSide, matrixSide, 
                                          sizeof(float), VARIABLE_UTILISATION);
   else if (strcmp(argv[ARG_TYPE], MONITOR) == 0)
      manager = new MemoryMonitor(matrixSide, matrixSide, sizeof(float),
                                                         MONITOR_UTILISATION);
   else
   {
      cerr << "Invalid Manager type!\n";
      exit(EXIT_FAILURE);
   }
   QueryPerformanceCounter(&end);
   /*cout << (double)(end.QuadPart - start.QuadPart) / freq.QuadPart <<
                                                                        "us,";*/

   /*2. load data*/
   fp = fopen(argv[FILE_NAME], READ_ONLY);
   if (fp == NULL)
   {
      cerr << "Unable to open input file!\n";
      exit(EXIT_FAILURE);
   }
   QueryPerformanceCounter(&start);
   i = 0;
   while (i < matrixSide && fgets(line, LINE_LENGTH, fp) != NULL)
   {
      j = 0;
      token = strtok(line, DELIM);
      do
      {
         temp[i * matrixSide + j] = atof(token);
         j++;
      } while (j < matrixSide && (token = strtok(NULL, DELIM)) != NULL);
      i++;
   }
   QueryPerformanceCounter(&end);
   fclose(fp);
   /*cout << (double)(end.QuadPart - start.QuadPart) / freq.QuadPart << "us,";*/
   
   /*3. request and set each element*/
   QueryPerformanceCounter(&start);
   for (i = 0; i < matrixSide; i++)
   {
      for (j = 0; j < matrixSide; j++)
      {
         if (temp[i * matrixSide + j] == 0)
            continue;
         manager->set(i, j, &temp[i * matrixSide + j], sizeof(float));
      }
   }
   QueryPerformanceCounter(&end);
   /*cout << (double)(end.QuadPart - start.QuadPart) / freq.QuadPart << "us,";*/

   /*4. access each element*/
   QueryPerformanceCounter(&start);
   for (i = 0; i < matrixSide; i++)
   {
      for (j = 0; j < matrixSide; j++)
      {
         if (manager->get(&data, i, j))
            cerr << data << " ";
         else
            cerr << EMPTY_ELT << " ";
      }
      cerr << "\n";
   }
   QueryPerformanceCounter(&end);
   /*cout << (double)(end.QuadPart - start.QuadPart) / freq.QuadPart << "us,";*/

   /*5. write results to disk*/
   fp = fopen(OUT_FILE, WRITE_ONLY);
   if (fp == NULL)
   {
      cerr << "Unable to open output file!\n";
      exit(EXIT_FAILURE);
   }
   QueryPerformanceCounter(&start);
   for (i = 0; i < matrixSide; i++)
   {
      for (j = 0; j < matrixSide; j++)
      {
         if (j != 0)
            fprintf(fp, ",");
         if (manager->get(&data, i, j))
            fprintf(fp, "%f", data);
         else
            fprintf(fp, "%f", EMPTY_ELT);
      }
      fprintf(fp, "\n");
   }
   QueryPerformanceCounter(&end);
   fclose(fp);
   unlink(OUT_FILE);
   /*cout << (double)(end.QuadPart - start.QuadPart) / freq.QuadPart << "us,";*/

   /*6. delete every second element, access every element*/
   QueryPerformanceCounter(&start);
   for (i = 0; i < matrixSide; i++)
   {
      for (j = 0; j < matrixSide; j+=2)
      {
         manager->del(i, j);
      }
   }
   QueryPerformanceCounter(&end);
   /*cout << (double)(end.QuadPart - start.QuadPart) / freq.QuadPart << "us,";*/

   /*7. access every element, test shows performance after deletion*/
   QueryPerformanceCounter(&start);
   for (i = 0; i < matrixSide; i++)
   {
      for (j = 0; j < matrixSide; j++)
      {
         if (manager->get(&data, i, j))
            cerr << data << " ";
         else
            cerr << EMPTY_ELT << " ";
      }
      cerr << "\n";
   }
   QueryPerformanceCounter(&end);
   /*cout << (double)(end.QuadPart - start.QuadPart) / freq.QuadPart << "us";*/

   /*free local memory*/
   delete[] temp;
   manager->cleanup();
   delete manager;

   cout << "Press enter to continue...";
   cin.ignore(1);
}
