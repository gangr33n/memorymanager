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
   float** temp;
   /*struct timeval start, end;*/
   FILE* fp;
   char line[LINE_LENGTH];
   char* token;
   float data;

   /*check for correct command line input*/
   if (argc != NUM_ARGS)
   {
      cerr << "Please select a memory manager by entering" <<
            "'memorymanager {fixed|fixed_monitor|variable|variable_monitor}" <<
                                                " {arraySize} {matrixFile}'\n";
      exit(EXIT_FAILURE);
   }
   matrixSide = atoi(argv[ARG_SIZE]);

   /*allocate space for temporary and pointer arrays*/
   temp = new float*[matrixSide];
   for (i = 0; i < matrixSide; i++)
      temp[i] = new float[matrixSide];
   
   /*1. initialise memory*/
   /*gettimeofday(&start, NULL);*/
   if (strcmp(argv[ARG_TYPE], FIXED) == 0)
      manager = new FixedSizeMemoryManager(matrixSide, sizeof(float));
   else if (strcmp(argv[ARG_TYPE], VARIABLE) == 0)
      manager = new VariableSizeMemoryManager(matrixSide, sizeof(float));
   else if (strcmp(argv[ARG_TYPE], MONITOR) == 0)
      manager = new MemoryMonitor(matrixSide, sizeof(float));
   else
   {
      cerr << "Invalid Manager type!\n";
      exit(EXIT_FAILURE);
   }
   /*gettimeofday(&end, NULL);
   cout << ((end.tv_sec*MILLION+end.tv_usec) -
                                 (start.tv_sec*MILLION+start.tv_usec)) << "us,";*/

   /*2. load data*/
   fp = fopen(argv[FILE_NAME], READ_ONLY);
   if (fp == NULL)
   {
      cerr << "Unable to open input file!\n";
      exit(EXIT_FAILURE);
   }
   /*gettimeofday(&start, NULL);*/
   i = 0;
   while (i < matrixSide && fgets(line, LINE_LENGTH, fp) != NULL)
   {
      j = 0;
      token = strtok(line, DELIM);
      do
      {
         temp[i][j] = atof(token);
         j++;
      } while (j < matrixSide && (token = strtok(NULL, DELIM)) != NULL);
      i++;
   }
   /*gettimeofday(&end, NULL);
   fclose(fp);
   cout << ((end.tv_sec*MILLION+end.tv_usec) -
                                 (start.tv_sec*MILLION+start.tv_usec)) << "us,";*/
   
   /*3. request and set each element*/
   /*gettimeofday(&start, NULL);*/
   for (i = 0; i < matrixSide; i++)
   {
      for (j = 0; j < matrixSide; j++)
      {
         if (temp[i][j] == 0)
            continue;
         manager->set(i, j, &temp[i][j], sizeof(float));
      }
   }
   /*gettimeofday(&end, NULL);
   cout << ((end.tv_sec*MILLION+end.tv_usec) -
                                 (start.tv_sec*MILLION+start.tv_usec)) << "us,";*/

   /*4. access each element*/
   /*gettimeofday(&start, NULL);*/
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
   /*gettimeofday(&end, NULL);
   cout << ((end.tv_sec*MILLION+end.tv_usec) -
                                 (start.tv_sec*MILLION+start.tv_usec)) << "us,";*/

   /*5. write results to disk*/
   fp = fopen(OUT_FILE, WRITE_ONLY);
   if (fp == NULL)
   {
      cerr << "Unable to open output file!\n";
      exit(EXIT_FAILURE);
   }
   /*gettimeofday(&start, NULL);*/
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
   /*gettimeofday(&end, NULL);
   fclose(fp);
   unlink(OUT_FILE);
   cout << ((end.tv_sec*MILLION+end.tv_usec) -
                                 (start.tv_sec*MILLION+start.tv_usec)) << "us,";*/

   /*6. delete every second element, access every element*/
   /*gettimeofday(&start, NULL);*/
   for (i = 0; i < matrixSide; i++)
   {
      for (j = 0; j < matrixSide; j+=2)
      {
         manager->del(i, j);
      }
   }
   /*gettimeofday(&end, NULL);
   cout << ((end.tv_sec*MILLION+end.tv_usec) -
                                 (start.tv_sec*MILLION+start.tv_usec)) << "us,";*/

   /*7. access every element, test shows performance after deletion*/
   /*gettimeofday(&start, NULL);*/
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
   /*gettimeofday(&end, NULL);
   cout << ((end.tv_sec*MILLION+end.tv_usec) -
                                 (start.tv_sec*MILLION+start.tv_usec)) << "us";

   /*free local memory*/
   for (i = 0; i < matrixSide; i++)
      delete[] temp[i];
   delete[] temp;
   manager->cleanup();
   delete manager;

   cout << "Press enter to continue...";
   cin.ignore(1);
}
