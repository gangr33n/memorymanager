#
# Author: Wayne Madden <s3197676@student.rmit.edu.au>
# Version: 0.1
#
# Free to re-use and reference from within code as long as the original owner
# is referenced as per GNU standards
#
# Makefile to compile the Memory Monitor program modules and test set that I
# am developing for my final year Design project at RMIT University
#

MAIN = Main.o
UTILS = Bitmap.o
MANAGERS = MemoryManager.o FixedSizeMemoryManager.o VariableSizeMemoryManager.o MemoryMonitor.o

BINARY = test
CC = g++
CC_FLAGS = -ansi -Wall -pedantic -gstabs

#####

all: $(MANAGERS) $(MAIN) $(UTILS)
	$(CC) $(MANAGERS) $(MAIN) $(UTILS) -o $(BINARY)

#####

Main.o: Main.cpp Main.h
	$(CC) -c $(CC_FLAGS) Main.cpp

#####

Bitmap.o: Bitmap.cpp Bitmap.h
	$(CC) -c $(CC_FLAGS) Bitmap.cpp

#####

MemoryManager.o: MemoryManager.cpp MemoryManager.h
	$(CC) -c $(CC_FLAGS) MemoryManager.cpp

FixedSizeMemoryManager.o: FixedSizeMemoryManager.cpp FixedSizeMemoryManager.h
	$(CC) -c $(CC_FLAGS) FixedSizeMemoryManager.cpp

VariableSizeMemoryManager.o: VariableSizeMemoryManager.cpp VariableSizeMemoryManager.h
	$(CC) -c $(CC_FLAGS) VariableSizeMemoryManager.cpp

MemoryMonitor.o: MemoryMonitor.cpp MemoryMonitor.h
	$(CC) -c $(CC_FLAGS) MemoryMonitor.cpp

#####

clean:
	rm -f *.o $(BINARY)
