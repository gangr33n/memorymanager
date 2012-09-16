/**
 * @file
 * @author Wayne Madden <s3197676@student.rmit.edu.au>
 * @version 0.2
 *
 * @section LICENSE
 * Free to re-use and reference from within code as long as the original owner
 * is referenced as per GNU standards
 *
 * @section DESCRIPTION
 * Header file for the main test harness for the Memory manager that I am
 * developing for my final year Design project at RMIT University
 */

#include <cstdlib>
#include <iostream>
#include <cstring>
#include <Windows.h>
#include <cstdio>

#include "FixedSizeMemoryManager.h"
#include "VariableSizeMemoryManager.h"
#include "MemoryMonitor.h"

using namespace std;

#define FALSE 0
#define TRUE 1

#define FAILURE 0
#define SUCCESS 1

#define NUM_ARGS 4
#define ARG_TYPE 1
#define FIXED "fixed"
#define VARIABLE "variable"
#define MONITOR "monitor"
#define ARG_SIZE 2
#define FILE_NAME 3

#define MILLION 1000000

#define READ_ONLY "r"
#define DELIM ", "
#define LINE_LENGTH 1000000

#define WRITE_ONLY "w"
#define OUT_FILE "testOut\manager.out"

#define EMPTY_ELT 0.0

#define VARIABLE_UTILISATION 1.0
#define MONITOR_UTILISATION 1.0
