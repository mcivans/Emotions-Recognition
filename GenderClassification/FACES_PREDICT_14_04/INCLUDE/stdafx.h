#pragma once
#define NOMINMAX 0
#include <vector>
#include <list>
#include <stdio.h>
#include <string>
#include <math.h>
#include <ctime>
#include <iostream>
#include <process.h>
#include <windows.h>
#include <fstream>
#include <algorithm>
#include <limits>
#include <locale.h>

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>
extern "C" {
#include <vl/generic.h>
#include <vl/stringop.h>
#include <vl/pgm.h>
#include <vl/sift.h>
#include <vl/getopt_long.h>
#include <vl/dsift.h>
}

#include "genderize.h"
#include "utils.h"