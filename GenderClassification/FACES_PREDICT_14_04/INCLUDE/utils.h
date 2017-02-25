#ifndef _LPR_UTILS_
#define _LPR_UTILS_

#include <stdafx.h>

using namespace std;
using namespace cv;

static void mat2gray(Mat A) {
	double minVal;
	double maxVal;
	minMaxIdx(A,&minVal,&maxVal);
	Mat onShow;
	A.convertTo(onShow,CV_32F);
	imshow("image-1",((onShow - minVal)/(maxVal - minVal + 0.00000001)));
	waitKey(0);
};

static vector<string> get_all_files_names_within_folder(string folder)
{
    vector<string> names;
    char search_path[200];
    sprintf(search_path, "%s/*.*", folder.c_str());
    WIN32_FIND_DATA fd; 
    HANDLE hFind = ::FindFirstFile(search_path, &fd); 
    if(hFind != INVALID_HANDLE_VALUE) { 
        do { 
            // read all (real) files in current folder
            // , delete '!' read other 2 default folder . and ..
            if(! (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) ) {
                names.push_back(fd.cFileName);
            }
        }while(::FindNextFile(hFind, &fd)); 
        ::FindClose(hFind); 
    } 
    return names;
}

static vector<vector<string>> readCSV(string fileName) {
	std::ifstream       file(fileName);
	
	vector<vector<string>> result;
	string line;
	while(getline(file,line))
	{
		vector<string> tokens;
		std::stringstream          lineStream(line);
		string                cell;

		while(std::getline(lineStream,cell,','))
		{
			tokens.push_back(cell);
		}
		result.push_back(tokens);
	}
	return result;
};

static void writeCSV(string fileName, Mat V) {
	fstream f;
	f.open(fileName, ios::out);
	//int total = 0;
	for(int i=0; i < V.rows; i++) {
		for (int j = 0; j < V.cols-1; j++)
			f << V.at<float>(i,j) << ',';
		f << V.at<float>(i,V.cols-1) << endl;
		//total++;
	}
	//cout<<"всего:"<<total;
	f.close();
};


#endif