#ifndef _GENDERIZE_
#define _GENDERIZE_

#include <stdafx.h>

using namespace std;
using namespace cv;

struct Feature {
	int startX;
	int startY;
	int endX;
	int endY;
	int type;
	Mat alfas;
};

class Genderize {
	Mat W;
	float rho;
	vector<Feature> features;
	Mat model;
	vector<Mat> gaborArray;
	vector<Mat> gaborFFTS;
	Mat HY;
	Mat HX;
	Mat ONE_1;
public:
	static const int SZ = 60;
	static const int OFFSET = 2;
	static const int GABOR_SIZE = 18;
	static const int SIFT_SIZE = 128;
	static const int HOG_SIZE = 32;
	//Genderize(String iniFile);
	Genderize(string fileName);
	int predictGender(Mat& A);	
	Mat generateGabors(vector<Mat> images);
private:
	Mat getFeatureVector(Mat& image, Mat& gaborScaledRow, Feature feature);
	Mat getDSIFT(Mat& G);
	Mat getIndirectHOG(Mat& image);
	Mat getHOGFeatureVector(Mat& image, int start_y,int start_x,int end_y,int end_x);
	Mat getSIFTFeatureVector(Mat& image,int start_y,int start_x,int end_y,int end_x);
	Mat getGaborFeatureVector(Mat& gaborScaledRow, int start_y,int start_x,int end_y,int end_x);
	void gaborFeatures(const cv::Mat& srcFFT, const vector<Mat>& gaborFFTArray, Mat*& gaborImages, int half_filter, int initialHeight, int initialWidth);
	vector<Mat> gaborFilterBank(int u, int v, int m, int n);
	void paddingWithFFT(Mat src, Mat& FFT);
	vector<Mat> prepareGabor(const vector<Mat>& gaborArray, int fftSize);
	vector<int> getGaborIndexes(int startX, int startY, int endX, int endY);
};
#endif