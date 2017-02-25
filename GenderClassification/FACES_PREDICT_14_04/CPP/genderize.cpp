#include "genderize.h"

/*
* Инициализация на основе распознавателя
*/
Genderize::Genderize(string fileName) {
	std::ifstream       file(fileName);
	gaborArray = gaborFilterBank(3,6,19,19);
	gaborFFTS = prepareGabor(gaborArray,64);
	string line;
	int featuresCount = 0;
	if (getline(file,line)) {
		std::stringstream lineStream(line);
		string            cell;

		while(std::getline(lineStream,cell,','))
		{
			featuresCount = stoi(cell);
			break;
		}
	}
	for(int i = 0; i<featuresCount;i++) {
		getline(file,line);
		std::stringstream lineStream(line);
		string            cell;
		Feature feature;
		getline(lineStream,cell,',');
		feature.type = stoi(cell);
		getline(lineStream,cell,',');
		feature.startY = stoi(cell);
		getline(lineStream,cell,',');
		feature.startX = stoi(cell);
		getline(lineStream,cell,',');
		feature.endY = stoi(cell);
		getline(lineStream,cell,',');
		feature.endX = stoi(cell);
		features.push_back(feature);
	}
	model = Mat::zeros(1,featuresCount+1,CV_32F);
	
	if (std::getline(file,line,'\n')) {
		std::stringstream lineStream(line);
		string            cell;
		int cntr = 0;
		while(std::getline(lineStream,cell,',')) {
			model.at<float>(0,cntr++) = (float)atof(cell.c_str());
		}
	}
	try {
		for(int i = 0; i<featuresCount;i++) {
			std::getline(file,line,'\n');
	//		cout << "i: "<<i<<"length: "<<line.length();
			std::stringstream lineStream(line);
			string            cell;
			vector<float> values; 
			while(getline(lineStream,cell,',')) {
				values.push_back((float)atof(cell.c_str()));
			}
			Mat alfaValues = Mat::zeros(values.size(),1,CV_32F);
			for (int j =0; j < values.size();j++) {
				alfaValues.at<float>(j,0) = values[j];
			}
			features[i].alfas = alfaValues;
		}
	} catch (Exception e) {
		cout << e.msg;
	}
	file.close();
	HY = Mat(3,1,CV_32F);
	HY.at<float>(0,0) = 1;
	HY.at<float>(1,0) = 0;
	HY.at<float>(2,0) = -1;
	HX = Mat(1,3,CV_32F);
	HX.at<float>(0,0) = -1;
	HX.at<float>(0,1) = 0;
	HX.at<float>(0,2) = 1;
	ONE_1 = Mat::ones(1,1,CV_32F);
};

/*
* Предсказание пола по изображению
*/
int Genderize::predictGender(Mat& image) {
	Mat values = Mat::zeros(model.cols,1,CV_32F);
	values.at<float>(0,0) = 1.0;
	Mat srcFFT;
	Mat* gaborImages = new Mat[18];
	paddingWithFFT(image,srcFFT);		
	gaborFeatures(srcFFT,gaborFFTS,gaborImages,9,SZ,SZ);
	Mat gaborScaledRow;
	Mat gabors = Mat::zeros(18,(SZ/2)*(SZ/2),CV_32F);
	for (int i = 0; i< 18;i++) {
		Mat scalled;
		resize(gaborImages[i],scalled,Size(SZ/2,SZ/2),0,0,INTER_LINEAR);
		gabors.row(i) = scalled.reshape(1,1) + 0;				
		scalled.release();
	}

	if (gaborScaledRow.rows == 0) {
		gaborScaledRow = gabors.reshape(1,1) + 0;
	} else {
		hconcat(gaborScaledRow,gabors.reshape(1,1)+0,gaborScaledRow);
	}
	gabors.release();

	for (int i =0; i < features.size();i++) {
		Feature feature = features[i];
		if (feature.type == -1) {
			continue;
		}
		Mat descriptor = getFeatureVector(image, gaborScaledRow, feature);
		hconcat(descriptor,Mat::ones(1,1,CV_32F),descriptor);
		Mat scalar = descriptor * feature.alfas;
		//cout<<scalar;
		values.at<float>(i+1,0) = scalar.at<float>(0,0);
	}
	//cout<<model;
	//cout<<values;
	Mat result = model * values;
	//cout<<result;
	return result.at<float>(0,0) > 0 ? 0 : 1;
};

/*
* Построение в зависимости от блока определенного вида особенностей и изображения - вектора описания
*/
Mat Genderize::getFeatureVector(Mat& image, Mat& gaborScaledRow, Feature feature) {
	int start_y = feature.startY;
	int start_x = feature.startX;
	int end_y = feature.endY;
	int end_x = feature.endX;
	Mat result;
	switch (feature.type) {
		case 1: result = getSIFTFeatureVector(image, start_y, start_x, end_y, end_x);
			break;
		case 2: result = getHOGFeatureVector(image, start_y, start_x, end_y, end_x);
			break;
		case 3: result = getGaborFeatureVector(gaborScaledRow, start_y, start_x, end_y, end_x);
			break;
	}
	return result;
};

/*
* Быстрый вариант DSIFT из библиотеки vlfeat
*/
Mat Genderize::getDSIFT(Mat& img) {
	VlDsiftDescriptorGeometry geom ;
	geom.numBinX = 4 ;
	geom.numBinY = 4 ;
	geom.numBinT = 8 ;
	geom.binSizeX = 4 ;
	geom.binSizeY = 4 ;
	int step [2] = {16,16} ;
	//VlDsiftFilter* dsiftFilt = vl_dsift_new_basic(img.cols, img.rows,16,4);	
	//  vl_dsift_set_flat_window(dsiftFilt, VL_FALSE) ;
	VlDsiftFilter* dsiftFilt = vl_dsift_new (img.cols,img.rows) ;
	vl_dsift_set_geometry(dsiftFilt, &geom) ;
	vl_dsift_set_steps(dsiftFilt, step[0], step[1]) ;
	vl_dsift_set_window_size(dsiftFilt, 2) ;
	vl_dsift_set_flat_window(dsiftFilt, 1) ;
	vl_dsift_process(dsiftFilt, img.ptr<float>(0));
	
	int nkeys = vl_dsift_get_keypoint_num (dsiftFilt) ;
	int descrSize = vl_dsift_get_descriptor_size (dsiftFilt) ;
	const float* descrs = vl_dsift_get_descriptors(dsiftFilt);
	Mat descriptor = Mat::zeros(1, descrSize*nkeys, CV_32F);
	for (int k = 0 ; k < nkeys ; ++k) {
		 float *tmpDescr = new float [descrSize] ;
		 vl_dsift_transpose_descriptor (tmpDescr,
			descrs + descrSize * k,
			geom.numBinT,
			geom.numBinX,
			geom.numBinY) ;
		for (int i = 0; i < descrSize; ++i) {
			//descriptor.at<float>(0,i) = descrs[i];
			descriptor.at<float>(0,descrSize * k + i) = tmpDescr[i] >= 0.5 ? 255 : floor(512.0F*tmpDescr[i]);
		}
	 }
 	vl_dsift_delete(dsiftFilt);
	//free space

	return descriptor;
};

/*
* Функция вычисления по блоку изображения - вектора ненаправленных HOG-дескрипторов
*/
Mat Genderize::getIndirectHOG(Mat& image) {
	int	nwin_x=2;
	int	nwin_y=2;
	int he = image.rows;
	int wi = image.cols;
	//cout<<Im;
	Mat descriptor(1, nwin_x*nwin_y*8, CV_32F);


	Mat grad_x,grad_y,angles(image.rows,image.cols,CV_32F),magnits(image.rows,image.cols,CV_32F);

	//sepFilter2D(Im, grad_x, Im.depth(), hx, one1, Point(-1,-1), 0,BORDER_REFLECT);
	//sepFilter2D(Im, grad_y, Im.depth(), one1, hy, Point(-1,-1), 0,BORDER_REFLECT);
	sepFilter2D(image, grad_x, image.depth(), HX, ONE_1, Point(-1,-1), 0,BORDER_CONSTANT);
	sepFilter2D(image, grad_y, image.depth(), ONE_1, HY, Point(-1,-1), 0,BORDER_CONSTANT);
	for (int y = 0; y < grad_x.rows; y++) {
		grad_x.at<float>(y,0) = image.at<float>(y,1);
		grad_x.at<float>(y,wi-1) = -image.at<float>(y,wi-2);
	}
	for (int x = 0; x < grad_y.cols; x++) {
		grad_y.at<float>(0,x) = -image.at<float>(1,x);
		grad_y.at<float>(he-1,x) = image.at<float>(he-2,x);
	}
	for (int y = 0; y < grad_x.rows; y++) {
		for (int x = 0; x < grad_x.cols; x++) {
			angles.at<float>(y,x)=atan2(grad_y.at<float>(y,x), grad_x.at<float>(y,x));
			magnits.at<float>(y,x)=sqrt(pow(grad_y.at<float>(y,x),2) + pow(grad_x.at<float>(y,x),2));
		}
	}
	
	int step_x = static_cast<int>(image.cols/(nwin_x));
	int step_y = static_cast<int>(image.rows/(nwin_y));
	int cont=0;
	
	int cntr = 0;
	float* intervals = new float[16];
	for(int i = 0; i < 16; i++){
		intervals[i] = CV_PI*((2.0*(i+1))/16 - 1);
	}
	for (int i=0; i < nwin_y; i++) {
		for(int j=0; j < nwin_x; j++) { 
			int zrs = 0;
			float magcounts[] = {0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0};
			for(int i1 = 0; i1 < step_y; i1++) {
				for(int j1 = 0; j1 < step_x; j1++) {
					float fa = angles.at<float>(i*step_y+i1, j*step_x+j1);
					int interval = static_cast<int>((fa+CV_PI)/((2*CV_PI/16)));
					if (interval < 0) {
						interval == 0;
					}
					if (fa == intervals[1] || fa == intervals[3]) {
						interval++;
					}
					if (fa == -intervals[15] || fa == intervals[15] || interval > 15 || interval < 0) {	
						continue;
					}				
					if (fa == intervals[13]) {	
						interval--;
					}
					//cout << interval<<" ";
					magcounts[interval] += magnits.at<float>(i*step_y+i1,j*step_x+j1);
				}
			}
			for (int i1 =0; i1 < 8; i1++) {
				descriptor.at<float>(0,cntr + i1) = magcounts[i1] + magcounts[8 + i1]; 
			}
			cntr = cntr+8;
		}
	}
	float L2Norm = 0;
	for (int i1 =0; i1 < HOG_SIZE; i1++) {
		L2Norm += descriptor.at<float>(i1) *descriptor.at<float>(i1);
	}
	//cout<<descriptor;
	L2Norm = sqrt(L2Norm) + 0.01;
	descriptor = descriptor / L2Norm;
	return descriptor;
};

/*
* Получение HOG-дескриптора для блока
*/
Mat Genderize::getHOGFeatureVector(Mat& image, int start_y,int start_x,int end_y,int end_x) {
	Mat subImage = image.rowRange(start_y-1,end_y).colRange(start_x-1,end_x);
	//cout<<subImage;
	Mat descriptor = getIndirectHOG(subImage);	
	return descriptor;
};

/*
* Получение DSIFT-дескриптора для блока
*/
Mat Genderize::getSIFTFeatureVector(Mat& image,int start_y,int start_x,int end_y,int end_x) {
	Mat subMat = image.rowRange(start_y-1,end_y).colRange(start_x-1,end_x).t()+0;
	Mat descriptor = getDSIFT(subMat);
	//cout<<descriptor;
	return descriptor;
};

/*
* Получение GABOR-дескриптора для блока (gaborScaledRow - 18 полных GABOR-изображений)
*/
Mat Genderize::getGaborFeatureVector(Mat& gaborScaledRow, int startY,int startX,int endY,int endX) {
	vector<int> indexes = getGaborIndexes(startY, startX, endY, endX); // Получение индексов по положению блока
	Mat descriptor = Mat::zeros(1,indexes.size(),CV_32F);
	for (int i = 0; i< indexes.size();i++) {
		descriptor.at<float>(0,i) = gaborScaledRow.at<float>(0,indexes[i]);
	}
	return descriptor;
};

vector<Mat> Genderize::gaborFilterBank(int u, int v, int m, int n) {

	vector<Mat>gaborArray;
	float fmax = 0.25;
	float gama = sqrt(2.0);
	float  eta = sqrt(2.0);

	for (int i = 1; i<= u;i++) {

		float fu = fmax/pow(sqrt(2.0),(i-1));
		float alpha = fu/gama;
		float beta = fu/eta;

		for(int j = 1; j <=v; j++) {
			float tetav = ((j-1)*CV_PI/v);
			Mat reFilter(n,m,CV_32F);
			Mat imFilter(n,m,CV_32F);

			for(int x = 1; x<=m;x++) {
				for(int y = 1; y<=n;y++) {
					float xprime = (x-((m+1.0)/2))*cos(tetav)+(y-((n+1.0)/2))*sin(tetav);
					float yprime = -(x-((m+1.0)/2))*sin(tetav)+(y-((n+1.0)/2))*cos(tetav);
					reFilter.at<float>(x-1,y-1) = (fu*fu/(CV_PI*gama*eta))*exp(-((alpha*alpha)*(xprime*xprime)+(beta*beta)*(yprime*yprime)))*cos(2*CV_PI*fu*xprime);
					imFilter.at<float>(x-1,y-1) = (fu*fu/(CV_PI*gama*eta))*exp(-((alpha*alpha)*(xprime*xprime)+(beta*beta)*(yprime*yprime)))*sin(2*CV_PI*fu*xprime);
				}
			}
			gaborArray.push_back(reFilter);
			gaborArray.push_back(imFilter);
		}
	}
	return gaborArray;
};

void Genderize::paddingWithFFT(Mat src, Mat& FFT) {
	Mat srss[] = {src, Mat::zeros(src.size(), CV_32F)};
	Mat complexS;
	merge(srss,2,complexS);
	Mat padded;
	copyMakeBorder(complexS, padded, OFFSET, OFFSET, OFFSET, OFFSET, BORDER_REFLECT, Scalar::all(0));
	dft(padded, FFT);
};

vector<Mat> Genderize::prepareGabor(const vector<Mat>& gaborArray, int fftSize) {
	vector<Mat> gaborFFTArray;
	Mat padded, complexF;
	for (int i =0; i < gaborArray.size(); i+=2) {
		Mat gabors[] = {gaborArray[i],gaborArray[i+1]};
		merge(gabors,2,complexF);
		copyMakeBorder(complexF, padded, fftSize-complexF.rows, 0,  fftSize-complexF.rows, 0, BORDER_CONSTANT, Scalar::all(0));
		Mat FFT;
		dft(padded, FFT);
		gaborFFTArray.push_back(FFT);
	};
	return gaborFFTArray;
};

void Genderize::gaborFeatures(const cv::Mat& srcFFT, const vector<Mat>& gaborFFTArray, Mat*& gaborImages, int half_filter, int initialHeight, int initialWidth) {
	int start = 12 - half_filter;
	for(int i =0 ; i < gaborFFTArray.size(); i++) { 
		Mat dst;
		Mat planes[] = {Mat::zeros(initialHeight,initialWidth,CV_32F), Mat::zeros(initialHeight,initialWidth, CV_32F)};
		mulSpectrums(srcFFT, gaborFFTArray[i], dst, 0, false);
		idft(dst, dst, DFT_SCALE);
		split(dst.rowRange(start,start+initialHeight).colRange(start,start+initialWidth), planes);  
		magnitude(planes[0], planes[1], planes[0]);
		gaborImages[i] = planes[0];
	};
};

/*
* Получение индексов в строке с 18 полными габорами по положению блока
*/
vector<int> Genderize::getGaborIndexes(int startY, int startX, int endY, int endX) {
	vector<int> indexes;
	endX = endX / 2; endY = endY / 2;
	startX = startX < 2 ? 1 : startX / 2;
	startY = startY < 2 ? 1 : startY / 2;
		
	int step2 = (endX - startX + 1) / 3;
	step2 = step2 < 1 ? 1 : (step2 > 3 ? 3 : step2);
	int step1 = (endY - startY + 1) / 3;
	step1 = step1 < 1 ? 1 : (step1 > 3 ? 3 : step1);
	vector<int> temp;
	for (int i = 1; i <= SZ/2; i++) {
		for (int j = 1; j <= SZ/2; j++) {
			if ((i >= startY && i <= endY && (i - startY) % step1 == 0) &&
				(j >= startX && j <= endX && (j - startX) % step2 == 0)){
				temp.push_back((i - 1) * ( SZ / 2 ) + j - 1);
			}
		}
	}
	for (int i = 0; i < 18; i++) {			
		for (int j = 0; j < temp.size(); j++) {
			indexes.push_back(temp[j] + i*(SZ/2)*(SZ/2));
		}
	}
	return indexes;
};

Mat Genderize::generateGabors(vector<Mat> images) {
	int N = images.size();	
	Mat features = Mat::zeros(N,18*(SZ/2)*(SZ/2),CV_32F);
	for(int i=0; i < N; i++) {
		cout<<i<<" ";
		Mat srcFFT;
		Mat* gaborImages = new Mat[18];
		paddingWithFFT(images[i],srcFFT);		
		gaborFeatures(srcFFT,gaborFFTS,gaborImages,9,SZ,SZ);
		Mat gaborScaledRow;
		Mat gabors = Mat::zeros(18,(SZ/2)*(SZ/2),CV_32F);
		for (int j = 0; j< 18;j++) {
			Mat scalled;
			resize(gaborImages[j],scalled,Size(SZ/2,SZ/2),0,0,INTER_LINEAR);
			gabors.row(j) = scalled.reshape(1,1) + 0;				
			scalled.release();
		}

		if (gaborScaledRow.rows == 0) {
			gaborScaledRow = gabors.reshape(1,1) + 0;
		} else {
			hconcat(gaborScaledRow,gabors.reshape(1,1)+0,gaborScaledRow);
		}
		gabors.release();
		features.row(i) = gaborScaledRow + 0;
		delete[] gaborImages;
	}
	return features;
};
