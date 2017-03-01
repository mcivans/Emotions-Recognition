#include <stdafx.h>
#include <direct.h>
#include <iostream>

using namespace std;
using namespace cv;

int main(int, char**)
{
	// Сгенерировать ли файл с Габорами
	//bool const GENERATE_GABORS = true;
	bool const GENERATE_GABORS = false;
	// Путь к файлу с моделью передается в конструкор
	Genderize* genderize = new Genderize("D:\\_Repositories\\GenderclassificationCAL\\15_1\\exp_smile15.csv");
	// Путь к папке с изображениями
	string sPath = "C:\\_Repositories\\Emotions_Recognition\\Selections\\15\\LDB";
	// Путь к файлу с Габорами
	string sGaborsFile = "C:\\_Repositories\\Emotions_Recognition\\Selections\\15\\GABORS_smile15.csv";
	// Осуществлять ли разбиение файлов из тестовой выборки по детектируемым для них классам
	bool MAKE_SEPARATION = true;
	//bool MAKE_SEPARATION = false;
	// Путь к папке с разбиением файлов
	string sSeperationFolder = "D:\\_Repositories\\GenderclassificationCAL\\Separation";
	// Формировать ли log-file с информацией об ошибках детектирования эмоций
	bool MAKE_ERRORS_LOG = true;
	vector<string> vct = get_all_files_names_within_folder(sPath);
	vector<Mat> images;
	vector<int> lbls;
	int errorsSmileCount = 0, errorsNotSmileCount = 0; // количество ошибочно детектированных файлов классам "Улыбки" и "Неулыбки"
	int imgs = vct.size();
	vector<string> errdetectfiles;
	vector<string> errdetectclass;

	if (MAKE_SEPARATION && !GENERATE_GABORS) {
		string sepfolder = sSeperationFolder + "\\Smile";
		mkdir(sepfolder.c_str());
		sepfolder = sSeperationFolder + "\\NotSmile";
		mkdir(sepfolder.c_str());
	}

	for(int i = 0; i < imgs; i++) {
		cout << i<< " ";
		string path = sPath + "\\"+ vct[i];	
		bool smile = vct[i].find("SMILE") != std::string::npos;
		//bool smile = vct[i].find("S") != std::string::npos;
		lbls.push_back( smile? 0 : 1);
		Mat A = imread(path);
		Mat G;
		cvtColor(A, G, CV_RGB2GRAY );
		Mat src_smile;
		G.convertTo(src_smile,CV_32F);

		if (GENERATE_GABORS)
			images.push_back(src_smile);
		else {
			int predict_label = genderize->predictGender(src_smile);
			
			if (MAKE_SEPARATION) {
				ifstream src(path, ios::binary);
				string destpath;
				if (predict_label == 0)
					destpath = sSeperationFolder + "\\Smile\\" + vct[i];
				else
					destpath = sSeperationFolder + "\\NotSmile\\" + vct[i];
				ofstream dest(destpath, ios::binary);
				dest << src.rdbuf();
			}

			if (lbls[i] != predict_label) {
				if (lbls[i] == 0) {
					errorsSmileCount++;
				}
				else {
					errorsNotSmileCount++;
				}
				cout << vct[i] << endl;
				errdetectfiles.push_back(vct[i]);
				if (predict_label == 0)
					errdetectclass.push_back("S");
				else
					errdetectclass.push_back("NS");
			}
		}

	}

	if (GENERATE_GABORS)
	{
		Mat gaborFeatures = genderize->generateGabors(images);
		writeCSV(sGaborsFile, gaborFeatures);
	}
	else {
		cout << "\nImages:   " << imgs << endl;
		cout << "\nAccuracy:" << endl;
		float smile_acc = (1.0 - float(errorsSmileCount) / imgs) * 100.0;
		float notsmile_acc = (1.0 - float(errorsNotSmileCount) / imgs) * 100.0;
		cout << "   Smile:      " << setprecision(5) << smile_acc << "%   ( " << imgs - errorsSmileCount << " imgs )" << endl;
		cout << "   NotSmile:   " << setprecision(5) << notsmile_acc << "%   ( " << imgs - errorsNotSmileCount << " imgs )" << endl;
	}

	if (MAKE_SEPARATION && MAKE_ERRORS_LOG && !GENERATE_GABORS) {
		string logpath = sSeperationFolder + "\\" + "detection_errors.log";
		ofstream logerrors(logpath, ios::out);
		logerrors << "\nImages:   " << imgs << endl;
		logerrors << "\nDetection errors\n" << "   Smile:    " << errorsSmileCount << " imgs\n   NotSmile: " << errorsNotSmileCount <<  " imgs" << endl;
		logerrors << endl;
		for (int i = 0; i < errdetectfiles.size(); i++)
			logerrors << setw(40) << errdetectfiles.at(i) << "     " << errdetectclass.at(i) << endl;
	}

	cout << "Press any key and Enter to continue..." << endl;
	char ch;
	cin >> ch;

	return 0;
}
