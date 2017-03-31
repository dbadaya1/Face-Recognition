#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>

#include <string>


#include "helper.hpp"
#include "eigenfaces.hpp"
#include "ImageItem.hpp"

using namespace std;
using namespace cv;



int main() {
	string prefix = "images/att_faces/";
	string csv_file = prefix + "csv_list.txt";

	
	vector<ImageItem> items;
	// read in the images
	try {
		items = Helper::readCsv(csv_file,prefix);
	}
	catch (exception& e) {
		cerr << "Error opening file \"" << csv_file << "\"." << endl;
		return 1;
	}

	
	ImageItem testData;
	testData.image = imread(prefix + "s5/5.pgm", IMREAD_GRAYSCALE);
	testData.label = -1;
	int actualLabel = 4;
	
	Eigenfaces eigenfaces(items, 80);

	double confidence = eigenfaces.predict(testData);
	cout << "actual=" << actualLabel << " / predicted=" << testData.label << endl;

	

	waitKey(0);

	return 0;
}