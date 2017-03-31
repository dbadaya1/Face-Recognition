
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <fstream>
#include <sstream>
#include <iostream>

using namespace cv;
using namespace std;

void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels) {
	
	ifstream file;
	file.exceptions(ifstream::failbit | ifstream::badbit);
	try {
		file.open(filename.c_str());
	}
	catch (const ifstream::failure& e) {
		cout << "Exception opening/reading file";
		cout << e.what();
		cerr << "\n Error: " << strerror(errno);

		return;
	}


	std::string line, path, classlabel;
	// For each line in the given file:
	while (file.good()) {
		std::getline(file, line);
		std::stringstream liness(line);
		// split line
		std::getline(liness, path, ';');
		std::getline(liness, classlabel);
		// push pack the data
		images.push_back(imread(path, 0));
		labels.push_back(atoi(classlabel.c_str()));
	}
}

// Normalizes a given image into a value range between 0 and 255.
Mat norm_0_255(const Mat& src) {
	// Create and return normalized image:
	Mat dst;
	switch (src.channels()) {
	case 1:
		cv::normalize(src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
		break;
	case 3:
		cv::normalize(src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
		break;
	default:
		src.copyTo(dst);
		break;
	}
	return dst;
}

// Converts the images given in src into a row matrix.
Mat asRowMatrix(const vector<Mat>& src, int rtype, double alpha = 1, double beta = 0) {
	// Number of samples:
	size_t n = src.size();
	// Return empty matrix if no matrices given:
	if (n == 0)
		return Mat();
	// dimensionality of (reshaped) samples
	size_t d = src[0].total();
	// Create resulting data matrix:
	Mat data(n, d, rtype);
	// Now copy data:
	for (int i = 0; i < n; i++) {
		//
		if (src[i].empty()) {
			string error_message = format("Image number %d was empty, please check your input data.", i);
			CV_Error(CV_StsBadArg, error_message);
		}
		// Make sure data can be reshaped, throw a meaningful exception if not!
		if (src[i].total() != d) {
			string error_message = format("Wrong number of elements in matrix #%d! Expected %d was %d.", i, d, src[i].total());
			CV_Error(CV_StsBadArg, error_message);
		}
		// Get a hold of the current row:
		Mat xi = data.row(i);
		// Make reshape happy by cloning for non-continuous matrices:
		if (src[i].isContinuous()) {
			src[i].reshape(1, 1).convertTo(xi, rtype, alpha, beta);
		}
		else {
			src[i].clone().reshape(1, 1).convertTo(xi, rtype, alpha, beta);
		}
	}
	return data;
}

int main() {
	// Holds some images:
	vector<Mat> db;

	// Load the greyscale images. The images in the example are
	// taken from the AT&T Facedatabase, which is publicly available
	// at:
	//
	//      http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html
	//
	// This is the path to where I stored the images, yours is different!
	//
	string prefix = "src/images/att_faces/";

	/*
	db.push_back(imread(prefix + "s1/1.pgm", IMREAD_GRAYSCALE));
	db.push_back(imread(prefix + "s1/2.pgm", IMREAD_GRAYSCALE));
	db.push_back(imread(prefix + "s1/3.pgm", IMREAD_GRAYSCALE));
	db.push_back(imread(prefix + "s1/4.pgm", IMREAD_GRAYSCALE));
	db.push_back(imread(prefix + "s1/5.pgm", IMREAD_GRAYSCALE));
	db.push_back(imread(prefix + "s1/6.pgm", IMREAD_GRAYSCALE));
	db.push_back(imread(prefix + "s1/7.pgm", IMREAD_GRAYSCALE));
	db.push_back(imread(prefix + "s1/8.pgm", IMREAD_GRAYSCALE));
	db.push_back(imread(prefix + "s1/9.pgm", IMREAD_GRAYSCALE));
	db.push_back(imread(prefix + "s1/10.pgm", IMREAD_GRAYSCALE));
	*/


	vector<int> labels;
	read_csv(prefix + "csv_list.txt", db, labels);
	

	
	for (const auto& person : labels) {
		// Build a matrix with the observations in row:
		Mat data = asRowMatrix(db, CV_32FC1);

		// Number of components to keep for the PCA:
		int num_components = 10;

		// Perform a PCA:
		PCA pca(data, Mat(), CV_PCA_DATA_AS_ROW, num_components);

		// And copy the PCA results:
		Mat mean = pca.mean.clone();
		Mat eigenvalues = pca.eigenvalues.clone();
		Mat eigenvectors = pca.eigenvectors.clone();

		// The mean face:
		imshow("Average", norm_0_255(mean.reshape(1, db[0].rows)));

		// The first eigenface:
		imshow("EigenFace_1", norm_0_255(pca.eigenvectors.row(0)).reshape(1, db[0].rows));
		
		imshow("Original", imread(prefix + "s1/1.pgm"));
		break;
	}
	waitKey(0);

	// Success!
	return 0;
}