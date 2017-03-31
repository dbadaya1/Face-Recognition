#pragma once
#include "opencv2/opencv.hpp"
#include <limits.h>
#include <vector>
#include "ImageItem.hpp"

using namespace std;
using namespace cv;

class Eigenfaces {
private:
	int _noOfComponents;
	double _threshold;
	vector<ImageItem> _projections;
	Mat _eigenvectors;
	Mat _eigenvalues;
	Mat _mean;

public:
	Eigenfaces(const std::vector<ImageItem>& items,int num_components = 0,double threshold = DBL_MAX)
		:_noOfComponents(num_components),_threshold(threshold) {
		train(items);
	}

	void train(const std::vector<ImageItem>& items);


	double predict(ImageItem & item);

	ImageItem project(const ImageItem & item);

	//! returns the eigenvectors of this PCA
	Mat eigenvectors() const { return _eigenvectors; }

	//! returns the eigenvalues of this PCA
	Mat eigenvalues() const { return _eigenvalues; }

	//! returns the mean of this PCA
	Mat mean() const { return _mean; }
};
