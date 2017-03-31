#include "Eigenfaces.hpp"
#include "Helper.hpp"

void Eigenfaces::train(const std::vector<ImageItem> &items) {
	int noOfSamples = items.size();
	int dimensions = items[0].image.total();

	cv::Mat pcaData(noOfSamples, dimensions, CV_8U);
	for (int i = 0; i < noOfSamples; i++) {
		if (items[i].image.total() != dimensions) {
			std::cout << "No. of pixels in image " << i << " wrong ";
			exit(1);
		}
		Helper::convertToRowMatrix(items[i].image,CV_8U).copyTo(pcaData.row(i));
	}


	// clip number of components to be valid
	if ((_noOfComponents <= 0) || (_noOfComponents > noOfSamples))
		_noOfComponents = noOfSamples;

	PCA pca(pcaData, noArray(), CV_PCA_DATA_AS_ROW, _noOfComponents);




	_mean = pca.mean.reshape(1, 1); 
//	_eigenvalues = pca.eigenvalues.clone(); // eigenvalues by row
	cv::transpose(pca.eigenvectors, _eigenvectors); // eigenvectors by column

	this->_projections.resize(noOfSamples);
	ImageItem item;
	for (int i = 0; i < noOfSamples; i++) {
		item.image = pcaData.row(i);
		item.label = items[i].label;
		this->_projections[i] = project(item);
	}
}


double Eigenfaces::predict(ImageItem& item) {
	if (_projections.empty()) {
		cout << "Projections empty \n";
	//	exit(1);
	}
	else if (_eigenvectors.rows != item.image.total()) {
		cout << " Training and test image must be of same size ";
		//exit(1);
	}

	ImageItem projection = project(item);  	// project into PCA subspace

	// find 1-nearest neighbor
	double minConfidence = DBL_MAX;
	int minClass = -1;
	for (int i = 0; i < _projections.size(); i++) {
		double dist = norm(_projections[i].image, projection.image, NORM_L2);
		if ((dist <= minConfidence) && (dist < _threshold)) {
			minConfidence = dist;
			minClass = _projections[i].label;
		}
	}
	item.label = minClass;
	return minConfidence;
}

ImageItem Eigenfaces::project(const ImageItem& item) {
	int dimensions = item.image.total();

	// make sure the data has the correct shape
	if (_eigenvectors.rows != dimensions) {
		string error_message = format("Wrong shapes for given matrices.");
		CV_Error(CV_StsBadArg, error_message);
	}

	// make sure mean is correct if not empty
	if (!_mean.empty() && (_mean.total() != dimensions)) {
		string error_message = format("Wrong mean shape for the given data matrix");
		CV_Error(CV_StsBadArg, error_message);
	}

	ImageItem projection;
	projection.label = item.label;

	Mat temp = item.image;
	temp.reshape(1, 1).convertTo(temp, _mean.type());
	// finally calculate projection as Y = (X-mean)*W
	subtract(temp, _mean, temp);
	gemm(temp, _eigenvectors, 1.0, noArray(), 0.0, projection.image);
	return projection;
}

