#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>

#include <string>


#include "helper.hpp"
#include "eigenfaces.hpp"

using namespace std;
using namespace cv;



int main() {

	string prefix = "src/images/";
	string csv_file = prefix + "csv_list.txt";

	Mat temp;
	Mat image = imread(prefix + "d.jpg");

	int rows = image.rows;
	int columns = image.cols;
	image.convertTo(temp, CV_32S);
	//image = image.reshape(1,1);

	//image.reshape(1, 1).convertTo(temp, CV_32FC1);

	cout << temp;
	imshow("1", image);                   // Show our image inside it.
	imwrite("test.jpg", temp);


	waitKey(1);

	return 0;
}