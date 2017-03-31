#pragma once

#include "opencv2/opencv.hpp"
#include "ImageItem.hpp"
#include <vector>
#include <string>

class Helper {
public:
	static cv::Mat convertToRowMatrix(const cv::Mat & source, int rtype);
	static std::vector<ImageItem> readCsv(const std::string & filename, const std::string & prefix);

};