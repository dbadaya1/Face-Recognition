#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include "ImageItem.hpp"


#include "Helper.hpp"

cv::Mat Helper::convertToRowMatrix(const cv::Mat& source, int rtype) {
	cv::Mat destination;
	if (source.isContinuous()) {
		source.reshape(1, 1).convertTo(destination, rtype);
	}
	else {
		source.clone().reshape(1, 1).convertTo(destination, rtype);
	}
	return destination;
}

std::vector<ImageItem> Helper::readCsv(const std::string& filename, const std::string& prefix) {
	std::ifstream file(filename.c_str(), std::ifstream::in);
	if (!file) {
		throw std::exception();
	}
	std::string line, path, classlabel;
	std::vector<ImageItem> items;
	while (std::getline(file, line)) {
		if (line.empty()) {
			break;
		}
		std::stringstream ss(line);

		ImageItem item;
		std::getline(ss, path, ';');
		std::getline(ss, classlabel);

		item.image = cv::imread(prefix + path, CV_LOAD_IMAGE_GRAYSCALE);
		item.label = stoi(classlabel);
		items.push_back(item);
	}
	return items;
}
