#include <iostream>
#include<string>
#include "bayesian_segnet.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
int main(int argc, char **argv)
{
	if(argc!=4){
		cerr <<endl << "usag: " << "image " << "protxt "<< "caffemodule"<<endl;
		return 0;
	}
	cv::Mat image = cv::imread(argv[1]);
	string  protxt_file = string(argv[2]);
	string caffemodule_file = string(argv[3]);
	BayesianSegNet *mpBayesianSegNet;
	BayesianSegNetParams params{protxt_file, caffemodule_file};
    mpBayesianSegNet = new BayesianSegNet(params);

/*
    void segmentImage(const cv::Mat &image,
                      MatXu &classes,
                      MatXd &confidence,
    MatXd &entropy);
    */
    MatXu classes;
    MatXd confidence;
    MatXd entropy;
    mpBayesianSegNet->segmentImage(image,classes,confidence,entropy);

    cv::Mat confidence_image = mpBayesianSegNet->generateConfidenceImage(confidence);
    cv::Mat entropy_image = mpBayesianSegNet->generateEntropyImage(entropy);
   // cv::Mat variance_image = mpBayesianSegNet->generateVarianceImage()
    cv::Mat segment_image = mpBayesianSegNet->generateSegmentedImage(classes,image);
    cv::imshow("segment",segment_image);
    cv::imshow("confidence",confidence_image);
    cv::imshow("entropy",entropy_image);
   // cv::imshow("2",classes);

    cv::waitKey(0);


	

	return 0;
}
