#include "opencv2/opencv.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

// https://docs.opencv.org/2.4/doc/tutorials/introduction/display_image/display_image.html de aici am citirea
// https://docs.opencv.org/3.4/d8/dbc/tutorial_histogram_calculation.html histograma

using namespace std;
using namespace cv;

int* histogram(Mat image, int *hist) {
    int pixel;

    for(int i = 0; i < image.rows; i++) {
        for(int j = 0; j < image.cols; j++) {
            pixel = image.at<uchar>(i, j);
            hist[pixel]++;  
        }
    }

   // for(int i = 0; i < 256; i++) {
    //    cout << hist[i] << " ";
 
  //  }
}

// https://www.geeksforgeeks.org/histogram-equalisation-in-c-image-processing/
// https://medium.com/@animeshsk3/back-to-basics-part-1-histogram-equalization-in-image-processing-f607f33c5d55
// wikipedia
// https://github.com/akanimax/multithreaded-histogram-equalization-cpp/blob/master/hist_equal.cpp
void histogramEqualization(Mat image) {
    int *hist = (int*) calloc(256, sizeof(int));
    int *equalizedHist = (int*) calloc(256, sizeof(int));

    Mat image_ycrcb;
    cvtColor(image, image_ycrcb, CV_BGR2YCrCb);

    vector<Mat> channels;
    split(image_ycrcb, channels);

    // calculam histograma imaginii
    histogram(channels[0], hist);

    int probability = 0;

    for(int i = 0; i < 256; i++) {
        equalizedHist[i] = round((hist[i] + probability) * 255 / (image.rows * image.cols));
        probability += hist[i];
    }

    for(int i = 0; i < image.rows; i++) {
        for(int j = 0; j < image.cols; j++) {
            channels[0].at<uchar>(i, j) = (uchar)(equalizedHist[channels[0].at<uchar>(i, j)]);
        }
    }

    merge(channels, image_ycrcb);

    cvtColor(image_ycrcb, image, CV_YCrCb2BGR);

    namedWindow("Histogram Equalization", WINDOW_AUTOSIZE);
    imshow("Histogram Equalization", image);

}

// https://digitalcommons.unf.edu/cgi/viewcontent.cgi?referer=https://www.google.com/&httpsredir=1&article=1264&context=etd&fbclid=IwAR0AA7GANkNKfCOLxPzH_-Vdr8yAdGmF8zvfJ16LD-yFuGhsQ659sckSGvo
void adaptiveHistogramEqualization(Mat image, int window) {
    int rank, pixels;
    Mat image_ycrcb;
    cvtColor(image, image_ycrcb, CV_BGR2YCrCb);

    vector<Mat> channels, equalizedChannels;
    split(image_ycrcb, channels);
    equalizedChannels = channels;

    for(int i = 0; i < image.rows; i++) {
        for(int j = 0; j < image.cols; j++) {
            rank = 0;
            pixels = 0;

            for(int k = i - window / 2; k < i + window / 2; k++) {
                for(int l = j - window / 2; l < j + window / 2; l++) {
                    if(k >= 0 && k < image.rows && l >= 0 && l < image.cols) {
                        if(channels[0].at<uchar>(i, j) > channels[0].at<uchar>(k, l)) {
                            rank++;
                        }

                        pixels++;
                    }
                }
            }

            if(pixels != 0) {
                equalizedChannels[0].at<uchar>(i, j) = round(rank * 255 / pixels);               
            }
        }
    }

    merge(equalizedChannels, image_ycrcb);
    cvtColor(image_ycrcb, image, CV_YCrCb2BGR);
    namedWindow("Adaptive Histogram Equalization", WINDOW_AUTOSIZE);
    imshow("Adaptive Histogram Equalization", image);
}


// aceasta este functia cu problema
void AHEAlgorithm(Mat image, int window) {
    int rank, pixel, firstCalc = 0, pixels;
    Mat image_ycrcb;
    int *hist = (int*) calloc(256, sizeof(int));
    int *equalizedHist = (int*) calloc(256, sizeof(int));

   cvtColor(image, image_ycrcb, CV_BGR2YCrCb);

    vector<Mat> channels, equalizedChannels;
    split(image_ycrcb, channels);
    split(image_ycrcb, equalizedChannels);
   // equalizedChannels = channels;

    for(int i = 0; i < image.rows; i++) {
        for(int j = 0; j < image.cols; j++) {
           // rank = 0;
            pixels = 0;

           // if(firstCalc == 0) {
            memset(hist, 0, sizeof(hist));
            //memset(equalizedHist, 0, sizeof(hist));
                
            for(int k = i - window / 2; k < i + window / 2; k++) {
                for(int l = j - window / 2; l < j + window / 2; l++) {
                    if(k >= 0 && k < image.rows && l >= 0 && l < image.cols) {
                        pixel = channels[0].at<uchar>(k, l);
                        hist[pixel]++;
                        pixels++;
                    }
                }
            }

            int CHist = 0;

            for(int k = 0; k < channels[0].at<uchar>(i, j); k++) {
                CHist += hist[k];
            }

            if(pixels != 0) {
                equalizedChannels[0].at<uchar>(i, j) = (uchar)round((float)CHist * 255 / pixels);              
            }

        }

    }

    merge(equalizedChannels, image_ycrcb);
    cvtColor(image_ycrcb, image, CV_YCrCb2BGR);
    namedWindow("Adaptive Histogram Equalization", WINDOW_AUTOSIZE);
    imshow("Adaptive Histogram Equalization", image);
}

int* histogram2(Mat image, int *hist, int x1, int x2, int y1, int y2) {
    int pixel;

    memset(hist, 0, sizeof(hist));

    for(int i = x1; i < x2; i++) {
        for(int j = y1; j < y2; j++) {
            pixel = image.at<uchar>(i, j);
            hist[pixel]++;  
        }
    }
}

void CLAHEAlgorithmNuMerge(Mat image, int window, float alpha) {
    int rank, pixels, excess;
    Mat image_ycrcb;
    cvtColor(image, image_ycrcb, CV_BGR2YCrCb);

    vector<Mat> channels, equalizedChannels;
    split(image_ycrcb, channels);
    equalizedChannels = channels;

    int *hist = (int*) calloc(256, sizeof(int));
    int *equalizedHist = (int*) calloc(256, sizeof(int));

    for(int i = 0; i < image.rows; i++) {
        for(int j = 0; j < image.cols; j++) {
            rank = 0;
            pixels = 0;
            excess = 0;

            memset(hist, 0, sizeof(hist));

            for(int k = i - window / 2; k < i + window / 2; k++) {
                for(int l = j - window / 2; l < j + window / 2; l++) {
                    if(k >= 0 && k < image.rows && l >= 0 && l < image.cols) {
                        hist[channels[0].at<uchar>(k, l)]++;
                        pixels++;
                    }
                }
            }

            int probability = 0;

            if(pixels != 0) {
                for(int k = 0; k < 256; k++) {
                    equalizedHist[k] = round((hist[k] + probability) * 255 / pixels);
                    probability += hist[k];
                }

                equalizedChannels[0].at<uchar>(i, j) = (uchar)(equalizedHist[channels[0].at<uchar>(i, j)]);             
            }
        }
    }
    
    merge(equalizedChannels, image_ycrcb);
    cvtColor(image_ycrcb, image, CV_YCrCb2BGR);
    namedWindow("Adaptive Histogram Equalization", WINDOW_AUTOSIZE);
    imshow("Adaptive Histogram Equalization", image); 
}

void CLAHEAlgorithm(Mat image, float beta) {
    Ptr<CLAHE> CLAHE = createCLAHE();
    Mat image_ycrcb;

    cvtColor(image, image_ycrcb, CV_BGR2YCrCb);
    vector<Mat> channels;
    split(image_ycrcb, channels);

    CLAHE->setClipLimit(beta);
    CLAHE->apply(channels[0], channels[0]);

    merge(channels, image_ycrcb);
    cvtColor(image_ycrcb, image, CV_YCrCb2BGR);
    namedWindow("CLAHE Algorithm", WINDOW_AUTOSIZE);
    imshow("CLAHE Algorithm", image);
}

void colorStretching(Mat image) {
    int *hist = (int*) calloc(256, sizeof(int));
    int *equalizedHist = (int*) calloc(256, sizeof(int));
    
    Mat image_ycrcb;
    int minim = 99999999, maxim = -1;
    cvtColor(image, image_ycrcb, CV_BGR2YCrCb);

    vector<Mat> channels;
    split(image_ycrcb, channels);

    // calculam histograma imaginii
    histogram(channels[0], hist);

    for(int i = 0; i < 256; i++) {
        if(hist[i] != 0) {
            if(minim > i) {
                minim = i;
            } else if(maxim < i) {
                maxim = i;
            }
        }
    }

    for(int i = 0; i < image.rows; i++) {
        for(int j = 0; j < image.cols; j++) {
            channels[0].at<uchar>(i, j) = (uchar)round(((float)channels[0].at<uchar>(i, j) - minim) / (maxim - minim) * 255);
            if(channels[0].at<uchar>(i, j) > 255) {
                channels[0].at<uchar>(i, j) = 255;
            } 
        } 
    }

    merge(channels, image_ycrcb);
    cvtColor(image_ycrcb, image, CV_YCrCb2BGR);
    namedWindow("Color Stretching", WINDOW_AUTOSIZE);
    imshow("Color Stretching", image);

}

void plotHistogram(Mat image) {
    // vectorul de 3 matrici (R, G, B) pentru fiecare canal
    vector<Mat> channels;
    // histogramele pt fiecare canal R, G, B
    Mat hist_r, hist_g, hist_b;
    float valRange[] = {0, 256};
    const float* range = {valRange};
    int size = 256;

    // despart cele 3 canale
    split(image, channels);

    // calculez histogramele pt fiecare canal
    calcHist(&channels[0], 1, 0, Mat(), hist_b, 1, &size, &range, true, false);
    calcHist(&channels[1], 1, 0, Mat(), hist_g, 1, &size, &range, true, false);
    calcHist(&channels[2], 1, 0, Mat(), hist_r, 1, &size, &range, true, false);

    // dimensiunile cu care voi reprezenta grafic histograma
    Mat histograma(500, 500, CV_8UC3, Scalar(0, 0, 0));

    // normalizez vectorii care reprezinta histogramele
    normalize(hist_b, hist_b, 0, histograma.rows, NORM_MINMAX, -1, Mat());
    normalize(hist_g, hist_g, 0, histograma.rows, NORM_MINMAX, -1, Mat());
    normalize(hist_r, hist_r, 0, histograma.rows, NORM_MINMAX, -1, Mat());

    int bins = cvRound((double) 500 / size);

    for( int i = 1; i < 256; i++ ) {
        line(histograma, Point( bins * (i - 1), 500 - cvRound(hist_b.at<float>(i - 1))),
              Point( bins * (i), 500 - cvRound(hist_b.at<float>(i))),
              Scalar(255, 0, 0), 2, 8, 0);
        line(histograma, Point( bins * (i - 1), 500 - cvRound(hist_g.at<float>(i - 1))),
              Point( bins * (i), 500 - cvRound(hist_g.at<float>(i))),
              Scalar(0, 255, 0), 2, 8, 0);
        line(histograma, Point( bins * (i - 1), 500 - cvRound(hist_r.at<float>(i - 1))),
              Point(bins * (i), 500 - cvRound(hist_r.at<float>(i))),
              Scalar(0, 0, 255), 2, 8, 0);
    }

    namedWindow("Histogram", WINDOW_AUTOSIZE);
    imshow("Histogram", histograma);
}

int main (int argc, char** argv) {
    Mat image;

    if (argc != 2) {
        cout << "No input or output image\n";
        return -1;
    }

    // Mat mat= imread(filename, CV_LOAD_IMAGE_ANYDEPTH);
    image = imread(argv[1], IMREAD_COLOR);

    if (!image.data) {
        cout << "Error\n";
        return -1;
    }

    namedWindow("Input image", WINDOW_AUTOSIZE);
    imshow("Input image", image);
   // imwrite("output", image);

    // calculate and plot histogram
    //plotHistogram(image);

    // calculate histogram
    //histogram(image);

    // histogram equalization algorithm
   // histogramEqualization(image);

    // adaptive histogram equalization algorithm
  //  adaptiveHistogramEqualization(image, 125);

    // CLAHE Algorithm
   // CLAHEAlgorithm(image, 4);

    // Color Stretching Algorithm
    //colorStretching(image);

    // SWAHE Algorithm
    AHEAlgorithm(image, 125);

    waitKey(0);
    return 0;
}
