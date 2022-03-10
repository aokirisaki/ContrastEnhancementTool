#include "opencv2/opencv.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <math.h>
#include <time.h>

using namespace std;
using namespace cv;

// generearea histogramei
int* histogram(Mat image, int *hist) {
    int pixel;

    // parcurg fiecare pixel si ii numar aparitiile in matricea pixelilor
    for(int i = 0; i < image.rows; i++) {
        for(int j = 0; j < image.cols; j++) {
            pixel = image.at<uchar>(i, j);
            hist[pixel]++;  
        }
    }
}

void histogramEqualization(Mat image) {
    int *hist = (int*) calloc(256, sizeof(int));
    int *equalizedHist = (int*) calloc(256, sizeof(int));

    // convertesc imaginea BGR primita ca parametru in YCrCb
    Mat image_ycrcb;
    cvtColor(image, image_ycrcb, CV_BGR2YCrCb);

    // despart canalele de culoare ale imaginii
    vector<Mat> channels;
    split(image_ycrcb, channels);

    // calculez histograma imaginii
    histogram(channels[0], hist);

    int probability = 0;

    // aplic algoritmul Histogram Equalization
    for(int i = 0; i < 256; i++) {
        equalizedHist[i] = round((hist[i] + probability) * 255 / (image.rows * image.cols));
        probability += hist[i];
    }

    // updatez valorile intensitatilor pixelilor la valorile calculate mai sus
    for(int i = 0; i < image.rows; i++) {
        for(int j = 0; j < image.cols; j++) {
            channels[0].at<uchar>(i, j) = (uchar)(equalizedHist[channels[0].at<uchar>(i, j)]);
        }
    }

    // unesc inapoi canalele de culoare
    merge(channels, image_ycrcb);

    // convertesc imaginea inapoi la BGR
    cvtColor(image_ycrcb, image, CV_YCrCb2BGR);

    // afisez imaginea modificata
    namedWindow("Histogram Equalization", WINDOW_AUTOSIZE);
    imshow("Histogram Equalization", image);
}

void adaptiveHistogramEqualization(Mat image, int window) {
    int rank, pixels;
    Mat image_ycrcb;

    // convertesc imaginea BGR primita ca parametru in YCrCb
    cvtColor(image, image_ycrcb, CV_BGR2YCrCb);

    // despart canalele de culoare ale imaginii
    vector<Mat> channels, equalizedChannels;
    split(image_ycrcb, channels);
    equalizedChannels = channels;

    // parcurg matricea de pixeli
    for(int i = 0; i < image.rows; i++) {
        for(int j = 0; j < image.cols; j++) {
            // initializez cu 0 rank-ul pixelului central al sub-imaginii
            // si nr de pixeli din sub-imagine
            rank = 0;
            pixels = 0;

            // parcurg fiecare pixel din sub-imagine
            for(int k = i - window / 2; k < i + window / 2; k++) {
                for(int l = j - window / 2; l < j + window / 2; l++) {
                    // daca ma aflu in matricea de pixeli (nu ies in afara)
                    if(k >= 0 && k < image.rows && l >= 0 && l < image.cols) {
                        // daca valoarea intensitatii pixelului central este mai mare ca
                        // cea a pixelului la care am ajuns cu parcurgerea
                        if(channels[0].at<uchar>(i, j) > channels[0].at<uchar>(k, l)) {
                            // creste rank-ul pixelului central
                            rank++;
                        }

                        pixels++;
                    }
                }
            }

            // daca am avut pixeli in sub-imagine
            if(pixels != 0) {
                // calculez noua valoare a intensitatii pixelului central
                equalizedChannels[0].at<uchar>(i, j) = round(rank * 255 / pixels);               
            }
        }
    }

    // unesc inapoi canalele de culoare
    merge(equalizedChannels, image_ycrcb);

    // convertesc imaginea inapoi la BGR
    cvtColor(image_ycrcb, image, CV_YCrCb2BGR);

    // afisez imaginea modificata
    namedWindow("Adaptive Histogram Equalization", WINDOW_AUTOSIZE);
    imshow("Adaptive Histogram Equalization", image);
}

// o alta versiune a algoritmului Adaptive Histogram Equalization
// fata de cea anterioara, este mai slab optimizata fiindca
// se calculeaza rank-ul fiecarui pixel din sub-imagine
void AHEAlgorithm(Mat image, int window) {
    int rank, pixel, firstCalc = 0, pixels, CHist;
    Mat image_ycrcb;
    int *hist = (int*) calloc(256, sizeof(int));

    // convertesc imaginea BGR primita ca parametru in YCrCb
   cvtColor(image, image_ycrcb, CV_BGR2YCrCb);

    // despart canalele de culoare ale imaginii
    vector<Mat> channels, equalizedChannels;
    split(image_ycrcb, channels);
    split(image_ycrcb, equalizedChannels);

    // parcurg matricea de pixeli
    for(int i = 0; i < image.rows; i++) {
        for(int j = 0; j < image.cols; j++) {
            // initializez nr de pixeli din sub-imagine
            pixels = 0;

            // initializez histograma sub-imaginii
            memset(hist, 0, 256 * sizeof(int));
            
            // parcurg fiecare pixel din sub-imagine
            for(int k = i - window / 2; k <= i + window / 2; k++) {
                for(int l = j - window / 2; l <= j + window / 2; l++) {
                    // daca ma aflu in matricea de pixeli (nu ies in afara)
                    if(k >= 0 && k < image.rows && l >= 0 && l < image.cols) {
                        // cresc rank-ul pixelului curent
                        pixel = channels[0].at<uchar>(k, l);
                        hist[pixel]++;
                        // cresc nr de pixeli din sub-imagine
                        pixels++;
                    }
                }
            }

            CHist = 0;

            // calculez noua intensitate a pixelului central
            // adunand in CHist rank-urile pentru pixelii cu intensitati mai mici decat a lui
            for(int k = 0; k <= channels[0].at<uchar>(i, j); k++) {
                CHist += hist[k];
            }

            if(pixels != 0) {
                equalizedChannels[0].at<uchar>(i, j) = (uchar)round((float)CHist * 255 / pixels);              
            }

        }

    }

    // unesc inapoi canalele de culoare
    merge(equalizedChannels, image_ycrcb);

    // convertesc imaginea inapoi la BGR
    cvtColor(image_ycrcb, image, CV_YCrCb2BGR);

    // afisez imaginea modificata
    namedWindow("Adaptive Histogram Equalization - neoptimizat", WINDOW_AUTOSIZE);
    imshow("Adaptive Histogram Equalization - neoptimizat", image);
}

void SWAHEAlgorithm(Mat image, int window) {
    int rank, pixel, firstCalc = 0, pixels, CHist = 0;
    Mat image_ycrcb;
    int *hist = (int*) calloc(256, sizeof(int));

    // convertesc imaginea BGR primita ca parametru in YCrCb
   cvtColor(image, image_ycrcb, CV_BGR2YCrCb);

   // despart canalele de culoare ale imaginii
    vector<Mat> channels, equalizedChannels;
    split(image_ycrcb, channels);
    split(image_ycrcb, equalizedChannels);

    // variabilele in care voi retine care este coloana cea mai din stanga a sub-imaginii
    // si care este cea mai din dreapta
    int last_column = 0, right_column;

    // parcurg matricea de pixeli
    for(int i = 0; i < image.rows; i++) {
        for(int j = 0; j < image.cols; j++) {
            if(last_column < j - window / 2) {
                // scot valorile corespunzatoare pixelilor din coloana cea mai din stanga
                // care nu face parte din noua sub-imagine
                for(int k = i - window / 2; k <= i + window / 2; k++) {
                    if(k >= 0 && k < image.rows) {
                        pixel = channels[0].at<uchar>(k, last_column);
                        hist[pixel]--;
                        pixels--;
                    }
                }

                // updatez care este noua coloana cea mai din stanga
                last_column = j - window / 2;
            }

            // adaug valorile corespunzatoare celei mai din dreapta coloane
            for(int k = i - window / 2; k <= i + window / 2; k++) {
                if(k >= 0 && k < image.rows) {
                    right_column = j + window / 2;
                    if(right_column < image.cols) {
                        pixel = channels[0].at<uchar>(k, right_column);
                        hist[pixel]++;
                        pixels++;
                    }
                }
            }

            // calculez valoarea intensitatii pixelului central ca la AHE
           CHist = 0;

            for(int k = 0; k <= channels[0].at<uchar>(i, j); k++) {
                CHist += hist[k];
            }

            equalizedChannels[0].at<uchar>(i, j) = (uchar)round((float)CHist * 255 / pixels);

        }
    }

    // unesc inapoi canalele de culoare
    merge(equalizedChannels, image_ycrcb);

    // convertesc imaginea inapoi la BGR
    cvtColor(image_ycrcb, image, CV_YCrCb2BGR);

    // afisez imaginea modificata
    namedWindow("Sliding Window Adaptive Histogram Equalization", WINDOW_AUTOSIZE);
    imshow("Sliding Window Adaptive Histogram Equalization", image);
}

void CLAHEAlgorithm(Mat image, int window, float clipping_limit) {
    int rank, pixel, firstCalc = 0, pixels, excess;
    Mat image_ycrcb;
    int *hist = (int*) calloc(256, sizeof(int));

    // convertesc imaginea BGR primita ca parametru in YCrCb
   cvtColor(image, image_ycrcb, CV_BGR2YCrCb);

    // despart canalele de culoare ale imaginii
    vector<Mat> channels, equalizedChannels;
    split(image_ycrcb, channels);
    split(image_ycrcb, equalizedChannels);

    // parcurg matricea de pixeli
    for(int i = 0; i < image.rows; i++) {
        for(int j = 0; j < image.cols; j++) {
            // initializez nr de pixeli din sub-imagine
            // si histograma sub-imaginii
            pixels = 0;
            memset(hist, 0, 256 * sizeof(int));
                
            // parcurg sub-imaginea pixel cu pixel
            for(int k = i - window / 2; k <= i + window / 2; k++) {
                for(int l = j - window / 2; l <= j + window / 2; l++) {
                    if(k >= 0 && k < image.rows && l >= 0 && l < image.cols) {
                        // calculez rank-ul fiecarui pixel
                        pixel = channels[0].at<uchar>(k, l);
                        hist[pixel]++;
                        pixels++;
                    }
                }
            }

            // variabila in care voi pastra excesul
            excess = 0;

            // daca o intensitate are rank mai mare ca clipping_limit
            // iau ce depaseste si adaug la exces
            for(int k = 0; k < 256; k++) {
                if(hist[k] > clipping_limit) {
                    excess = excess + hist[k] - clipping_limit;
                    hist[k] = clipping_limit;
                }
            }

            int m = excess / 255;

            // distribui excesul la valorile care sunt sub clipping_limit
            for(int k = 0; k < 256; k++) {
                if(hist[k] < clipping_limit - m) {
                    hist[k] += m;
                    excess -= m;
                } else if(hist[k] < clipping_limit){
                    excess = excess - clipping_limit + hist[k];
                    hist[k] = clipping_limit;
                }
            }

            // distribui pana excesul a fost tot distribuit
            while(excess > 0) {
                for(int k = 0; k < 256; k++) {
                    if(excess > 0) {
                        if(hist[k] < clipping_limit) {
                            hist[k]++;
                            excess--;
                        }
                    }
                }
            }

            // calculez noua valoare a pixelului central
            int CHist = 0;

            for(int k = 0; k <= channels[0].at<uchar>(i, j); k++) {
                CHist += hist[k];
            }

            if(pixels != 0) {
                equalizedChannels[0].at<uchar>(i, j) = (uchar)round((float)CHist * 255 / pixels);              
            }

        }

    }

    // unesc inapoi canalele de culoare
    merge(equalizedChannels, image_ycrcb);
    
    // convertesc imaginea inapoi la BGR
    cvtColor(image_ycrcb, image, CV_YCrCb2BGR);

    // afisez imaginea modificata
    namedWindow("Contrast Limited Adaptive Histogram Equalization", WINDOW_AUTOSIZE);
    imshow("Contrast Limited Adaptive Histogram Equalization", image); 
}

// CLAHE - versiunea cu implementarea din openCV 
void CLAHEAlgorithmOpenCV(Mat image, float beta) {
    Ptr<CLAHE> CLAHE = createCLAHE();
    Mat image_ycrcb;

    // convertesc imaginea din BGR in YCrCb
    cvtColor(image, image_ycrcb, CV_BGR2YCrCb);
    vector<Mat> channels;
    split(image_ycrcb, channels);

    // stabilesc limita de clipping
    CLAHE->setClipLimit(beta);

    // aplix algoritmul CLAHE asupra canalului cu intensitatile
    CLAHE->apply(channels[0], channels[0]);

    // unesc inapoi canalele
    merge(channels, image_ycrcb);

    // convertesc imaginea inapoi la BGR
    cvtColor(image_ycrcb, image, CV_YCrCb2BGR);

    // afisez imaginea modificata
    namedWindow("CLAHE - openCV", WINDOW_AUTOSIZE);
    imshow("CLAHE - openCV", image);
}

void colorStretching(Mat image) {
    int *hist = (int*) calloc(256, sizeof(int));
    int *equalizedHist = (int*) calloc(256, sizeof(int));
    
    Mat image_ycrcb;
    // initializez valorile minime si maxime ale intensitatilor pixelilor
    int minim = 99999999, maxim = -1;

    // convertesc imaginea din BGR in YCrCb
    cvtColor(image, image_ycrcb, CV_BGR2YCrCb);

    // despart canalele
    vector<Mat> channels;
    split(image_ycrcb, channels);

    // calculez histograma imaginii
    histogram(channels[0], hist);

    // caut valoarea minima si valoarea maxima din histograma
    for(int i = 0; i < 256; i++) {
        if(hist[i] != 0) {
            if(minim > i) {
                minim = i;
            } else if(maxim < i) {
                maxim = i;
            }
        }
    }

    // calculez noile valori ale intensitatilor pixelilor pe baza minimului si a maximului
    for(int i = 0; i < image.rows; i++) {
        for(int j = 0; j < image.cols; j++) {
            channels[0].at<uchar>(i, j) = (uchar)round(((float)channels[0].at<uchar>(i, j) - minim) / (maxim - minim) * 255);
            if(channels[0].at<uchar>(i, j) > 255) {
                channels[0].at<uchar>(i, j) = 255;
            } 
        } 
    }

    // unesc inapoi canalele
    merge(channels, image_ycrcb);

    // convertesc inapoi imaginea la BGR
    cvtColor(image_ycrcb, image, CV_YCrCb2BGR);

    // afisez imaginea modificata
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
    Mat histograma(500, 500, CV_8UC3, Scalar(255, 255, 255));

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
    imwrite( "./histogram.png", histograma);

}

int main (int argc, char** argv) {
    Mat image;

    if (argc < 3) {
        cout << "No input or output image or no specified algorithm\n";
        return -1;
    }

    // citesc imaginea
    image = imread(argv[1], IMREAD_COLOR);

    if (!image.data) {
        cout << "Error\n";
        return -1;
    }

    // afisez imaginea originala
    namedWindow("Input image", WINDOW_AUTOSIZE);
    imshow("Input image", image);

    clock_t begin = clock();

    if(strcmp(argv[2], "histeq") == 0) {
        histogramEqualization(image);
    } else if(strcmp(argv[2], "colorstr") == 0) {
        colorStretching(image);
    } else if (strcmp(argv[2], "AHE") == 0) {
        if(argc < 4) {
            cout << "No window size specified\n";
            return -1;
        }
        
        adaptiveHistogramEqualization(image, std::stoi(argv[3]));
    } else if (strcmp(argv[2], "AHE2") == 0) {
        if(argc < 4) {
            cout << "No window size specified\n";
            return -1;
        }

        AHEAlgorithm(image, std::stoi(argv[3]));
    } else if (strcmp(argv[2], "SWAHE") == 0) {
        if(argc < 4) {
            cout << "No window size specified\n";
            return -1;
        }
cout << std::stoi(argv[3]) << "\n";
        SWAHEAlgorithm(image, std::stoi(argv[3]));
    } else if (strcmp(argv[2], "CLAHE") == 0) {
        if(argc < 5) {
            cout << "No window size or clipping limit specified\n";
            return -1;
        }

        CLAHEAlgorithm(image, std::stoi(argv[3]), std::stoi(argv[4]));
    } else if (strcmp(argv[2], "CLAHEopenCV") == 0) {
        if(argc < 4) {
            cout << "No clipping limit specified\n";
            return -1;
        }

        CLAHEAlgorithmOpenCV(image, std::stoi(argv[3]));
    }

    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    cout << time_spent;

    imwrite( "./output_image.png", image);
    plotHistogram(image);

    return 0;
}
