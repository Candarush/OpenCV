#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
#include <iostream>
#include <vector>
#include <thread>
#include <string>

using namespace std;
using namespace cv;

void FilterCloud(Mat* mat)
{
    int i,j;
    for (i=0; i<mat->cols; i++) {
        for (j=0; j<mat->rows; j++) {
            Vec3b p = mat->at<Vec3b>(j,i);
            if (p[0] < 120)
            {
                p = Vec3b(255,255,255);
            }
            else
            {
                p = Vec3b(0,0,0);
            }
        }
    }
}

void ColorCheckThread(mutex* m,int* completionMarker,Mat* inImage, Mat* outImage, float** output, int curCellX, int curCellY, int cellSizeX, int cellSizeY)
{
    
    int i,j;
    int cloudPoints = 0;
    
    for (i=curCellY*cellSizeY;i<(curCellY+1)*cellSizeY;i++)
    {
        for (j=curCellX*cellSizeX;j<(curCellX+1)*cellSizeX;j++)
        {
            Vec3b p = inImage->at<Vec3b>(i,j);
            if (p[2]>120)
            {
                cloudPoints++;
            }
        }
    }
    double cloudness = (double)cloudPoints/(cellSizeX*cellSizeY);
    (output[curCellY])[curCellX] = cloudness;
    int greyScale = 255*cloudness;
    m->lock();
    rectangle(*outImage, Point(curCellX*cellSizeX,curCellY*cellSizeY), Point((curCellX+1)*cellSizeX,(curCellY+1)*cellSizeY), Vec3b(greyScale,greyScale,greyScale), FILLED);
    (*completionMarker)--;
    m->unlock();
}

void CompareHistThread(mutex* m,int* completionMarker, Mat* inHist , Mat* inImage, Mat* outImage, float** output, int curCellX, int curCellY, int cellSizeX, int cellSizeY)
{
    Mat section = (*inImage)(Rect(curCellX*cellSizeX, curCellY*cellSizeY, cellSizeX, cellSizeY));
    Mat greyMat;
    Mat rgbchannel[3];
    split( section, rgbchannel );
    greyMat = rgbchannel[2];
    Mat greyHist;
    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };
    bool uniform = true, accumulate = false;
    Mat hist = Mat(Size(256,256),CV_8UC3);
    calcHist( &greyMat, 1, 0, Mat(), greyHist, 1, &histSize, &histRange, uniform, accumulate );
    normalize(greyHist, greyHist, 0, 256, NORM_MINMAX, -1, Mat() );
    
    MatND secHist = MatND(greyHist);
    MatND comHist = MatND(*inHist);
    
    double cloudness = compareHist(secHist, comHist, HISTCMP_BHATTACHARYYA);
    (output[curCellY])[curCellX] = cloudness;
    int greyScale = 255*cloudness;
    m->lock();
    rectangle(*outImage, Point(curCellX*cellSizeX,curCellY*cellSizeY), Point((curCellX+1)*cellSizeX,(curCellY+1)*cellSizeY), Vec3b(greyScale,greyScale,greyScale), FILLED);
    (*completionMarker)--;
    m->unlock();
}

void ColorCheck(int maxthreads, int gridSizeX, int gridSizeY, string imagePath) {
    ///
    
    int i,j,threads = 0;
    float** output = new float*[gridSizeY];
    for (i=0;i<gridSizeY;i++)
    {
        output[i] = new float[gridSizeX];
    }
    
    for (i=0;i<gridSizeX;i++)
    {
        for (j=0;j<gridSizeY;j++)
        {
            (output[j])[i]=0;
        }
    }
    
    Mat src = imread(imagePath, IMREAD_COLOR);
    if (src.empty())
    {
        cout << "Картинка не найдена!" << endl;
        return;
    }
    
    Mat resultImage = Mat(src.clone());
    int imageSizeX = src.cols;
    int imageSizeY = src.rows;
    cout << "Размер фото: " << imageSizeX << " " << imageSizeY << endl;

    int cellSizeX = imageSizeX/gridSizeY;
    int cellSizeY = imageSizeY/gridSizeX;
    
    vector<thread> computeVector;
    mutex compMutex;

    for (i=0;i<gridSizeX;i++)
    {
        for (j=0;j<gridSizeY;j++)
        {
            while (threads >= maxthreads)
            {
                
            }
            threads++; computeVector.push_back(thread(ColorCheckThread,&compMutex,&threads,&src,&resultImage,output,i,j,cellSizeY,cellSizeX));
            computeVector.back().detach();
        }
    }
    while (threads > 0)
    {
        //Ждем
    }
    vconcat(src ,resultImage, resultImage);
    imshow("Image", resultImage);
    
    //Чистка
    for (int i=0; i<gridSizeY; ++i) {
      free(output[i]);
    }
    free(output);
}

void CompareHistograms(int maxthreads, int gridSizeX, int gridSizeY, string imagePath)
{
    
    Mat src = imread(imagePath, IMREAD_COLOR ).clone();
    if( src.empty() )
    {
        cout << "Картинка не найдена!" << endl;
        return;
    }
    
    Mat compareMat = imread(imagePath, IMREAD_COLOR ).clone();
    if( compareMat.empty() )
    {
        cout << "Картинка не найдена!" << endl;
        return;
    }
    Mat greyMat;
    Mat rgbchannel[3];
    split( compareMat, rgbchannel );
    greyMat = rgbchannel[2];
    Mat greyHist;
    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };
    bool uniform = true, accumulate = false;
    Mat hist = Mat(Size(256,256),CV_8UC3);
    calcHist( &greyMat, 1, 0, Mat(), greyHist, 1, &histSize, &histRange, uniform, accumulate );
    normalize(greyHist, greyHist, 0, 256, NORM_MINMAX, -1, Mat() );
    
    Mat resultImage = Mat(src.clone());
    int imageSizeX = src.cols;
    int imageSizeY = src.rows;
    cout << "Размер фото: " << imageSizeX << " " << imageSizeY << endl;

    int cellSizeX = imageSizeX/gridSizeY;
    int cellSizeY = imageSizeY/gridSizeX;
    
    int i,j,threads = 0;
    float** output = new float*[gridSizeY];
    for (i=0;i<gridSizeY;i++)
    {
        output[i] = new float[gridSizeX];
    }
    
    vector<thread> computeVector;
    mutex compMutex;
    
    for (i=0;i<gridSizeX;i++)
    {
        for (j=0;j<gridSizeY;j++)
        {
            while (threads >= maxthreads)
            {
                
            }
            threads++; computeVector.push_back(thread(CompareHistThread,&compMutex,&threads,&greyHist,&src,&resultImage,output,i,j,cellSizeY,cellSizeX));
            computeVector.back().detach();
        }
    }
    while (threads > 0)
    {
        //Ждем
    }
    vconcat(src ,resultImage, resultImage);
    imshow("Image", resultImage);
    
    //Чистка
    for (int i=0; i<gridSizeY; ++i) {
      free(output[i]);
    }
    free(output);
}

void OpticalFlow(string imagePath, int speed)
{
    
    VideoCapture capture(imagePath);
    if (!capture.isOpened()){
        cout << "Видео не найдено!" << endl;
        return;
    }

    int i;
    Mat frame1, prvs;
    capture >> frame1;
    frame1 = frame1.clone();
    FilterCloud(&frame1);
    resize(frame1, frame1, Size(640, 320), 0, 0, INTER_CUBIC);
    cvtColor(frame1, prvs, COLOR_BGR2GRAY);
    while(true){
        Mat frame2,frame2unresized, next;
        for (i=1; i<speed; i++) {
            capture.grab();
        }
        capture >> frame2unresized;
        frame2unresized = frame2unresized.clone();
        FilterCloud(&frame2unresized);
        if (frame2unresized.empty())
            break;
        resize(frame2unresized, frame2, Size(640, 320), 0, 0, INTER_CUBIC);
        cvtColor(frame2, next, COLOR_BGR2GRAY);
        Mat flow(prvs.size(), CV_32FC2);
        calcOpticalFlowFarneback(prvs, next, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
        // visualization
        Mat flow_parts[2];
        split(flow, flow_parts);
        Mat magnitude, angle, magn_norm;
        cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);
        normalize(magnitude, magn_norm, 0.0f, 1.0f, NORM_MINMAX);
        angle *= ((1.f / 360.f) * (180.f / 255.f));
        //build hsv image
        Mat _hsv[3], hsv, hsv8, bgr;
        _hsv[0] = angle;
        _hsv[1] = Mat::ones(angle.size(), CV_32F);
        _hsv[2] = magn_norm*3;
        merge(_hsv, 3, hsv);
        hsv.convertTo(hsv8, CV_8U, 255.0);
        cvtColor(hsv8, bgr, COLOR_HSV2BGR);
        resize(bgr, bgr, Size(frame2unresized.cols, frame2unresized.rows), 0, 0, INTER_CUBIC);
        vconcat(bgr, frame2unresized, bgr);
        imshow("OpticFlow", bgr);
        int keyboard = waitKey(1);
        if (keyboard == 'q' || keyboard == 27)
            break;
        prvs = next;
    }
}


int main() {

    ColorCheck(8, 20, 20, "/Users/vladislavakimov/Documents/xCode/OpenCVtesting/OpenCVtesting/Samples/Foto3.png");
    
    //CompareHistograms(8,20,20,"/Users/vladislavakimov/Documents/xCode/OpenCVtesting/OpenCVtesting/Samples/Foto3.png");
    
    //OpticalFlow("/Users/vladislavakimov/Documents/xCode/OpenCVtesting/OpenCVtesting/Samples/Clouds.mp4", 5);
    
    waitKey();

}
