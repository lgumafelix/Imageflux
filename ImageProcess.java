/*
 * Lucky Gumafelix
 * CSC 492 - Senior Design Project
 * Imageflux: Parking Detection System using Image Processing
 * Dr. Mohsen Beheshti
 */
package imageflux;

import com.googlecode.javacpp.Loader;
import com.googlecode.javacv.cpp.opencv_core.CvPoint;
import com.googlecode.javacv.cpp.opencv_core.CvScalar;
import com.googlecode.javacv.cpp.opencv_core.CvSeq;
import com.googlecode.javacv.cpp.opencv_core.IplImage;

import static com.googlecode.javacv.cpp.opencv_core.*;
import static com.googlecode.javacv.cpp.opencv_imgproc.*;
import static com.googlecode.javacv.cpp.opencv_highgui.*;
import java.nio.ByteBuffer;

public class ImageProcess {

    protected IplImage src, grayImage, smoothImage, erodeImage, thresholdImage, cannyImage, contourImage, resultImage;
    protected int maxLevel = 0;
    static int maxThresh = 600;
    public static int availableSpace = 0;
    public static int occupiedSpace = 0;
    public static int totalSpace = 0;
    CvMemStorage storage = CvMemStorage.create();
    CvSeq seq = cvCreateSeq(0, Loader.sizeof(CvContour.class), Loader.sizeof(CvSeq.class), storage);
    CvSeq corner;
    static CvPoint endPoints[], midPoints[], branchPoints[], topEndPoint[], bottomEndPoint[], mergeEndPoints[];

    public ImageProcess(IplImage source) {
        src = source.clone();
        grayImage = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 1);
        smoothImage = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 1);
        erodeImage = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 1);
        thresholdImage = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 1);
        cannyImage = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 1);
        contourImage = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 1);
        resultImage = source.clone();
    }

    public IplImage cvtToGray() {
        cvCvtColor(src, grayImage, CV_BGR2GRAY);
        return grayImage;
    }

    public IplImage SmoothIplImage() {
        cvtToGray();
        cvSmooth(grayImage, smoothImage, CV_GAUSSIAN, 3);
        return smoothImage;
    }

    public IplImage ThresholdIplImage() {
        cvtToGray();
        SmoothIplImage();
        cvThreshold(smoothImage, thresholdImage, 60, 120, CV_THRESH_TOZERO);
        return thresholdImage;
    }

    public IplImage ErodeIplImage() {
        cvtToGray();
        SmoothIplImage();
        ThresholdIplImage();
        cvErode(thresholdImage, erodeImage, null, 1);
        return erodeImage;
    }

    public IplImage CannyIplImage() {
        cvtToGray();
        SmoothIplImage();
        ThresholdIplImage();
        ErodeIplImage();
        Canny(erodeImage, cannyImage, 80, 200, 3, true);
        return cannyImage;
    }

    public CvSeq FindContourIplImage(int maxLevel) {
        cvtToGray();
        SmoothIplImage();
        ErodeIplImage();
        ThresholdIplImage();
        CannyIplImage();
        cvFindContours(cannyImage, storage, seq, Loader.sizeof(CvContour.class), CV_RETR_LIST, CV_LINK_RUNS);
        //cvDrawContours(src, seq, CvScalar.GREEN, CvScalar.RED, maxLevel, 2, 8, CvPoint.ZERO);
        //cvApproxPoly(seq, Loader.sizeof(CvContour.class), storage, CV_POLY_APPROX_DP, 8, 0);
        return seq;
    }

    public void DrawContours() {
        FindContourIplImage(0);
        cvDrawContours(src, seq, CvScalar.GREEN, CvScalar.RED, maxLevel, 2, 8, CvPoint.ZERO);
        cvShowImage("src", src);
        release("Original Image", src);
    }

    public void showOriginalImage() {
        cvShowImage("Original Image", src);
        release("Original Image", src);
    }

    public void showGrayImage() {
        cvShowImage("Gray Image", grayImage);
        release("Gray Image", grayImage);
    }

    public void showSmoothImage() {
        cvShowImage("Smooth Image", smoothImage);
        release("Smooth Image", smoothImage);
    }

    public void showErodeImage() {
        cvShowImage("Erode Image", erodeImage);
        release("Erode Image", erodeImage);
    }

    public void showThresholdImage() {
        cvShowImage("Threshold Image", thresholdImage);
        release("Threshold Image", thresholdImage);
    }

    public void showCannyImage() {
        cvShowImage("Canny Image", cannyImage);
        release("Canny Image", cannyImage);
    }

    public void release(String s, IplImage img) {
        cvWaitKey();
        cvReleaseImage(img);
        cvDestroyWindow(s);
    }

    public void DetectGivenPoints() {
        // counter variables
        int i, j, k;
        cvtToGray();
        SmoothIplImage();
        ErodeIplImage();
        ThresholdIplImage();
        CannyIplImage();
        CvSeq seq = FindContourIplImage(maxLevel);
        //cvFindContours(cannyImage, storage, seq, Loader.sizeof(CvContour.class), CV_RETR_LIST, CV_LINK_RUNS);
        corner = cvApproxPoly(seq, Loader.sizeof(CvContour.class), cvCreateMemStorage(0), CV_POLY_APPROX_DP, cvContourPerimeter(seq) * 0.02, 0);

        endPoints = new CvPoint[corner.total()];
        midPoints = new CvPoint[corner.total()];

        // loop through ALL the detected points (endpoints and branchpoints) and store them to both array
        // we will process them separately later
        for (i = 0; i < corner.total(); i++) {
            endPoints[i] = new CvPoint(cvGetSeqElem(corner, i));
            midPoints[i] = new CvPoint(cvGetSeqElem(corner, i));
            //cvDrawCircle(src, endPoints[i], 4, CvScalar.RED, -1, 8, 0);
            //System.out.println("[" + i + "] X value = " + endPoints[i].x() + "\t; Y value = " + endPoints[i].y());
        }

        // draw all TOP POINTS
        for (i = 0; i < (corner.total() / 2); i++) {
            //cvDrawCircle(src, endPoints[i], 4, CvScalar.BLUE, -1, 8, 0);
            //System.out.println("[" + i + "] X value = " + endPoints[i].x() + "\t; Y value = " + endPoints[i].y());
        }
        // draw all TOP ENDPOINTS only
        /*
         * Point #11 needs to be manually drawn/detected here.
         * see lines 176 to 177.
         */
        int topEndpointsCount = 0;
        for (i = 0; i < (corner.total() / 2); i += 3) {
            if (i % 3 == 0) {
                topEndpointsCount++;
                //cvDrawCircle(src, endPoints[i], 4, CvScalar.BLUE, -1, 8, 0);
                //System.out.println("[" + i + "] X value = " + endPoints[i].x() + "\t; Y value = " + endPoints[i].y());
            }
        }
        //cvDrawCircle(image, endPoints[11], 4, CvScalar.BLUE, -1, 8, 0);
        //System.out.println("[" + 11 + "] X value = " + endPoints[11].x() + "\t; Y value = " + endPoints[11].y());

        // set coordinates for TOP ENDPOINTS 
        topEndPoint = new CvPoint[5];
        for (i = 0; i < topEndPoint.length; i++) {
            for (j = 0; j < (corner.total() / 2); j += 3) {
                topEndPoint[i] = new CvPoint(endPoints[j].x(), endPoints[j].y());
                //cvDrawCircle(src, topEndPoint[i], 4, CvScalar.BLUE, -1, 8, 0);
                //System.out.println("topEndPoint[" + i + "] (" + topEndPoint[i].x() + ", " + topEndPoint[i].y() + ")");
                i++;
            }
        }
        /* Again, manually add and draw point #11 to topEnPoint[4] = (601, 61)
         */
        topEndPoint[4] = new CvPoint(endPoints[11].x(), endPoints[11].y());
        //cvDrawCircle(src, topEndPoint[4], 4, CvScalar.BLUE, -1, 8, 0);
        //System.out.println("topEndPoint[" + 4 + "] (" + topEndPoint[4].x() + ", " + topEndPoint[4].y() + ")");

        // print out ALL top endpoints
        for (i = 0; i < topEndPoint.length; i++) {
            //cvDrawCircle(src, topEndPoint[i], 4, CvScalar.BLUE, -1, 8, 0);
            //System.out.println("topEndPoint[" + i + "] (" + topEndPoint[i].x() + ", " + topEndPoint[i].y() + ")");
        }

        // draw all BOTTOM POINTS
        for (i = (corner.total() / 2); i < corner.total(); i++) {
            //cvDrawCircle(src, endPoints[i], 4, CvScalar.BLUE, -1, 8, 0);
            //System.out.println("[" + i + "] X value = " + endPoints[i].x() + "\t; Y value = " + endPoints[i].y());
        }

        // draw all BOTTOM ENDPOINTS only
        int bottomEndPointsCount = 0;
        for (i = (corner.total() / 2); i < corner.total(); i += 3) {
            bottomEndPointsCount++;
            //cvDrawCircle(src, endPoints[i], 4, CvScalar.BLUE, -1, 8, 0);
            //System.out.println("[" + i + "] X value = " + endPoints[i].x() + "\t; Y value = " + endPoints[i].y());
        }
        //System.out.println("bottomEndPointsCounts = " + bottomEndPointsCount);

        // set coordinates BOTTOM ENDPOINTS 
        bottomEndPoint = new CvPoint[5];
        for (i = 0; i < bottomEndPoint.length; i++) {
            for (j = (corner.total() / 2); j < corner.total(); j += 3) {
                bottomEndPoint[i] = new CvPoint(endPoints[j].x(), endPoints[j].y());
                //cvDrawCircle(src, bottompEndPoint[i], 4, CvScalar.BLUE, -1, 8, 0);
                //System.out.println("bottompEndPoint[" + i + "] (" + bottompEndPoint[i].x() + ", " + bottompEndPoint[i].y() + ")");
                i++;
            }
        }

        // print out ALL bottom endpoints
        for (i = 0; i < bottomEndPoint.length; i++) {
            //cvDrawCircle(src, bottompEndPoint[i], 4, CvScalar.BLUE, -1, 8, 0);
            //System.out.println("bottomEndPoint[" + i + "] (" + bottompEndPoint[i].x() + ", " + bottompEndPoint[i].y() + ")");
        }

        // now merge the topEndPoints and bottomEndPoints into one array
        int mergeTopPointSize = 10;
        mergeEndPoints = new CvPoint[mergeTopPointSize];
        for (i = 0; i < mergeEndPoints.length; i++) {
            // first add the top endpoints
            for (j = i; j < topEndpointsCount; j++) {
                mergeEndPoints[i] = new CvPoint(topEndPoint[j].x(), topEndPoint[j].y());
                //cvDrawCircle(src, mergeEndPoints[i], 4, CvScalar.BLUE, -1, 8, 0);
                //System.out.println("mergeEndPoints[" + i + "] (" + mergeEndPoints[i].x() + ", " + mergeEndPoints[i].y() + ")");
                i++;
            }
        }
        /* Manually add Point #11
         * mergeEndPoint[4] = topEndPoint[4]
         */
        mergeEndPoints[4] = new CvPoint(topEndPoint[4].x(), topEndPoint[4].y());
        //cvDrawCircle(src, mergeEndPoints[4], 4, CvScalar.BLUE, -1, 8, 0);
        //System.out.println("mergeEndPoints[" + 4 + "] (" + mergeEndPoints[4].x() + ", " + mergeEndPoints[4].y() + ")");

        // then add the bottom endpoints
        for (i = bottomEndPointsCount; i < mergeEndPoints.length; i++) {
            for (j = 0; j < bottomEndPointsCount; j++) {
                mergeEndPoints[i] = new CvPoint(bottomEndPoint[j].x(), bottomEndPoint[j].y());
                i++;
            }
        }
        //try printing ALL ENDPOINTS (top & bottom)
        for (i = 0; i < mergeEndPoints.length; i++) {
            //cvDrawCircle(src, mergeEndPoints[i], 5, CvScalar.BLUE, -1, 8, 0);
            //System.out.println("mergeEndPoint[" + i + "] (" + mergeEndPoints[i].x() + ", " + mergeEndPoints[i].y() + ")");
        }

        // MANUAL SETUP OF BRANCHPOINTS
        branchPoints = new CvPoint[8];
        branchPoints[0] = new CvPoint(endPoints[2].x(), endPoints[2].y());
        branchPoints[1] = new CvPoint(endPoints[5].x(), endPoints[5].y());
        branchPoints[2] = new CvPoint(endPoints[8].x(), endPoints[8].y());
        branchPoints[3] = new CvPoint(endPoints[13].x(), endPoints[13].y());
        //cvDrawCircle(src, branchPoints[0], 4, CvScalar.BLUE, -1, 8, 0);
        //System.out.println("branchPoints[" + 0 + "] (" + branchPoints[0].x() + ", " + branchPoints[0].y() + ")");

        // do remaining bottom branch points
        for (i = (branchPoints.length) / 2; i < branchPoints.length; i++) {
            for (j = (corner.total() / 2) + 2; j < corner.total(); j += 3) {
                //System.out.println("i = "+i+"\tj = "+j);
                branchPoints[i] = new CvPoint(endPoints[j].x(), endPoints[j].y());
                //cvDrawCircle(src, branchPoints[i], 4, CvScalar.BLUE, -1, 8, 0);
                //System.out.println("branchPoints[" + i + "] (" + branchPoints[i].x() + ", " + branchPoints[i].y() + ")");
                i++;
            }
        }

        // print all branchpoints
        for (i = 0; i < branchPoints.length; i++) {
            //cvDrawCircle(src, branchPoints[i], 5, CvScalar.BLUE, -1, 8, 0);
            //System.out.println("branchPoints[" + i + "] (" + branchPoints[i].x() + ", " + branchPoints[i].y() + ")");
        }
    }

    public void DetectPerfectPoints() {
        // best case scenario detecting all 26 points

        int i, j, k;
        cvtToGray();
        SmoothIplImage();
        ErodeIplImage();
        ThresholdIplImage();
        CannyIplImage();
        CvSeq seq = FindContourIplImage(maxLevel);
        //cvFindContours(cannyImage, storage, seq, Loader.sizeof(CvContour.class), CV_RETR_LIST, CV_LINK_RUNS);
        corner = cvApproxPoly(seq, Loader.sizeof(CvContour.class), cvCreateMemStorage(0), CV_POLY_APPROX_DP, cvContourPerimeter(seq) * 0.02, 0);

        endPoints = new CvPoint[corner.total()];
        midPoints = new CvPoint[corner.total()];

        // loop through ALL the detected points (endpoints and branchpoints) and store them to both array
        // we will process them separately later
        for (i = 0; i < corner.total(); i++) {
            endPoints[i] = new CvPoint(cvGetSeqElem(corner, i));
            midPoints[i] = new CvPoint(cvGetSeqElem(corner, i));
            //cvDrawCircle(src, endPoints[i], 4, CvScalar.RED, -1, 8, 0);
            //System.out.println("[" + i + "] X value = " + endPoints[i].x() + "\t; Y value = " + endPoints[i].y());
        }

        // draw all TOP POINTS
        for (i = 0; i < (corner.total() / 2); i++) {
            //cvDrawCircle(image, endPoints[i], 4, CvScalar.BLUE, -1, 8, 0);
            //System.out.println("[" + i + "] X value = " + endPoints[i].x() + "\t; Y value = " + endPoints[i].y());
        }

        // draw all TOP ENDPOINTS only
        int topEndpointsCount = 0;
        for (i = 0; i < (corner.total() / 2); i += 3) {
            if (i % 3 == 0) {
                topEndpointsCount++;
                //cvDrawCircle(image, endPoints[i], 4, CvScalar.BLUE, -1, 8, 0);
                //System.out.println("[" + i + "] X value = " + endPoints[i].x() + "\t; Y value = " + endPoints[i].y());
            }
        }
        //System.out.println("topEndpointsCount = " + topEndpointsCount);

        // set coordinates for TOP ENDPOINTS 
        CvPoint topEndPoint[] = new CvPoint[topEndpointsCount];
        for (i = 0; i < topEndPoint.length; i++) {
            for (j = 0; j < (corner.total() / 2); j += 3) {
                topEndPoint[i] = new CvPoint(endPoints[j].x(), endPoints[j].y());
                //cvDrawCircle(image, topEndPoint[i], 4, CvScalar.BLUE, -1, 8, 0);
                //System.out.println("topEndPoint[" + i + "] (" + topEndPoint[i].x() + ", " + topEndPoint[i].y() + ")");
                i++;
            }
        }

        // print out ALL top endpoints
        for (i = 0; i < topEndPoint.length; i++) {
            //cvDrawCircle(image, topEndPoint[i], 4, CvScalar.BLUE, -1, 8, 0);
            //System.out.println("topEndPoint[" + i + "] (" + topEndPoint[i].x() + ", " + topEndPoint[i].y() + ")");
        }

        // draw all BOTTOM POINTS
        for (i = (corner.total() / 2); i < corner.total(); i++) {
            //cvDrawCircle(image, endPoints[i], 4, CvScalar.BLUE, -1, 8, 0);
            //System.out.println("[" + i + "] X value = " + endPoints[i].x() + "\t; Y value = " + endPoints[i].y());
        }

        // draw all BOTTOM ENDPOINTS only
        int bottomEndPointsCount = 0;
        for (i = (corner.total() / 2); i < corner.total(); i += 3) {
            bottomEndPointsCount++;
            //cvDrawCircle(image, endPoints[i], 4, CvScalar.BLUE, -1, 8, 0);
            //System.out.println("[" + i + "] X value = " + endPoints[i].x() + "\t; Y value = " + endPoints[i].y());
        }
        //System.out.println("bottomEndPointsCounts = " + bottomEndPointsCount);

        // set coordinate for BOTTOM ENDPOINTS 
        CvPoint bottompEndPoint[] = new CvPoint[bottomEndPointsCount];
        for (i = 0; i < bottompEndPoint.length; i++) {
            for (j = (corner.total() / 2); j < corner.total(); j += 3) {
                bottompEndPoint[i] = new CvPoint(endPoints[j].x(), endPoints[j].y());
                //cvDrawCircle(image, bottompEndPoint[i], 4, CvScalar.BLUE, -1, 8, 0);
                //System.out.println("bottompEndPoint[" + i + "] (" + bottompEndPoint[i].x() + ", " + bottompEndPoint[i].y() + ")");
                i++;
            }
        }

        // print out ALL bottom endpoints
        for (i = 0; i < bottompEndPoint.length; i++) {
            //cvDrawCircle(image, bottompEndPoint[i], 4, CvScalar.BLUE, -1, 8, 0);
            //System.out.println("bottomEndPoint[" + i + "] (" + bottompEndPoint[i].x() + ", " + bottompEndPoint[i].y() + ")");
        }

        // now merge the topEndPoints and bottomEndPoints into one array
        int mergeTopPointSize = topEndpointsCount + bottomEndPointsCount;
        mergeEndPoints = new CvPoint[mergeTopPointSize];
        for (i = 0; i < mergeEndPoints.length; i++) {
            // first add the top endpoints
            for (j = i; j < topEndpointsCount; j++) {
                mergeEndPoints[i] = new CvPoint(topEndPoint[j].x(), topEndPoint[j].y());
                i++;
            }
        }
        for (i = bottomEndPointsCount; i < mergeEndPoints.length; i++) {
            // then add the bottcom endpoints
            for (k = 0; k < bottomEndPointsCount; k++) {
                mergeEndPoints[i] = new CvPoint(bottompEndPoint[k].x(), bottompEndPoint[k].y());
                i++;
            }
        }

        //try printing ALL ENDPOINTS (top & bottom)
        for (i = 0; i < mergeEndPoints.length; i++) {
            //cvDrawCircle(image, mergeEndPoints[i], 5, CvScalar.BLUE, -1, 8, 0);
            //System.out.println("mergeEndPoint[" + i + "] (" + mergeEndPoints[i].x() + ", " + mergeEndPoints[i].y() + ")");
        }

        //setup the branch points taken from the top points
        branchPoints = new CvPoint[8];
        // setup top branchpoints (2,5,8,11)
        for (i = 0; i < branchPoints.length / 2; i++) {
            for (j = 2; j < (endPoints.length / 2); j += 3) {
                branchPoints[i] = new CvPoint(endPoints[j].x(), endPoints[j].y());
                //cvDrawCircle(image, branchPoints[i], 4, CvScalar.BLUE, -1, 8, 0);
                //System.out.println("branchPoints[" + i + "] (" + branchPoints[i].x() + ", " + branchPoints[i].y() + ")");
                i++;
            }
        }
        // setup bottom branchpoints (15,18,21,24)
        for (i = branchPoints.length / 2; i < branchPoints.length; i++) {
            for (j = (endPoints.length / 2) + 2; j < endPoints.length; j += 3) {
                branchPoints[i] = new CvPoint(endPoints[j].x(), endPoints[j].y());
                //cvDrawCircle(image, branchPoints[i], 4, CvScalar.BLUE, -1, 8, 0);
                //System.out.println("branchPoints[" + i + "] (" + branchPoints[i].x() + ", " + branchPoints[i].y() + ")");
                i++;
            }
        }

        // draw all branchpoints
        for (i = 0; i < branchPoints.length; i++) {
            //cvDrawCircle(image, branchPoints[i], 4, CvScalar.BLUE, -1, 8, 0);
            //System.out.println("branchPoints[" + i + "] (" + branchPoints[i].x() + ", " + branchPoints[i].y() + ")");
        }
    }

    public IplImage computeParkingSpace() {
        // reset the parking spaces everytime a new ImageProcess is invoked
        availableSpace = 0;
        occupiedSpace = 0;
        totalSpace = 0;
        
        cvtToGray();
        SmoothIplImage();
        ErodeIplImage();
        ThresholdIplImage();
        CannyIplImage();
        FindContourIplImage(maxLevel);
        //DetectGivenPoints();
        DetectPerfectPoints();

        // do the top endpoints + branchpoints
        for (int i = 0; i < 4; i++) {
            calculatetopPixel(resultImage, cannyImage, mergeEndPoints[i], branchPoints[i]);
        }
        // do the bottom endpoints + branchpoints
        for (int i = (branchPoints.length / 2); i < branchPoints.length; i++) {
            calculatebottomPixel(resultImage, cannyImage, mergeEndPoints[i + 1], branchPoints[i]);
        }
        return resultImage;
    }

    private void calculatetopPixel(IplImage src, IplImage canny, CvPoint endPoint, CvPoint branchPoint) {
        /* src = image to draw the results
         * canny = image used to calculaate the black and white pixels of the parking
         * endPoint & branchPOint - opposite points to draw the rectangle on the region of interest
         */

        ByteBuffer buffer = canny.getByteBuffer();
        int whitepix = 0;
        int blackpix = 0;
        int rows = canny.height();
        int cols = canny.width();

        //System.out.println("The Pair of points:");
        //System.out.println("endpoint (" + endPoint.x() + ", " + endPoint.y() + ") and branchpoint (" + branchPoint.x() + ", " + branchPoint.y() + ")");
        int vec[][] = new int[rows][cols];
        int rowStart, rowEnd, colStart, colEnd, rowHeight, colWidth;

        rowStart = endPoint.y();
        colStart = endPoint.x();
        rowEnd = branchPoint.y();
        colEnd = branchPoint.x();
        rowHeight = rowEnd - rowStart;
        colWidth = colEnd - colStart;

        //System.out.println("width = " + colWidth);
        //System.out.println("height = " + rowHeight);
        for (int i = rowStart; i < rowEnd; i++) {
            for (int j = colStart; j < colEnd; j++) {
                int index = ((i * canny.widthStep()) + (j * canny.nChannels()));
                vec[i][j] = (buffer.get(index) & 0xFF);
                // compare to black (0)
                if (vec[i][j] == 0) {
                    blackpix++;
                } // compare to white (255)
                else if (vec[i][j] == 255) {
                    whitepix++;
                }
            }
        }
        // if white pixels within the rectangular region is greater than threshold level, it is occupied
        if (whitepix > maxThresh) {
            cvRectangle(src, endPoint, branchPoint, CvScalar.RED, 4, 8, 0);
            occupiedSpace++;
        } // if not, then it is available
        else {
            cvRectangle(src, endPoint, branchPoint, CvScalar.GREEN, 4, 8, 0);
            availableSpace++;
        }
        totalSpace = occupiedSpace + availableSpace;
    }

    private void calculatebottomPixel(IplImage src, IplImage canny, CvPoint endPoint, CvPoint branchPoint) {
        /* src = image to draw the results
         * canny = image used to calculaate the black and white pixels of the parking
         * endPoint & branchPOint - opposite points to draw the rectangle on the region of interest
         */

        ByteBuffer buffer = canny.getByteBuffer();
        int whitepix = 0;
        int blackpix = 0;
        int rows = canny.height();
        int cols = canny.width();

        //System.out.println("The Pair of points:");
        //System.out.println("endpoint (" + endPoint.x() + ", " + endPoint.y() + ") and branchpoint (" + branchPoint.x() + ", " + branchPoint.y() + ")");
        int vec[][] = new int[rows][cols];
        int rowStart, rowEnd, colStart, colEnd, rowHeight, colWidth;

        rowStart = branchPoint.y();
        colStart = branchPoint.x();
        rowEnd = endPoint.y();
        colEnd = endPoint.x();
        rowHeight = rowEnd - rowStart;
        colWidth = colEnd - colStart;

        //System.out.println("width = " + colWidth);
        //System.out.println("height = " + rowHeight);
        for (int i = rowStart; i < rowEnd; i++) {
            for (int j = colStart; j < colEnd; j++) {
                int index = ((i * canny.widthStep()) + (j * canny.nChannels()));
                vec[i][j] = (buffer.get(index) & 0xFF);
                // compare to black (0)
                if (vec[i][j] == 0) {
                    blackpix++;
                } // compare to white (255)
                else if (vec[i][j] == 255) {
                    whitepix++;
                }
            }
        }
        // if white pixels within the rectangular region is greater than threshold level, it is occupied
        if (whitepix > maxThresh) {
            cvRectangle(src, endPoint, branchPoint, CvScalar.RED, 4, 8, 0);
            occupiedSpace++;
        } // if not, then it is available
        else {
            cvRectangle(src, endPoint, branchPoint, CvScalar.GREEN, 4, 8, 0);
            availableSpace++;
        }
        totalSpace = occupiedSpace + availableSpace;
    }

    public IplImage drawEndPoints() {
        DetectPerfectPoints();
        // draw ALL ENDPOINTS (top & bottom)
        for (int i = 0; i < mergeEndPoints.length; i++) {
            cvDrawCircle(src, mergeEndPoints[i], 5, CvScalar.RED, -1, 8, 0);
            //System.out.println("mergeEndPoint[" + i + "] (" + mergeEndPoints[i].x() + ", " + mergeEndPoints[i].y() + ")");
        }
        return src;
    }

    public IplImage drawBranchPoints() {
        DetectPerfectPoints();
        // draw ALL BRANCHPOINTS
        for (int i = 0; i < branchPoints.length; i++) {
            cvDrawCircle(src, branchPoints[i], 5, CvScalar.BLUE, -1, 8, 0);
            //System.out.println("branchPoints[" + i + "] (" + branchPoints[i].x() + ", " + branchPoints[i].y() + ")");
        }
        return src;
    }

    public IplImage drawAllPoints() {
        drawEndPoints();
        // draw ALL ENDPOINTS (top & bottom)
        for (int i = 0; i < mergeEndPoints.length; i++) {
            cvDrawCircle(src, mergeEndPoints[i], 5, CvScalar.RED, -1, 8, 0);
            //System.out.println("mergeEndPoint[" + i + "] (" + mergeEndPoints[i].x() + ", " + mergeEndPoints[i].y() + ")");
        }
        // draw ALL BRANCHPOINTS
        for (int i = 0; i < branchPoints.length; i++) {
            cvDrawCircle(src, branchPoints[i], 5, CvScalar.BLUE, -1, 8, 0);
            //System.out.println("branchPoints[" + i + "] (" + branchPoints[i].x() + ", " + branchPoints[i].y() + ")");
        }
        return src;
    }
}
