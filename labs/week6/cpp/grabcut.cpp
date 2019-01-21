/*
 Copyright 2017 BIG VISION LLC ALL RIGHTS RESERVED

 This program is distributed WITHOUT ANY WARRANTY to the
 Plus and Premium membership students of the online course
 titled "Computer Visionfor Faces" by Satya Mallick for
 personal non-commercial use.

 Sharing this code is strictly prohibited without written
 permission from Big Vision LLC.

 For licensing and other inquiries, please email
 spmallick@bigvisionllc.com

 */

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>

using namespace std;
using namespace cv;

// Instructions
static void help()
{
    cout << "========================================================================"
            "\n Interactive Image Segmentation using GrabCut algorithm.\n "
            "This sample shows interactive image segmentation using grabcut algorithm.\n"
            "\nUSAGE :\n"
            "\t./grabcut <filename>\n"
            "README FIRST :\n"
            "\tSelect a rectangular area around the object you want to segment.\nThen press -n to segment the object (once or a few times)"
            "\nFor any finer touch-ups,  you can press any of the keys below and draw lines on"
            "\nthe areas you want. Then again press -n for updating the output.\n" <<
            "\nHot keys: \n"
            "\tESC - To Exit\n"
            "\tr - To reset the setup\n"
            "\tn - To update the segmentation\n"
            "\n"
            "\tKEY 0 - To select areas of sure background 1\n"
            "\tKEY 1 - To select areas of sure foreground 2\n"
            "\n"
            "\tKEY 2 - To select areas of probable background 3\n"
            "\tKEY 3 - To select areas of probable foreground 4\n"
            "=======================================================================" << endl;
}

// Convention of defining color in opencv is BGR
const Scalar RED = Scalar(0,0,255);
const Scalar PINK = Scalar(230,130,255);
const Scalar BLUE = Scalar(255,0,0);
const Scalar LIGHTBLUE = Scalar(255,255,160);
const Scalar GREEN = Scalar(0,255,0);

int value;

// On every area selection, changes are updated through binmask
static void getBinMask( const Mat& comMask, Mat& binMask )
{
    if( comMask.empty() || comMask.type()!=CV_8UC1 )
        CV_Error( Error::StsBadArg, "comMask is empty or has incorrect type (not CV_8UC1)" );
    if( binMask.empty() || binMask.rows!=comMask.rows || binMask.cols!=comMask.cols )
        binMask.create( comMask.size(), CV_8UC1 );
    binMask = comMask & 1;
}

class Interface
{
public:
    enum{ NOT_SET = 0, IN_PROCESS = 1, SET = 2 };
    static const int radius = 2;
    static const int thickness = -1;

    void reset();
    void setImageAndWinName( const Mat& _image, const string& _winName );
    void showImage() const;
    void mouseClick( int event, int x, int y, int flags, void* param );
    int nextIter();
    int getIterCount() const { return iterCount; }

private:
    void setRectInMask();
    void setLblsInMask( int flags, Point p, bool isPr );

    const string* winName;
    const Mat* image;
    Mat mask;
    Mat bgdModel, fgdModel;

    uchar rectState, lblsState, prLblsState;
    bool isInitialized;

    Rect rect;
    vector<Point> FG_PIXEL, BG_PIXEL, prFG_PIXEL, prBG_PIXEL;
    int iterCount;
};

// Reset everything
void Interface::reset()
{
    if( !mask.empty() )
        mask.setTo(Scalar::all(GC_BGD));
    BG_PIXEL.clear(); FG_PIXEL.clear();
    prBG_PIXEL.clear();  prFG_PIXEL.clear();

    isInitialized = false;
    rectState = NOT_SET;
    lblsState = NOT_SET;
    prLblsState = NOT_SET;
    iterCount = 0;
}

// Seting up Image
void Interface::setImageAndWinName( const Mat& _image, const string& _winName  )
{
    if( _image.empty() || _winName.empty() )
        return;
    image = &_image;
    winName = &_winName;
    mask.create( image->size(), CV_8UC1);
    reset();
}

void Interface::showImage() const
{
    if( image->empty() || winName->empty() )
        return;

    Mat res;
    Mat binMask;
    if( !isInitialized )
        image->copyTo( res );               // Creating copy of Image to work with
    else
    {
        getBinMask( mask, binMask );
        image->copyTo( res, binMask );
    }

    // Selecting coloured Pixel for selected mouse events
    // e.g blue for background aarea selection
    vector<Point>::const_iterator it;
    for( it = BG_PIXEL.begin(); it != BG_PIXEL.end(); ++it )
        circle( res, *it, radius, BLUE, thickness );
    for( it = FG_PIXEL.begin(); it != FG_PIXEL.end(); ++it )
        circle( res, *it, radius, RED, thickness );
    for( it = prBG_PIXEL.begin(); it != prBG_PIXEL.end(); ++it )
        circle( res, *it, radius, LIGHTBLUE, thickness );
    for( it = prFG_PIXEL.begin(); it != prFG_PIXEL.end(); ++it )
        circle( res, *it, radius, PINK, thickness );

    if( rectState == IN_PROCESS || rectState == SET )
        rectangle( res, Point( rect.x, rect.y ), Point(rect.x + rect.width, rect.y + rect.height ), GREEN, 2);

    imshow( *winName, res );
}

void Interface::setRectInMask()
{
    CV_Assert( !mask.empty() );
    mask.setTo( GC_BGD );
    rect.x = max(0, rect.x);
    rect.y = max(0, rect.y);
    rect.width = min(rect.width, image->cols-rect.x);
    rect.height = min(rect.height, image->rows-rect.y);
    (mask(rect)).setTo( Scalar(GC_PR_FGD) );
}

// Applying grabcut to selected areas by keys
void Interface::setLblsInMask( int flags, Point p, bool isPr )
{
    vector<Point> *bpxls, *fpxls;
    uchar bvalue, fvalue;
    if( !isPr )
    {
        bpxls = &BG_PIXEL;
        fpxls = &FG_PIXEL;
        bvalue = GC_BGD;
        fvalue = GC_FGD;
    }
    else
    {
        bpxls = &prBG_PIXEL;
        fpxls = &prFG_PIXEL;
        bvalue = GC_PR_BGD;
        fvalue = GC_PR_FGD;
    }

    // When area is sure background or foreground and we have to select background
    // i.e sure area is selected by variable lblsState and background by value 1 and 3
    if( flags & (value == 1 || value ==3) )
    {
        bpxls->push_back(p);
        circle( mask, p, radius, bvalue, thickness );
    }
    if( flags & (value == 2 || value ==4)  )
    {
        fpxls->push_back(p);
        circle( mask, p, radius, fvalue, thickness );
    }
}

void Interface::mouseClick( int event, int x, int y, int flags, void* )
{
    //Set labels for events and then push labels(IN_PROCESS) to mask
    switch( event )
    {
    case EVENT_LBUTTONDOWN:                          // set rect or GC_BGD(GC_FGD) labels
        {

            if( rectState == NOT_SET )
            {
                rectState = IN_PROCESS;
                rect = Rect( x, y, 1, 1 );
            }
            if ( (value == 1 || value ==2) && rectState == SET ) // If it is sure area
                lblsState = IN_PROCESS;
            if ( (value == 3 || value ==4) && rectState == SET ) // Probable prFGD/prBGD
                prLblsState = IN_PROCESS;
        }
        break;

    case EVENT_LBUTTONUP:
        if( rectState == IN_PROCESS )
        {
            rect = Rect( Point(rect.x, rect.y), Point(x,y) );
            rectState = SET;
            setRectInMask();
            CV_Assert( BG_PIXEL.empty() && FG_PIXEL.empty() && prBG_PIXEL.empty() && prFG_PIXEL.empty() );
            showImage();
        }
        if( lblsState == IN_PROCESS )
        {
            setLblsInMask(flags, Point(x,y), false);
            lblsState = SET;
            showImage();
        }
        if( prLblsState == IN_PROCESS )
        {
            setLblsInMask(flags, Point(x,y), true);
            prLblsState = SET;
            showImage();
        }
        break;

    case EVENT_MOUSEMOVE:
        if( rectState == IN_PROCESS )
        {
            rect = Rect( Point(rect.x, rect.y), Point(x,y) );
            CV_Assert( BG_PIXEL.empty() && FG_PIXEL.empty() && prBG_PIXEL.empty() && prFG_PIXEL.empty() );
            showImage();
        }
        else if( lblsState == IN_PROCESS )
        {
            setLblsInMask(flags, Point(x,y), false);
            showImage();
        }
        else if( prLblsState == IN_PROCESS )
        {
            setLblsInMask(flags, Point(x,y), true);
            showImage();
        }
        break;
    }
}

// Grabing image on each updation of foreground,background Labels ie (lblsState) or probable ...(prlblsState)
int Interface::nextIter()
{
    if( isInitialized )
        grabCut( *image, mask, rect, bgdModel, fgdModel, 1 );
    else
    {
        if( rectState != SET )   // Rectangle State not defined
            return iterCount;

        if( lblsState == SET || prLblsState == SET )
            grabCut( *image, mask, rect, bgdModel, fgdModel, 1, GC_INIT_WITH_MASK );
        else
            grabCut( *image, mask, rect, bgdModel, fgdModel, 1, GC_INIT_WITH_RECT );

        isInitialized = true;
    }
    iterCount++;

    BG_PIXEL.clear(); FG_PIXEL.clear();
    prBG_PIXEL.clear(); prFG_PIXEL.clear();

    return iterCount;
}

Interface app;

static void on_mouse( int event, int x, int y, int flags, void* param )
{
    app.mouseClick( event, x, y, flags, param );
}

int main( int argc, char** argv )
{
    // Reading file from command line

    cv::CommandLineParser parser(argc, argv, "{help h||}{@input||}");
    if (parser.has("help"))
    {
        help();
        return 0;
    }
    string filename = parser.get<string>("@input");

    if( filename.empty() )
    {

        help();
        cout << "\n >Empty filename \n" "Call Program by :"
            "\t./grabcut <image_name>" <<endl;
        return 0;
    }

    Mat image = imread( filename, 1 );
    if( image.empty() )
    {
        cout << "\n >Couldn't read image filename " << filename << endl;
        return 1;
    }

    help();

    const string winName = "image";
    namedWindow(winName, WINDOW_AUTOSIZE );
    setMouseCallback( winName, on_mouse, 0 );
    app.setImageAndWinName( image, winName );
    app.showImage();

    // Key bindings
    for(;;)
    {
        char key = (char)waitKey(0);
        switch( key )
        {case '0':                                   //BGD
            value = 1;
            cout << " Using BLUE color,  >mark background regions with left mouse button \n"<<endl;
            break;
        case '1':                                   //FGD
            value = 2;
            cout << " Using RED color, >mark foreground regions with left mouse button \n" <<endl;
            break;
        case '2':                                   //prBGD
            value = 3;
            cout << " Using LIGHTBLUE color, >mark probable Background regions with left mouse button \n" <<endl;
            break;
        case '3':                                   //prFGD
            value = 4;
            cout << " Using PINK color, >mark probable foreground regions with left mouse button \n" <<endl;
            break;
        case '\x1b':                                // Exit
            cout << "Exiting ..." << endl;
            goto exit_main;
        case 'r':                                   //Reset
            cout << "resetting \n" << endl;
            app.reset();
            app.showImage();
            break;

        case 'n':
            int iterCount = app.getIterCount();
            cout << iterCount+1 << "> For finer touchups,  mark foreground and background after pressing keys 0-3 and again press n \n" << endl;
            int newIterCount = app.nextIter();
            if( newIterCount > iterCount )          // Counting Iteration for changes
            {
                app.showImage();
            }
            else
                cout << "rect must be determined>" << endl;
            break;
        }
    }

exit_main:
    destroyWindow( winName );
    return 0;
}
