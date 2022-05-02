#include "Tracking.h"

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>
#include <unistd.h>
#include"ORBmatcher.h"
#include"FrameDrawer.h"
#include"Converter.h"
#include"Initializer.h"
#include"G2oTypes.h"
#include"Optimizer.h"
#include"PnPsolver.h"

#include<iostream>

#include<mutex>
#include<chrono>
#include <include/CameraModels/Pinhole.h>
#include <include/CameraModels/KannalaBrandt8.h>
#include <include/MLPnPsolver.h>


using namespace std;

namespace ORB_SLAM3
{

//构造Tracking线程,在构造函数的时候初始化一些参数
/********************************************************
    pSys:Examples主函数中构造的system类指针
    pVoc：词袋指针（ORBVocabulary类指针）
    pFrameDrawer：显示窗口 FrameDrawer类指针
    pMapDrawer ：显示窗口 MapDrawer类指针
    pKFDB ：关键帧数据库类指针
    strSettingPath ：.yaml文件路径
    sensor：传感器类型 = 3
    _nameSeq:空string
********************************************************/
/*******初始化*************************************
    mState：当前tracking状态初始化为NO_IMAGES_YET（没有图像传入）
    mSensor = 3 单目惯性
    mTrackedFr：跟踪的图像帧数量 初始化为0
    mbStep:初始化为false
    mbOnlyTracking：是否只进行跟踪不进行局部建图，初始化为false，表示既进行建图又进行跟踪
    mbMapUpdated：
    mbVO:初始化为false
    mpORBVocabulary：词袋指针
    mpKeyFrameDB：关键帧数据集指针
    mpInitializer：初始化指针
    mpSystem：System指针
    mnLastRelocFrameId:上一个重定位帧的ID 初始化为0
    time_recently_lost：初始化为5
    mnInitialFrameId：初始图像帧ID，初始化为0
    mbCreatedMap:初始化为false，表明当前地图未被构造
    mnFirstFrameId：第一个图像帧ID，初始化为0
********************************************************/
Tracking::Tracking(System *pSys, ORBVocabulary* pVoc, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Atlas *pAtlas, KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor, const string &_nameSeq):
    mState(NO_IMAGES_YET), mSensor(sensor), mTrackedFr(0), mbStep(false),
    mbOnlyTracking(false), mbMapUpdated(false), mbVO(false), mpORBVocabulary(pVoc),
    mpKeyFrameDB(pKFDB),mpInitializer(static_cast<Initializer*>(NULL)), 
    mpSystem(pSys), mpViewer(NULL),mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer),
    mpAtlas(pAtlas), mnLastRelocFrameId(0), time_recently_lost(5.0),
    mnInitialFrameId(0), mbCreatedMap(false), mnFirstFrameId(0), mpCamera2(nullptr)
{
    // Load camera parameters from settings file
    //加载标定的一些参数，类型：只读
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

    //相机畸变参数
    cv::Mat DistCoef = cv::Mat::zeros(4,1,CV_32F);

    //相机类型
    string sCameraName = fSettings["Camera.type"];
    //EUROC是针孔相机
    if(sCameraName == "PinHole")
    {
        float fx = fSettings["Camera.fx"];
        float fy = fSettings["Camera.fy"];
        float cx = fSettings["Camera.cx"];
        float cy = fSettings["Camera.cy"];

        vector<float> vCamCalib{fx,fy,cx,cy};//相机信息

        mpCamera = new Pinhole(vCamCalib);//相机模型

        mpAtlas->AddCamera(mpCamera);

        DistCoef.at<float>(0) = fSettings["Camera.k1"];
        DistCoef.at<float>(1) = fSettings["Camera.k2"];
        DistCoef.at<float>(2) = fSettings["Camera.p1"]; 
        DistCoef.at<float>(3) = fSettings["Camera.p2"];
    }
    if(sCameraName == "KannalaBrandt8")
    {
        float fx = fSettings["Camera.fx"];
        float fy = fSettings["Camera.fy"];
        float cx = fSettings["Camera.cx"];
        float cy = fSettings["Camera.cy"];

        float K1 = fSettings["Camera.k1"];
        float K2 = fSettings["Camera.k2"];
        float K3 = fSettings["Camera.k3"];
        float K4 = fSettings["Camera.k4"];

        vector<float> vCamCalib{fx,fy,cx,cy,K1,K2,K3,K4};

        mpCamera = new KannalaBrandt8(vCamCalib);

        mpAtlas->AddCamera(mpCamera);

        if(sensor==System::STEREO || sensor==System::IMU_STEREO){
            //Right camera
            fx = fSettings["Camera2.fx"];
            fy = fSettings["Camera2.fy"];
            cx = fSettings["Camera2.cx"];
            cy = fSettings["Camera2.cy"];

            K1 = fSettings["Camera2.k1"];
            K2 = fSettings["Camera2.k2"];
            K3 = fSettings["Camera2.k3"];
            K4 = fSettings["Camera2.k4"];

            cout << endl << "Camera2 Parameters: " << endl;
            cout << "- fx: " << fx << endl;
            cout << "- fy: " << fy << endl;
            cout << "- cx: " << cx << endl;
            cout << "- cy: " << cy << endl;

            vector<float> vCamCalib2{fx,fy,cx,cy,K1,K2,K3,K4};

            mpCamera2 = new KannalaBrandt8(vCamCalib2);

            mpAtlas->AddCamera(mpCamera2);

            int leftLappingBegin = fSettings["Camera.lappingBegin"];
            int leftLappingEnd = fSettings["Camera.lappingEnd"];

            int rightLappingBegin = fSettings["Camera2.lappingBegin"];
            int rightLappingEnd = fSettings["Camera2.lappingEnd"];

            static_cast<KannalaBrandt8*>(mpCamera)->mvLappingArea[0] = leftLappingBegin;
            static_cast<KannalaBrandt8*>(mpCamera)->mvLappingArea[1] = leftLappingEnd;

            static_cast<KannalaBrandt8*>(mpCamera2)->mvLappingArea[0] = rightLappingBegin;
            static_cast<KannalaBrandt8*>(mpCamera2)->mvLappingArea[1] = rightLappingEnd;

            fSettings["Tlr"] >> mTlr;
            cout << "- mTlr: \n" << mTlr << endl;
            mpFrameDrawer->both = true;
        }
    }

    float fx = fSettings["Camera.fx"];//焦距
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];//光心
    float cy = fSettings["Camera.cy"];

    //构造内参矩阵
    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    const float k3 = fSettings["Camera.k3"];

    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    //畸变矩阵
    DistCoef.copyTo(mDistCoef);
    //基线
    mbf = fSettings["Camera.bf"];
    //帧速
    float fps = fSettings["Camera.fps"];
    if(fps==0)
        fps=30;

    // Max/Min Frames to insert keyframes and to check relocalisation
    //限制 关键帧中最少有0帧，最多30帧
    mMinFrames = 0;
    mMaxFrames = fps;

    cout << endl << "Camera Parameters: " << endl;
    cout << "- fx: " << fx << " ";
    cout << "- fy: " << fy << " ";
    cout << "- cx: " << cx << " ";
    cout << "- cy: " << cy << endl;
    cout << "- bf: " << mbf << " ";
    cout << "- k1: " << DistCoef.at<float>(0) << " ";
    cout << "- k2: " << DistCoef.at<float>(1) << endl;


    cout << "- p1: " << DistCoef.at<float>(2) << " ";
    cout << "- p2: " << DistCoef.at<float>(3) << endl;

    if(DistCoef.rows==5)
        cout << "- k3: " << DistCoef.at<float>(4) << endl;



    cout << "- fps: " << fps << " ";


    int nRGB = fSettings["Camera.RGB"];
    mbRGB = nRGB; //色彩类习惯

    if(mbRGB)
        cout << "- color order: RGB " << endl;
    else
        cout << "- color order: BGR (ignored if grayscale)" << endl;

    // Load ORB parameters
    //ORB参数
    int nFeatures = fSettings["ORBextractor.nFeatures"]; //特征个数
    float fScaleFactor = fSettings["ORBextractor.scaleFactor"];//相邻层的尺度因子
    int nLevels = fSettings["ORBextractor.nLevels"];//金字塔层数
    int fIniThFAST = fSettings["ORBextractor.iniThFAST"]; //提取FAST角点的初始阈值
    int fMinThFAST = fSettings["ORBextractor.minThFAST"]; //更小的阈值

    //构建ORB特征提取器
    //输入参数：ORB特征的参数.yaml文件
    mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(sensor==System::STEREO || sensor==System::IMU_STEREO)
        mpORBextractorRight = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    //对于单目IMU 特征点数量加倍,初始ORB特征提取器
    if(sensor==System::MONOCULAR || sensor==System::IMU_MONOCULAR)
        mpIniORBextractor = new ORBextractor(5*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    initID = 0; lastID = 0;

    cout << endl << "ORB Extractor Parameters: " << endl;
    cout << "- Number of Features: " << nFeatures << endl;
    cout << "- Scale Levels: " << nLevels << endl;
    cout << "- Scale Factor: " << fScaleFactor << endl;
    cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
    cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

    if(sensor==System::STEREO || sensor==System::RGBD || sensor==System::IMU_STEREO)
    {
        mThDepth = mbf*(float)fSettings["ThDepth"]/fx;
        cout << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
    }

    if(sensor==System::RGBD)
    {
        mDepthMapFactor = fSettings["DepthMapFactor"];
        if(fabs(mDepthMapFactor)<1e-5)
            mDepthMapFactor=1;
        else
            mDepthMapFactor = 1.0f/mDepthMapFactor;
    }

    //输出标定信息
    if(sensor==System::IMU_MONOCULAR || sensor==System::IMU_STEREO)
    {
        cv::Mat Tbc;
        //Tbc:相机到IMU的变换矩阵
        fSettings["Tbc"] >> Tbc;
        cout << endl;

        cout << "Left camera to Imu Transform (Tbc): " << endl << Tbc << endl;

        float freq, Ng, Na, Ngw, Naw;
        fSettings["IMU.Frequency"] >> freq;
        fSettings["IMU.NoiseGyro"] >> Ng;
        fSettings["IMU.NoiseAcc"] >> Na;
        fSettings["IMU.GyroWalk"] >> Ngw;
        fSettings["IMU.AccWalk"] >> Naw;

        const float sf = sqrt(freq);
        cout << endl;
        cout << "IMU frequency: " << freq << " Hz" << endl;
        cout << "IMU gyro noise: " << Ng << " rad/s/sqrt(Hz)" << endl;
        cout << "IMU gyro walk: " << Ngw << " rad/s^2/sqrt(Hz)" << endl;
        cout << "IMU accelerometer noise: " << Na << " m/s^2/sqrt(Hz)" << endl;
        cout << "IMU accelerometer walk: " << Naw << " m/s^3/sqrt(Hz)" << endl;
        cout<<"=========================================================="<<endl<<endl;
        //根据标定的参数，计算噪声协方差和游走协方差
        mpImuCalib = new IMU::Calib(Tbc,Ng*sf,Na*sf,Ngw/sf,Naw/sf);

        //根据偏置和标定进行新的预积分
        //IMU：：Bias()构造加速度和陀螺仪偏置均为0
        mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(),*mpImuCalib);

        mnFramesToResetIMU = mMaxFrames;
    }

    //所有代码中都没使用这个参数
    mbInitWith3KFs = false;


    //Test Images
    if((mSensor == System::STEREO || mSensor == System::IMU_STEREO) && sCameraName == "PinHole")
    {
        cv::Mat K_l, K_r, P_l, P_r, R_l, R_r, D_l, D_r;
        fSettings["LEFT.K"] >> K_l;
        fSettings["RIGHT.K"] >> K_r;

        fSettings["LEFT.P"] >> P_l;
        fSettings["RIGHT.P"] >> P_r;

        fSettings["LEFT.R"] >> R_l;
        fSettings["RIGHT.R"] >> R_r;

        fSettings["LEFT.D"] >> D_l;
        fSettings["RIGHT.D"] >> D_r;

        int rows_l = fSettings["LEFT.height"];
        int cols_l = fSettings["LEFT.width"];
        int rows_r = fSettings["RIGHT.height"];
        int cols_r = fSettings["RIGHT.width"];

        // M1r y M2r son los outputs (igual para l)
        // M1r y M2r son las matrices relativas al mapeo correspondiente a la rectificación de mapa en el eje X e Y respectivamente
        cv::initUndistortRectifyMap(K_l,D_l,R_l,P_l.rowRange(0,3).colRange(0,3),cv::Size(cols_l,rows_l),CV_32F,M1l,M2l);
        cv::initUndistortRectifyMap(K_r,D_r,R_r,P_r.rowRange(0,3).colRange(0,3),cv::Size(cols_r,rows_r),CV_32F,M1r,M2r);
    }
    else
    {
        int cols = 752; //相机width
        int rows = 480;
        cv::Mat R_l = cv::Mat::eye(3, 3, CV_32F);
    }

    mnNumDataset = 0;

    //f_track_stats.open("tracking_stats"+ _nameSeq + ".txt");
    /*f_track_stats.open("tracking_stats.txt");
    f_track_stats << "# timestamp, Num KF local, Num MP local, time" << endl;
    f_track_stats << fixed ;*/

#ifdef SAVE_TIMES
    f_track_times.open("tracking_times.txt");
    f_track_times << "# ORB_Ext(ms), Stereo matching(ms), Preintegrate_IMU(ms), Pose pred(ms), LocalMap_track(ms), NewKF_dec(ms)" << endl;
    f_track_times << fixed ;
#endif
}

Tracking::~Tracking()
{
    //f_track_stats.close();
#ifdef SAVE_TIMES
    f_track_times.close();
#endif
}

void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}

void Tracking::SetLoopClosing(LoopClosing *pLoopClosing)
{
    mpLoopClosing=pLoopClosing;
}

void Tracking::SetViewer(Viewer *pViewer)
{
    mpViewer=pViewer;
}

void Tracking::SetStepByStep(bool bSet)
{
    bStepByStep = bSet;
}



cv::Mat Tracking::GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp, string filename)
{
    mImGray = imRectLeft;
    cv::Mat imGrayRight = imRectRight;
    mImRight = imRectRight;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGB2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGR2GRAY);
        }
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGBA2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGRA2GRAY);
        }
    }

    if (mSensor == System::STEREO && !mpCamera2)
        mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera);
    else if(mSensor == System::STEREO && mpCamera2)
        mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera,mpCamera2,mTlr);
    else if(mSensor == System::IMU_STEREO && !mpCamera2)
        mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera,&mLastFrame,*mpImuCalib);
    else if(mSensor == System::IMU_STEREO && mpCamera2)
        mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera,mpCamera2,mTlr,&mLastFrame,*mpImuCalib);

    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    mCurrentFrame.mNameFile = filename;
    mCurrentFrame.mnDataset = mnNumDataset;

    Track();

    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    double t_track = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t1 - t0).count();

    /*cout << "trracking time: " << t_track << endl;
    f_track_stats << setprecision(0) << mCurrentFrame.mTimeStamp*1e9 << ",";
    f_track_stats << mvpLocalKeyFrames.size() << ",";
    f_track_stats << mvpLocalMapPoints.size() << ",";
    f_track_stats << setprecision(6) << t_track << endl;*/

#ifdef SAVE_TIMES
    f_track_times << mCurrentFrame.mTimeORB_Ext << ",";
    f_track_times << mCurrentFrame.mTimeStereoMatch << ",";
    f_track_times << mTime_PreIntIMU << ",";
    f_track_times << mTime_PosePred << ",";
    f_track_times << mTime_LocalMapTrack << ",";
    f_track_times << mTime_NewKF_Dec << ",";
    f_track_times << t_track << endl;
#endif

    return mCurrentFrame.mTcw.clone();
}


cv::Mat Tracking::GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp, string filename)
{  cout<<"RGB-D "<<endl;
    mImGray = imRGB;
    cv::Mat imDepth = imD;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }

    if((fabs(mDepthMapFactor-1.0f)>1e-5) || imDepth.type()!=CV_32F)
        imDepth.convertTo(imDepth,CV_32F,mDepthMapFactor);

    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    mCurrentFrame = Frame(mImGray,imDepth,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

    mCurrentFrame.mNameFile = filename;
    mCurrentFrame.mnDataset = mnNumDataset;


    Track();

    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    double t_track = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t1 - t0).count();

    /*f_track_stats << setprecision(0) << mCurrentFrame.mTimeStamp*1e9 << ",";
    f_track_stats << mvpLocalKeyFrames.size() << ",";
    f_track_stats << mvpLocalMapPoints.size() << ",";
    f_track_stats << setprecision(6) << t_track << endl;*/

#ifdef SAVE_TIMES
    f_track_times << mCurrentFrame.mTimeORB_Ext << ",";
    f_track_times << mCurrentFrame.mTimeStereoMatch << ",";
    f_track_times << mTime_PreIntIMU << ",";
    f_track_times << mTime_PosePred << ",";
    f_track_times << mTime_LocalMapTrack << ",";
    f_track_times << mTime_NewKF_Dec << ",";
    f_track_times << t_track << endl;
#endif

    return mCurrentFrame.mTcw.clone();
}

//利用图像和IMU求解位姿 
/*********************************
 * im：当前帧图像 
 * timestamp:当前图像帧的时间戳 
 * filename：空string
*********************************/
cv::Mat Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp, string filename)
{
    //im:原图，mImGray：灰度图
    mImGray = im;

    //第一步：将图像转为灰度图
    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }

    
    //第二步：构造当前图像帧Frame类
    if (mSensor == System::MONOCULAR)
    {
        if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET ||(lastID - initID) < mMaxFrames)
            mCurrentFrame = Frame(mImGray,timestamp,mpIniORBextractor,mpORBVocabulary,mpCamera,mDistCoef,mbf,mThDepth);
        else
            mCurrentFrame = Frame(mImGray,timestamp,mpORBextractorLeft,mpORBVocabulary,mpCamera,mDistCoef,mbf,mThDepth);
    }
    else if(mSensor == System::IMU_MONOCULAR)
    {
        /********************************************************************
         * 构建Tracking线程时，状态mSate初始化为NO_IMAGES_YET，表示还没有图像帧进入
         * mStete的NOT_INITIALIZED表示初始化还未成功
        *********************************************************************/
        if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET)
        {
            cout << "init extractor" << endl;
            /********************************************************************
                Frame类构造函数
                mImGray：当前图像帧的灰度图
                timestamp:当前图像帧的时间戳
                mpIniORBextractor：Tracking线程构造时创建的ORB特征初始提取器，属于ORBextractor类指针
                mpORBVocabulary:System.cc中构建传入Tracking线程中的ORB词袋类指针
                mpCamera：Tracking线程构造时创建，包括内参矩阵，fx,fy,cx,cy
                mDistCoef：畸变矩阵，Tracking线程构造时创建
                mbf:基线
                mThDepth：双目中使用，此处为空
                mLastFrame：上一图像帧，Frame类，构造函数中需要传入Frame类指针，因此此处为&mLastFrame
                mpImuCalib：IMU标定参数，本身的性质是IMU::Calib类指针，
                            但Frame构造函数中是IMU::Calib类，因此使用*mpImuCalib
            *******************************************************************/
            mCurrentFrame = Frame(mImGray,timestamp,mpIniORBextractor,mpORBVocabulary,mpCamera,mDistCoef,mbf,mThDepth,&mLastFrame,*mpImuCalib);
            /*********************************************
                Frame构造函数实现了对输入图像帧的ORB特征点提取、
                描述子的计算，并将特征点分配至网格以加快匹配速度，
                同时将当前图像帧的速度初始化为上一图像帧的速度
             ********************************************/
        }
        else //其他状态：OK，RECENTLY_LOST等
        {
            /********************************************************************
                mpORBextractorLeft：属于ORBextractor类指针
                mLastFrame：上一图像帧，Frame类
            *******************************************************************/
            mCurrentFrame = Frame(mImGray,timestamp,mpORBextractorLeft,mpORBVocabulary,mpCamera,mDistCoef,mbf,mThDepth,&mLastFrame,*mpImuCalib);
            /*********************************************
                Frame构造函数实现了对输入图像帧的ORB特征点提取、
                描述子的计算，并将特征点分配至网格以加快匹配速度，
                同时将当前图像帧的速度初始化为上一图像帧的速度
            ********************************************/
        }
    }
    

    //t0存储未初始化时的第1帧图像时间戳
    if (mState==NO_IMAGES_YET)
        t0=timestamp;

    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    cout<<" image mTimeStamp"<<mCurrentFrame.mTimeStamp-(1.341800e+09)<<endl;
    mCurrentFrame.mNameFile = filename; //为空
    mCurrentFrame.mnDataset = mnNumDataset;  //为空

    //当前图像帧的ID赋值上一个ID,初始帧Id为0
    lastID = mCurrentFrame.mnId; 

    //第三步：跟踪
    Track();

    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    double t_track = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t1 - t0).count();

    /*f_track_stats << setprecision(0) << mCurrentFrame.mTimeStamp*1e9 << ",";
    f_track_stats << mvpLocalKeyFrames.size() << ",";
    f_track_stats << mvpLocalMapPoints.size() << ",";
    f_track_stats << setprecision(6) << t_track << endl;*/

#ifdef SAVE_TIMES
    f_track_times << mCurrentFrame.mTimeORB_Ext << ",";
    f_track_times << mCurrentFrame.mTimeStereoMatch << ",";
    f_track_times << mTime_PreIntIMU << ",";
    f_track_times << mTime_PosePred << ",";
    f_track_times << mTime_LocalMapTrack << ",";
    f_track_times << mTime_NewKF_Dec << ",";
    f_track_times << t_track << endl;
#endif

    return mCurrentFrame.mTcw.clone();
}

//从传感器数据中加载IMU测量量，放入IMU序列
void Tracking::GrabImuData(const IMU::Point &imuMeasurement)
{
    /*********************************
        imuMeasurement：ORB_SLAM3::IMU::Point类
        cv::Point3f类型的加速度a
        cv::Point3f类型的角速度w
        double 类型的时间戳t
    ***********************************/
    unique_lock<mutex> lock(mMutexImuQueue);
    //mlQueueImuData中存储了两帧之间所有的IMU源数据,std::list
    /***************************************************************************
     上一帧图像的时间戳 < IMU测量量的时间戳 <= 当前图像的时间戳 
    ****************************************************************************/
    mlQueueImuData.push_back(imuMeasurement);
}


// 对IMU进行预积分
void Tracking::PreintegrateIMU()
{
    /***********************输入的IMU*************************
        上一帧图像的时间戳 < IMU测量量的时间戳 <= 当前图像的时间戳
    ********************************************************/


    //若当前图像帧不存在上一帧，即当前图像是第一帧图像
    //上一帧不存在,说明两帧之间没有imu数据，不进行预积分
    //mbImuPreintegrated标志位设置为真
    if(!mCurrentFrame.mpPrevFrame)
    {
        Verbose::PrintMess("non prev frame ", Verbose::VERBOSITY_NORMAL);
        mCurrentFrame.setIntegrated(); //设置mbImuPreintegrated = true;
        return; //返回
    }


    //mvImuFromLastFrame存储实际用于MU预积分求解的IMU测量量
    mvImuFromLastFrame.clear(); //来自上一帧的IMU
    //mvImuFromLastFrame向量的大小为当前帧和上一帧之间的IMU数据量
    mvImuFromLastFrame.reserve(mlQueueImuData.size()); 
    if(mlQueueImuData.size() == 0)
    {
        Verbose::PrintMess("Not IMU data in mlQueueImuData!!", Verbose::VERBOSITY_NORMAL);
        mCurrentFrame.setIntegrated();//设置mbImuPreintegrated = true;
        return;
    }

    //IMU数据传递
    int N=0;  //存储有效IMU个数
    while(true)
    {   
        // 数据还没有时,会等待一段时间,直到mlQueueImuData中有imu数据.一开始不需要等待
        //初始化bSleep为false
        bool bSleep = false;
        {
            unique_lock<mutex> lock(mMutexImuQueue);
            //IMU数据向量不为空
            if(!mlQueueImuData.empty())
            {
                IMU::Point* m = &mlQueueImuData.front();
                cout.precision(17);
                if(m->t<mCurrentFrame.mpPrevFrame->mTimeStamp-0.001l) //情况很少出现
                {
                    mlQueueImuData.pop_front();
                }
                else if(m->t<mCurrentFrame.mTimeStamp-0.001l)
                { N++;
                    mvImuFromLastFrame.push_back(*m);
                    mlQueueImuData.pop_front();
                }
                else
                {   N++;
                    mvImuFromLastFrame.push_back(*m);
                    break;
                }
            }
            else  //IMU数据向量为空，标识位为true，等待
            {
                break;
                bSleep = true;
            }
        }
        if(bSleep)
            usleep(500);
    }


    //构建中值预积分:偏置为上一帧
    const int n = mvImuFromLastFrame.size()-1;// m个imu组数据会有m-1个预积分量
    //构造IMU预积分处理器并初始化标定参数
    /**********输入**************
     * mLastFrame.mImuBias：上一图像帧的偏置
     * mCurrentFrame.mImuCalib：IMU标定参数
    **********************/
    IMU::Preintegrated* pImuPreintegratedFromLastFrame = new IMU::Preintegrated(mLastFrame.mImuBias,mCurrentFrame.mImuCalib);
    

    //遍历IMU数据并计算
    for(int i=0; i<n; i++)
    {
        float tstep;
        cv::Point3f acc, angVel;
        if((i==0) && (i<(n-1)))
        {
            //onenote详细分解
            float tab = mvImuFromLastFrame[i+1].t-mvImuFromLastFrame[i].t;
            float tini = mvImuFromLastFrame[i].t-mCurrentFrame.mpPrevFrame->mTimeStamp;
            acc = (mvImuFromLastFrame[i].a+mvImuFromLastFrame[i+1].a-
                    (mvImuFromLastFrame[i+1].a-mvImuFromLastFrame[i].a)*(tini/tab))*0.5f;
            angVel = (mvImuFromLastFrame[i].w+mvImuFromLastFrame[i+1].w-
                    (mvImuFromLastFrame[i+1].w-mvImuFromLastFrame[i].w)*(tini/tab))*0.5f;
            tstep = mvImuFromLastFrame[i+1].t-mCurrentFrame.mpPrevFrame->mTimeStamp;
        }
        else if(i<(n-1))
        {
            acc = (mvImuFromLastFrame[i].a+mvImuFromLastFrame[i+1].a)*0.5f;
            angVel = (mvImuFromLastFrame[i].w+mvImuFromLastFrame[i+1].w)*0.5f;
            tstep = mvImuFromLastFrame[i+1].t-mvImuFromLastFrame[i].t;
        }
        else if((i>0) && (i==(n-1)))
        {
            
            float tab = mvImuFromLastFrame[i+1].t-mvImuFromLastFrame[i].t;
            float tend = mvImuFromLastFrame[i+1].t-mCurrentFrame.mTimeStamp;
            
            acc = (mvImuFromLastFrame[i].a+mvImuFromLastFrame[i+1].a-
                    (mvImuFromLastFrame[i+1].a-mvImuFromLastFrame[i].a)*(tend/tab))*0.5f;
            angVel = (mvImuFromLastFrame[i].w+mvImuFromLastFrame[i+1].w-
                    (mvImuFromLastFrame[i+1].w-mvImuFromLastFrame[i].w)*(tend/tab))*0.5f;
            tstep = mCurrentFrame.mTimeStamp-mvImuFromLastFrame[i].t;
        }
        else if((i==0) && (i==(n-1)))
        {
            acc = mvImuFromLastFrame[i].a;
            angVel = mvImuFromLastFrame[i].w;
            tstep = mCurrentFrame.mTimeStamp-mCurrentFrame.mpPrevFrame->mTimeStamp;
        }
        
        //依次进行积分计算
        if (!mpImuPreintegratedFromLastKF)
            cout << "mpImuPreintegratedFromLastKF does not exist" << endl;
        //预积分计算
        //相对于上一个关键帧帧进行预积分
        mpImuPreintegratedFromLastKF->IntegrateNewMeasurement(acc,angVel,tstep); 
        //相对于上一个图像帧进行预积分
        pImuPreintegratedFromLastFrame->IntegrateNewMeasurement(acc,angVel,tstep);
    }
    
    //mCurrentFrame.mpImuPreintegratedFrame中存储上一图像帧到当前图像帧之间的预积分值
    mCurrentFrame.mpImuPreintegratedFrame = pImuPreintegratedFromLastFrame;
    //mCurrentFrame.mpImuPreintegrated中存储上一关键帧到当前关键帧之间的预积分值
    mCurrentFrame.mpImuPreintegrated = mpImuPreintegratedFromLastKF;
    mCurrentFrame.mpLastKeyFrame = mpLastKeyFrame;

    //设置为已预积分状态
    mCurrentFrame.setIntegrated();  //mbImuPreintegrated = true;


    Verbose::PrintMess("Preintegration is finished!! ", Verbose::VERBOSITY_DEBUG);
}

//利用IMU预测状态
/***********************************
 * 用途1：恒速模型，计算速度
 * 用途2：跟踪丢失时先进行IMU预测状态
 * *******************************/
bool Tracking::PredictStateIMU()
{   
    //如果当前图像帧不存在前一帧，直接返回
    if(!mCurrentFrame.mpPrevFrame)
    {
        Verbose::PrintMess("No last frame", Verbose::VERBOSITY_NORMAL);
        return false;
    }


    cout<<"地图更新"<<mbMapUpdated<<endl;
    if(mbMapUpdated && mpLastKeyFrame)
    {
        const cv::Mat twb1 = mpLastKeyFrame->GetImuPosition();
        const cv::Mat Rwb1 = mpLastKeyFrame->GetImuRotation();
        const cv::Mat Vwb1 = mpLastKeyFrame->GetVelocity();

        const cv::Mat Gz = (cv::Mat_<float>(3,1) << 0,0,-IMU::GRAVITY_VALUE);
        const float t12 = mpImuPreintegratedFromLastKF->dT;
        // cout<<"ori "<<mpImuPreintegratedFromLastKF->GetOriginalDeltaPosition().t()<<endl;
        // cout<<"BIA "<<mpImuPreintegratedFromLastKF->GetDeltaPosition(mpLastKeyFrame->GetImuBias()).t()<<endl;
        // sleep(1);
        cv::Mat Rwb2 = IMU::NormalizeRotation(Rwb1*mpImuPreintegratedFromLastKF->GetDeltaRotation(mpLastKeyFrame->GetImuBias()));
        cv::Mat twb2 = twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1*mpImuPreintegratedFromLastKF->GetDeltaPosition(mpLastKeyFrame->GetImuBias());
        cv::Mat Vwb2 = Vwb1 + t12*Gz + Rwb1*mpImuPreintegratedFromLastKF->GetDeltaVelocity(mpLastKeyFrame->GetImuBias());
        mCurrentFrame.SetImuPoseVelocity(Rwb2,twb2,Vwb2);
        mCurrentFrame.mPredRwb = Rwb2.clone();
        mCurrentFrame.mPredtwb = twb2.clone();
        mCurrentFrame.mPredVwb = Vwb2.clone();
        mCurrentFrame.mImuBias = mpLastKeyFrame->GetImuBias();
        mCurrentFrame.mPredBias = mCurrentFrame.mImuBias;
        return true;
    }
    else if(!mbMapUpdated)
    {
        const cv::Mat twb1 = mLastFrame.GetImuPosition();
        const cv::Mat Rwb1 = mLastFrame.GetImuRotation();
        const cv::Mat Vwb1 = mLastFrame.mVw;
        const cv::Mat Gz = (cv::Mat_<float>(3,1) << 0,0,-IMU::GRAVITY_VALUE);
        const float t12 = mCurrentFrame.mpImuPreintegratedFrame->dT;

        cv::Mat Rwb2 = IMU::NormalizeRotation(Rwb1*mCurrentFrame.mpImuPreintegratedFrame->GetDeltaRotation(mLastFrame.mImuBias));
        cv::Mat twb2 = twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1*mCurrentFrame.mpImuPreintegratedFrame->GetDeltaPosition(mLastFrame.mImuBias);
        cv::Mat Vwb2 = Vwb1 + t12*Gz + Rwb1*mCurrentFrame.mpImuPreintegratedFrame->GetDeltaVelocity(mLastFrame.mImuBias);

        mCurrentFrame.SetImuPoseVelocity(Rwb2,twb2,Vwb2);
        mCurrentFrame.mPredRwb = Rwb2.clone();
        mCurrentFrame.mPredtwb = twb2.clone();
        mCurrentFrame.mPredVwb = Vwb2.clone();
        mCurrentFrame.mImuBias =mLastFrame.mImuBias;
        mCurrentFrame.mPredBias = mCurrentFrame.mImuBias;
        return true;
    }
    else
        cout << "not IMU prediction!!" << endl;

    return false;
}


void Tracking::ComputeGyroBias(const vector<Frame*> &vpFs, float &bwx,  float &bwy, float &bwz)
{
    const int N = vpFs.size();
    vector<float> vbx;
    vbx.reserve(N);
    vector<float> vby;
    vby.reserve(N);
    vector<float> vbz;
    vbz.reserve(N);

    cv::Mat H = cv::Mat::zeros(3,3,CV_32F);
    cv::Mat grad  = cv::Mat::zeros(3,1,CV_32F);
    for(int i=1;i<N;i++)
    {
        Frame* pF2 = vpFs[i];
        Frame* pF1 = vpFs[i-1];
        cv::Mat VisionR = pF1->GetImuRotation().t()*pF2->GetImuRotation();
        cv::Mat JRg = pF2->mpImuPreintegratedFrame->JRg;
        cv::Mat E = pF2->mpImuPreintegratedFrame->GetUpdatedDeltaRotation().t()*VisionR;
        cv::Mat e = IMU::LogSO3(E);
        assert(fabs(pF2->mTimeStamp-pF1->mTimeStamp-pF2->mpImuPreintegratedFrame->dT)<0.01);

        cv::Mat J = -IMU::InverseRightJacobianSO3(e)*E.t()*JRg;
        grad += J.t()*e;
        H += J.t()*J;
    }

    cv::Mat bg = -H.inv(cv::DECOMP_SVD)*grad;
    bwx = bg.at<float>(0);
    bwy = bg.at<float>(1);
    bwz = bg.at<float>(2);

    for(int i=1;i<N;i++)
    {
        Frame* pF = vpFs[i];
        pF->mImuBias.bwx = bwx;
        pF->mImuBias.bwy = bwy;
        pF->mImuBias.bwz = bwz;
        pF->mpImuPreintegratedFrame->SetNewBias(pF->mImuBias);
        pF->mpImuPreintegratedFrame->Reintegrate();
    }
}

void Tracking::ComputeVelocitiesAccBias(const vector<Frame*> &vpFs, float &bax,  float &bay, float &baz)
{
    const int N = vpFs.size();
    const int nVar = 3*N +3; // 3 velocities/frame + acc bias
    const int nEqs = 6*(N-1);

    cv::Mat J(nEqs,nVar,CV_32F,cv::Scalar(0));
    cv::Mat e(nEqs,1,CV_32F,cv::Scalar(0));
    cv::Mat g = (cv::Mat_<float>(3,1)<<0,0,-IMU::GRAVITY_VALUE);

    for(int i=0;i<N-1;i++)
    {
        Frame* pF2 = vpFs[i+1];
        Frame* pF1 = vpFs[i];
        cv::Mat twb1 = pF1->GetImuPosition();
        cv::Mat twb2 = pF2->GetImuPosition();
        cv::Mat Rwb1 = pF1->GetImuRotation();
        cv::Mat dP12 = pF2->mpImuPreintegratedFrame->GetUpdatedDeltaPosition();
        cv::Mat dV12 = pF2->mpImuPreintegratedFrame->GetUpdatedDeltaVelocity();
        cv::Mat JP12 = pF2->mpImuPreintegratedFrame->JPa;
        cv::Mat JV12 = pF2->mpImuPreintegratedFrame->JVa;
        float t12 = pF2->mpImuPreintegratedFrame->dT;
        // Position p2=p1+v1*t+0.5*g*t^2+R1*dP12
        J.rowRange(6*i,6*i+3).colRange(3*i,3*i+3) += cv::Mat::eye(3,3,CV_32F)*t12;
        J.rowRange(6*i,6*i+3).colRange(3*N,3*N+3) += Rwb1*JP12;
        e.rowRange(6*i,6*i+3) = twb2-twb1-0.5f*g*t12*t12-Rwb1*dP12;
        // Velocity v2=v1+g*t+R1*dV12
        J.rowRange(6*i+3,6*i+6).colRange(3*i,3*i+3) += -cv::Mat::eye(3,3,CV_32F);
        J.rowRange(6*i+3,6*i+6).colRange(3*(i+1),3*(i+1)+3) += cv::Mat::eye(3,3,CV_32F);
        J.rowRange(6*i+3,6*i+6).colRange(3*N,3*N+3) -= Rwb1*JV12;
        e.rowRange(6*i+3,6*i+6) = g*t12+Rwb1*dV12;
    }

    cv::Mat H = J.t()*J;
    cv::Mat B = J.t()*e;
    cv::Mat x(nVar,1,CV_32F);
    cv::solve(H,B,x);

    bax = x.at<float>(3*N);
    bay = x.at<float>(3*N+1);
    baz = x.at<float>(3*N+2);

    for(int i=0;i<N;i++)
    {
        Frame* pF = vpFs[i];
        x.rowRange(3*i,3*i+3).copyTo(pF->mVw);
        if(i>0)
        {
            pF->mImuBias.bax = bax;
            pF->mImuBias.bay = bay;
            pF->mImuBias.baz = baz;
            pF->mpImuPreintegratedFrame->SetNewBias(pF->mImuBias);
        }
    }
}

void Tracking::ResetFrameIMU()
{
    // TODO To implement...
}

/*******************************************************
 * 跟踪过程，包括恒速模型跟踪、参考关键帧跟踪、局部地图跟踪
 * track包含两部分：估计运动、跟踪局部地图
 * Step 1：初始化
 * Step 2：跟踪
 * Step 3：记录位姿信息，用于轨迹复现
 *******************************************************/
void Tracking::Track()
{
#ifdef SAVE_TIMES
    mTime_PreIntIMU = 0;
    mTime_PosePred = 0;
    mTime_LocalMapTrack = 0;
    mTime_NewKF_Dec = 0;
#endif
    cout<<"-------------当前图像帧:"<<mCurrentFrame.mnId<<"-------------"<<endl;
    if (bStepByStep)
    {   
        while(!mbStep)//mbStep在Tracking创建时初始为false
            usleep(500);
        mbStep = false;
    }

    //如果LocalMapping线程中检测到IMU有问题，重置当前地图
    if(mpLocalMapper->mbBadImu)//构建LocalMapping线程时，已初始化为false
    {
        cout << "TRACK: Reset map because local mapper set the bad imu flag " << endl;
        mpSystem->ResetActiveMap();
        return;
    }

    // 从Atlas中取出当前active的地图
    Map* pCurrentMap = mpAtlas->GetCurrentMap();

    //如果有图像且不是第一个图像，操作IMU数据
    //处理时间戳异常情况
    if(mState!=NO_IMAGES_YET) 
    {   
        //上一图像帧的时间戳大于当前图像时间戳
        if(mLastFrame.mTimeStamp>mCurrentFrame.mTimeStamp)
        {   
            //出错了，清除imu数据，创建新的子地图
            cerr << "ERROR: Frame with a timestamp older than previous frame detected!" << endl;
            unique_lock<mutex> lock(mMutexImuQueue);
            mlQueueImuData.clear();
            CreateMapInAtlas();
            return;
        }
        //当前图像时间戳比上一图像时间戳大1s，说明时间戳明显跳变了，重置地图后直接返回
        else if(mCurrentFrame.mTimeStamp>mLastFrame.mTimeStamp+1.0)
        {   
            cout << "id last: " << mLastFrame.mnId << "    id curr: " << mCurrentFrame.mnId << endl;
            //根据是否是imu模式,进行imu的补偿
            if(mpAtlas->isInertial())
            {
                // 如果当前地图imu成功初始化
                if(mpAtlas->isImuInitialized())
                {
                    cout << "Timestamp jump detected. State set to LOST. Reseting IMU integration..." << endl;
                    // IMU完成第2阶段BA（在localmapping线程里）
                    if(!pCurrentMap->GetIniertialBA2())
                    {
                        // 如果当前子图中imu没有经过BA2，重置active地图
                        mpSystem->ResetActiveMap();
                    }
                    else
                    {
                        // 如果当前子图中imu进行了BA2，重新创建新的子图
                        CreateMapInAtlas();
                    }
                }
                else
                {
                    // 如果当前子图中imu还没有初始化，重置active地图
                    cout << "Timestamp jump detected, before IMU initialization. Reseting..." << endl;
                    mpSystem->ResetActiveMap();
                }
            }
            // 不跟踪直接返回
            return;
        }
    }



    //单目IMU或双目IMU且上一关键帧存在
    //在上个阶段，单目初始化中，若初始三角化成功，则mpLastKeyFrame=当前关键帧
    if ((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO) && mpLastKeyFrame)
    {
        mCurrentFrame.SetNewBias(mpLastKeyFrame->GetImuBias()); //设定新的偏置
    }

    //如果是第一张图像,设置当前状态为未初始化NOT_INITIALIZED
    if(mState==NO_IMAGES_YET)
    {
        mState = NOT_INITIALIZED;  //未初始化
    }

    //保存当前状态到上一个图像帧处理状态mLastProcessedState 
    mLastProcessedState=mState;

    //如果是单目IMU或双目IMU且还没创建地图的状态，对IMU数据进行预积分
    //CreateMapInAtlas函数中对mbCreatedMap状态进行改变
    // mbCreatedMap:初始化为false，表明当前地图未被构造
    if ((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO) && !mbCreatedMap)
    {
#ifdef SAVE_TIMES
        std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
#endif
        PreintegrateIMU(); //IMU预积分
#ifdef SAVE_TIMES
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

        mTime_PreIntIMU = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t1 - t0).count();
#endif


    }
    


    mbCreatedMap = false;//false表示未建图状态

    // Get Map Mutex -> Map cannot be changed
    unique_lock<mutex> lock(pCurrentMap->mMutexMapUpdate);

    mbMapUpdated = false;//false表示地图不更新


    // 判断地图id是否更新了
    int nCurMapChangeIndex = pCurrentMap->GetMapChangeIndex();
    int nMapChangeIndex = pCurrentMap->GetLastMapChange();
    if(nCurMapChangeIndex>nMapChangeIndex)
    {
        // cout << "Map update detected" << endl;
        // 检测到地图更新了
        pCurrentMap->SetLastMapChange(nCurMapChangeIndex);
        mbMapUpdated = true;
    }



    //非初始化状态（第一帧进入后已变为了未初始化状态）
    if(mState==NOT_INITIALIZED)
    {
        if(mSensor==System::STEREO || mSensor==System::RGBD || mSensor==System::IMU_STEREO)
            StereoInitialization();
        else
        {
            MonocularInitialization(); //单目初始化操作
        }

        mpFrameDrawer->Update(this);

        //如果状态不是OK，当前帧赋给上一帧,直接返回
        if(mState!=OK) // If rightly initialized, mState=OK
        {
            mLastFrame = Frame(mCurrentFrame);
            return;
        }

        //若当前地图是第一个地图，第一帧的ID为当前帧的ID
        if(mpAtlas->GetAllMaps().size() == 1)
        {
            mnFirstFrameId = mCurrentFrame.mnId;
        }
    }
    else //初始化成功，正式进入跟踪过程
    {
        // System is initialized. Track Frame.
        bool bOK;

        // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
        //初始化Tracking线程时，mbOnlyTracking被设为了false
        //mbOnlyTracking为false时表示既跟踪，又建图，为true时表示只跟踪不建图
        //在viewer中有个开关ActivateLocalizationMode，可以控制是否开启mbOnlyTracking
        if(!mbOnlyTracking)
        {
#ifdef SAVE_TIMES
        std::chrono::steady_clock::time_point timeStartPosePredict = std::chrono::steady_clock::now();
#endif

            // State OK
            // Local Mapping is activated. This is the normal behaviour, unless
            // you explicitly activate the "only tracking" mode.
            //若可以进行正常跟踪，即单目视觉初始化成功
            if(mState==OK)
            {

                // Local Mapping might have changed some MapPoints tracked in last frame
                //检查上一帧中被替换的地图点，将地图点替换成新的地图点
                //局部建图线程则可能会对原有的地图点进行替换.在这里进行检查
                CheckReplacedInLastFrame();

                //若运动模型是空的且IMU未初始化，或者刚完成重定位，则跟踪参考关键帧
                if((mVelocity.empty() && !pCurrentMap->isImuInitialized()) || mCurrentFrame.mnId<mnLastRelocFrameId+2)
                {   cout<<"跟踪参考关键帧"<<endl;
                    bOK = TrackReferenceKeyFrame();
                    /******************************************
                     * 如果参考关键帧中的地图点和当前图像帧中的特征点匹配成功数量少于15，bOK=false
                     * 在TrackReferenceKeyFrame中通过3D-2D重投影误差实现了对当前图像帧的位姿估计
                     * 并且剔除了误差过大的地图点
                     * bOK=true
                     * *****************************************/
                    // if(bOK)
                    // {
                    //     cout<<"TRUE"<<endl;
                    // }
                    // else
                    // {
                    //     cout<<"FALSE"<<endl;
                    // }
                }
                else
                {   cout<<"跟踪恒速模型"<<endl;
                    //Verbose::PrintMess("TRACK: Track with motion model", Verbose::VERBOSITY_DEBUG);
                    // 用恒速模型跟踪。
                    // 假设上上帧到上一帧的位姿=上一帧的位姿到当前帧位姿
                    // 根据恒速模型设定当前帧的初始位姿，用最近的普通帧来跟踪当前的普通帧
                    // 通过投影的方式在参考帧中找当前帧特征点的匹配点，优化每个特征点所对应3D点的投影误差即可得到位姿
                    bOK = TrackWithMotionModel();
                    if(!bOK)
                        bOK = TrackReferenceKeyFrame();
                }


                if (!bOK)
                {
                    if ( mCurrentFrame.mnId<=(mnLastRelocFrameId+mnFramesToResetIMU) &&
                         (mSensor==System::IMU_MONOCULAR || mSensor==System::IMU_STEREO))
                    {
                        mState = LOST;
                    }
                    else if(pCurrentMap->KeyFramesInMap()>10)
                    {
                        cout << "KF in map: " << pCurrentMap->KeyFramesInMap() << endl;
                        mState = RECENTLY_LOST;
                        mTimeStampLost = mCurrentFrame.mTimeStamp;
                        //mCurrentFrame.SetPose(mLastFrame.mTcw);
                    }
                    else
                    {
                        mState = LOST;
                    }
                }
            }
            
            //其他状态（跟踪不正常）
            else
            {
                //如果是近期丢失状态
                if (mState == RECENTLY_LOST)
                {
                    Verbose::PrintMess("Lost for a short time", Verbose::VERBOSITY_NORMAL);

                    //设置bOK为true
                    bOK = true;

                    //若是IMU模式，试图用IMU预测位姿
                    if((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO))
                    {   
                        //若IMU已初始化，用IMU数据预测位姿
                        if(pCurrentMap->isImuInitialized())
                            PredictStateIMU();
                        else
                            bOK = false;

                        // 如果IMU模式下当前帧距离跟丢帧超过5s还没有找回
                        if (mCurrentFrame.mTimeStamp-mTimeStampLost>time_recently_lost)
                        {
                            mState = LOST;
                            Verbose::PrintMess("Track Lost...", Verbose::VERBOSITY_NORMAL);
                            bOK=false;
                        }
                    }

                    else
                    {
                        // TODO fix relocalization
                        bOK = Relocalization();
                        if(!bOK)
                        {
                            mState = LOST;
                            Verbose::PrintMess("Track Lost...", Verbose::VERBOSITY_NORMAL);
                            bOK=false;
                        }
                    }
                }
                else if (mState == LOST)
                {

                    Verbose::PrintMess("A new map is started...", Verbose::VERBOSITY_NORMAL);

                    if (pCurrentMap->KeyFramesInMap()<10)
                    {
                        mpSystem->ResetActiveMap();
                        cout << "Reseting current map..." << endl;
                    }else
                        CreateMapInAtlas();

                    if(mpLastKeyFrame)
                        mpLastKeyFrame = static_cast<KeyFrame*>(NULL);

                    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

                    return;
                }
            
            }


#ifdef SAVE_TIMES
        std::chrono::steady_clock::time_point timeEndPosePredict = std::chrono::steady_clock::now();

        mTime_PosePred = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(timeEndPosePredict - timeStartPosePredict).count();
#endif

        }
        
        else
        {
            // Localization Mode: Local Mapping is deactivated (TODO Not available in inertial mode)
            if(mState==LOST)
            {
                if(mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO)
                    Verbose::PrintMess("IMU. State LOST", Verbose::VERBOSITY_NORMAL);
                bOK = Relocalization();
            }
            else
            {
                if(!mbVO)
                {
                    // In last frame we tracked enough MapPoints in the map
                    if(!mVelocity.empty())
                    {   
                        bOK = TrackWithMotionModel();
                    }
                    else
                    {   
                        bOK = TrackReferenceKeyFrame();
                    }
                }
                else
                {
                    // In last frame we tracked mainly "visual odometry" points.

                    // We compute two camera poses, one from motion model and one doing relocalization.
                    // If relocalization is sucessfull we choose that solution, otherwise we retain
                    // the "visual odometry" solution.

                    bool bOKMM = false;
                    bool bOKReloc = false;
                    vector<MapPoint*> vpMPsMM;
                    vector<bool> vbOutMM;
                    cv::Mat TcwMM;
                    if(!mVelocity.empty())
                    {
                        bOKMM = TrackWithMotionModel();
                        vpMPsMM = mCurrentFrame.mvpMapPoints;
                        vbOutMM = mCurrentFrame.mvbOutlier;
                        TcwMM = mCurrentFrame.mTcw.clone();
                    }
                    bOKReloc = Relocalization();

                    if(bOKMM && !bOKReloc)
                    {
                        mCurrentFrame.SetPose(TcwMM);
                        mCurrentFrame.mvpMapPoints = vpMPsMM;
                        mCurrentFrame.mvbOutlier = vbOutMM;

                        if(mbVO)
                        {
                            for(int i =0; i<mCurrentFrame.N; i++)
                            {
                                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                                {
                                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                                }
                            }
                        }
                    }
                    else if(bOKReloc)
                    {
                        mbVO = false;
                    }

                    bOK = bOKReloc || bOKMM;
                }
            }
        }




        if(!mCurrentFrame.mpReferenceKF)
            mCurrentFrame.mpReferenceKF = mpReferenceKF;

        // If we have an initial estimation of the camera pose and matching. Track the local map.
        //在跟踪得到当前帧初始姿态后，现在对local map进行跟踪得到更多的匹配，并优化当前位姿
        //上面是对两帧之间进行跟踪，得到初始位姿，这这部分进行更多的跟踪，优化位姿
        if(!mbOnlyTracking)  //若为定位+建图模式
        {   

            if(bOK)
            {
#ifdef SAVE_TIMES
                std::chrono::steady_clock::time_point time_StartTrackLocalMap = std::chrono::steady_clock::now();
#endif             
                //局部地图跟踪
                bOK = TrackLocalMap();
                /*********************************************************************
                判断局部地图是否跟踪成功
                第一种：当前帧的ID距离上次重定位帧的ID之间的时间太短，且当前跟踪匹配到的地图点少于50
                第二种：在情况一不满足的情况下，若当前跟踪匹配上的地图点少于30则认为跟踪失败                
                ********************************************************************/
#ifdef SAVE_TIMES
                std::chrono::steady_clock::time_point time_EndTrackLocalMap = std::chrono::steady_clock::now();

                mTime_LocalMapTrack = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndTrackLocalMap - time_StartTrackLocalMap).count();
#endif

            }
            if(!bOK)
            {
                //cout << "Fail to track local map!" << endl;
            }
        }
        else  //若是纯定位模式
        {
            // mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
            // a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
            // the camera we will use the local map again.
            
            /**********************************************************************
             * mbVO在tracking线程初始化阶段，便初始为false
             * mbVO为true表示，地图中的地图点存在少量的匹配，无法获得局部地图，因此要执行跟踪
             * 一旦系统重定位了相机，则再次使用局部地图
            **********************************************************************/
            if(bOK && !mbVO)
                bOK = TrackLocalMap();
        }

        //若跟踪成功
        if(bOK)
            mState = OK;  //设置当前的状态为OK
        else if (mState == OK)  //若状态已经是OK （当第一阶段跟踪成功，第二阶段局部地图跟踪失败时执行）
        {
            if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO)
            {
                Verbose::PrintMess("Track lost for less than one second...", Verbose::VERBOSITY_NORMAL);
                //若当前地图没有的IMU初始化未成功或当前地图的惯性BA优化未执行
                if(!pCurrentMap->isImuInitialized() || !pCurrentMap->GetIniertialBA2())
                {
                    //cout << "IMU is not or recently initialized. Reseting active map..." << endl;
                    mpSystem->ResetActiveMap();// mbResetActiveMap = true;
                    
                }

                mState=RECENTLY_LOST;
            }
            else
                mState=LOST; // visual to lost  //更改状态为丢失
            
            //如果当前图像帧距离上次重定位帧超过1s,用当前图像帧的时间戳更新lost时间戳
            if(mCurrentFrame.mnId>mnLastRelocFrameId+mMaxFrames)
            {
                mTimeStampLost = mCurrentFrame.mTimeStamp;
            }
        }

        // Save frame if recent relocalization, since they are used for IMU reset (as we are making copy, it shluld be once mCurrFrame is completely modified)
        //若刚刚发生重定位且IMU已初始化
        if((mCurrentFrame.mnId<(mnLastRelocFrameId+mnFramesToResetIMU))   //如果当前图像帧距离上次重定位图像帧小于1S
        && (mCurrentFrame.mnId > mnFramesToResetIMU)   //且当前图像帧运行超过1S
        && ((mSensor == System::IMU_MONOCULAR) || (mSensor == System::IMU_STEREO))   //IMU模式
        && pCurrentMap->isImuInitialized())  //IMU已初始化
        {   
            //保存上一图像帧和当前图像帧
            // TODO check this situation
            Verbose::PrintMess("Saving pointer to frame. imu needs reset...", Verbose::VERBOSITY_NORMAL);
            Frame* pF = new Frame(mCurrentFrame);
            pF->mpPrevFrame = new Frame(mLastFrame);

            //重置IMU
            // Load preintegration
            pF->mpImuPreintegratedFrame = new IMU::Preintegrated(mCurrentFrame.mpImuPreintegratedFrame);
        }


        //若当前地图已完成IMU初始化
        if(pCurrentMap->isImuInitialized())
        {
            if(bOK)
            {
                if(mCurrentFrame.mnId==(mnLastRelocFrameId+mnFramesToResetIMU))
                {
                    cout << "RESETING FRAME!!!" << endl;
                    ResetFrameIMU();
                }
                else if(mCurrentFrame.mnId>(mnLastRelocFrameId+30))
                    mLastBias = mCurrentFrame.mImuBias;
            }
        }


        // Update drawer
        //更新图像
        mpFrameDrawer->Update(this);
        //更新地图
        if(!mCurrentFrame.mTcw.empty())
            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);



        //若跟踪成功或状态为刚刚丢失
        if(bOK || mState==RECENTLY_LOST)
        {
            // Update motion model
            //若当前帧和上一帧的位姿不为空
            if(!mLastFrame.mTcw.empty() && !mCurrentFrame.mTcw.empty())
            {   
                //构建上一帧的姿态
                cv::Mat LastTwc = cv::Mat::eye(4,4,CV_32F);
                mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0,3).colRange(0,3));
                mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0,3).col(3));
                //更新运动模型
                mVelocity = mCurrentFrame.mTcw*LastTwc;
            }
            else
                mVelocity = cv::Mat();
            
            //若是IMU 单目，用IMU积分的位姿显示？
            if(mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO)
                mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

            // Clean VO matches
            //清除观测不到的地图点
            for(int i=0; i<mCurrentFrame.N; i++)
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(pMP)
                    if(pMP->Observations()<1)  //观测小于1,说明此地图点没有被观测到
                    {
                        mCurrentFrame.mvbOutlier[i] = false;  //设置此点对应的特征点为内点
                        mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);  //此地图点为空
                    }
            }

            // Delete temporal MapPoints
            //清除临时地图点
            for(list<MapPoint*>::iterator lit = mlpTemporalPoints.begin(), lend =  mlpTemporalPoints.end(); lit!=lend; lit++)
            {
                MapPoint* pMP = *lit;
                delete pMP;
            }
            mlpTemporalPoints.clear();

#ifdef SAVE_TIMES
            std::chrono::steady_clock::time_point timeStartNewKF = std::chrono::steady_clock::now();
#endif


            //判断是否需要添加关键帧 关键帧标志位
            bool bNeedKF = NeedNewKeyFrame();



#ifdef SAVE_TIMES
            std::chrono::steady_clock::time_point timeEndNewKF = std::chrono::steady_clock::now();

            mTime_NewKF_Dec = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(timeEndNewKF - timeStartNewKF).count();
#endif



            // Check if we need to insert a new keyframe
            //根据条件来判断是否插入关键帧
            // 需要同时满足下面条件1和2
            // 条件1：bNeedKF=true，需要插入关键帧
            // 条件2：bOK=true跟踪成功 或 IMU模式下的RECENTLY_LOST模式
            if(bNeedKF && (bOK|| (mState==RECENTLY_LOST && (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO))))
                CreateNewKeyFrame();

            // We allow points with high innovation (considererd outliers by the Huber Function)
            // pass to the new keyframe, so that bundle adjustment will finally decide
            // if they are outliers or not. We don't want next frame to estimate its position
            // with those points so we discard them in the frame. Only has effect if lastframe is tracked
            //删除在BA中被检测为外点的地图点
            for(int i=0; i<mCurrentFrame.N;i++)
            {
                if(mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                    mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
            }
        }

        // Reset if the camera get lost soon after initialization
        //若为Lost状态
        if(mState==LOST)
        {   
            //若地图中关键帧的个数小于5个，重置地图，退出跟踪
            if(pCurrentMap->KeyFramesInMap()<=5)
            {
                mpSystem->ResetActiveMap();
                return;
            }

            //若为IMU模式且IMU初始化未成功
            if ((mSensor == System::IMU_MONOCULAR) || (mSensor == System::IMU_STEREO))
                if (!pCurrentMap->isImuInitialized())
                {
                    Verbose::PrintMess("Track lost before IMU initialisation, reseting...", Verbose::VERBOSITY_QUIET);
                    mpSystem->ResetActiveMap();
                    return;
                }

            //创建新的地图
            CreateMapInAtlas();
        }

        //若当前图像帧的参考帧未设置，设置参考帧为当前图像帧的参考帧
        if(!mCurrentFrame.mpReferenceKF)
            mCurrentFrame.mpReferenceKF = mpReferenceKF;

        mLastFrame = Frame(mCurrentFrame);
    }

    // mState的历史变化--mState=没有图像->（当输入第一张图像）->mState=未初始化
    //               --mState=未初始化-> 单目初始化（MonocularInitialization）->初始化成功->mState=OK
    //                                                                    -> 未成功  
    //               --mState=OK或近期丢失 -> 当前图像帧位姿求解成功-> 存储lost状态
   


    //记录位姿信息，用于轨迹复现
    if(mState==OK || mState==RECENTLY_LOST)
    {
        // Store frame pose information to retrieve the complete camera trajectory afterwards.
        //若当前图像帧的位姿已求出，则存储姿态
        if(!mCurrentFrame.mTcw.empty())
        {   
            //Tcr = Tc_iw*Twc_r
            cv::Mat Tcr = mCurrentFrame.mTcw*mCurrentFrame.mpReferenceKF->GetPoseInverse();
            mlRelativeFramePoses.push_back(Tcr);
            mlpReferences.push_back(mCurrentFrame.mpReferenceKF);
            mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
            mlbLost.push_back(mState==LOST);
        }
        else
        {
            // This can happen if tracking is lost
            // 如果跟踪失败，则相对位姿使用上一次值
            mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
            mlpReferences.push_back(mlpReferences.back());
            mlFrameTimes.push_back(mlFrameTimes.back());
            mlbLost.push_back(mState==LOST);
        }

    }


}


void Tracking::StereoInitialization()
{
    if(mCurrentFrame.N>500)
    {
        if (mSensor == System::IMU_STEREO)
        {
            if (!mCurrentFrame.mpImuPreintegrated || !mLastFrame.mpImuPreintegrated)
            {
                cout << "not IMU meas" << endl;
                return;
            }

            if (cv::norm(mCurrentFrame.mpImuPreintegratedFrame->avgA-mLastFrame.mpImuPreintegratedFrame->avgA)<0.5)
            {
                cout << cv::norm(mCurrentFrame.mpImuPreintegratedFrame->avgA) << endl;
                cout << "not enough acceleration" << endl;
                return;
            }

            if(mpImuPreintegratedFromLastKF)
                delete mpImuPreintegratedFromLastKF;

            mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(),*mpImuCalib);
            mCurrentFrame.mpImuPreintegrated = mpImuPreintegratedFromLastKF;
        }

        // Set Frame pose to the origin (In case of inertial SLAM to imu)
        if (mSensor == System::IMU_STEREO)
        {
            cv::Mat Rwb0 = mCurrentFrame.mImuCalib.Tcb.rowRange(0,3).colRange(0,3).clone();
            cv::Mat twb0 = mCurrentFrame.mImuCalib.Tcb.rowRange(0,3).col(3).clone();
            mCurrentFrame.SetImuPoseVelocity(Rwb0, twb0, cv::Mat::zeros(3,1,CV_32F));
        }
        else
            mCurrentFrame.SetPose(cv::Mat::eye(4,4,CV_32F));

        // Create KeyFrame
        KeyFrame* pKFini = new KeyFrame(mCurrentFrame,mpAtlas->GetCurrentMap(),mpKeyFrameDB);

        // Insert KeyFrame in the map
        mpAtlas->AddKeyFrame(pKFini);

        // Create MapPoints and asscoiate to KeyFrame
        if(!mpCamera2){
            for(int i=0; i<mCurrentFrame.N;i++)
            {
                float z = mCurrentFrame.mvDepth[i];
                if(z>0)
                {
                    cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                    MapPoint* pNewMP = new MapPoint(x3D,pKFini,mpAtlas->GetCurrentMap());
                    pNewMP->AddObservation(pKFini,i);
                    pKFini->AddMapPoint(pNewMP,i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpAtlas->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i]=pNewMP;
                }
            }
        } else{
            for(int i = 0; i < mCurrentFrame.Nleft; i++){
                int rightIndex = mCurrentFrame.mvLeftToRightMatch[i];
                if(rightIndex != -1){
                    cv::Mat x3D = mCurrentFrame.mvStereo3Dpoints[i];

                    MapPoint* pNewMP = new MapPoint(x3D,pKFini,mpAtlas->GetCurrentMap());

                    pNewMP->AddObservation(pKFini,i);
                    pNewMP->AddObservation(pKFini,rightIndex + mCurrentFrame.Nleft);

                    pKFini->AddMapPoint(pNewMP,i);
                    pKFini->AddMapPoint(pNewMP,rightIndex + mCurrentFrame.Nleft);

                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpAtlas->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i]=pNewMP;
                    mCurrentFrame.mvpMapPoints[rightIndex + mCurrentFrame.Nleft]=pNewMP;
                }
            }
        }

        Verbose::PrintMess("New Map created with " + to_string(mpAtlas->MapPointsInMap()) + " points", Verbose::VERBOSITY_QUIET);

        mpLocalMapper->InsertKeyFrame(pKFini);

        mLastFrame = Frame(mCurrentFrame);
        mnLastKeyFrameId=mCurrentFrame.mnId;
        mpLastKeyFrame = pKFini;
        mnLastRelocFrameId = mCurrentFrame.mnId;

        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints=mpAtlas->GetAllMapPoints();
        mpReferenceKF = pKFini;
        mCurrentFrame.mpReferenceKF = pKFini;

        mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints);

        mpAtlas->GetCurrentMap()->mvpKeyFrameOrigins.push_back(pKFini);

        mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

        mState=OK;
    }
}

//单目初始化，并行地计算基础矩阵和单应性矩阵，选取其中一个模型，恢复出最开始两帧之间的相对姿态以及点云
//得到初始两帧的匹配、相对运动、初始MapPoints
void Tracking::MonocularInitialization()
{
    //若是第一次进行初始化操作
    if(!mpInitializer)
    {    
        // Set Reference Frame
        //设置参考帧且要求当前帧的特征点数量大于100
        if(mCurrentFrame.mvKeys.size()>100)
        {   
            mInitialFrame = Frame(mCurrentFrame); //设置当前帧为初始帧
            mLastFrame = Frame(mCurrentFrame);  //设置上一帧为当前帧
            //将当前图像帧的关键点存为mvbPrevMatched，即对于第一帧来说，之前匹配的关键点向量值为当前图像帧的关键点内容
            //mvbPrevMatched:  std::vector<cv::Point2f>
            mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size()); 
            for(size_t i=0; i<mCurrentFrame.mvKeysUn.size(); i++)
                mvbPrevMatched[i]=mCurrentFrame.mvKeysUn[i].pt;

            if(mpInitializer)
                delete mpInitializer;
            
            //构建新的初始化器，固定初始化的参考帧为当前帧（初始帧），sigma:1.0  200：迭代次数
            /**********************************************
                mCurrentFrame:输入为参考图像帧（初始化器未构造成功时，当前帧=初始帧=参考帧=上一帧）
                1：测量误差
                200：RANSAC最大迭代次数
            ***********************************************/
            mpInitializer =  new Initializer(mCurrentFrame,1.0,200);

            //将mvIniMatches向量全部赋值为-1，表示没有匹配，mvIniMatches中存储的是匹配的点的id
            fill(mvIniMatches.begin(),mvIniMatches.end(),-1); 

            if (mSensor == System::IMU_MONOCULAR)
            {
                if(mpImuPreintegratedFromLastKF)
                {
                    delete mpImuPreintegratedFromLastKF;
                }
                //构建新的预积分求解 偏置初始化为（0,0,0,0,0,0）
                mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(),*mpImuCalib);
                mCurrentFrame.mpImuPreintegrated = mpImuPreintegratedFromLastKF;

            }
            return;
        }
    }
    else //非第一次进行初始化，即初始化器已被创建
    {   
        //若当前图像帧的特征点总数小于100,且当前帧和前一帧图像之间的时间大于1s,则重新构造初始化器
        if (((int)mCurrentFrame.mvKeys.size()<=100)||((mSensor == System::IMU_MONOCULAR)&&(mLastFrame.mTimeStamp-mInitialFrame.mTimeStamp>1.0)))
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);

            return;
        }
        
        // Find correspondences
        //在mInitialFrame初始帧与mCurrentFrame当前帧中进行匹配
        /**************************
         * 输入：0.9  最佳和次佳的特征点评分的比值阈值
         * true: 是否检查特征点方向，true表示检查
         *************************/
        ORBmatcher matcher(0.9,true);

        /************当前图像帧和初始图像帧进行匹配**************************
         * mInitialFrame:初始帧图像（设置为第一张图像）
         * mCurrentFrame：当前帧
         * mvbPrevMatched：初始化时构建，存储的是初始图像帧的特征点坐标，用于区域搜索
         * mvIniMatches：初始是vector中全为false，表示未匹配
         * 100：搜索窗口大小
        ***************************************************************/
        int nmatches = matcher.SearchForInitialization(mInitialFrame,mCurrentFrame,mvbPrevMatched,mvIniMatches,100);
        /*****************匹配后****************
         * mvbPrevMatched:容量为mInitialFrame特征点数量
                          mvbPrevMatched[i]的索引i是mInitialFrame特征点序号
                          mvbPrevMatched[i]中的内容是匹配的mCurrentFrame的特征点坐标值
         * mvIniMatches: 容量为mInitialFrame特征点的数量,表示参考帧F1和F2匹配关系
                         mvIniMatches[i]的索引i是mInitialFrame特征点序号
                         mvIniMatches[i]值保存的是匹配好的F2特征点索引
        ***************************************/

        // Check if there are enough correspondences
        //检查匹配结果是否足够
        if(nmatches<100)
        {   
            //若匹配数目不够，删除初始化指针，重新进入初始化状态
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);
            std::cout<<"特征点数目不够，未进行三角化"<<endl;
            return;
        }

        cv::Mat Rcw; // Current Camera Rotation
        cv::Mat tcw; // Current Camera Translation
        vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)





        //判断是否进行三角化位姿估计，通过H模型或F模型进行单目初始化，得到两帧间相对运动、初始MapPoints
        /**************************************************************
         * mInitialFrame.mvKeysUn:初始帧特征点
         * mCurrentFrame.mvKeysUn：当前帧特征点
         * mvIniMatches：表示参考帧F1和F2匹配关系
                        mvIniMatches[i]的索引i是mInitialFrame特征点序号
                        mvIniMatches[i]值保存的是匹配好的F2特征点索引
         * Rcw,tcw: 待输出 初始化得到的相机的位姿
         * mvIniP3D：std::vector<cv::Point3f>待输出  进行三角化得到的空间点集合      
         * vbTriangulated: 待输出，对应于mvIniMatches来说，其中哪些点被三角化了
        *****************************************************************/
        if(mpCamera->ReconstructWithTwoViews(mInitialFrame.mvKeysUn,
                                             mCurrentFrame.mvKeysUn,
                                             mvIniMatches,
                                             Rcw,tcw,
                                             mvIniP3D,
                                             vbTriangulated))
        {
            cout<<"三角化成功"<<endl;


            



            for(size_t i=0, iend=mvIniMatches.size(); i<iend;i++)
            {
                if(mvIniMatches[i]>=0 && !vbTriangulated[i])
                {
                    mvIniMatches[i]=-1;
                    nmatches--;
                }
            }
            /******************************************************************
             * mvIniMatches：mvIniMatches[i]中i代表mInitialFrame的特征点序号
                             mvIniMatches[i]若为-1，表示初始帧中特征点i未成功匹配
                             若不为-1,则存储mCurrentFrame中匹配到的特征点序号
             * mvIniMatches.size() 
               = mInitialFrame.mvKeysUn.size()
               = mvIniP3D.size()
            ******************************************************************/

            // Set Frame Poses
            //将初始化的第一帧设置为世界坐标系
            //设置初始帧的姿态为单位阵
            mInitialFrame.SetPose(cv::Mat::eye(4,4,CV_32F));

            //设置解算得到的姿态为当前图像帧的姿态
            cv::Mat Tcw = cv::Mat::eye(4,4,CV_32F); //世界坐标系到相机坐标系
            Rcw.copyTo(Tcw.rowRange(0,3).colRange(0,3));
            tcw.copyTo(Tcw.rowRange(0,3).col(3));
            mCurrentFrame.SetPose(Tcw);

            // 创建初始化地图点MapPoints
            //为关键帧初始化生成对应的mapPoints
            CreateInitialMapMonocular();
            /****************************************
             * 生成初始关键帧pKFini：初始图像帧
             * 生成当前关键帧pKFcur：当前图像帧
             * 生成上一图像帧mLastFrame：当前图像帧
             * **************************************/

            // Just for video
            // bStepByStep = true;
        }
        else
        {
            std::cout<<"三角化不成功"<<endl;
        }
    }
}


//单目相机初始化成功后用三角化得到的点生成地图点
void Tracking::CreateInitialMapMonocular()
{
   // Create KeyFrames
    //初始关键帧，当前关键帧
    //构造函数中，设置关键帧的位姿为传入图像帧的位姿
    KeyFrame* pKFini = new KeyFrame(mInitialFrame,mpAtlas->GetCurrentMap(),mpKeyFrameDB);
    KeyFrame* pKFcur = new KeyFrame(mCurrentFrame,mpAtlas->GetCurrentMap(),mpKeyFrameDB);

    if(mSensor == System::IMU_MONOCULAR)
        pKFini->mpImuPreintegrated = (IMU::Preintegrated*)(NULL);


    //将初始关键帧,当前关键帧的描述子转为BoW
    pKFini->ComputeBoW();
    pKFcur->ComputeBoW();

    // Insert KFs in the map
    //向地图中插入关键帧
    mpAtlas->AddKeyFrame(pKFini);
    mpAtlas->AddKeyFrame(pKFcur);

    //遍历所有匹配的特征点，分别创建特征点对应的mapPoints
    for(size_t i=0; i<mvIniMatches.size();i++)
    {   
        /*******************************************************************
            mvIniMatches：mvIniMatches[i]中i代表mInitialFrame的特征点序号
            mvIniMatches[i]若为-1，表示初始帧中特征点i未成功匹配
            若不为-1,则存储mCurrentFrame中匹配到的特征点序号
        *******************************************************************/

        if(mvIniMatches[i]<0)
            continue;
        
        //Create MapPoint.
        //用三角化点mvIniP3D初始化为空间点的世界坐标
        cv::Mat worldPos(mvIniP3D[i]);   //ReconstructWithTwoViews中计算的mvInip3D
        //用3D点构造地图点
        MapPoint* pMP = new MapPoint(worldPos,pKFcur,mpAtlas->GetCurrentMap());


        /**********为该MapPoint添加属性****************
            a.观测到该MapPoint的关键帧
            b.该MapPoint的描述子
            c.该MapPoint的平均观测方向和深度范围
        ********************************************/
        //给关键帧加上地图点
        //关键帧能观测到很多个地图点pMP，将此关键帧与地图点关联，添加地图点以及2D特征点的关系
        pKFini->AddMapPoint(pMP,i);  //i：初始关键帧对应的特征点ID
        pKFcur->AddMapPoint(pMP,mvIniMatches[i]);  //mvIniMatches[i]：当前帧匹配的特征点ID

        //地图点的属性，添加能观测到此地图点的关键帧
        pMP->AddObservation(pKFini,i);
        pMP->AddObservation(pKFcur,mvIniMatches[i]);

        //计算用于描述该地图点的描述子，选取最好的用来表征此地图点的描述子
        pMP->ComputeDistinctiveDescriptors();

        //更新pMP的平均观测方向和观测距离的范围
        pMP->UpdateNormalAndDepth();

        //Fill Current Frame structure
        //一个关键帧上的特征点是由多个地图点投影而形成，将地图点与关键帧中的特征点相关联
        //初始图像帧中的第i个特征点，对应当前图像帧中的第mvIniMatches[i]个特征点
        //mvpMapPoints[j]的j表示此图像帧中的第j个特征点，对应的是mvpMapPoints[j]地图点
        mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
        //设置当前地图点不是外点false
        mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

        //Add to Map
        //将此地图点加入地图中
        mpAtlas->AddMapPoint(pMP);
    }


    // Update Connections
    // 统计这个关键帧与其他关键帧共视点的个数，若大于阈值，关联这两个关键帧
    //在3D点和关键帧之间建立边，每个边有一个权重，边的权重是该关键帧与当前帧公共3D点的个数
    pKFini->UpdateConnections();
    pKFcur->UpdateConnections();

    std::set<MapPoint*> sMPs;
    sMPs = pKFini->GetMapPoints(); //初始关键帧的地图点




    // Bundle Adjustment
    //对上面的只有两个关键帧和地图点的地图进行全局BA优化
   // Verbose::PrintMess("New Map created with " + to_string(mpAtlas->MapPointsInMap()) + " points", Verbose::VERBOSITY_QUIET);
    //全局BA 迭代次数20
    Optimizer::GlobalBundleAdjustemnt(mpAtlas->GetCurrentMap(),20);


    //输出当前关键帧中地图点的数量
    pKFcur->PrintPointDistribution();
    
    //求初始关键帧的中值场景深度
    float medianDepth = pKFini->ComputeSceneMedianDepth(2);//2代表单目情况
    float invMedianDepth; //平均场景逆深度
    if(mSensor == System::IMU_MONOCULAR)//若是单目IMU
        invMedianDepth = 4.0f/medianDepth; // 4.0f
    else
        invMedianDepth = 1.0f/medianDepth;

    //若中值深度小于0，且当前关键帧中的地图点数量小于50,则进行重置ActiveMap
    if(medianDepth<0 || pKFcur->TrackedMapPoints(1)<50) // TODO Check, originally 100 tracks
    {
        Verbose::PrintMess("Wrong initialization, reseting...", Verbose::VERBOSITY_NORMAL);
        mpSystem->ResetActiveMap();
        return;
    }

    // Scale initial baseline
    //尺度初始化，归一化平面
    cv::Mat Tc2w = pKFcur->GetPose(); //世界到相机坐标系
    Tc2w.col(3).rowRange(0,3) = Tc2w.col(3).rowRange(0,3)*invMedianDepth;
    pKFcur->SetPose(Tc2w);  //只对旋转进行归一化了

    // Scale points
    //3D点加上尺度信息，归一化平面
    vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();  //初始关键帧的所有地图点
    for(size_t iMP=0; iMP<vpAllMapPoints.size(); iMP++)
    {
        if(vpAllMapPoints[iMP])
        {
            MapPoint* pMP = vpAllMapPoints[iMP];
            pMP->SetWorldPos(pMP->GetWorldPos()*invMedianDepth);
            pMP->UpdateNormalAndDepth();
        }
    }

    //关联当前帧和初始帧
    if (mSensor == System::IMU_MONOCULAR)
    {
        pKFcur->mPrevKF = pKFini;
        pKFini->mNextKF = pKFcur;
        pKFcur->mpImuPreintegrated = mpImuPreintegratedFromLastKF;

        mpImuPreintegratedFromLastKF = new IMU::Preintegrated(pKFcur->mpImuPreintegrated->GetUpdatedBias(),pKFcur->mImuCalib);
    }

    //局部地图加入关键帧
    //LocalMapping线程中的mlNewKeyFrames列表不为空，可以开始处理关键帧
    mpLocalMapper->InsertKeyFrame(pKFini);
    mpLocalMapper->InsertKeyFrame(pKFcur);
    mpLocalMapper->mFirstTs=pKFcur->mTimeStamp;

    mCurrentFrame.SetPose(pKFcur->GetPose());
    mnLastKeyFrameId=mCurrentFrame.mnId;
    mpLastKeyFrame = pKFcur;
    mnLastRelocFrameId = mInitialFrame.mnId;

    mvpLocalKeyFrames.push_back(pKFcur);
    mvpLocalKeyFrames.push_back(pKFini);
    mvpLocalMapPoints=mpAtlas->GetAllMapPoints();
    mpReferenceKF = pKFcur;  //当前帧为参考关键帧
    //当前帧的参考关键帧为当前关键帧
    mCurrentFrame.mpReferenceKF = pKFcur;

    // Compute here initial velocity
    //计算初始速度为空
    vector<KeyFrame*> vKFs = mpAtlas->GetAllKeyFrames();  //获得当前地图中的所有关键帧

    cv::Mat deltaT = vKFs.back()->GetPose()*vKFs.front()->GetPoseInverse();
    mVelocity = cv::Mat();
    Eigen::Vector3d phi = LogSO3(Converter::toMatrix3d(deltaT.rowRange(0,3).colRange(0,3)));


    double aux = (mCurrentFrame.mTimeStamp-mLastFrame.mTimeStamp)/(mCurrentFrame.mTimeStamp-mInitialFrame.mTimeStamp);
    phi *= aux;

    mLastFrame = Frame(mCurrentFrame);

    mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints);

    mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

    mpAtlas->GetCurrentMap()->mvpKeyFrameOrigins.push_back(pKFini);

    //初始完成
    mState=OK;

    initID = pKFcur->mnId;
}


//在Atlas中创建新的地图
void Tracking::CreateMapInAtlas()
{
    //当前图像帧的ID传入上一初始帧ID
    mnLastInitFrameId = mCurrentFrame.mnId;

    //创建新地图并加入地图集
    mpAtlas->CreateNewMap();

    //如果有IMU
    if (mSensor==System::IMU_STEREO || mSensor == System::IMU_MONOCULAR)
    {
        //Map中的mIsInertial设置为true，表示系统中有IMU传感器
        mpAtlas->SetInertialSensor();  

    }
    //设置初始化状态为false
    mbSetInit=false;

    //初始图像帧的ID为当前图像帧ID+1
    mnInitialFrameId = mCurrentFrame.mnId+1;
    //修改Tracking线程的状态
    mState = NO_IMAGES_YET;

    // Restart the variable with information about the last KF
    //用上一关键帧的信息重新计算变量
    mVelocity = cv::Mat();
    //当前图像帧的ID传入上一定位帧ID
    mnLastRelocFrameId = mnLastInitFrameId; // The last relocation KF_id is the current id, because it is the new starting point for new map
    Verbose::PrintMess("First frame id in map: " + to_string(mnLastInitFrameId+1), Verbose::VERBOSITY_NORMAL);
    //mvVO标志位为false
    mbVO = false; // Init value for know if there are enough MapPoints in the last KF

    //如果是单目模式，判断是否完成i初始化，如果完成初始化，删除初始化指针，设置初始化指针为空
    if(mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR)
    {
        if(mpInitializer)
            delete mpInitializer;
        mpInitializer = static_cast<Initializer*>(NULL);
    }

    //如果是IMU模式且IMU预积分已实现，删除当前IMU预积分，重新进行IMU预积分构造
    if((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO ) && mpImuPreintegratedFromLastKF)
    {
        delete mpImuPreintegratedFromLastKF;
        mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(),*mpImuCalib);
    }

    //如果上一关键帧存在，初始化上一关键帧为空
    if(mpLastKeyFrame)
        mpLastKeyFrame = static_cast<KeyFrame*>(NULL);

    //如果参考关键帧存在，初始化参考关键帧为空
    if(mpReferenceKF)
        mpReferenceKF = static_cast<KeyFrame*>(NULL);

    mLastFrame = Frame();
    mCurrentFrame = Frame();
    //清空初始匹配点
    mvIniMatches.clear();

    //已完成地图构建
    mbCreatedMap = true;

}

void Tracking::CheckReplacedInLastFrame()
{
    for(int i =0; i<mLastFrame.N; i++)
    {
        MapPoint* pMP = mLastFrame.mvpMapPoints[i];

        if(pMP)
        {
            MapPoint* pRep = pMP->GetReplaced();
            if(pRep)
            {
                mLastFrame.mvpMapPoints[i] = pRep;
            }
        }
    }
}

//按照参考关键帧来进行跟踪，用参考关键帧的地图点来对当前普通帧进行匹配优化位姿
//单目-IMU，返回true
bool Tracking::TrackReferenceKeyFrame()
{
    //将当前图像帧的描述子转化为BO向量
    //计算词袋向量mBowVec和特征向量mFeatVec
    mCurrentFrame.ComputeBoW();

    // We perform first an ORB matching with the reference keyframe
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.7,true);

    vector<MapPoint*> vpMapPointMatches;


    //参考关键帧与当前帧进行基于词袋模型的匹配，对关键帧的特征点进行跟踪
    /****************输入时vpMapPointMatches为空************/
    int nmatches = matcher.SearchByBoW(mpReferenceKF,mCurrentFrame,vpMapPointMatches);
    /*****************vpMapPointMatches********************
     * 存储大小：与当前图像帧的特征点数量一致
     * vpMapPointMatches的索引：当前图像帧中特征点的序号
     * vpMapPointMatches[i]：匹配成功的特征点的对应的地图点(来自参考关键帧)
    ************************************************/



    if(nmatches<15)
    {
        cout << "TRACK_REF_KF: Less than 15 matches!!\n";
        return false;
    }

    mCurrentFrame.mvpMapPoints = vpMapPointMatches;//更新当前图像帧的地图点为包括匹配信息的地图点
    mCurrentFrame.SetPose(mLastFrame.mTcw);  //用上一帧的位姿设置当前图像帧的初始位姿

    //mCurrentFrame.PrintPointDistribution();


    // cout << " TrackReferenceKeyFrame mLastFrame.mTcw:  " << mLastFrame.mTcw << endl;
    //通过3D-2D的重投影误差获得优化位姿
    //优化过程标记重投影误差过大的特征点为外点
    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    //根据优化过程中标记的外点特征点，进行特征点剔除
    int nmatchesMap = 0;
    //遍历当前帧中所有的特征点
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        //if(i >= mCurrentFrame.Nleft) break;
        //如果当前图像帧存在此特征点对应的地图点
        if(mCurrentFrame.mvpMapPoints[i])
        {   
            //如果此特征点被标记为外点
            if(mCurrentFrame.mvbOutlier[i])
            {   
                //提取此外点特征点对应的地图点
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                //对当前地图点重新初始化，清除此地图点
                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;//将此特征点设为内点
                //此时，此特征点不存在对应的地图点
                if(i < mCurrentFrame.Nleft)
                {
                    pMP->mbTrackInView = false;  
                }
                else{
                    pMP->mbTrackInViewR = false;
                }
                pMP->mbTrackInView = false;//此特征点不再进行跟踪
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;  //匹配数量-1
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }

    // TODO check these conditions
    if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO)
        return true;
    else
        return nmatchesMap>=10;
}

//更新上一个图像帧
void Tracking::UpdateLastFrame()
{
    // Update pose according to reference keyframe
    //提取上一个图像帧的参考关键帧
    KeyFrame* pRef = mLastFrame.mpReferenceKF;

    //Tlr：参考关键帧到上一个图像帧的变换
    cv::Mat Tlr = mlRelativeFramePoses.back();
    //Tci-1, w = Tc_i-1,r*Trw
    //设置上一个图像帧的位姿
    mLastFrame.SetPose(Tlr*pRef->GetPose());

    //如果上一个图像帧为关键帧，或者单目/单目imu，则返回
    if(mnLastKeyFrameId==mLastFrame.mnId || mSensor==System::MONOCULAR || mSensor==System::IMU_MONOCULAR || !mbOnlyTracking)
        return;

    // Create "visual odometry" MapPoints
    // We sort points according to their measured depth by the stereo/RGB-D sensor
    vector<pair<float,int> > vDepthIdx;
    vDepthIdx.reserve(mLastFrame.N);
    for(int i=0; i<mLastFrame.N;i++)
    {
        float z = mLastFrame.mvDepth[i];
        if(z>0)
        {
            vDepthIdx.push_back(make_pair(z,i));
        }
    }

    if(vDepthIdx.empty())
        return;

    sort(vDepthIdx.begin(),vDepthIdx.end());

    // We insert all close points (depth<mThDepth)
    // If less than 100 close points, we insert the 100 closest ones.
    int nPoints = 0;
    for(size_t j=0; j<vDepthIdx.size();j++)
    {
        int i = vDepthIdx[j].second;

        bool bCreateNew = false;

        MapPoint* pMP = mLastFrame.mvpMapPoints[i];
        if(!pMP)
            bCreateNew = true;
        else if(pMP->Observations()<1)
        {
            bCreateNew = true;
        }

        if(bCreateNew)
        {
            cv::Mat x3D = mLastFrame.UnprojectStereo(i);
            MapPoint* pNewMP = new MapPoint(x3D,mpAtlas->GetCurrentMap(),&mLastFrame,i);

            mLastFrame.mvpMapPoints[i]=pNewMP;

            mlpTemporalPoints.push_back(pNewMP);
            nPoints++;
        }
        else
        {
            nPoints++;
        }

        if(vDepthIdx[j].first>mThDepth && nPoints>100)
        {
            break;
        }
    }
}


//恒速模型进行跟踪图像帧
bool Tracking::TrackWithMotionModel()
{   
    //构建匹配器
    ORBmatcher matcher(0.9,true);

    // Update last frame pose according to its reference keyframe
    // Create "visual odometry" points if in Localization Mode
    //设置上一个图像帧的位姿，为参考关键帧到上一图像帧的变换×世界坐标系到参考关键真之间的变换
    UpdateLastFrame();


    // 根据IMU或者恒速模型得到当前帧的初始位姿
    if (mpAtlas->isImuInitialized() && (mCurrentFrame.mnId>mnLastRelocFrameId+mnFramesToResetIMU))
    {   cout<<"IMU 估计位姿"<<endl;
        // 若IMU完成初始化 并且 距离重定位挺久不需要重置IMU，用IMU来估计位姿
        PredictStateIMU();
        return true;
    }
    else
    {   //其他情况，用恒速模型得到当前图像帧的初始位姿
        cout<<"恒速模型估计位姿"<<endl;
        cout<<mVelocity<<endl;
        mCurrentFrame.SetPose(mVelocity*mLastFrame.mTcw);
    }

    //初始化当前图像帧的地图点
    fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));

    // Project points seen in previous frame
    int th;

    //设置匹配阈值
    if(mSensor==System::STEREO)
        th=7;
    else
        th=15;


    //投影法特征匹配 如果数量少，增加阈值进行特征点匹配
    int nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,th,mSensor==System::MONOCULAR || mSensor==System::IMU_MONOCULAR);

    // If few matches, uses a wider window search
    if(nmatches<20)
    {
        Verbose::PrintMess("Not enough matches, wider window search!!", Verbose::VERBOSITY_NORMAL);
        fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));

        nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,2*th,mSensor==System::MONOCULAR || mSensor==System::IMU_MONOCULAR);
        Verbose::PrintMess("Matches with wider search: " + to_string(nmatches), Verbose::VERBOSITY_NORMAL);

    }

    if(nmatches<20)
    {
        Verbose::PrintMess("Not enough matches!!", Verbose::VERBOSITY_NORMAL);
        if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO)
            return true;
        else
            return false;
    }

    
    // Optimize frame pose with all matches
    //利用3D-2D的重投影误差对当前图像帧的位姿进行优化[与Trackreference一样]
    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                if(i < mCurrentFrame.Nleft){
                    pMP->mbTrackInView = false;
                }
                else{
                    pMP->mbTrackInViewR = false;
                }
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }

    if(mbOnlyTracking)
    {
        mbVO = nmatchesMap<10;
        return nmatches>20;
    }

    if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO)
        return true;
    else
        return nmatchesMap>=10;
}

//用局部地图进行跟踪
bool Tracking::TrackLocalMap()
{

    // We have an estimation of the camera pose and some map points tracked in the frame.
    // We retrieve the local map and try to find matches to points in the local map.
    //跟踪的图像帧数量，在构造Tracking线程时初始化为0
    mTrackedFr++;

    //更新局部地图: 更新局部地图中的关键帧UpdateLocalKeyFrames()和地图点UpdateLocalPoints()
    UpdateLocalMap();

    // 筛选局部地图中新增的在视野范围内的地图点，投影到当前帧搜索匹配，得到更多的匹配关系
    cout<<"   [局部地图跟踪 TrackLocalMap ]    "<<endl;
    SearchLocalPoints();

    // TOO check outliers before PO
    //查看内点和外点数目
    //aux1：当前图像帧中的地图点数目
    //aux2：当前图像帧中地图点的外点数目
    int aux1 = 0, aux2=0;
    for(int i=0; i<mCurrentFrame.N; i++)
        if( mCurrentFrame.mvpMapPoints[i])
        {
            aux1++;
            if(mCurrentFrame.mvbOutlier[i])
                aux2++;
        }

    //重新进行优化当前图像帧的位姿
    int inliers;
    //IMU初始化未成功，只优化当前图像帧的位姿
    if (!mpAtlas->isImuInitialized())
        Optimizer::PoseOptimization(&mCurrentFrame);
    else 
    {   //如果距离上次重定位时间比较近，IMU数量少，则不使用IMU数据，只进行位姿优化
        if(mCurrentFrame.mnId<=mnLastRelocFrameId+mnFramesToResetIMU)
        {
            Verbose::PrintMess("TLM: PoseOptimization ", Verbose::VERBOSITY_DEBUG);
            Optimizer::PoseOptimization(&mCurrentFrame);
        }
        else //否则，使用IMU数据进行优化
        {
            //如果没有更新地图
            if(!mbMapUpdated) //  && (mnMatchesInliers>30))
            {   
                //用IMU和上一个图像帧进行位姿优化
                Verbose::PrintMess("TLM: PoseInertialOptimizationLastFrame ", Verbose::VERBOSITY_DEBUG);
                inliers = Optimizer::PoseInertialOptimizationLastFrame(&mCurrentFrame); // , !mpLastKeyFrame->GetMap()->GetIniertialBA1());
            }
            else
            {
                Verbose::PrintMess("TLM: PoseInertialOptimizationLastKeyFrame ", Verbose::VERBOSITY_DEBUG);
                inliers = Optimizer::PoseInertialOptimizationLastKeyFrame(&mCurrentFrame); // , !mpLastKeyFrame->GetMap()->GetIniertialBA1());
            }
        }
    }

    //遍历当前帧的地图点，统计地图点内点的数量和外点的数量
    aux1 = 0, aux2 = 0;
    for(int i=0; i<mCurrentFrame.N; i++)
        if( mCurrentFrame.mvpMapPoints[i])
        {
            aux1++;
            if(mCurrentFrame.mvbOutlier[i])
                aux2++;
        }

    //匹配成功的内点数量
    mnMatchesInliers = 0;  



    // Update MapPoints Statistics
    //更新地图点的被观测程度
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {   

            if(!mCurrentFrame.mvbOutlier[i])
            {   
                // 若当前帧的地图点可以被当前帧观测到，其被观测统计量加1
                mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                //若是在纯定位过程
                if(!mbOnlyTracking)
                {      
                    // 如果该地图点被相机观测数目nObs大于0，匹配内点计数+1
                    // nObs： 被观测到的相机数目，单目+1，双目或RGB-D则+2
                    if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                        mnMatchesInliers++;
                }
                else
                    mnMatchesInliers++;
            }
            else if(mSensor==System::STEREO)
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
        }
    }

    // Decide if the tracking was succesful
    // More restrictive if there was a relocalization recently
    //更新局部地图中的匹配内点数量
    mpLocalMapper->mnMatchesInliers=mnMatchesInliers;

    //若刚刚发生了重定位，且匹配的内点数少于50,认为跟踪失败
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && mnMatchesInliers<50)
        return false;

    //若在Recently_lost状态下，且匹配内点数超过10个，则认为是跟踪成功
    if((mnMatchesInliers>10)&&(mState==RECENTLY_LOST))
        return true;

    //单目IMU模式
    if (mSensor == System::IMU_MONOCULAR)
    {   
        //若匹配成功的内点数目小于15,则认为跟踪失败
        if(mnMatchesInliers<15)
        {
            return false;
        }
        else
            return true;
    }
    else if (mSensor == System::IMU_STEREO)
    {
        if(mnMatchesInliers<15)
        {
            return false;
        }
        else
            return true;
    }
    else
    {
        if(mnMatchesInliers<30)
            return false;
        else
            return true;
    }
}

bool Tracking::NeedNewKeyFrame()
{
    if(((mSensor == System::IMU_MONOCULAR) || (mSensor == System::IMU_STEREO)) && !mpAtlas->GetCurrentMap()->isImuInitialized())
    {
        if (mSensor == System::IMU_MONOCULAR && (mCurrentFrame.mTimeStamp-mpLastKeyFrame->mTimeStamp)>=0.25)
            return true;
        else if (mSensor == System::IMU_STEREO && (mCurrentFrame.mTimeStamp-mpLastKeyFrame->mTimeStamp)>=0.25)
            return true;
        else
            return false;
    }

    if(mbOnlyTracking)
        return false;

    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
    {
        return false;
    }

    // Return false if IMU is initialazing
    if (mpLocalMapper->IsInitializing())
        return false;
    const int nKFs = mpAtlas->KeyFramesInMap();

    // Do not insert keyframes if not enough frames have passed from last relocalisation
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && nKFs>mMaxFrames)
    {
        return false;
    }

    // Tracked MapPoints in the reference keyframe
    int nMinObs = 3;
    if(nKFs<=2)
        nMinObs=2;
    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

    // Local Mapping accept keyframes?
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

    // Check how many "close" points are being tracked and how many could be potentially created.
    int nNonTrackedClose = 0;
    int nTrackedClose= 0;

    if(mSensor!=System::MONOCULAR && mSensor!=System::IMU_MONOCULAR)
    {
        int N = (mCurrentFrame.Nleft == -1) ? mCurrentFrame.N : mCurrentFrame.Nleft;
        for(int i =0; i<N; i++)
        {
            if(mCurrentFrame.mvDepth[i]>0 && mCurrentFrame.mvDepth[i]<mThDepth)
            {
                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                    nTrackedClose++;
                else
                    nNonTrackedClose++;

            }
        }
    }

    bool bNeedToInsertClose;
    bNeedToInsertClose = (nTrackedClose<100) && (nNonTrackedClose>70);

    // Thresholds
    float thRefRatio = 0.75f;
    if(nKFs<2)
        thRefRatio = 0.4f;

    if(mSensor==System::MONOCULAR)
        thRefRatio = 0.9f;

    if(mpCamera2) thRefRatio = 0.75f;

    if(mSensor==System::IMU_MONOCULAR)
    {
        if(mnMatchesInliers>350) // Points tracked from the local map
            thRefRatio = 0.75f;
        else
            thRefRatio = 0.90f;
    }

    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    const bool c1a = mCurrentFrame.mnId>=mnLastKeyFrameId+mMaxFrames;
    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
    const bool c1b = ((mCurrentFrame.mnId>=mnLastKeyFrameId+mMinFrames) && bLocalMappingIdle);
    //Condition 1c: tracking is weak
    const bool c1c = mSensor!=System::MONOCULAR && mSensor!=System::IMU_MONOCULAR && mSensor!=System::IMU_STEREO && (mnMatchesInliers<nRefMatches*0.25 || bNeedToInsertClose) ;
    // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
    const bool c2 = (((mnMatchesInliers<nRefMatches*thRefRatio || bNeedToInsertClose)) && mnMatchesInliers>15);

    // Temporal condition for Inertial cases
    bool c3 = false;
    if(mpLastKeyFrame)
    {
        if (mSensor==System::IMU_MONOCULAR)
        {
            if ((mCurrentFrame.mTimeStamp-mpLastKeyFrame->mTimeStamp)>=0.5)
                c3 = true;
        }
        else if (mSensor==System::IMU_STEREO)
        {
            if ((mCurrentFrame.mTimeStamp-mpLastKeyFrame->mTimeStamp)>=0.5)
                c3 = true;
        }
    }

    bool c4 = false;
    if ((((mnMatchesInliers<75) && (mnMatchesInliers>15)) || mState==RECENTLY_LOST) && ((mSensor == System::IMU_MONOCULAR))) // MODIFICATION_2, originally ((((mnMatchesInliers<75) && (mnMatchesInliers>15)) || mState==RECENTLY_LOST) && ((mSensor == System::IMU_MONOCULAR)))
        c4=true;
    else
        c4=false;

    if(((c1a||c1b||c1c) && c2)||c3 ||c4)
    {
        // If the mapping accepts keyframes, insert keyframe.
        // Otherwise send a signal to interrupt BA
        if(bLocalMappingIdle)
        {
            return true;
        }
        else
        {
            mpLocalMapper->InterruptBA();
            if(mSensor!=System::MONOCULAR  && mSensor!=System::IMU_MONOCULAR)
            {
                if(mpLocalMapper->KeyframesInQueue()<3)
                    return true;
                else
                    return false;
            }
            else
                return false;
        }
    }
    else
        return false;
}

void Tracking::CreateNewKeyFrame()
{
    if(mpLocalMapper->IsInitializing())
        return;

    if(!mpLocalMapper->SetNotStop(true))
        return;

    KeyFrame* pKF = new KeyFrame(mCurrentFrame,mpAtlas->GetCurrentMap(),mpKeyFrameDB);

    if(mpAtlas->isImuInitialized())
        pKF->bImu = true;

    pKF->SetNewBias(mCurrentFrame.mImuBias);
    mpReferenceKF = pKF;
    mCurrentFrame.mpReferenceKF = pKF;

    if(mpLastKeyFrame)
    {
        pKF->mPrevKF = mpLastKeyFrame;
        mpLastKeyFrame->mNextKF = pKF;
    }
    else
        Verbose::PrintMess("No last KF in KF creation!!", Verbose::VERBOSITY_NORMAL);

    // Reset preintegration from last KF (Create new object)
    if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO)
    {
        mpImuPreintegratedFromLastKF = new IMU::Preintegrated(pKF->GetImuBias(),pKF->mImuCalib);
    }

    if(mSensor!=System::MONOCULAR && mSensor != System::IMU_MONOCULAR) // TODO check if incluide imu_stereo
    {
        mCurrentFrame.UpdatePoseMatrices();
        // cout << "create new MPs" << endl;
        // We sort points by the measured depth by the stereo/RGBD sensor.
        // We create all those MapPoints whose depth < mThDepth.
        // If there are less than 100 close points we create the 100 closest.
        int maxPoint = 100;
        if(mSensor == System::IMU_STEREO)
            maxPoint = 100;

        vector<pair<float,int> > vDepthIdx;
        int N = (mCurrentFrame.Nleft != -1) ? mCurrentFrame.Nleft : mCurrentFrame.N;
        vDepthIdx.reserve(mCurrentFrame.N);
        for(int i=0; i<N; i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                vDepthIdx.push_back(make_pair(z,i));
            }
        }

        if(!vDepthIdx.empty())
        {
            sort(vDepthIdx.begin(),vDepthIdx.end());

            int nPoints = 0;
            for(size_t j=0; j<vDepthIdx.size();j++)
            {
                int i = vDepthIdx[j].second;

                bool bCreateNew = false;

                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(!pMP)
                    bCreateNew = true;
                else if(pMP->Observations()<1)
                {
                    bCreateNew = true;
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
                }

                if(bCreateNew)
                {
                    cv::Mat x3D;

                    if(mCurrentFrame.Nleft == -1){
                        x3D = mCurrentFrame.UnprojectStereo(i);
                    }
                    else{
                        x3D = mCurrentFrame.UnprojectStereoFishEye(i);
                    }

                    MapPoint* pNewMP = new MapPoint(x3D,pKF,mpAtlas->GetCurrentMap());
                    pNewMP->AddObservation(pKF,i);

                    //Check if it is a stereo observation in order to not
                    //duplicate mappoints
                    if(mCurrentFrame.Nleft != -1 && mCurrentFrame.mvLeftToRightMatch[i] >= 0){
                        mCurrentFrame.mvpMapPoints[mCurrentFrame.Nleft + mCurrentFrame.mvLeftToRightMatch[i]]=pNewMP;
                        pNewMP->AddObservation(pKF,mCurrentFrame.Nleft + mCurrentFrame.mvLeftToRightMatch[i]);
                        pKF->AddMapPoint(pNewMP,mCurrentFrame.Nleft + mCurrentFrame.mvLeftToRightMatch[i]);
                    }

                    pKF->AddMapPoint(pNewMP,i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpAtlas->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i]=pNewMP;
                    nPoints++;
                }
                else
                {
                    nPoints++; // TODO check ???
                }

                if(vDepthIdx[j].first>mThDepth && nPoints>maxPoint)
                {
                    break;
                }
            }

            Verbose::PrintMess("new mps for stereo KF: " + to_string(nPoints), Verbose::VERBOSITY_NORMAL);

        }
    }


    mpLocalMapper->InsertKeyFrame(pKF);

    mpLocalMapper->SetNotStop(false);

    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKF;
    //cout  << "end creating new KF" << endl;
}


//用局部地图点进行投影匹配，得到更多的匹配关系
void Tracking::SearchLocalPoints()
{
    // Do not search map points already matched
    //遍历当前图像帧中的地图点，标记当前图像帧中的所有好的地图点都不需被再次匹配
    for(vector<MapPoint*>::iterator vit=mCurrentFrame.mvpMapPoints.begin(), vend=mCurrentFrame.mvpMapPoints.end(); vit!=vend; vit++)
    {   
        //提取此地图点
        MapPoint* pMP = *vit;
        if(pMP)  //如果地图点存在
        {
            if(pMP->isBad())  //地图点是坏点
            {
                *vit = static_cast<MapPoint*>(NULL);
            }
            else
            {   
                //该地图点可被当前图像帧观测到，因此能观测到该特征点的帧数+1
                pMP->IncreaseVisible();
                //标记此地图点可被当前图像帧观测到
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                //由于已经被当前图像帧观测到了，因此已完成了投影匹配，标记为false，在后续阶段不需要再次操作
                pMP->mbTrackInViewR = false;
            }
        }
    }

    //进行投影匹配的地图点的数目
    int nToMatch=0;

    // Project points in frame and check its visibility
    //遍历所有局部地图中的地图点，筛选需要进行投影匹配的地图点，并添加在当前图像帧中的投影坐标
    for(vector<MapPoint*>::iterator vit=mvpLocalMapPoints.begin(), vend=mvpLocalMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;

        //说明此地图点已被当前图像帧观测到，因此一定在图像的视野范围内，不被选择
        if(pMP->mnLastFrameSeen == mCurrentFrame.mnId)
            continue;
        if(pMP->isBad())
            continue;

        // Project (this fills MapPoint variables for matching)
        // 判断地图点是否在在当前帧视野内
        //0.5:夹角余弦
        //若此地图点在视野范围内则在函数中设置跟踪标志位mbTrackInView为true
        //isInFrustum()函数中，若pMP地图点是当前图像帧中的视野范围内，则会设置此地图点在当前图像帧
        //的投影，即pMP->mTrackProjX, pMP->mTrackProjY
        if(mCurrentFrame.isInFrustum(pMP,0.5))
        {   
            //进入此if，说明pMP地图点在当前图像帧的视野范围内
            pMP->IncreaseVisible(); //可观测到该地图点的图像帧数量+1
            //局部地图内参与投影匹配的数量+1
            nToMatch++;
        }
        
        //pMP->mbTrackInView为true表示该地图点需要被跟踪
        if(pMP->mbTrackInView)
        {   
            //添加该地图点在当前图像帧中的投影坐标
            mCurrentFrame.mmProjectPoints[pMP->mnId] = cv::Point2f(pMP->mTrackProjX, pMP->mTrackProjY);
        }
    }


    //需要进行投影匹配的数目>0
    if(nToMatch>0)
    {   
        ORBmatcher matcher(0.8);

        int th = 1;

        if(mSensor==System::RGBD)
            th=3;

        //已进行单目初始化，则修改阈值
        if(mpAtlas->isImuInitialized())
        {
            if(mpAtlas->GetCurrentMap()->GetIniertialBA2())
                th=2;
            else
                th=3;
        }
        
        //若未进行imu初始化，且是IMU模式，则修改阈值
        else if(!mpAtlas->isImuInitialized() && (mSensor==System::IMU_MONOCULAR || mSensor==System::IMU_STEREO))
        {
            th=10;
        }

        // If the camera has been relocalised recently, perform a coarser search
        // 如果不久前进行过重定位，那么进行一个更加宽泛的搜索，阈值需要增大
        if(mCurrentFrame.mnId<mnLastRelocFrameId+2)
            th=5;

        if(mState==LOST || mState==RECENTLY_LOST) // Lost for less than 1 second
            th=15; // 15
        
        
        /*******************阈值设计*****************
         * 1、RGB-D模式：th=3
         * 2、已实现IMU初始化且惯性BA完成：th=2
         *    已实现IMU初始化但惯性BA未完成：th=3
         * 3、IMU未成功进行初始化但是传感器中包括IMU：th=10
         * 4、如果不久前刚进行过重定位：th=5
         * 5、若状态是丢失或者是近期丢失：th=15  
         * 6、其他情况：th=1   
        ********************************************/


        /**************当前图像帧和局部地图的地图点进行匹配*****************
         * mCurrentFrame：当前图像帧
         * mvpLocalMapPoints：局部地图中的地图点
         * th:阈值
         * mpLocalMapper->mbFarPoints :yaml文件中设置，但没有传入，则为false
         * mpLocalMapper->mThFarPoints：yaml文件中设置，但没有传入，则为false
        *****************************************************/
        int matches = matcher.SearchByProjection(mCurrentFrame, mvpLocalMapPoints, th, 
                                                mpLocalMapper->mbFarPoints, 
                                                mpLocalMapper->mThFarPoints);



                                            
    }
}


//更新局部地图中的关键帧UpdateLocalKeyFrames()和地图点UpdateLocalPoints()
void Tracking::UpdateLocalMap()
{
    // This is for visualization
    //// 设置参考地图点用于绘图显示局部地图点（红色）
    mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints);

    // Update
    //更新局部关键帧,
    UpdateLocalKeyFrames();
    /**************************
        添加一级共视关键帧、
        二级共视关键帧和一级的子、
        父关键帧
        以及IMU模式下的临时关键帧
    ***************************/


    //更新局部关键点
    UpdateLocalPoints();
}

//更新局部地图中的地图点，Tracking::UpdateLocalMap()函数中用到
void Tracking::UpdateLocalPoints()
{   
    // 清楚当前局部地图点
    mvpLocalMapPoints.clear();

    int count_pts = 0;

    //遍历局部关键帧
    for(vector<KeyFrame*>::const_reverse_iterator itKF=mvpLocalKeyFrames.rbegin(), itEndKF=mvpLocalKeyFrames.rend(); itKF!=itEndKF; ++itKF)
    {   
        //提取局部关键帧
        KeyFrame* pKF = *itKF;

        //提取局部关键帧中的地图点
        const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

        // 遍历所有的地图点
        for(vector<MapPoint*>::const_iterator itMP=vpMPs.begin(), itEndMP=vpMPs.end(); itMP!=itEndMP; itMP++)
        {
            //取出地图点
            MapPoint* pMP = *itMP;
            if(!pMP)
                continue;
            
            // 用该地图点的成员变量mnTrackReferenceForFrame 记录当前帧的id
            // 表示它已经是当前帧的局部地图点了，可以防止重复添加局部地图点
            if(pMP->mnTrackReferenceForFrame==mCurrentFrame.mnId) 
                continue;
            
            //如果词地图点不是坏点，添加地图点
            if(!pMP->isBad())
            {
                count_pts++;
                mvpLocalMapPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame=mCurrentFrame.mnId;
            }
        }
    }
}


//Tracking::TrackLocalMap()函数中更新局部关键帧
//方法是遍历当前帧的地图点，将观测到这些地图点的关键帧
//和相邻的关键帧及其父子关键帧，作为mvpLocalKeyFrames
void Tracking::UpdateLocalKeyFrames()
{
    // Each map point vote for the keyframes in which it has been observed
    //记录所有能观测到当前帧地图点的关键帧
    map<KeyFrame*,int> keyframeCounter;

    //这个判断与是否进行恒速模型跟踪和参考关键帧跟踪一致
    //如果IMU未初始化或刚刚完成重定位
    if(!mpAtlas->isImuInitialized() || (mCurrentFrame.mnId<mnLastRelocFrameId+2))
    {   
        //遍历当前图像帧的特征点/地图点（特征点数量=地图点数量，但地图点可以不存在）
        for(int i=0; i<mCurrentFrame.N; i++)
        {   
            //取出特征点i对应的地图点
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            if(pMP)//如果地图点存在
            {   
                //若该地图点不是坏点
                if(!pMP->isBad())
                {   
                    //提取此地图点的观测（可观测到的关键帧及对应的特征点序号）
                    const map<KeyFrame*,tuple<int,int>> observations = pMP->GetObservations();

                    for(map<KeyFrame*,tuple<int,int>>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
                    {
                        /************************************
	                        map[key] = value，当要插入的键存在时，会覆盖键对应的原来的值。
                            如果键不存在，则添加一组键值对
	                        it->first 是地图点看到的关键帧，同一个关键帧看到的地图点会累加到该关键帧计数
	                        keyframeCounter 第一个参数表示观测到当前地图点的某个关键帧，
                                            第2个参数表示该关键帧看到了多少当前帧(mCurrentFrame)的地图点，也就是共视程度
                        ************************************/
                        keyframeCounter[it->first]++;
                    }
                }
                else//若该地图点是坏点
                {   //删除此地图点
                    mCurrentFrame.mvpMapPoints[i]=NULL;
                }
            }
        }
    }
    else //其他情况
    {   
        //由于mLastFrame中存储的是上一帧跟踪成功后的数据
        //也因为此时当前帧还没进行匹配
        //因此遍历上一帧的特征点/地图点
        for(int i=0; i<mLastFrame.N; i++)
        {
            // Using lastframe since current frame has not matches yet
            //如果上一帧的该地图点存在
            if(mLastFrame.mvpMapPoints[i])
            {
                // 提取该地图点
                MapPoint* pMP = mLastFrame.mvpMapPoints[i];
                if(!pMP)
                    continue;
                if(!pMP->isBad()) //如果不是坏点
                {
                    const map<KeyFrame*,tuple<int,int>> observations = pMP->GetObservations();
                    for(map<KeyFrame*,tuple<int,int>>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
                        keyframeCounter[it->first]++;
                }
                else
                {
                    // MODIFICATION
                    mLastFrame.mvpMapPoints[i]=NULL;
                }
            }
        }
    }


    //存储具有最多观测次数（max）的关键帧
    int max=0;
    KeyFrame* pKFmax= static_cast<KeyFrame*>(NULL);

    // 更新局部关键帧（mvpLocalKeyFrames）
    mvpLocalKeyFrames.clear();  //清空所有的局部关键帧
    mvpLocalKeyFrames.reserve(3*keyframeCounter.size());//先申请三倍内存，不够可以后面扩容



    //局部关键帧共存在三种类型，分别添加
    //类型1：一级共视关键帧，添加能观测到当前图像帧的地图点的关键帧
    /**********************************************************************************
    keyframeCounter 第一个参数表示观测到当前地图点的某个关键帧，
                    第2个参数表示该关键帧看到了多少当前帧(mCurrentFrame)的地图点，也就是共视程度
    **********************************************************************************/
    for(map<KeyFrame*,int>::const_iterator it=keyframeCounter.begin(), itEnd=keyframeCounter.end(); it!=itEnd; it++)
    {   
        //观测到当前帧中某个地图点的某个关键帧
        KeyFrame* pKF = it->first;

        if(pKF->isBad())
            continue;

        //寻找具有最大观测数目max的关键帧pKF
        if(it->second>max)
        {
            max=it->second;
            pKFmax=pKF;
        }

        mvpLocalKeyFrames.push_back(pKF); //将此关键帧添加进局部关键帧
        // 用该关键帧的成员变量mnTrackReferenceForFrame 记录当前帧的id
        // 表示它已经是当前帧的局部关键帧了，可以防止重复添加局部关键帧
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
    }

    //根据已经添加的局部关键帧，寻找更多的局部关键帧进行添加
    //类型2：二级共视关键帧：添加一级共视关键帧的共视关键帧，每个一级共视关键帧添加一个二级共视关键帧
    //类型3：添加一级共视关键帧的子关键帧、父关键帧添加作为局部关键帧, 每个一级共识关键帧添加一个子关键帧和一个父关键帧
    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        // Limit the number of keyframes
        //限制局部关键帧的数量不能大于80
        if(mvpLocalKeyFrames.size()>80) // 80
            break;

        //提取当前局部关键帧集合中的关键帧
        KeyFrame* pKF = *itKF;

        //二级共视关键帧
        //寻找当前关键帧的共视关键帧，寻找10个，如果共视帧不满10个，返回所有具有共视关系的关键帧
        const vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);
        //vNeighs中的关键帧是按照共视程度从大到小排序的，靠前的关键帧共视程度大

        for(vector<KeyFrame*>::const_iterator itNeighKF=vNeighs.begin(), itEndNeighKF=vNeighs.end(); itNeighKF!=itEndNeighKF; itNeighKF++)
        {
            KeyFrame* pNeighKF = *itNeighKF;
            if(!pNeighKF->isBad())//如果不是坏的关键帧
            {   
                //如果此关键帧没有被添加过
                if(pNeighKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pNeighKF); //添加
                    pNeighKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }


        //类型三 一级共视关键帧的子关键帧
        const set<KeyFrame*> spChilds = pKF->GetChilds();
        for(set<KeyFrame*>::const_iterator sit=spChilds.begin(), send=spChilds.end(); sit!=send; sit++)
        {
            KeyFrame* pChildKF = *sit;
            if(!pChildKF->isBad())
            {
                if(pChildKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pChildKF);
                    pChildKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        //类型三 一级共视关键帧的子关键帧
        KeyFrame* pParent = pKF->GetParent();
        if(pParent)
        {
            if(pParent->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(pParent);
                pParent->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                break;
            }
        }
    }


    //在IMU-单目模式下，mvpLocalKeyFrames增加临时关键帧（数量最多20个）
    if((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO) &&mvpLocalKeyFrames.size()<80)
    {
        //cout << "CurrentKF: " << mCurrentFrame.mnId << endl;
        //取出当前图像帧的上一个关键帧
        KeyFrame* tempKeyFrame = mCurrentFrame.mpLastKeyFrame;

        const int Nd = 20;
        for(int i=0; i<Nd; i++)
        {
            if (!tempKeyFrame)
                break;
            //cout << "tempKF: " << tempKeyFrame << endl;
            if(tempKeyFrame->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(tempKeyFrame);
                tempKeyFrame->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                tempKeyFrame=tempKeyFrame->mPrevKF;
            }
        }
    }

    //更新当前帧的参考关键帧，与当前图像帧共视程度最高的关键帧作为参考关键帧
    if(pKFmax)
    {
        mpReferenceKF = pKFmax;
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }
}


//重定位
bool Tracking::Relocalization()
{   
    
    Verbose::PrintMess("Starting relocalization", Verbose::VERBOSITY_NORMAL);

    //计算当前图像帧的词袋模型
    // Compute Bag of Words Vector
    mCurrentFrame.ComputeBoW();

    // Relocalization is performed when tracking is lost
    // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
    //寻找与当前图像帧相似的候选关键帧
    vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame, mpAtlas->GetCurrentMap());

    if(vpCandidateKFs.empty()) 
    {
        Verbose::PrintMess("There are not candidates", Verbose::VERBOSITY_NORMAL);
        return false;
    }

    //候选关键帧的数量
    const int nKFs = vpCandidateKFs.size();

    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.75,true);

    //构造每个关键帧的解算器
    vector<MLPnPsolver*> vpMLPnPsolvers;
    vpMLPnPsolvers.resize(nKFs);

    //存储每个关键帧和当前图像帧中特征点的匹配关系
    vector<vector<MapPoint*> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);

    //存储丢弃的关键帧的ID
    vector<bool> vbDiscarded;
    vbDiscarded.resize(nKFs);

    int nCandidates=0;

    //遍历候选关键帧
    for(int i=0; i<nKFs; i++)
    {
        KeyFrame* pKF = vpCandidateKFs[i];

        if(pKF->isBad())
            vbDiscarded[i] = true;
        else
        {
            //利用词袋模型进行3D-2D的特征匹配，与跟踪参考关键帧的方法一样
            int nmatches = matcher.SearchByBoW(pKF,mCurrentFrame,vvpMapPointMatches[i]);
            if(nmatches<15)
            {
                vbDiscarded[i] = true;
                continue;
            }
            else  //若匹配成功数量大于15,构建求解器，求解相对位姿
            {
                MLPnPsolver* pSolver = new MLPnPsolver(mCurrentFrame,vvpMapPointMatches[i]);
                //设置求解器的参数等
                // * 0.99:迭代过程中随机选取的点均为内点的概率
                // * 10：最小内点数目
                // * 300：最大迭代次数
                // * 6：每次随机选取的特征点数量
                // * 0.5：内点占整个数据集的比例
                // * 5.991：判断RANSAC的误差阈值
                pSolver->SetRansacParameters(0.99,10,300,6,0.5,5.991);  //This solver needs at least 6 points
                vpMLPnPsolvers[i] = pSolver;
            }
        }
    }

    // Alternatively perform some iterations of P4P RANSAC
    // Until we found a camera pose supported by enough inliers
    // 足够的内点才能匹配使用PNP算法，MLPnP需要至少6个点

    bool bMatch = false;  //表示是否找到能匹配上的关键帧 false表示还没找到
    ORBmatcher matcher2(0.9,true);

    //若候选帧数目>0且还未匹配到关键帧，则继续循环
    while(nCandidates>0 && !bMatch)
    {
        //遍历候选关键帧
        for(int i=0; i<nKFs; i++)
        {
            if(vbDiscarded[i])
                continue;

            // 内点标记
            vector<bool> vbInliers;
            // 内点数
            int nInliers;
            // bNoMore为true表示需要更多的点来进行迭代，此次失败
            bool bNoMore;

            MLPnPsolver* pSolver = vpMLPnPsolvers[i];

            // Perform 5 Ransac Iterations
            // 执行5次RANSAC迭代
            cv::Mat Tcw = pSolver->iterate(5,bNoMore,vbInliers,nInliers);

            // If Ransac reachs max. iterations discard keyframe
            if(bNoMore)
            {
                vbDiscarded[i]=true;
                nCandidates--;
            }

            // If a Camera Pose is computed, optimize
            //如果求解的位姿不为空
            if(!Tcw.empty())
            {   
                //设置当前图像帧的位姿
                Tcw.copyTo(mCurrentFrame.mTcw);

                //存储找到的地图点
                set<MapPoint*> sFound;

                //迭代产生的内点个数
                const int np = vbInliers.size();

                for(int j=0; j<np; j++)
                {
                    //若为内点
                    if(vbInliers[j])
                    {
                        //构建当前图像帧的地图点
                        mCurrentFrame.mvpMapPoints[j]=vvpMapPointMatches[i][j];
                        //找到的地图点+1
                        sFound.insert(vvpMapPointMatches[i][j]);
                    }
                    else
                        mCurrentFrame.mvpMapPoints[j]=NULL;
                }
                
                //返回剔除坏点后的地图点个数
                int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                
                if(nGood<10)
                    continue;

                for(int io =0; io<mCurrentFrame.N; io++)
                    if(mCurrentFrame.mvbOutlier[io])
                        mCurrentFrame.mvpMapPoints[io]=static_cast<MapPoint*>(NULL);

                // If few inliers, search by projection in a coarse window and optimize again
                //如果内点个数小于50,则进行再次匹配
                if(nGood<50)
                {
                    //3D-2D匹配
                    //10：匹配时搜索范围，会乘以金字塔尺度
                    //100：匹配的ORB描述子距离应该小于这个阈值  
                    //投影法：和恒速模型一样 
                    int nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,10,100);

                    if(nadditional+nGood>=50)
                    {
                        nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points
                        if(nGood>30 && nGood<50)
                        {
                            sFound.clear();
                            for(int ip =0; ip<mCurrentFrame.N; ip++)
                                if(mCurrentFrame.mvpMapPoints[ip])
                                    sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                            nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,3,64);

                            // Final optimization
                            if(nGood+nadditional>=50)
                            {
                                nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                                for(int io =0; io<mCurrentFrame.N; io++)
                                    if(mCurrentFrame.mvbOutlier[io])
                                        mCurrentFrame.mvpMapPoints[io]=NULL;
                            }
                        }
                    }
                }


                // If the pose is supported by enough inliers stop ransacs and continue
                if(nGood>=50)
                {
                    bMatch = true;
                    break;
                }
            }
        }
    }

    if(!bMatch)
    {
        return false;
    }
    else
    {
        mnLastRelocFrameId = mCurrentFrame.mnId;
        cout << "Relocalized!!" << endl;
        return true;
    }

}

void Tracking::Reset(bool bLocMap)
{
    Verbose::PrintMess("System Reseting", Verbose::VERBOSITY_NORMAL);

    if(mpViewer)
    {
        mpViewer->RequestStop();
        while(!mpViewer->isStopped())
            usleep(3000);
    }

    // Reset Local Mapping
    if (!bLocMap)
    {
        Verbose::PrintMess("Reseting Local Mapper...", Verbose::VERBOSITY_NORMAL);
        mpLocalMapper->RequestReset();
        Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);
    }


    // Reset Loop Closing
    Verbose::PrintMess("Reseting Loop Closing...", Verbose::VERBOSITY_NORMAL);
    mpLoopClosing->RequestReset();
    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

    // Clear BoW Database
    Verbose::PrintMess("Reseting Database...", Verbose::VERBOSITY_NORMAL);
    mpKeyFrameDB->clear();
    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

    // Clear Map (this erase MapPoints and KeyFrames)
    mpAtlas->clearAtlas();
    mpAtlas->CreateNewMap();
    if (mSensor==System::IMU_STEREO || mSensor == System::IMU_MONOCULAR)
        mpAtlas->SetInertialSensor();
    mnInitialFrameId = 0;

    KeyFrame::nNextId = 0;
    Frame::nNextId = 0;
    mState = NO_IMAGES_YET;

    if(mpInitializer)
    {
        delete mpInitializer;
        mpInitializer = static_cast<Initializer*>(NULL);
    }
    mbSetInit=false;

    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();
    mCurrentFrame = Frame();
    mnLastRelocFrameId = 0;
    mLastFrame = Frame();
    mpReferenceKF = static_cast<KeyFrame*>(NULL);
    mpLastKeyFrame = static_cast<KeyFrame*>(NULL);
    mvIniMatches.clear();

    if(mpViewer)
        mpViewer->Release();

    Verbose::PrintMess("   End reseting! ", Verbose::VERBOSITY_NORMAL);
}

void Tracking::ResetActiveMap(bool bLocMap)
{
    Verbose::PrintMess("Active map Reseting", Verbose::VERBOSITY_NORMAL);
    if(mpViewer)
    {
        mpViewer->RequestStop();
        while(!mpViewer->isStopped())
            usleep(3000);
    }

    Map* pMap = mpAtlas->GetCurrentMap();

    if (!bLocMap)
    {
        Verbose::PrintMess("Reseting Local Mapper...", Verbose::VERBOSITY_NORMAL);
        mpLocalMapper->RequestResetActiveMap(pMap);
        Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);
    }

    // Reset Loop Closing
    Verbose::PrintMess("Reseting Loop Closing...", Verbose::VERBOSITY_NORMAL);
    mpLoopClosing->RequestResetActiveMap(pMap);
    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

    // Clear BoW Database
    Verbose::PrintMess("Reseting Database", Verbose::VERBOSITY_NORMAL);
    mpKeyFrameDB->clearMap(pMap); // Only clear the active map references
    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

    // Clear Map (this erase MapPoints and KeyFrames)
    mpAtlas->clearMap();


    //KeyFrame::nNextId = mpAtlas->GetLastInitKFid();
    //Frame::nNextId = mnLastInitFrameId;
    mnLastInitFrameId = Frame::nNextId;
    mnLastRelocFrameId = mnLastInitFrameId;
    mState = NO_IMAGES_YET; //NOT_INITIALIZED;

    if(mpInitializer)
    {
        delete mpInitializer;
        mpInitializer = static_cast<Initializer*>(NULL);
    }

    list<bool> lbLost;
    // lbLost.reserve(mlbLost.size());
    unsigned int index = mnFirstFrameId;
    cout << "mnFirstFrameId = " << mnFirstFrameId << endl;
    for(Map* pMap : mpAtlas->GetAllMaps())
    {
        if(pMap->GetAllKeyFrames().size() > 0)
        {
            if(index > pMap->GetLowerKFID())
                index = pMap->GetLowerKFID();
        }
    }

    //cout << "First Frame id: " << index << endl;
    int num_lost = 0;
    cout << "mnInitialFrameId = " << mnInitialFrameId << endl;

    for(list<bool>::iterator ilbL = mlbLost.begin(); ilbL != mlbLost.end(); ilbL++)
    {
        if(index < mnInitialFrameId)
            lbLost.push_back(*ilbL);
        else
        {
            lbLost.push_back(true);
            num_lost += 1;
        }

        index++;
    }
    cout << num_lost << " Frames had been set to lost" << endl;

    mlbLost = lbLost;

    mnInitialFrameId = mCurrentFrame.mnId;
    mnLastRelocFrameId = mCurrentFrame.mnId;

    mCurrentFrame = Frame();
    mLastFrame = Frame();
    mpReferenceKF = static_cast<KeyFrame*>(NULL);
    mpLastKeyFrame = static_cast<KeyFrame*>(NULL);
    mvIniMatches.clear();

    if(mpViewer)
        mpViewer->Release();

    Verbose::PrintMess("   End reseting! ", Verbose::VERBOSITY_NORMAL);
}

vector<MapPoint*> Tracking::GetLocalMapMPS()
{
    return mvpLocalMapPoints;
}

void Tracking::ChangeCalibration(const string &strSettingPath)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    Frame::mbInitialComputations = true;
}

void Tracking::InformOnlyTracking(const bool &flag)
{
    mbOnlyTracking = flag;
}

void Tracking::UpdateFrameIMU(const float s, const IMU::Bias &b, KeyFrame* pCurrentKeyFrame)
{
    Map * pMap = pCurrentKeyFrame->GetMap();
    unsigned int index = mnFirstFrameId;
    list<ORB_SLAM3::KeyFrame*>::iterator lRit = mlpReferences.begin();
    list<bool>::iterator lbL = mlbLost.begin();
    for(list<cv::Mat>::iterator lit=mlRelativeFramePoses.begin(),lend=mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lbL++)
    {
        if(*lbL)
            continue;

        KeyFrame* pKF = *lRit;

        while(pKF->isBad())
        {
            pKF = pKF->GetParent();
        }

        if(pKF->GetMap() == pMap)
        {
            (*lit).rowRange(0,3).col(3)=(*lit).rowRange(0,3).col(3)*s;
        }
    }

    mLastBias = b;

    mpLastKeyFrame = pCurrentKeyFrame;

    mLastFrame.SetNewBias(mLastBias);
    mCurrentFrame.SetNewBias(mLastBias);

    cv::Mat Gz = (cv::Mat_<float>(3,1) << 0, 0, -IMU::GRAVITY_VALUE);

    cv::Mat twb1;
    cv::Mat Rwb1;
    cv::Mat Vwb1;
    float t12;

    while(!mCurrentFrame.imuIsPreintegrated())
    {
        usleep(500);
    }


    if(mLastFrame.mnId == mLastFrame.mpLastKeyFrame->mnFrameId)
    {
        mLastFrame.SetImuPoseVelocity(mLastFrame.mpLastKeyFrame->GetImuRotation(),
                                      mLastFrame.mpLastKeyFrame->GetImuPosition(),
                                      mLastFrame.mpLastKeyFrame->GetVelocity());
    }
    else
    {
        twb1 = mLastFrame.mpLastKeyFrame->GetImuPosition();
        Rwb1 = mLastFrame.mpLastKeyFrame->GetImuRotation();
        Vwb1 = mLastFrame.mpLastKeyFrame->GetVelocity();
        t12 = mLastFrame.mpImuPreintegrated->dT;

        mLastFrame.SetImuPoseVelocity(Rwb1*mLastFrame.mpImuPreintegrated->GetUpdatedDeltaRotation(),
                                      twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1*mLastFrame.mpImuPreintegrated->GetUpdatedDeltaPosition(),
                                      Vwb1 + Gz*t12 + Rwb1*mLastFrame.mpImuPreintegrated->GetUpdatedDeltaVelocity());
    }

    if (mCurrentFrame.mpImuPreintegrated)
    {
        twb1 = mCurrentFrame.mpLastKeyFrame->GetImuPosition();
        Rwb1 = mCurrentFrame.mpLastKeyFrame->GetImuRotation();
        Vwb1 = mCurrentFrame.mpLastKeyFrame->GetVelocity();
        t12 = mCurrentFrame.mpImuPreintegrated->dT;

        mCurrentFrame.SetImuPoseVelocity(Rwb1*mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaRotation(),
                                      twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1*mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaPosition(),
                                      Vwb1 + Gz*t12 + Rwb1*mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaVelocity());
    }

    mnFirstImuFrameId = mCurrentFrame.mnId;
}


cv::Mat Tracking::ComputeF12(KeyFrame *&pKF1, KeyFrame *&pKF2)
{
    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();

    cv::Mat R12 = R1w*R2w.t();
    cv::Mat t12 = -R1w*R2w.t()*t2w+t1w;

    cv::Mat t12x = Converter::tocvSkewMatrix(t12);

    const cv::Mat &K1 = pKF1->mK;
    const cv::Mat &K2 = pKF2->mK;


    return K1.t().inv()*t12x*R12*K2.inv();
}


void Tracking::CreateNewMapPoints()
{
    // Retrieve neighbor keyframes in covisibility graph
    const vector<KeyFrame*> vpKFs = mpAtlas->GetAllKeyFrames();

    ORBmatcher matcher(0.6,false);

    cv::Mat Rcw1 = mpLastKeyFrame->GetRotation();
    cv::Mat Rwc1 = Rcw1.t();
    cv::Mat tcw1 = mpLastKeyFrame->GetTranslation();
    cv::Mat Tcw1(3,4,CV_32F);
    Rcw1.copyTo(Tcw1.colRange(0,3));
    tcw1.copyTo(Tcw1.col(3));
    cv::Mat Ow1 = mpLastKeyFrame->GetCameraCenter();

    const float &fx1 = mpLastKeyFrame->fx;
    const float &fy1 = mpLastKeyFrame->fy;
    const float &cx1 = mpLastKeyFrame->cx;
    const float &cy1 = mpLastKeyFrame->cy;
    const float &invfx1 = mpLastKeyFrame->invfx;
    const float &invfy1 = mpLastKeyFrame->invfy;

    const float ratioFactor = 1.5f*mpLastKeyFrame->mfScaleFactor;

    int nnew=0;

    // Search matches with epipolar restriction and triangulate
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF2 = vpKFs[i];
        if(pKF2==mpLastKeyFrame)
            continue;

        // Check first that baseline is not too short
        cv::Mat Ow2 = pKF2->GetCameraCenter();
        cv::Mat vBaseline = Ow2-Ow1;
        const float baseline = cv::norm(vBaseline);

        if((mSensor!=System::MONOCULAR)||(mSensor!=System::IMU_MONOCULAR))
        {
            if(baseline<pKF2->mb)
            continue;
        }
        else
        {
            const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
            const float ratioBaselineDepth = baseline/medianDepthKF2;

            if(ratioBaselineDepth<0.01)
                continue;
        }

        // Compute Fundamental Matrix
        cv::Mat F12 = ComputeF12(mpLastKeyFrame,pKF2);

        // Search matches that fullfil epipolar constraint
        vector<pair<size_t,size_t> > vMatchedIndices;
        matcher.SearchForTriangulation(mpLastKeyFrame,pKF2,F12,vMatchedIndices,false);

        cv::Mat Rcw2 = pKF2->GetRotation();
        cv::Mat Rwc2 = Rcw2.t();
        cv::Mat tcw2 = pKF2->GetTranslation();
        cv::Mat Tcw2(3,4,CV_32F);
        Rcw2.copyTo(Tcw2.colRange(0,3));
        tcw2.copyTo(Tcw2.col(3));

        const float &fx2 = pKF2->fx;
        const float &fy2 = pKF2->fy;
        const float &cx2 = pKF2->cx;
        const float &cy2 = pKF2->cy;
        const float &invfx2 = pKF2->invfx;
        const float &invfy2 = pKF2->invfy;

        // Triangulate each match
        const int nmatches = vMatchedIndices.size();
        for(int ikp=0; ikp<nmatches; ikp++)
        {
            const int &idx1 = vMatchedIndices[ikp].first;
            const int &idx2 = vMatchedIndices[ikp].second;

            const cv::KeyPoint &kp1 = mpLastKeyFrame->mvKeysUn[idx1];
            const float kp1_ur=mpLastKeyFrame->mvuRight[idx1];
            bool bStereo1 = kp1_ur>=0;

            const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];
            const float kp2_ur = pKF2->mvuRight[idx2];
            bool bStereo2 = kp2_ur>=0;

            // Check parallax between rays
            cv::Mat xn1 = (cv::Mat_<float>(3,1) << (kp1.pt.x-cx1)*invfx1, (kp1.pt.y-cy1)*invfy1, 1.0);
            cv::Mat xn2 = (cv::Mat_<float>(3,1) << (kp2.pt.x-cx2)*invfx2, (kp2.pt.y-cy2)*invfy2, 1.0);

            cv::Mat ray1 = Rwc1*xn1;
            cv::Mat ray2 = Rwc2*xn2;
            const float cosParallaxRays = ray1.dot(ray2)/(cv::norm(ray1)*cv::norm(ray2));

            float cosParallaxStereo = cosParallaxRays+1;
            float cosParallaxStereo1 = cosParallaxStereo;
            float cosParallaxStereo2 = cosParallaxStereo;

            if(bStereo1)
                cosParallaxStereo1 = cos(2*atan2(mpLastKeyFrame->mb/2,mpLastKeyFrame->mvDepth[idx1]));
            else if(bStereo2)
                cosParallaxStereo2 = cos(2*atan2(pKF2->mb/2,pKF2->mvDepth[idx2]));

            cosParallaxStereo = min(cosParallaxStereo1,cosParallaxStereo2);

            cv::Mat x3D;
            if(cosParallaxRays<cosParallaxStereo && cosParallaxRays>0 && (bStereo1 || bStereo2 || cosParallaxRays<0.9998))
            {
                // Linear Triangulation Method
                cv::Mat A(4,4,CV_32F);
                A.row(0) = xn1.at<float>(0)*Tcw1.row(2)-Tcw1.row(0);
                A.row(1) = xn1.at<float>(1)*Tcw1.row(2)-Tcw1.row(1);
                A.row(2) = xn2.at<float>(0)*Tcw2.row(2)-Tcw2.row(0);
                A.row(3) = xn2.at<float>(1)*Tcw2.row(2)-Tcw2.row(1);

                cv::Mat w,u,vt;
                cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);

                x3D = vt.row(3).t();

                if(x3D.at<float>(3)==0)
                    continue;

                // Euclidean coordinates
                x3D = x3D.rowRange(0,3)/x3D.at<float>(3);

            }
            else if(bStereo1 && cosParallaxStereo1<cosParallaxStereo2)
            {
                x3D = mpLastKeyFrame->UnprojectStereo(idx1);
            }
            else if(bStereo2 && cosParallaxStereo2<cosParallaxStereo1)
            {
                x3D = pKF2->UnprojectStereo(idx2);
            }
            else
                continue; //No stereo and very low parallax

            cv::Mat x3Dt = x3D.t();

            //Check triangulation in front of cameras
            float z1 = Rcw1.row(2).dot(x3Dt)+tcw1.at<float>(2);
            if(z1<=0)
                continue;

            float z2 = Rcw2.row(2).dot(x3Dt)+tcw2.at<float>(2);
            if(z2<=0)
                continue;

            //Check reprojection error in first keyframe
            const float &sigmaSquare1 = mpLastKeyFrame->mvLevelSigma2[kp1.octave];
            const float x1 = Rcw1.row(0).dot(x3Dt)+tcw1.at<float>(0);
            const float y1 = Rcw1.row(1).dot(x3Dt)+tcw1.at<float>(1);
            const float invz1 = 1.0/z1;

            if(!bStereo1)
            {
                float u1 = fx1*x1*invz1+cx1;
                float v1 = fy1*y1*invz1+cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                if((errX1*errX1+errY1*errY1)>5.991*sigmaSquare1)
                    continue;
            }
            else
            {
                float u1 = fx1*x1*invz1+cx1;
                float u1_r = u1 - mpLastKeyFrame->mbf*invz1;
                float v1 = fy1*y1*invz1+cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                float errX1_r = u1_r - kp1_ur;
                if((errX1*errX1+errY1*errY1+errX1_r*errX1_r)>7.8*sigmaSquare1)
                    continue;
            }

            //Check reprojection error in second keyframe
            const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];
            const float x2 = Rcw2.row(0).dot(x3Dt)+tcw2.at<float>(0);
            const float y2 = Rcw2.row(1).dot(x3Dt)+tcw2.at<float>(1);
            const float invz2 = 1.0/z2;
            if(!bStereo2)
            {
                float u2 = fx2*x2*invz2+cx2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                if((errX2*errX2+errY2*errY2)>5.991*sigmaSquare2)
                    continue;
            }
            else
            {
                float u2 = fx2*x2*invz2+cx2;
                float u2_r = u2 - mpLastKeyFrame->mbf*invz2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                float errX2_r = u2_r - kp2_ur;
                if((errX2*errX2+errY2*errY2+errX2_r*errX2_r)>7.8*sigmaSquare2)
                    continue;
            }

            //Check scale consistency
            cv::Mat normal1 = x3D-Ow1;
            float dist1 = cv::norm(normal1);

            cv::Mat normal2 = x3D-Ow2;
            float dist2 = cv::norm(normal2);

            if(dist1==0 || dist2==0)
                continue;

            const float ratioDist = dist2/dist1;
            const float ratioOctave = mpLastKeyFrame->mvScaleFactors[kp1.octave]/pKF2->mvScaleFactors[kp2.octave];

            if(ratioDist*ratioFactor<ratioOctave || ratioDist>ratioOctave*ratioFactor)
                continue;

            // Triangulation is succesfull
            MapPoint* pMP = new MapPoint(x3D,mpLastKeyFrame,mpAtlas->GetCurrentMap());

            pMP->AddObservation(mpLastKeyFrame,idx1);
            pMP->AddObservation(pKF2,idx2);

            mpLastKeyFrame->AddMapPoint(pMP,idx1);
            pKF2->AddMapPoint(pMP,idx2);

            pMP->ComputeDistinctiveDescriptors();

            pMP->UpdateNormalAndDepth();

            mpAtlas->AddMapPoint(pMP);
            nnew++;
        }
    }
    TrackReferenceKeyFrame();
}

void Tracking::NewDataset()
{
    mnNumDataset++;
}

int Tracking::GetNumberDataset()
{
    return mnNumDataset;
}

int Tracking::GetMatchesInliers()
{
    return mnMatchesInliers;
}

} //namespace ORB_SLAM
