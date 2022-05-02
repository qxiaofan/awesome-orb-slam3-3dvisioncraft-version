/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2020 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/


#include "LocalMapping.h"
#include "LoopClosing.h"
#include "ORBmatcher.h"
#include "Optimizer.h"
#include "Converter.h"

#include<mutex>
#include<chrono>

namespace ORB_SLAM3
{
//LocalMapping类的构造函数
/********************************************************
    pSys: System类当前SLAM系统 
    pAtlas：Atlas类指针
    bMonocular，bInertial:传感器类型单目IMU
    strSequence
********************************************************/
/****************初始化**********************************
    mbResetRequested：   请求当前线程复位的标志。true，表示一直请求复位，但复位还未完成；表示复位完成为false
    mbResetRequestedActiveMap ：初始化为false
    mbFinishRequested：请求终止当前线程的标志，初始化为false。只是请求，不一定终止，终止要看mbFinished
    mbFinished: 判断最终LocalMapping::Run() 是否完成的标志。初始化为true
    bInitializing：
    mbAbortBA：是否流产BA优化的标志位
    mbStopped：为true表示可以并终止localmapping 线程，初始化为false
    mbStopRequested：外部线程调用，为true，表示外部线程请求停止局部建图线程，初始化为false
    mbNotStop：true，表示不要停止 localmapping 线程，因为要插入关键帧了。需要和 mbStopped 结合使用
    mbAcceptKeyFrames：  true，允许接受关键帧。tracking 和local mapping 之间的关键帧调度
    mbNewInit：初始化为false
    mIdxInit:初始化为0
    mScale：初始化为1
    mInitSect：初始化为0
    mbNotBA1,mbNotBA2：初始化为true
    mIdxIteration:初始化为0
    infoInertial：IMU相关9×9矩阵     
********************************************************/
LocalMapping::LocalMapping(System* pSys, Atlas *pAtlas, const float bMonocular, bool bInertial, const string &_strSeqName):
    mpSystem(pSys), mbMonocular(bMonocular), mbInertial(bInertial), mbResetRequested(false), 
    mbResetRequestedActiveMap(false), mbFinishRequested(false), mbFinished(true), 
    mpAtlas(pAtlas), bInitializing(false),mbAbortBA(false), mbStopped(false), 
    mbStopRequested(false), mbNotStop(false), mbAcceptKeyFrames(true),
    mbNewInit(false), mIdxInit(0), mScale(1.0), mInitSect(0), mbNotBA1(true),
    mbNotBA2(true), mIdxIteration(0), infoInertial(Eigen::MatrixXd::Zero(9,9))
{
    mnMatchesInliers = 0;

    //IMU状态 为false表示正常，为true表示不正常
    mbBadImu = false;

    mTinit = 0.f;

    mNumLM = 0;
    mNumKFCulling=0;

    //DEBUG: times and data from LocalMapping in each frame

    strSequence = "";//_strSeqName;

    //f_lm.open("localMapping_times" + strSequence + ".txt");
    /*f_lm.open("localMapping_times.txt");

    f_lm << "# Timestamp KF, Num CovKFs, Num KFs, Num RecentMPs, Num MPs, processKF, MPCulling, CreateMP, SearchNeigh, BA, KFCulling, [numFixKF_LBA]" << endl;
    f_lm << fixed;*/
}

void LocalMapping::SetLoopCloser(LoopClosing* pLoopCloser)
{
    mpLoopCloser = pLoopCloser;
}

void LocalMapping::SetTracker(Tracking *pTracker)
{
    mpTracker=pTracker;
}

void LocalMapping::Run()
{

    //初始化完成标志位为false
    mbFinished = false;

    while(1)
    {
        // Tracking will see that Local Mapping is busy
        //设置关键帧接收标志位为false，表示不接收关键帧
        SetAcceptKeyFrames(false);

        // Check if there are keyframes in the queue
        //查看关键帧列表和IMU状态，若关键帧列表不为空且IMU状态正常
        if(CheckNewKeyFrames() && !mbBadImu)
        {
            // std::cout << "LM" << std::endl;
            std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();

            // BoW conversion and insertion in Map
            //处理列表中的关键帧
            ProcessNewKeyFrame();
            std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

            // Check recent MapPoints
            //检查mlpRecentAddedMapPoints中的地图点，并将不好的剔除，保留需要继续进行检测的点
            //（mlpRecentAddedMapPoints在processNetKeyFrame中进行更新）
            MapPointCulling();
            std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

            // Triangulate new MapPoints
            //通过三角化恢复新的地图点
            CreateNewMapPoints();
            std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();

            // Save here:
            // # Cov KFs
            // # tot Kfs
            // # recent added MPs
            // # tot MPs
            // # localMPs in LBA
            // # fixedKFs in LBA

            mbAbortBA = false;

            //如果mlNewKeyFrames列表为空了
            if(!CheckNewKeyFrames())
            {   

                //对于初始阶段，列表中有初始关键帧和当前关键帧，对这两个关键帧处理后，才
                //进行融合操作，由于接下来的BA优化只对列表中的最新关键帧进行优化，所以在
                //融合算法中只对最新的关键帧进行地图点融合


                // Find more matches in neighbor keyframes and fuse point duplications
                //检查并融合当前关键帧与相邻关键帧中重复的地图点
                SearchInNeighbors();
            }

            std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
            std::chrono::steady_clock::time_point t5 = t4, t6 = t4;
            // mbAbortBA = false;

            //DEBUG--
            /*f_lm << setprecision(0);
            f_lm << mpCurrentKeyFrame->mTimeStamp*1e9 << ",";
            f_lm << mpCurrentKeyFrame->GetVectorCovisibleKeyFrames().size() << ",";
            f_lm << mpCurrentKeyFrame->GetMap()->GetAllKeyFrames().size() << ",";
            f_lm << mlpRecentAddedMapPoints.size() << ",";
            f_lm << mpCurrentKeyFrame->GetMap()->GetAllMapPoints().size() << ",";*/
            //--
            int num_FixedKF_BA = 0;

            //若mlNewKeyFrames列表为空，且回环检测没有请求停止局部建图线程（mbStopRequested标志位=false）
            if(!CheckNewKeyFrames() && !stopRequested())
            {
                //地图中关键帧的数量>2
                if(mpAtlas->KeyFramesInMap()>2)
                {
                    //若是IMU模式且当前关键帧所在的地图已完成初始化
                    if(mbInertial && mpCurrentKeyFrame->GetMap()->isImuInitialized())
                    {
                        float dist = cv::norm(mpCurrentKeyFrame->mPrevKF->GetCameraCenter() - mpCurrentKeyFrame->GetCameraCenter()) +
                                cv::norm(mpCurrentKeyFrame->mPrevKF->mPrevKF->GetCameraCenter() - mpCurrentKeyFrame->mPrevKF->GetCameraCenter());

                        if(dist>0.05)
                            mTinit += mpCurrentKeyFrame->mTimeStamp - mpCurrentKeyFrame->mPrevKF->mTimeStamp;
                        if(!mpCurrentKeyFrame->GetMap()->GetIniertialBA2())
                        {
                            if((mTinit<10.f) && (dist<0.02))
                            {
                                cout << "Not enough motion for initializing. Reseting..." << endl;
                                unique_lock<mutex> lock(mMutexReset);
                                mbResetRequestedActiveMap = true;
                                mpMapToReset = mpCurrentKeyFrame->GetMap();
                                mbBadImu = true;
                            }
                        }

                        bool bLarge = ((mpTracker->GetMatchesInliers()>75)&&mbMonocular)||((mpTracker->GetMatchesInliers()>100)&&!mbMonocular);
                        Optimizer::LocalInertialBA(mpCurrentKeyFrame, &mbAbortBA, mpCurrentKeyFrame->GetMap(), bLarge, !mpCurrentKeyFrame->GetMap()->GetIniertialBA2());
                    }
                    else   //否则，进入局部优化
                    {   cout<<"      局部BA优化        "<<endl;
                        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
                        Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame,&mbAbortBA, mpCurrentKeyFrame->GetMap(),num_FixedKF_BA);
                        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
                    }
                }

                t5 = std::chrono::steady_clock::now();

                // Initialize IMU here
                //若IMU模式且还没进行初始化，则进行初始化
                if(!mpCurrentKeyFrame->GetMap()->isImuInitialized() && mbInertial)
                {
                    if (mbMonocular)
                        InitializeIMU(1e2, 1e10, true);
                    else
                        InitializeIMU(1e2, 1e5, true);
                }


                // Check redundant local Keyframes
                //删除冗余关键帧
                KeyFrameCulling();

                t6 = std::chrono::steady_clock::now();

                //如果是IMU模式，且初始化成功时间较短
                if ((mTinit<100.0f) && mbInertial)
                {
                    if(mpCurrentKeyFrame->GetMap()->isImuInitialized() && mpTracker->mState==Tracking::OK) // Enter here everytime local-mapping is called
                    {
                        if(!mpCurrentKeyFrame->GetMap()->GetIniertialBA1())
                        {
                            if (mTinit>5.0f)
                            {
                                cout << "start VIBA 1" << endl;
                                mpCurrentKeyFrame->GetMap()->SetIniertialBA1();
                                if (mbMonocular)
                                    InitializeIMU(1.f, 1e5, true); // 1.f, 1e5
                                else
                                    InitializeIMU(1.f, 1e5, true); // 1.f, 1e5

                                cout << "end VIBA 1" << endl;
                            }
                        }
                        //else if (mbNotBA2){
                        else if(!mpCurrentKeyFrame->GetMap()->GetIniertialBA2())
                        {
                            if (mTinit>15.0f)
                            { // 15.0f
                                cout << "start VIBA 2" << endl;
                                mpCurrentKeyFrame->GetMap()->SetIniertialBA2();
                                if (mbMonocular)
                                    InitializeIMU(0.f, 0.f, true); // 0.f, 0.f
                                else
                                    InitializeIMU(0.f, 0.f, true);

                                cout << "end VIBA 2" << endl;
                            }
                        }

                        // scale refinement
                        if (((mpAtlas->KeyFramesInMap())<=100) &&
                                ((mTinit>25.0f && mTinit<25.5f)||
                                (mTinit>35.0f && mTinit<35.5f)||
                                (mTinit>45.0f && mTinit<45.5f)||
                                (mTinit>55.0f && mTinit<55.5f)||
                                (mTinit>65.0f && mTinit<65.5f)||
                                (mTinit>75.0f && mTinit<75.5f)))
                        {
                            cout << "start scale ref" << endl;
                            if (mbMonocular)
                                ScaleRefinement();
                            cout << "end scale ref" << endl;
                        }
                    }
                }
            }

            std::chrono::steady_clock::time_point t7 = std::chrono::steady_clock::now();

            mpLoopCloser->InsertKeyFrame(mpCurrentKeyFrame);
            std::chrono::steady_clock::time_point t8 = std::chrono::steady_clock::now();

            double t_procKF = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t1 - t0).count();
            double t_MPcull = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t2 - t1).count();
            double t_CheckMP = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t3 - t2).count();
            double t_searchNeigh = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t4 - t3).count();
            double t_Opt = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t5 - t4).count();
            double t_KF_cull = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t6 - t5).count();
            double t_Insert = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t8 - t7).count();

            //DEBUG--
            /*f_lm << setprecision(6);
            f_lm << t_procKF << ",";
            f_lm << t_MPcull << ",";
            f_lm << t_CheckMP << ",";
            f_lm << t_searchNeigh << ",";
            f_lm << t_Opt << ",";
            f_lm << t_KF_cull << ",";
            f_lm << setprecision(0) << num_FixedKF_BA << "\n";*/
            //--

        }
        //若mbStopRequestedw为正且mbNotStop为false，且IMU状态正常
        else if(Stop() && !mbBadImu)
        {
            // Safe area to stop
            while(isStopped() && !CheckFinish())
            {
                // cout << "LM: usleep if is stopped" << endl;
                usleep(3000);
            }
            if(CheckFinish())
                break;
        }

        // 查看是否有复位线程的请求
        ResetIfRequested();

        // Tracking will see that Local Mapping is busy
        //设置关键帧接收标志位为true，表示接收关键帧
        SetAcceptKeyFrames(true);

        //若当前线程结束，跳出主循环
        if(CheckFinish())
            break;

        // cout << "LM: normal usleep" << endl;
        usleep(3000);
    }

    //f_lm.close();

    SetFinish();
}

void LocalMapping::InsertKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexNewKFs);
    mlNewKeyFrames.push_back(pKF);
    mbAbortBA=true;
}


bool LocalMapping::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexNewKFs);
    return(!mlNewKeyFrames.empty());
}


/***********************处理列表中的关键帧********************
 * 对关键帧列表中新插入的关键帧进行处理，更新地图点的平均观测方向
 * 等，若是当前关键帧新生成的地图点，存入mlpRecentAddedMapPoints
 * 中等待下一步核验 
 * 在Tracking线程初始化时，添加了初始关键帧和当前关键帧两个，list先入先出原则
**********************************************************/
void LocalMapping::ProcessNewKeyFrame()
{
    //cout << "ProcessNewKeyFrame: " << mlNewKeyFrames.size() << endl;
    {
        unique_lock<mutex> lock(mMutexNewKFs);
        //从关键帧队列中选出队列中的第一个关键帧作为当前关键帧
        mpCurrentKeyFrame = mlNewKeyFrames.front();  
        //关键帧列表剔除第一个关键帧
        mlNewKeyFrames.pop_front();
    }

    // Compute Bags of Words structures
    //计算当前关键帧的词袋模型
    mpCurrentKeyFrame->ComputeBoW();

    // Associate MapPoints to the new keyframe and update normal and descriptor
    //提取和当前关键帧成功匹配的地图点
    //返回KeyFrame的mvpMapPoints，数目与特征点数量一致，若特征点未成功生成地图点，则为空
    const vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();

    //遍历当前关键帧的地图点
    for(size_t i=0; i<vpMapPointMatches.size(); i++)
    {
        //提取当前地图点
        MapPoint* pMP = vpMapPointMatches[i];
        //若地图点存在
        if(pMP)
        {
            //若地图点不是坏点
            if(!pMP->isBad())
            {   
                //pMP->IsInKeyFrame(mpCurrentKeyFrame)是可以检测当前地图点的观测中是否存在当前关键帧
                //若存在当前关键帧，返回1,不存在返回0.
                //按理来说，应该是存在当前观测的，因为是当前图像帧中提取出的特征点，但以防万一，如果没有，
                //添加这个关键帧观测
                if(!pMP->IsInKeyFrame(mpCurrentKeyFrame))
                {
                    //地图点添加观测三部曲：添加-更新深度-更新描述子
                    //为此地图点添加当前关键帧作为观测
                    pMP->AddObservation(mpCurrentKeyFrame, i);
                    //更新此地图点的平均观测方向和深度
                    pMP->UpdateNormalAndDepth();
                    //更新地图点的最佳描述子
                    pMP->ComputeDistinctiveDescriptors();
                }
                //若此地图点是来自当前关键帧
                else // this can only happen for new stereo points inserted by the Tracking
                {
                    //暂存入近期增加的地图点集合中，等待后续查验
                    mlpRecentAddedMapPoints.push_back(pMP);
                }
            }
        }
    
    }
    
    // Update links in the Covisibility Graph
    //更新共视图
    mpCurrentKeyFrame->UpdateConnections();

    // Insert Keyframe in Map
    //将此关键帧加入当前地图中
    mpAtlas->AddKeyFrame(mpCurrentKeyFrame);
}

void LocalMapping::EmptyQueue()
{
    while(CheckNewKeyFrames())
        ProcessNewKeyFrame();
}

/***********************新增地图点核验与剔除*************
 * 对mlpRecentAddedMapPoints中的地图点进行检测和剔除
 * 保留需要进行进一步检测的点
 * 剔除地图点的依据：
 * 1、IncreaseFound/IncreaseVisible < 0.25
 * 2、观测到该点的关键帧过少
************************************************/
void LocalMapping::MapPointCulling()
{   
    // Check Recent Added MapPoints
    //新增地图点迭代器
    list<MapPoint*>::iterator lit = mlpRecentAddedMapPoints.begin();
    
    //当前关键帧的关键帧ID（此关键帧生成的这些地图点）
    const unsigned long int nCurrentKFid = mpCurrentKeyFrame->mnId;

    //观测阈值设定
    int nThObs;
    if(mbMonocular)  //单目
        nThObs = 2;
    else
        nThObs = 3; // MODIFICATION_STEREO_IMU here 3
    //阈值
    const int cnThObs = nThObs;
    
    //新增地图点数目大小
    int borrar = mlpRecentAddedMapPoints.size();

    //遍历所有的新增地图点
    while(lit!=mlpRecentAddedMapPoints.end())
    {   
        //地图点指针
        MapPoint* pMP = *lit;

        //若是坏点，剔除
        if(pMP->isBad())
            lit = mlpRecentAddedMapPoints.erase(lit);
        // 跟踪到该地图点的图像帧数与可观测到该地图点的帧数之比小于25%  
        else if(pMP->GetFoundRatio()<0.25f) //(被找到/被看到)
        {   
            //消除此地图点的观测，并设置此地图点是坏点
            pMP->SetBadFlag();  
            //剔除
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        //若该地图点已在关键帧中被超过2个关键帧跟踪但观测到此地图点的关键帧数目没有达到阈值
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=2 && pMP->Observations()<=cnThObs)
        {
            //消除此地图点的观测，并设置此地图点是坏点
            pMP->SetBadFlag();
            //剔除
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        //若该地图点以被超过3个关键帧跟踪，则不再对其进行检测
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=3)
        {   //剔除
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        //
        else
        {
            lit++;
            borrar--;
        }
    }
    //cout << "erase MP: " << borrar << endl;
}


/**********************通过三角化生成新的地图点********************
 * 对当前处理的关键帧提取共视关键帧，对当前关键帧和共视或周围关键帧
 * 进行特征匹配，构造更多地图点
**************************************************************/
void LocalMapping::CreateNewMapPoints()
{   
    // Retrieve neighbor keyframes in covisibility graph
    //nn:搜索的最佳共视关键帧的数目
    int nn = 10;
    // For stereo inertial case
    //对单目的情况，需要更多的共视关键帧
    if(mbMonocular)
        nn=20;

    //在当前关键帧的共视关键帧中提取共视程度最高的nn个关键帧
    vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);

    //如果是IMU模式，则添加更多的关键帧存入附近关键帧
    if (mbInertial)
    {   
        //当前关键帧
        KeyFrame* pKF = mpCurrentKeyFrame;
        int count=0;
        //若上一步提取的关键帧数量没有达到要求且当前关键帧存在前一关键帧且添加的帧数没有超过阈值
        while((vpNeighKFs.size()<=nn)&&(pKF->mPrevKF)&&(count++<nn))
        {   
            //在当前附近关键帧向量中找当前关键帧的前一关键帧
            vector<KeyFrame*>::iterator it = std::find(vpNeighKFs.begin(), vpNeighKFs.end(), pKF->mPrevKF);
            //若成立，表示从头找到尾都没找到当前关键帧的前一关键帧，则将前一关键帧加入附近关键帧向量中
            if(it==vpNeighKFs.end())
                vpNeighKFs.push_back(pKF->mPrevKF);
            //更新关键帧
            pKF = pKF->mPrevKF;
        }
    }


    //匹配阈值
    float th = 0.6f;

    //匹配中不进行方向检测
    ORBmatcher matcher(th,false);

    //世界坐标系到当前关键帧坐标系的旋转
    cv::Mat Rcw1 = mpCurrentKeyFrame->GetRotation();
    //当前关键帧坐标系到世界坐标系的旋转
    cv::Mat Rwc1 = Rcw1.t();
    //当前关键帧坐标系下，世界坐标系的位置
    cv::Mat tcw1 = mpCurrentKeyFrame->GetTranslation();
    //世界坐标系到当前关键帧坐标系的位姿
    cv::Mat Tcw1(3,4,CV_32F);
    Rcw1.copyTo(Tcw1.colRange(0,3));
    tcw1.copyTo(Tcw1.col(3));
    //当前关键帧的光心在世界坐标系下的位置
    cv::Mat Ow1 = mpCurrentKeyFrame->GetCameraCenter();

    //内参
    const float &fx1 = mpCurrentKeyFrame->fx;
    const float &fy1 = mpCurrentKeyFrame->fy;
    const float &cx1 = mpCurrentKeyFrame->cx;
    const float &cy1 = mpCurrentKeyFrame->cy;
    const float &invfx1 = mpCurrentKeyFrame->invfx;
    const float &invfy1 = mpCurrentKeyFrame->invfy;

    //尺度因子，用于后面的点深度的验证，1.5是经验值
    const float ratioFactor = 1.5f*mpCurrentKeyFrame->mfScaleFactor;

    // Search matches with epipolar restriction and triangulate
    //遍历周围的关键帧
    for(size_t i=0; i<vpNeighKFs.size(); i++)
    {   
        //若当前待处理关键帧数量不为空，返回，先处理关键帧
        if(i>0 && CheckNewKeyFrames())// && (mnMatchesInliers>50))
            return;
        
        //周围关键帧
        KeyFrame* pKF2 = vpNeighKFs[i];

        //构造相机模型
        GeometricCamera* pCamera1 = mpCurrentKeyFrame->mpCamera, *pCamera2 = pKF2->mpCamera;

        // Check first that baseline is not too short
        //获得此周围关键帧在世界坐标系下相机光心的位置
        cv::Mat Ow2 = pKF2->GetCameraCenter();
        //基线向量，此周围关键帧的光心和当前关键帧的光心在世界坐标系下的连线
        cv::Mat vBaseline = Ow2-Ow1;
        //基线长度
        const float baseline = cv::norm(vBaseline);

        //根据基线长度等信息判断是否跳过此关键帧匹配
        //如果不是单目
        if(!mbMonocular)
        {
            if(baseline<pKF2->mb)
            continue;
        }
        //如果是单目情况
        else
        {   
            //计算周围关键帧的场景深度
            const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
            //基线长度/景深
            const float ratioBaselineDepth = baseline/medianDepthKF2;

            //若比例过小，说明两个关键帧距离太近，恢复的3D点不准，跳过此关键帧
            if(ratioBaselineDepth<0.01)
                continue;
        }

        // Compute Fundamental Matrix
        //计算当前关键帧和此周围关键帧之间的基础矩阵F
        cv::Mat F12 = ComputeF12(mpCurrentKeyFrame,pKF2);
        

        // Search matches that fullfil epipolar constraint
        //存储匹配关系的向量
        vector<pair<size_t,size_t> > vMatchedIndices;
        

        /**********同时满足以下条件，则bCoarse标志位为true*************************
         * 1、有IMU
         * 2、[当前关键帧的地图没有进行惯性BA2但进行了惯性BA1]或当前Tracking的状态是近期丢失 
        *************************************************************************/
        bool bCoarse = mbInertial &&
                ((!mpCurrentKeyFrame->GetMap()->GetIniertialBA2() && mpCurrentKeyFrame->GetMap()->GetIniertialBA1())||
                 mpTracker->mState==Tracking::RECENTLY_LOST);
        
        //当前关键帧和此周围关键帧进行特征匹配
        //vMatchedIndices  fisrt表示当前关键帧pKF1中特征点的ID
        //                 second表示周围关键帧pKF2中特征点ID
        //vMatchedIndices.size() = nmatchs
        matcher.SearchForTriangulation(mpCurrentKeyFrame,pKF2,F12,vMatchedIndices,false,bCoarse);

        //世界坐标系到周围关键帧坐标系的旋转
        cv::Mat Rcw2 = pKF2->GetRotation();
        cv::Mat Rwc2 = Rcw2.t();
        //周围关键帧坐标系下，当前世界坐标系的位置
        cv::Mat tcw2 = pKF2->GetTranslation();
        //世界坐标系到周围关键帧坐标系的姿态
        cv::Mat Tcw2(3,4,CV_32F);
        Rcw2.copyTo(Tcw2.colRange(0,3));
        tcw2.copyTo(Tcw2.col(3));

        //内参
        const float &fx2 = pKF2->fx;
        const float &fy2 = pKF2->fy;
        const float &cx2 = pKF2->cx;
        const float &cy2 = pKF2->cy;
        const float &invfx2 = pKF2->invfx;
        const float &invfy2 = pKF2->invfy;

        //对每对匹配成功的点生成三角化
        // Triangulate each match
        const int nmatches = vMatchedIndices.size();
        for(int ikp=0; ikp<nmatches; ikp++)
        {
            const int &idx1 = vMatchedIndices[ikp].first;
            const int &idx2 = vMatchedIndices[ikp].second;

            const cv::KeyPoint &kp1 = (mpCurrentKeyFrame -> NLeft == -1) ? mpCurrentKeyFrame->mvKeysUn[idx1]
                                                                         : (idx1 < mpCurrentKeyFrame -> NLeft) ? mpCurrentKeyFrame -> mvKeys[idx1]
                                                                                                               : mpCurrentKeyFrame -> mvKeysRight[idx1 - mpCurrentKeyFrame -> NLeft];
            const float kp1_ur=mpCurrentKeyFrame->mvuRight[idx1];
            bool bStereo1 = (!mpCurrentKeyFrame->mpCamera2 && kp1_ur>=0);
            const bool bRight1 = (mpCurrentKeyFrame -> NLeft == -1 || idx1 < mpCurrentKeyFrame -> NLeft) ? false
                                                                               : true;

            const cv::KeyPoint &kp2 = (pKF2 -> NLeft == -1) ? pKF2->mvKeysUn[idx2]
                                                            : (idx2 < pKF2 -> NLeft) ? pKF2 -> mvKeys[idx2]
                                                                                     : pKF2 -> mvKeysRight[idx2 - pKF2 -> NLeft];

            const float kp2_ur = pKF2->mvuRight[idx2];
            bool bStereo2 = (!pKF2->mpCamera2 && kp2_ur>=0);
            const bool bRight2 = (pKF2 -> NLeft == -1 || idx2 < pKF2 -> NLeft) ? false
                                                                               : true;

            if(mpCurrentKeyFrame->mpCamera2 && pKF2->mpCamera2)
            {
                if(bRight1 && bRight2)
                {
                    Rcw1 = mpCurrentKeyFrame->GetRightRotation();
                    Rwc1 = Rcw1.t();
                    tcw1 = mpCurrentKeyFrame->GetRightTranslation();
                    Tcw1 = mpCurrentKeyFrame->GetRightPose();
                    Ow1 = mpCurrentKeyFrame->GetRightCameraCenter();

                    Rcw2 = pKF2->GetRightRotation();
                    Rwc2 = Rcw2.t();
                    tcw2 = pKF2->GetRightTranslation();
                    Tcw2 = pKF2->GetRightPose();
                    Ow2 = pKF2->GetRightCameraCenter();

                    pCamera1 = mpCurrentKeyFrame->mpCamera2;
                    pCamera2 = pKF2->mpCamera2;
                }
                else if(bRight1 && !bRight2)
                {
                    Rcw1 = mpCurrentKeyFrame->GetRightRotation();
                    Rwc1 = Rcw1.t();
                    tcw1 = mpCurrentKeyFrame->GetRightTranslation();
                    Tcw1 = mpCurrentKeyFrame->GetRightPose();
                    Ow1 = mpCurrentKeyFrame->GetRightCameraCenter();

                    Rcw2 = pKF2->GetRotation();
                    Rwc2 = Rcw2.t();
                    tcw2 = pKF2->GetTranslation();
                    Tcw2 = pKF2->GetPose();
                    Ow2 = pKF2->GetCameraCenter();

                    pCamera1 = mpCurrentKeyFrame->mpCamera2;
                    pCamera2 = pKF2->mpCamera;
                }
                else if(!bRight1 && bRight2)
                {
                    Rcw1 = mpCurrentKeyFrame->GetRotation();
                    Rwc1 = Rcw1.t();
                    tcw1 = mpCurrentKeyFrame->GetTranslation();
                    Tcw1 = mpCurrentKeyFrame->GetPose();
                    Ow1 = mpCurrentKeyFrame->GetCameraCenter();

                    Rcw2 = pKF2->GetRightRotation();
                    Rwc2 = Rcw2.t();
                    tcw2 = pKF2->GetRightTranslation();
                    Tcw2 = pKF2->GetRightPose();
                    Ow2 = pKF2->GetRightCameraCenter();

                    pCamera1 = mpCurrentKeyFrame->mpCamera;
                    pCamera2 = pKF2->mpCamera2;
                }
                else
                {
                    Rcw1 = mpCurrentKeyFrame->GetRotation();
                    Rwc1 = Rcw1.t();
                    tcw1 = mpCurrentKeyFrame->GetTranslation();
                    Tcw1 = mpCurrentKeyFrame->GetPose();
                    Ow1 = mpCurrentKeyFrame->GetCameraCenter();

                    Rcw2 = pKF2->GetRotation();
                    Rwc2 = Rcw2.t();
                    tcw2 = pKF2->GetTranslation();
                    Tcw2 = pKF2->GetPose();
                    Ow2 = pKF2->GetCameraCenter();

                    pCamera1 = mpCurrentKeyFrame->mpCamera;
                    pCamera2 = pKF2->mpCamera;
                }
            }

            // Check parallax between rays
            cv::Mat xn1 = pCamera1->unprojectMat(kp1.pt);
            cv::Mat xn2 = pCamera2->unprojectMat(kp2.pt);

            cv::Mat ray1 = Rwc1*xn1;
            cv::Mat ray2 = Rwc2*xn2;
            const float cosParallaxRays = ray1.dot(ray2)/(cv::norm(ray1)*cv::norm(ray2));

            float cosParallaxStereo = cosParallaxRays+1;
            float cosParallaxStereo1 = cosParallaxStereo;
            float cosParallaxStereo2 = cosParallaxStereo;

            if(bStereo1)
                cosParallaxStereo1 = cos(2*atan2(mpCurrentKeyFrame->mb/2,mpCurrentKeyFrame->mvDepth[idx1]));
            else if(bStereo2)
                cosParallaxStereo2 = cos(2*atan2(pKF2->mb/2,pKF2->mvDepth[idx2]));

            cosParallaxStereo = min(cosParallaxStereo1,cosParallaxStereo2);

            cv::Mat x3D;
            if(cosParallaxRays<cosParallaxStereo && cosParallaxRays>0 && (bStereo1 || bStereo2 ||
               (cosParallaxRays<0.9998 && mbInertial) || (cosParallaxRays<0.9998 && !mbInertial)))
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
                x3D = mpCurrentKeyFrame->UnprojectStereo(idx1);
            }
            else if(bStereo2 && cosParallaxStereo2<cosParallaxStereo1)
            {
                x3D = pKF2->UnprojectStereo(idx2);
            }
            else
            {
                continue; //No stereo and very low parallax
            }

            cv::Mat x3Dt = x3D.t();

            if(x3Dt.empty()) continue;
            //Check triangulation in front of cameras
            float z1 = Rcw1.row(2).dot(x3Dt)+tcw1.at<float>(2);
            if(z1<=0)
                continue;

            float z2 = Rcw2.row(2).dot(x3Dt)+tcw2.at<float>(2);
            if(z2<=0)
                continue;

            //Check reprojection error in first keyframe
            const float &sigmaSquare1 = mpCurrentKeyFrame->mvLevelSigma2[kp1.octave];
            const float x1 = Rcw1.row(0).dot(x3Dt)+tcw1.at<float>(0);
            const float y1 = Rcw1.row(1).dot(x3Dt)+tcw1.at<float>(1);
            const float invz1 = 1.0/z1;

            if(!bStereo1)
            {
                cv::Point2f uv1 = pCamera1->project(cv::Point3f(x1,y1,z1));
                float errX1 = uv1.x - kp1.pt.x;
                float errY1 = uv1.y - kp1.pt.y;

                if((errX1*errX1+errY1*errY1)>5.991*sigmaSquare1)
                    continue;

            }
            else
            {
                float u1 = fx1*x1*invz1+cx1;
                float u1_r = u1 - mpCurrentKeyFrame->mbf*invz1;
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
                cv::Point2f uv2 = pCamera2->project(cv::Point3f(x2,y2,z2));
                float errX2 = uv2.x - kp2.pt.x;
                float errY2 = uv2.y - kp2.pt.y;
                if((errX2*errX2+errY2*errY2)>5.991*sigmaSquare2)
                    continue;
            }
            else
            {
                float u2 = fx2*x2*invz2+cx2;
                float u2_r = u2 - mpCurrentKeyFrame->mbf*invz2;
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

            if(mbFarPoints && (dist1>=mThFarPoints||dist2>=mThFarPoints)) // MODIFICATION
                continue;

            const float ratioDist = dist2/dist1;
            const float ratioOctave = mpCurrentKeyFrame->mvScaleFactors[kp1.octave]/pKF2->mvScaleFactors[kp2.octave];

            if(ratioDist*ratioFactor<ratioOctave || ratioDist>ratioOctave*ratioFactor)
                continue;


            // 形成的新的3D点 
            // Triangulation is succesfull
            MapPoint* pMP = new MapPoint(x3D,mpCurrentKeyFrame,mpAtlas->GetCurrentMap());




            pMP->AddObservation(mpCurrentKeyFrame,idx1);            
            pMP->AddObservation(pKF2,idx2);

            mpCurrentKeyFrame->AddMapPoint(pMP,idx1);
            pKF2->AddMapPoint(pMP,idx2);

            pMP->ComputeDistinctiveDescriptors();

            pMP->UpdateNormalAndDepth();

            mpAtlas->AddMapPoint(pMP);
            mlpRecentAddedMapPoints.push_back(pMP);
        }
    
    }


}

/*****************根据匹配关系对当前关键帧的地图点以及周围关键帧的地图点进行融合**********
******************选择最佳的特征来表征地图点****************************************/
void LocalMapping::SearchInNeighbors()
{
    // Retrieve neighbor keyframes
    //周围共视关键帧的数量
    int nn = 10;
    if(mbMonocular)
        nn=20;

    // 提取当前关键帧的共视关键帧
    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);
    
    vector<KeyFrame*> vpTargetKFs;
    //遍历共视关键帧，挑选没有与当前关键帧进行过融合的关键帧
    for(vector<KeyFrame*>::const_iterator vit=vpNeighKFs.begin(), vend=vpNeighKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;

        //pKFi->mnFuseTargetForKF:构建KeyFrame时初始化为0
        //pKFi不是坏帧且此周围关键帧没有与当前关键帧进行过融合
        if(pKFi->isBad() || pKFi->mnFuseTargetForKF == mpCurrentKeyFrame->mnId)
            continue;

        //存储没有进行过融合的关键帧
        vpTargetKFs.push_back(pKFi);
        //存储当前关键帧的ID，已证明此周围关键帧已和当前关键帧进行过融合
        pKFi->mnFuseTargetForKF = mpCurrentKeyFrame->mnId;
    }

    // Add some covisible of covisible
    // Extend to some second neighbors if abort is not requested
    //遍历周围关键帧中没有与当前关键帧进行过融合的关键帧
    for(int i=0, imax=vpTargetKFs.size(); i<imax; i++)
    {
        //获取与此周围关键帧具有最佳共视关系的20个关键帧（子关键帧）
        const vector<KeyFrame*> vpSecondNeighKFs = vpTargetKFs[i]->GetBestCovisibilityKeyFrames(20);
        
        //遍历此二级关键帧
        for(vector<KeyFrame*>::const_iterator vit2=vpSecondNeighKFs.begin(), vend2=vpSecondNeighKFs.end(); vit2!=vend2; vit2++)
        {
            KeyFrame* pKFi2 = *vit2;

            //若此二级关键帧是坏帧，或已与当前关键帧进行了融合，或此二级关键帧就是当前关键帧，则跳过
            if(pKFi2->isBad() || pKFi2->mnFuseTargetForKF==mpCurrentKeyFrame->mnId || pKFi2->mnId==mpCurrentKeyFrame->mnId)
                continue;

            //向目标关键帧vector中添加此二级关键帧
            vpTargetKFs.push_back(pKFi2);
            pKFi2->mnFuseTargetForKF=mpCurrentKeyFrame->mnId;
        }
        if (mbAbortBA)
            break;
    }

    // Extend to temporal neighbors
    //若是IMU模式
    if(mbInertial)
    {
        KeyFrame* pKFi = mpCurrentKeyFrame->mPrevKF;
        while(vpTargetKFs.size()<20 && pKFi)
        {
            if(pKFi->isBad() || pKFi->mnFuseTargetForKF==mpCurrentKeyFrame->mnId)
            {
                pKFi = pKFi->mPrevKF;
                continue;
            }
            vpTargetKFs.push_back(pKFi);
            pKFi->mnFuseTargetForKF=mpCurrentKeyFrame->mnId;
            pKFi = pKFi->mPrevKF;
        }
    }


    //匹配
    // Search matches by projection from current KF in target KFs
    ORBmatcher matcher;
    //获取当前关键帧的所有地图点
    vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    //遍历所有的目标关键帧，此中包括当前关键帧的周围关键帧，以及周围关键帧的周围关键帧
    //用当前关键帧中的地图点与周围关键帧的地图点融合
    for(vector<KeyFrame*>::iterator vit=vpTargetKFs.begin(), vend=vpTargetKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;


        // 将地图点投影到关键帧中进行匹配和融合；融合策略如下
        // 1.如果地图点能匹配关键帧的特征点，并且该点有对应的地图点，那么选择观测数目多的替换两个地图点
        // 2.如果地图点能匹配关键帧的特征点，并且该点没有对应的地图点，那么为该点添加该投影地图点
        // 注意这个时候对地图点融合的操作是立即生效的
        
        /********根据匹配关系判断当前关键帧的地图点和此周围关键帧的地图点是否是同一个地图点****
         * pKFi:当前关键帧的周围关键帧
         * vpMapPointMatches：当前关键帧的地图点
        **************************************************************************/
        matcher.Fuse(pKFi,vpMapPointMatches);
        if(pKFi->NLeft != -1) matcher.Fuse(pKFi,vpMapPointMatches,true);
    }

    if (mbAbortBA)
        return;

    // Search matches by projection from target KFs in current KF
    //用周围关键帧的地图点与当前关键帧的地图点进行融合
    vector<MapPoint*> vpFuseCandidates;
    vpFuseCandidates.reserve(vpTargetKFs.size()*vpMapPointMatches.size());

    //遍历所有的目标关键帧
    //用周围关键帧的地图点与当前关键帧的地图点进行融合
    for(vector<KeyFrame*>::iterator vitKF=vpTargetKFs.begin(), vendKF=vpTargetKFs.end(); vitKF!=vendKF; vitKF++)
    {
        KeyFrame* pKFi = *vitKF;

        //获取目标关键帧的地图点
        vector<MapPoint*> vpMapPointsKFi = pKFi->GetMapPointMatches();

        //遍历此目标关键帧的地图点，将目标关键帧中待融合的地图点加入候选融合点vector中
        for(vector<MapPoint*>::iterator vitMP=vpMapPointsKFi.begin(), vendMP=vpMapPointsKFi.end(); vitMP!=vendMP; vitMP++)
        {
            MapPoint* pMP = *vitMP;
            if(!pMP)
                continue;
            if(pMP->isBad() || pMP->mnFuseCandidateForKF == mpCurrentKeyFrame->mnId)
                continue;
            pMP->mnFuseCandidateForKF = mpCurrentKeyFrame->mnId;
            vpFuseCandidates.push_back(pMP);
        }
    }

    //融合此候选点和当前关键帧
    matcher.Fuse(mpCurrentKeyFrame,vpFuseCandidates);
    if(mpCurrentKeyFrame->NLeft != -1) matcher.Fuse(mpCurrentKeyFrame,vpFuseCandidates,true);


    // Update points
    //更新匹配点
    //获取当前关键帧的所有地图点
    vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    //更新当前关键帧的地图点的最小描述子和平均观测方向以及深度
    for(size_t i=0, iend=vpMapPointMatches.size(); i<iend; i++)
    {
        MapPoint* pMP=vpMapPointMatches[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                pMP->ComputeDistinctiveDescriptors();
                pMP->UpdateNormalAndDepth();
            }
        }
    }

    // Update connections in covisibility graph
    //更新当前关键帧的共视图
    mpCurrentKeyFrame->UpdateConnections();
}

//计算关键帧之间的基础矩阵
cv::Mat LocalMapping::ComputeF12(KeyFrame *&pKF1, KeyFrame *&pKF2)
{
    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();

    //PK2到PK1的旋转
    cv::Mat R12 = R1w*R2w.t();
    //PK1坐标系下，PK2的位置
    cv::Mat t12 = -R1w*R2w.t()*t2w+t1w;


    cv::Mat t12x = SkewSymmetricMatrix(t12);

    const cv::Mat &K1 = pKF1->mpCamera->toK();
    const cv::Mat &K2 = pKF2->mpCamera->toK();

    // E：本质矩阵 E=-t^R
    // F: 基础矩阵 F=K^-T*E*K^-1

    return K1.t().inv()*t12x*R12*K2.inv();
}

void LocalMapping::RequestStop()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopRequested = true;
    unique_lock<mutex> lock2(mMutexNewKFs);
    mbAbortBA = true;
}

bool LocalMapping::Stop()
{
    unique_lock<mutex> lock(mMutexStop);
    if(mbStopRequested && !mbNotStop)
    {
        mbStopped = true;
        cout << "Local Mapping STOP" << endl;
        return true;
    }

    return false;
}

bool LocalMapping::isStopped()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

bool LocalMapping::stopRequested()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopRequested;
}

void LocalMapping::Release()
{
    unique_lock<mutex> lock(mMutexStop);
    unique_lock<mutex> lock2(mMutexFinish);
    if(mbFinished)
        return;
    mbStopped = false;
    mbStopRequested = false;
    for(list<KeyFrame*>::iterator lit = mlNewKeyFrames.begin(), lend=mlNewKeyFrames.end(); lit!=lend; lit++)
        delete *lit;
    mlNewKeyFrames.clear();

    cout << "Local Mapping RELEASE" << endl;
}

bool LocalMapping::AcceptKeyFrames()
{
    unique_lock<mutex> lock(mMutexAccept);
    return mbAcceptKeyFrames;
}

void LocalMapping::SetAcceptKeyFrames(bool flag)
{
    unique_lock<mutex> lock(mMutexAccept);
    mbAcceptKeyFrames=flag;
}

bool LocalMapping::SetNotStop(bool flag)
{
    unique_lock<mutex> lock(mMutexStop);

    if(flag && mbStopped)
        return false;

    mbNotStop = flag;

    return true;
}

void LocalMapping::InterruptBA()
{
    mbAbortBA = true;
}

void LocalMapping::KeyFrameCulling()
{
    // Check redundant keyframes (only local keyframes)
    // A keyframe is considered redundant if the 90% of the MapPoints it sees, are seen
    // in at least other 3 keyframes (in the same or finer scale)
    // We only consider close stereo points
    const int Nd = 21; // MODIFICATION_STEREO_IMU 20 This should be the same than that one from LIBA
    mpCurrentKeyFrame->UpdateBestCovisibles();
    vector<KeyFrame*> vpLocalKeyFrames = mpCurrentKeyFrame->GetVectorCovisibleKeyFrames();

    float redundant_th;
    if(!mbInertial)
        redundant_th = 0.9;
    else if (mbMonocular)
        redundant_th = 0.9;
    else
        redundant_th = 0.5;

    const bool bInitImu = mpAtlas->isImuInitialized();
    int count=0;

    // Compoute last KF from optimizable window:
    unsigned int last_ID;
    if (mbInertial)
    {
        int count = 0;
        KeyFrame* aux_KF = mpCurrentKeyFrame;
        while(count<Nd && aux_KF->mPrevKF)
        {
            aux_KF = aux_KF->mPrevKF;
            count++;
        }
        last_ID = aux_KF->mnId;
    }



    for(vector<KeyFrame*>::iterator vit=vpLocalKeyFrames.begin(), vend=vpLocalKeyFrames.end(); vit!=vend; vit++)
    {
        count++;
        KeyFrame* pKF = *vit;

        if((pKF->mnId==pKF->GetMap()->GetInitKFid()) || pKF->isBad())
            continue;
        const vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();

        int nObs = 3;
        const int thObs=nObs;
        int nRedundantObservations=0;
        int nMPs=0;
        for(size_t i=0, iend=vpMapPoints.size(); i<iend; i++)
        {
            MapPoint* pMP = vpMapPoints[i];
            if(pMP)
            {
                if(!pMP->isBad())
                {
                    if(!mbMonocular)
                    {
                        if(pKF->mvDepth[i]>pKF->mThDepth || pKF->mvDepth[i]<0)
                            continue;
                    }

                    nMPs++;
                    if(pMP->Observations()>thObs)
                    {
                        const int &scaleLevel = (pKF -> NLeft == -1) ? pKF->mvKeysUn[i].octave
                                                                     : (i < pKF -> NLeft) ? pKF -> mvKeys[i].octave
                                                                                          : pKF -> mvKeysRight[i].octave;
                        const map<KeyFrame*, tuple<int,int>> observations = pMP->GetObservations();
                        int nObs=0;
                        for(map<KeyFrame*, tuple<int,int>>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
                        {
                            KeyFrame* pKFi = mit->first;
                            if(pKFi==pKF)
                                continue;
                            tuple<int,int> indexes = mit->second;
                            int leftIndex = get<0>(indexes), rightIndex = get<1>(indexes);
                            int scaleLeveli = -1;
                            if(pKFi -> NLeft == -1)
                                scaleLeveli = pKFi->mvKeysUn[leftIndex].octave;
                            else {
                                if (leftIndex != -1) {
                                    scaleLeveli = pKFi->mvKeys[leftIndex].octave;
                                }
                                if (rightIndex != -1) {
                                    int rightLevel = pKFi->mvKeysRight[rightIndex - pKFi->NLeft].octave;
                                    scaleLeveli = (scaleLeveli == -1 || scaleLeveli > rightLevel) ? rightLevel
                                                                                                  : scaleLeveli;
                                }
                            }

                            if(scaleLeveli<=scaleLevel+1)
                            {
                                nObs++;
                                if(nObs>thObs)
                                    break;
                            }
                        }
                        if(nObs>thObs)
                        {
                            nRedundantObservations++;
                        }
                    }
                }
            }
        }

        if(nRedundantObservations>redundant_th*nMPs)
        {
            if (mbInertial)
            {
                if (mpAtlas->KeyFramesInMap()<=Nd)
                    continue;

                if(pKF->mnId>(mpCurrentKeyFrame->mnId-2))
                    continue;

                if(pKF->mPrevKF && pKF->mNextKF)
                {
                    const float t = pKF->mNextKF->mTimeStamp-pKF->mPrevKF->mTimeStamp;

                    if((bInitImu && (pKF->mnId<last_ID) && t<3.) || (t<0.5))
                    {
                        pKF->mNextKF->mpImuPreintegrated->MergePrevious(pKF->mpImuPreintegrated);
                        pKF->mNextKF->mPrevKF = pKF->mPrevKF;
                        pKF->mPrevKF->mNextKF = pKF->mNextKF;
                        pKF->mNextKF = NULL;
                        pKF->mPrevKF = NULL;
                        pKF->SetBadFlag();
                    }
                    else if(!mpCurrentKeyFrame->GetMap()->GetIniertialBA2() && (cv::norm(pKF->GetImuPosition()-pKF->mPrevKF->GetImuPosition())<0.02) && (t<3))
                    {
                        pKF->mNextKF->mpImuPreintegrated->MergePrevious(pKF->mpImuPreintegrated);
                        pKF->mNextKF->mPrevKF = pKF->mPrevKF;
                        pKF->mPrevKF->mNextKF = pKF->mNextKF;
                        pKF->mNextKF = NULL;
                        pKF->mPrevKF = NULL;
                        pKF->SetBadFlag();
                    }
                }
            }
            else
            {
                pKF->SetBadFlag();
            }
        }
        if((count > 20 && mbAbortBA) || count>100) // MODIFICATION originally 20 for mbabortBA check just 10 keyframes
        {
            break;
        }
    }
}


cv::Mat LocalMapping::SkewSymmetricMatrix(const cv::Mat &v)
{
    return (cv::Mat_<float>(3,3) <<             0, -v.at<float>(2), v.at<float>(1),
            v.at<float>(2),               0,-v.at<float>(0),
            -v.at<float>(1),  v.at<float>(0),              0);
}

void LocalMapping::RequestReset()
{
    {
        unique_lock<mutex> lock(mMutexReset);
        cout << "LM: Map reset recieved" << endl;
        mbResetRequested = true;
    }
    cout << "LM: Map reset, waiting..." << endl;

    while(1)
    {
        {
            unique_lock<mutex> lock2(mMutexReset);
            if(!mbResetRequested)
                break;
        }
        usleep(3000);
    }
    cout << "LM: Map reset, Done!!!" << endl;
}

void LocalMapping::RequestResetActiveMap(Map* pMap)
{
    {
        unique_lock<mutex> lock(mMutexReset);
        cout << "LM: Active map reset recieved" << endl;
        mbResetRequestedActiveMap = true;
        mpMapToReset = pMap;
    }
    cout << "LM: Active map reset, waiting..." << endl;

    while(1)
    {
        {
            unique_lock<mutex> lock2(mMutexReset);
            if(!mbResetRequestedActiveMap)
                break;
        }
        usleep(3000);
    }
    cout << "LM: Active map reset, Done!!!" << endl;
}

void LocalMapping::ResetIfRequested()
{
    bool executed_reset = false;
    {
        unique_lock<mutex> lock(mMutexReset);
        if(mbResetRequested)
        {
            executed_reset = true;

            cout << "LM: Reseting Atlas in Local Mapping..." << endl;
            mlNewKeyFrames.clear();
            mlpRecentAddedMapPoints.clear();
            mbResetRequested=false;
            mbResetRequestedActiveMap = false;

            // Inertial parameters
            mTinit = 0.f;
            mbNotBA2 = true;
            mbNotBA1 = true;
            mbBadImu=false;

            mIdxInit=0;

            cout << "LM: End reseting Local Mapping..." << endl;
        }

        if(mbResetRequestedActiveMap) {
            executed_reset = true;
            cout << "LM: Reseting current map in Local Mapping..." << endl;
            mlNewKeyFrames.clear();
            mlpRecentAddedMapPoints.clear();

            // Inertial parameters
            mTinit = 0.f;
            mbNotBA2 = true;
            mbNotBA1 = true;
            mbBadImu=false;

            mbResetRequestedActiveMap = false;
            cout << "LM: End reseting Local Mapping..." << endl;
        }
    }
    if(executed_reset)
        cout << "LM: Reset free the mutex" << endl;

}

void LocalMapping::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool LocalMapping::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void LocalMapping::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;    
    unique_lock<mutex> lock2(mMutexStop);
    mbStopped = true;
}

bool LocalMapping::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

void LocalMapping::InitializeIMU(float priorG, float priorA, bool bFIBA)
{
    if (mbResetRequested)
        return;

    float minTime;
    int nMinKF;
    if (mbMonocular)
    {
        minTime = 2.0;
        nMinKF = 10;
    }
    else
    {
        minTime = 1.0;
        nMinKF = 10;
    }


    if(mpAtlas->KeyFramesInMap()<nMinKF)
        return;

    // Retrieve all keyframe in temporal order
    list<KeyFrame*> lpKF;
    KeyFrame* pKF = mpCurrentKeyFrame;
    while(pKF->mPrevKF)
    {
        lpKF.push_front(pKF);
        pKF = pKF->mPrevKF;
    }
    lpKF.push_front(pKF);
    vector<KeyFrame*> vpKF(lpKF.begin(),lpKF.end());

    if(vpKF.size()<nMinKF)
        return;

    mFirstTs=vpKF.front()->mTimeStamp;
    if(mpCurrentKeyFrame->mTimeStamp-mFirstTs<minTime)
        return;

    bInitializing = true;

    while(CheckNewKeyFrames())
    {
        ProcessNewKeyFrame();
        vpKF.push_back(mpCurrentKeyFrame);
        lpKF.push_back(mpCurrentKeyFrame);
    }

    const int N = vpKF.size();
    IMU::Bias b(0,0,0,0,0,0);

    // Compute and KF velocities mRwg estimation
    if (!mpCurrentKeyFrame->GetMap()->isImuInitialized())
    {
        cv::Mat cvRwg;
        cv::Mat dirG = cv::Mat::zeros(3,1,CV_32F);
        for(vector<KeyFrame*>::iterator itKF = vpKF.begin(); itKF!=vpKF.end(); itKF++)
        {
            if (!(*itKF)->mpImuPreintegrated)
                continue;
            if (!(*itKF)->mPrevKF)
                continue;

            dirG -= (*itKF)->mPrevKF->GetImuRotation()*(*itKF)->mpImuPreintegrated->GetUpdatedDeltaVelocity();
            cv::Mat _vel = ((*itKF)->GetImuPosition() - (*itKF)->mPrevKF->GetImuPosition())/(*itKF)->mpImuPreintegrated->dT;
            (*itKF)->SetVelocity(_vel);
            (*itKF)->mPrevKF->SetVelocity(_vel);
        }

        dirG = dirG/cv::norm(dirG);
        cv::Mat gI = (cv::Mat_<float>(3,1) << 0.0f, 0.0f, -1.0f);
        cv::Mat v = gI.cross(dirG);
        const float nv = cv::norm(v);
        const float cosg = gI.dot(dirG);
        const float ang = acos(cosg);
        cv::Mat vzg = v*ang/nv;
        cvRwg = IMU::ExpSO3(vzg);
        mRwg = Converter::toMatrix3d(cvRwg);
        mTinit = mpCurrentKeyFrame->mTimeStamp-mFirstTs;
    }
    else
    {
        mRwg = Eigen::Matrix3d::Identity();
        mbg = Converter::toVector3d(mpCurrentKeyFrame->GetGyroBias());
        mba = Converter::toVector3d(mpCurrentKeyFrame->GetAccBias());
    }

    mScale=1.0;

    mInitTime = mpTracker->mLastFrame.mTimeStamp-vpKF.front()->mTimeStamp;

    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    Optimizer::InertialOptimization(mpAtlas->GetCurrentMap(), mRwg, mScale, mbg, mba, mbMonocular, infoInertial, false, false, priorG, priorA);
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    /*cout << "scale after inertial-only optimization: " << mScale << endl;
    cout << "bg after inertial-only optimization: " << mbg << endl;
    cout << "ba after inertial-only optimization: " << mba << endl;*/


    if (mScale<1e-1)
    {
        cout << "scale too small" << endl;
        bInitializing=false;
        return;
    }



    // Before this line we are not changing the map

    unique_lock<mutex> lock(mpAtlas->GetCurrentMap()->mMutexMapUpdate);
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    if ((fabs(mScale-1.f)>0.00001)||!mbMonocular)
    {
        mpAtlas->GetCurrentMap()->ApplyScaledRotation(Converter::toCvMat(mRwg).t(),mScale,true);
        mpTracker->UpdateFrameIMU(mScale,vpKF[0]->GetImuBias(),mpCurrentKeyFrame);
    }
    std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();

    // Check if initialization OK
    if (!mpAtlas->isImuInitialized())
        for(int i=0;i<N;i++)
        {
            KeyFrame* pKF2 = vpKF[i];
            pKF2->bImu = true;
        }

    /*cout << "Before GIBA: " << endl;
    cout << "ba: " << mpCurrentKeyFrame->GetAccBias() << endl;
    cout << "bg: " << mpCurrentKeyFrame->GetGyroBias() << endl;*/

    std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
    if (bFIBA)
    {
        if (priorA!=0.f)
            Optimizer::FullInertialBA(mpAtlas->GetCurrentMap(), 100, false, 0, NULL, true, priorG, priorA);
        else
            Optimizer::FullInertialBA(mpAtlas->GetCurrentMap(), 100, false, 0, NULL, false);
    }

    std::chrono::steady_clock::time_point t5 = std::chrono::steady_clock::now();

    // If initialization is OK
    mpTracker->UpdateFrameIMU(1.0,vpKF[0]->GetImuBias(),mpCurrentKeyFrame);
    if (!mpAtlas->isImuInitialized())
    {
        cout << "IMU in Map " << mpAtlas->GetCurrentMap()->GetId() << " is initialized" << endl;
        mpAtlas->SetImuInitialized();
        mpTracker->t0IMU = mpTracker->mCurrentFrame.mTimeStamp;
        mpCurrentKeyFrame->bImu = true;
    }


    mbNewInit=true;
    mnKFs=vpKF.size();
    mIdxInit++;

    for(list<KeyFrame*>::iterator lit = mlNewKeyFrames.begin(), lend=mlNewKeyFrames.end(); lit!=lend; lit++)
    {
        (*lit)->SetBadFlag();
        delete *lit;
    }
    mlNewKeyFrames.clear();

    mpTracker->mState=Tracking::OK;
    bInitializing = false;


    /*cout << "After GIBA: " << endl;
    cout << "ba: " << mpCurrentKeyFrame->GetAccBias() << endl;
    cout << "bg: " << mpCurrentKeyFrame->GetGyroBias() << endl;
    double t_inertial_only = std::chrono::duration_cast<std::chrono::duration<double> >(t1 - t0).count();
    double t_update = std::chrono::duration_cast<std::chrono::duration<double> >(t3 - t2).count();
    double t_viba = std::chrono::duration_cast<std::chrono::duration<double> >(t5 - t4).count();
    cout << t_inertial_only << ", " << t_update << ", " << t_viba << endl;*/

    mpCurrentKeyFrame->GetMap()->IncreaseChangeIndex();

    return;
}

void LocalMapping::ScaleRefinement()
{
    // Minimum number of keyframes to compute a solution
    // Minimum time (seconds) between first and last keyframe to compute a solution. Make the difference between monocular and stereo
    // unique_lock<mutex> lock0(mMutexImuInit);
    if (mbResetRequested)
        return;

    // Retrieve all keyframes in temporal order
    list<KeyFrame*> lpKF;
    KeyFrame* pKF = mpCurrentKeyFrame;
    while(pKF->mPrevKF)
    {
        lpKF.push_front(pKF);
        pKF = pKF->mPrevKF;
    }
    lpKF.push_front(pKF);
    vector<KeyFrame*> vpKF(lpKF.begin(),lpKF.end());

    while(CheckNewKeyFrames())
    {
        ProcessNewKeyFrame();
        vpKF.push_back(mpCurrentKeyFrame);
        lpKF.push_back(mpCurrentKeyFrame);
    }

    const int N = vpKF.size();

    mRwg = Eigen::Matrix3d::Identity();
    mScale=1.0;

    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    Optimizer::InertialOptimization(mpAtlas->GetCurrentMap(), mRwg, mScale);
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    cout<<" Before bias "<<mpCurrentKeyFrame->GetImuBias()<<endl;
    if (mScale<1e-1) // 1e-1
    {
        cout << "scale too small" << endl;
        bInitializing=false;
        return;
    }

    // Before this line we are not changing the map
    unique_lock<mutex> lock(mpAtlas->GetCurrentMap()->mMutexMapUpdate);
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    if ((fabs(mScale-1.f)>0.00001)||!mbMonocular)
    {
        mpAtlas->GetCurrentMap()->ApplyScaledRotation(Converter::toCvMat(mRwg).t(),mScale,true);
        mpTracker->UpdateFrameIMU(mScale,mpCurrentKeyFrame->GetImuBias(),mpCurrentKeyFrame);
    }

    cout<<"bias "<<mpCurrentKeyFrame->GetImuBias()<<endl;
    std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();

    for(list<KeyFrame*>::iterator lit = mlNewKeyFrames.begin(), lend=mlNewKeyFrames.end(); lit!=lend; lit++)
    {
        (*lit)->SetBadFlag();
        delete *lit;
    }
    mlNewKeyFrames.clear();

    double t_inertial_only = std::chrono::duration_cast<std::chrono::duration<double> >(t1 - t0).count();

    // To perform pose-inertial opt w.r.t. last keyframe
    mpCurrentKeyFrame->GetMap()->IncreaseChangeIndex();

    return;
}



bool LocalMapping::IsInitializing()
{
    return bInitializing;
}


double LocalMapping::GetCurrKFTime()
{

    if (mpCurrentKeyFrame)
    {
        return mpCurrentKeyFrame->mTimeStamp;
    }
    else
        return 0.0;
}

KeyFrame* LocalMapping::GetCurrKF()
{
    return mpCurrentKeyFrame;
}

} //namespace ORB_SLAM
