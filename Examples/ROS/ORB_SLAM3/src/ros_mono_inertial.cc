/**
* This file is added by xiefei2929@126.com
*/


#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include<ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>

#include <opencv2/opencv.hpp>
#include<opencv2/core/core.hpp>
#include <std_msgs/Header.h>
#include <sensor_msgs/Imu.h> 

#include"../../../include/System.h"

using namespace std;

queue<sensor_msgs::ImuConstPtr> imu_buf;  //IMU信号序列
queue<sensor_msgs::ImageConstPtr> img0_buf;//图像序列
std::mutex m_buf;
cv::Mat M1l,M2l,M1r,M2r;
double pretime=0;
ORB_SLAM3::System* mpSLAM;


//图像信号读取
void img0_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    m_buf.lock();
    img0_buf.push(img_msg);
    m_buf.unlock();
}


//从图像信号序列中获取图像
cv::Mat getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg)
{
    //利用cv_bridge从传感器中获取图像
    // Copy the ros image message to cv::Mat.
    cv_bridge::CvImageConstPtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvShare(img_msg);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
    }


    cv::Mat img = cv_ptr->image.clone();
    return img;
}

//IMU信号读取
void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    m_buf.lock();
    imu_buf.push(imu_msg);
    m_buf.unlock();
    return;
}

//信号处理
void sync_process()
{
    while(1)
    {
        
        cv::Mat im;
        /*********************************
            std_msgs::Header 消息中主要有：
            unit32 seq 存储原始数据类型unit32
            time stamp 时间戳信息
            string frame_id 坐标系名称
        ***********************************/
        std_msgs::Header header;
        double time = 0;
        //make sure got enough imu frame before a image frame
        //确保在一个图像帧到来之前有足够的IMU
        if (!img0_buf.empty()&&imu_buf.size()>15)
        {
            // m_buf.lock();
            time = img0_buf.front()->header.stamp.toSec(); //图像的时间戳
            header = img0_buf.front()->header; //图像的header
            im = getImageFromMsg(img0_buf.front()); //从图像信号序列中获取图像
            img0_buf.pop(); //剔除buf中已经输入系统的图像
    
            /*********************************
                ORB_SLAM3::IMU::Point：
                cv::Point3f类型的加速度a
                cv::Point3f类型的角速度w
                double 类型的时间戳t
            ***********************************/
            vector<ORB_SLAM3::IMU::Point> vImuMeas;
            
            if(!imu_buf.empty())
            {
                // Load imu measurements from previous frame
                vImuMeas.clear();

                /***************************************************************************
                    上一帧图像的时间戳 < IMU测量量的时间戳 <= 当前图像的时间戳 存入IMU测量vector向量vImuMeas
                ****************************************************************************/
                while(imu_buf.front()->header.stamp.toSec()<=time&&imu_buf.front()->header.stamp.toSec()>pretime)
                {
                    double t = imu_buf.front()->header.stamp.toSec(); //时间戳
                    double dx = imu_buf.front()->linear_acceleration.x;  //加速度
                    double dy = imu_buf.front()->linear_acceleration.y;
                    double dz = imu_buf.front()->linear_acceleration.z;
                    double rx = imu_buf.front()->angular_velocity.x;  //角速度
                    double ry = imu_buf.front()->angular_velocity.y;
                    double rz = imu_buf.front()->angular_velocity.z;
                    // printf("%f %f %f %f %f %f %f \n",dx,dy,dz,rx,ry,rz,t);
                    vImuMeas.push_back(ORB_SLAM3::IMU::Point(dx,dy,dz,rx,ry,rz,t));
                    imu_buf.pop();
                }
            }
            pretime = time;
            /********************************************
             * im:输入图像
             * time:输入图像的时间戳
             * vImuMeas:当前图像和上一张图像之间的IMU测量量
            ********************************************/
            mpSLAM->TrackMonocular(im,time,vImuMeas); //进入Tracking线程
            // m_buf.unlock();
        }

        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "Mono_inertial");//第三个参数是节点名
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

    if(argc != 3)
    {   //三个参数 单/双目 词袋位置 标定文件
        cerr << endl << "Usage: rosrun ORB_SLAM3 Stereo path_to_vocabulary path_to_settings" << endl;
        ros::shutdown();
        return 1;
    }    

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    //创建SLAM系统，创建线程并启动
    /*********************************
     * 输入system构造函数：词袋模型的文件，标定文件，传感器类型，bUseViewer=true 使用可视化界面
    ***********************************/
    ORB_SLAM3::System SLAM(argv[1],argv[2],ORB_SLAM3::System::IMU_MONOCULAR,true);
    mpSLAM = &SLAM;


    ROS_WARN("waiting for image and imu...");



    ros::Subscriber sub_imu = n.subscribe("/imu0", 2000, imu_callback, ros::TransportHints().tcpNoDelay());
    ros::Subscriber sub_img0 = n.subscribe("/cam0/image_raw", 100, img0_callback);

    //信号同步过程
    std::thread sync_thread{sync_process};
  

    ros::spin();
    
    SLAM.Shutdown();
    SLAM.SaveTrajectoryTUM("dP_VIO.txt");
    return 0;
}


