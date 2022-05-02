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


#include "ORBmatcher.h"

#include<limits.h>

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"

#include<stdint-gcc.h>

using namespace std;

namespace ORB_SLAM3
{

const int ORBmatcher::TH_HIGH = 100;
const int ORBmatcher::TH_LOW = 50;
const int ORBmatcher::HISTO_LENGTH = 30;

ORBmatcher::ORBmatcher(float nnratio, bool checkOri): mfNNratio(nnratio), mbCheckOrientation(checkOri)
{
}


/**************图像帧和地图点集合进行匹配*****************
 * F：图像帧
 * vpMapPoints：地图点集合
 * th:阈值
 * bFarPoints :yaml文件中设置，但没有传入，则为false
 * ThFarPoints：yaml文件中设置，但没有传入，则为false
 * 
 * 使用：在局部地图跟踪过程中使用，在恒速模型跟踪和参考关键帧跟踪
 * 之后进行，此时传入的图像帧应以实现了地图点匹配，具有地图点
 * 
 * 方法：将地图点投影至当前图像帧的像素坐标系，以此为中心在其周围
 * 的搜索半径范围内进行搜索获得候选特征点，通过对比描述子距离实现
 * 最后的匹配
*****************************************************/
int ORBmatcher::SearchByProjection(Frame &F, const vector<MapPoint*> &vpMapPoints, const float th, const bool bFarPoints, const float thFarPoints)
{   
    //设置匹配成功数量=0
    int nmatches=0, left = 0, right = 0;

    
    //若阈值不为1,则需要扩大搜索范围
    const bool bFactor = th!=1.0;

    //遍历当前局部地图中的所有地图点
    for(size_t iMP=0; iMP<vpMapPoints.size(); iMP++)
    {   
        //提取地图点
        MapPoint* pMP = vpMapPoints[iMP];

        //mvTracKInView表示是否进行跟踪，若此地图点被标记为不跟踪，跳过
        if(!pMP->mbTrackInView && !pMP->mbTrackInViewR)
            continue;
        
        //若传入的两个标志为为true，跳过
        if(bFarPoints && pMP->mTrackDepth>thFarPoints)
            continue;

        //若当前的地图点是坏点，跳过
        if(pMP->isBad())
            continue;

        //若当前地图点的跟踪标志位为true，表示跟踪此地图点
        if(pMP->mbTrackInView)
        {
            //获得当前地图点的跟踪尺度层数（类似于图像金字塔中的金字塔层数）
            const int &nPredictedLevel = pMP->mnTrackScaleLevel;

            // The size of the window will depend on the viewing direction
            //搜索半径和地图点的视场角有关
            float r = RadiusByViewingCos(pMP->mTrackViewCos);

            //若扩大搜索阈值，则增加搜索半径
            if(bFactor)
                r*=th;

            //寻找当前图像中地图点的投影点在半径范围内的特征点，并存储ID
            const vector<size_t> vIndices =
                    F.GetFeaturesInArea(pMP->mTrackProjX,pMP->mTrackProjY,
                    r*F.mvScaleFactors[nPredictedLevel],nPredictedLevel-1,
                    nPredictedLevel);

            //若搜索成功
            if(!vIndices.empty())
            {   
                //获取地图点的描述子
                const cv::Mat MPdescriptor = pMP->GetDescriptor();


                int bestDist=256;
                int bestLevel= -1;
                int bestDist2=256;
                int bestLevel2 = -1;
                int bestIdx =-1 ;

                // Get best and second matches with near keypoints
                //遍历候选关键点，选出与当前地图点的描述子距离最小的关键点
                for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
                {
                    const size_t idx = *vit;

                    if(F.mvpMapPoints[idx])
                        if(F.mvpMapPoints[idx]->Observations()>0)
                            continue;

                    if(F.Nleft == -1 && F.mvuRight[idx]>0)
                    {
                        const float er = fabs(pMP->mTrackProjXR-F.mvuRight[idx]);
                        if(er>r*F.mvScaleFactors[nPredictedLevel])
                            continue;
                    }

                    const cv::Mat &d = F.mDescriptors.row(idx);

                    const int dist = DescriptorDistance(MPdescriptor,d);

                    if(dist<bestDist)
                    {
                        bestDist2=bestDist;
                        bestDist=dist;
                        bestLevel2 = bestLevel;
                        bestLevel = (F.Nleft == -1) ? F.mvKeysUn[idx].octave
                                                    : (idx < F.Nleft) ? F.mvKeys[idx].octave
                                                                      : F.mvKeysRight[idx - F.Nleft].octave;
                        bestIdx=idx;
                    }
                    else if(dist<bestDist2)
                    {
                        bestLevel2 = (F.Nleft == -1) ? F.mvKeysUn[idx].octave
                                                     : (idx < F.Nleft) ? F.mvKeys[idx].octave
                                                                       : F.mvKeysRight[idx - F.Nleft].octave;
                        bestDist2=dist;
                    }
                }

                // Apply ratio to second match (only if best and second are in the same scale level)
                if(bestDist<=TH_HIGH)
                {
                    if(bestLevel==bestLevel2 && bestDist>mfNNratio*bestDist2)
                        continue;

                    if(bestLevel!=bestLevel2 || bestDist<=mfNNratio*bestDist2)
                    {
                        //存储当前图像帧中匹配成功的地图点
                        F.mvpMapPoints[bestIdx]=pMP;

                        if(F.Nleft != -1 && F.mvLeftToRightMatch[bestIdx] != -1){ //Also match with the stereo observation at right camera
                            F.mvpMapPoints[F.mvLeftToRightMatch[bestIdx] + F.Nleft] = pMP;
                            nmatches++;
                            right++;
                        }

                        nmatches++;
                        left++;
                    }
                }
            }
        }

        //传感器不是双目鱼眼相机，则F.Nleft=-1
        if(F.Nleft != -1 && pMP->mbTrackInViewR)
        {
            const int &nPredictedLevel = pMP->mnTrackScaleLevelR;
            if(nPredictedLevel != -1){
                float r = RadiusByViewingCos(pMP->mTrackViewCosR);

                const vector<size_t> vIndices =
                        F.GetFeaturesInArea(pMP->mTrackProjXR,pMP->mTrackProjYR,r*F.mvScaleFactors[nPredictedLevel],nPredictedLevel-1,nPredictedLevel,true);

                if(vIndices.empty())
                    continue;

                const cv::Mat MPdescriptor = pMP->GetDescriptor();

                int bestDist=256;
                int bestLevel= -1;
                int bestDist2=256;
                int bestLevel2 = -1;
                int bestIdx =-1 ;

                // Get best and second matches with near keypoints
                for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
                {
                    const size_t idx = *vit;

                    if(F.mvpMapPoints[idx + F.Nleft])
                        if(F.mvpMapPoints[idx + F.Nleft]->Observations()>0)
                            continue;


                    const cv::Mat &d = F.mDescriptors.row(idx + F.Nleft);

                    const int dist = DescriptorDistance(MPdescriptor,d);

                    if(dist<bestDist)
                    {
                        bestDist2=bestDist;
                        bestDist=dist;
                        bestLevel2 = bestLevel;
                        bestLevel = F.mvKeysRight[idx].octave;
                        bestIdx=idx;
                    }
                    else if(dist<bestDist2)
                    {
                        bestLevel2 = F.mvKeysRight[idx].octave;
                        bestDist2=dist;
                    }
                }

                // Apply ratio to second match (only if best and second are in the same scale level)
                if(bestDist<=TH_HIGH)
                {
                    if(bestLevel==bestLevel2 && bestDist>mfNNratio*bestDist2)
                        continue;

                    if(F.Nleft != -1 && F.mvRightToLeftMatch[bestIdx] != -1){ //Also match with the stereo observation at right camera
                        F.mvpMapPoints[F.mvRightToLeftMatch[bestIdx]] = pMP;
                        nmatches++;
                        left++;
                    }


                    F.mvpMapPoints[bestIdx + F.Nleft]=pMP;
                    nmatches++;
                    right++;
                }
            }
        }
    
    }
    return nmatches;
}




float ORBmatcher::RadiusByViewingCos(const float &viewCos)
{
    if(viewCos>0.998)
        return 2.5;
    else
        return 4.0;
}


bool ORBmatcher::CheckDistEpipolarLine(const cv::KeyPoint &kp1,const cv::KeyPoint &kp2,const cv::Mat &F12,const KeyFrame* pKF2, const bool b1)
{
    // Epipolar line in second image l = x1'F12 = [a b c]
    const float a = kp1.pt.x*F12.at<float>(0,0)+kp1.pt.y*F12.at<float>(1,0)+F12.at<float>(2,0);
    const float b = kp1.pt.x*F12.at<float>(0,1)+kp1.pt.y*F12.at<float>(1,1)+F12.at<float>(2,1);
    const float c = kp1.pt.x*F12.at<float>(0,2)+kp1.pt.y*F12.at<float>(1,2)+F12.at<float>(2,2);

    const float num = a*kp2.pt.x+b*kp2.pt.y+c;

    const float den = a*a+b*b;

    if(den==0)
        return false;

    const float dsqr = num*num/den;

    if(!b1)
        return dsqr<3.84*pKF2->mvLevelSigma2[kp2.octave];
    else
        return dsqr<6.63*pKF2->mvLevelSigma2[kp2.octave];
}

bool ORBmatcher::CheckDistEpipolarLine2(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &F12, const KeyFrame *pKF2, const float unc)
{
    // Epipolar line in second image l = x1'F12 = [a b c]
    const float a = kp1.pt.x*F12.at<float>(0,0)+kp1.pt.y*F12.at<float>(1,0)+F12.at<float>(2,0);
    const float b = kp1.pt.x*F12.at<float>(0,1)+kp1.pt.y*F12.at<float>(1,1)+F12.at<float>(2,1);
    const float c = kp1.pt.x*F12.at<float>(0,2)+kp1.pt.y*F12.at<float>(1,2)+F12.at<float>(2,2);

    const float num = a*kp2.pt.x+b*kp2.pt.y+c;

    const float den = a*a+b*b;

    if(den==0)
        return false;

    const float dsqr = num*num/den;

    if(unc==1.f)
        return dsqr<3.84*pKF2->mvLevelSigma2[kp2.octave];
    else
        return dsqr<3.84*pKF2->mvLevelSigma2[kp2.octave]*unc;
}


/***************利用词袋模型对关键帧的特征点进行跟踪************
*******************匹配关键帧和图像帧************************
 * 输入：pKF：需要跟踪的关键帧
 *      F：  待匹配的当前图像帧
 *      vpMapPointMatches: 当前图像帧中的地图点对应的匹配，NULL表示未匹配
 * 返回：int 匹配成功的数量
 * 
 * 使用：1、跟踪参考关键帧
****************************************************************/
int ORBmatcher::SearchByBoW(KeyFrame* pKF,Frame &F, vector<MapPoint*> &vpMapPointMatches)
{   
    //提取关键帧的地图点
    const vector<MapPoint*> vpMapPointsKF = pKF->GetMapPointMatches();

    //初始化vpMapPointMatches，内存大小与输入图像帧的特征点数目一致，假定所有的特征点都没被匹配过
    vpMapPointMatches = vector<MapPoint*>(F.N,static_cast<MapPoint*>(NULL));
    //vpMapPointMatches的索引值为当前图像帧的特征点ID
    //vpMapPointMatches[i]存储的内容为匹配成功的地图点（关键帧中）

    //提取关键帧的特征向量(相当于图像的描述子)
    // FeatureVector:  std::map<NodeId, std::vector<unsigned int> > 
    //NodeId:所处的词袋模型层数，vector<unsigned int> 处于当前层数的特征点ID
    const DBoW2::FeatureVector &vFeatVecKF = pKF->mFeatVec;

    int nmatches=0;

    //构造直方图(HISTO_LENGTH=30)
    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500); //每个直方图中分出500个空间
    const float factor = 1.0f/HISTO_LENGTH;

    // We perform the matching over ORB that belong to the same vocabulary node (at a certain level)
    //将属于同一节点（即同一层）的ORB特征进行匹配
    DBoW2::FeatureVector::const_iterator KFit = vFeatVecKF.begin();  //关键帧的特征向量的开端
    DBoW2::FeatureVector::const_iterator Fit = F.mFeatVec.begin();   //输入图像的特征向量开端
    DBoW2::FeatureVector::const_iterator KFend = vFeatVecKF.end();   //关键帧的特征向量的终点
    DBoW2::FeatureVector::const_iterator Fend = F.mFeatVec.end();    //输入图像的特征向量终点

    vector<int> matches12 = vector<int>(pKF->N,-1);

    //遍历所有的关键帧特征向量
    while(KFit != KFend && Fit != Fend)
    {
        //判断关键帧的特征点所在的节点层数是否等于图像帧的特征点所在的节点层数
        if(KFit->first == Fit->first)
        {
            //关键帧在此层中的特征点ID集合
            const vector<unsigned int> vIndicesKF = KFit->second;
            //图像帧在此层中的特征点ID集合
            const vector<unsigned int> vIndicesF = Fit->second;

            //遍历此关键帧这一节点层的所有特征点
            for(size_t iKF=0; iKF<vIndicesKF.size(); iKF++)
            {
                //提取关键帧的特征点ID
                const unsigned int realIdxKF = vIndicesKF[iKF];

                //提取此特征点对应的地图点
                MapPoint* pMP = vpMapPointsKF[realIdxKF];
                
                //如果此地图点不存在或为坏点，跳过
                if(!pMP)
                    continue;

                if(pMP->isBad())
                    continue;


                //提取关键帧图像中此特征点对应的描述子行
                const cv::Mat &dKF= pKF->mDescriptors.row(realIdxKF);

                int bestDist1=256;  //最佳距离 
                int bestIdxF =-1 ;  //最佳匹配的图像特征点ID
                int bestDist2=256;  //次佳距离

                int bestDist1R=256;
                int bestIdxFR =-1 ;
                int bestDist2R=256;

                //遍历当前图像帧中此节点层的所有特征点
                for(size_t iF=0; iF<vIndicesF.size(); iF++)
                {
                    //非鱼眼相机都是-1
                    if(F.Nleft == -1)  
                    {   
                        //提取当前图像帧中此节点层的特征点ID
                        const unsigned int realIdxF = vIndicesF[iF];

                        //如果地图点存在，说明已经被匹配过了
                        //（进入此匹配函数时已经vpMapPointMatches初始化为空）
                        if(vpMapPointMatches[realIdxF])
                            continue;

                        //提取图像帧的此特征点对应的描述子
                        const cv::Mat &dF = F.mDescriptors.row(realIdxF);

                        //计算描述子的距离
                        const int dist =  DescriptorDistance(dKF,dF);

                        //选出图像帧中所有特征点与此关键帧特征点描述子距离最近和次近的特征
                        if(dist<bestDist1)
                        {
                            bestDist2=bestDist1;
                            bestDist1=dist;
                            bestIdxF=realIdxF;
                        }
                        else if(dist<bestDist2)
                        {
                            bestDist2=dist;
                        }



                    }
                    else
                    {
                        const unsigned int realIdxF = vIndicesF[iF];

                        if(vpMapPointMatches[realIdxF])
                            continue;

                        const cv::Mat &dF = F.mDescriptors.row(realIdxF);

                        const int dist =  DescriptorDistance(dKF,dF);

                        if(realIdxF < F.Nleft && dist<bestDist1){
                            bestDist2=bestDist1;
                            bestDist1=dist;
                            bestIdxF=realIdxF;
                        }
                        else if(realIdxF < F.Nleft && dist<bestDist2){
                            bestDist2=dist;
                        }

                        if(realIdxF >= F.Nleft && dist<bestDist1R){
                            bestDist2R=bestDist1R;
                            bestDist1R=dist;
                            bestIdxFR=realIdxF;
                        }
                        else if(realIdxF >= F.Nleft && dist<bestDist2R){
                            bestDist2R=dist;
                        }
                    }

                }


                //根据阈值和角度投票剔除误匹配
                //首先匹配距离必须小于设定的阈值，TH_LOW=50
                if(bestDist1<=TH_LOW)
                {   
                    //定义ORBmatcher(float nnratio, bool checkOri)
                    //其中nnratio=mfNNratio
                    //checkOri = mbCheckOrientation
                    //最佳匹配的距离要明显优于次佳匹配的距离
                    if(static_cast<float>(bestDist1)<mfNNratio*static_cast<float>(bestDist2))
                    {   
                        //设置此关键帧特征点对应的地图点，为图像帧的特征点对应的地图点
                        vpMapPointMatches[bestIdxF]=pMP;

                        /************************************************
                         * 判断关键帧是否有相机2
                         *      若没有，则关键点kp=pKF->mvKeysUn[realIdxKF]
                         *      若有，继续判断
                        *************************************************/
                       //KP = 此关键帧的特征点
                        const cv::KeyPoint &kp =
                                (!pKF->mpCamera2) ? pKF->mvKeysUn[realIdxKF] :
                                (realIdxKF >= pKF -> NLeft) ? pKF -> mvKeysRight[realIdxKF - pKF -> NLeft]
                                                            : pKF -> mvKeys[realIdxKF];
                        
                        //若进行方向判断
                        //定义ORBmatcher(float nnratio, bool checkOri)
                        //其中nnratio=mfNNratio
                        //checkOri = mbCheckOrientation
                        if(mbCheckOrientation)
                        {   
                            //Fkp为图像帧中最佳匹配的特征点
                            cv::KeyPoint &Fkp =
                                    (!pKF->mpCamera2 || F.Nleft == -1) ? F.mvKeys[bestIdxF] :
                                    (bestIdxF >= F.Nleft) ? F.mvKeysRight[bestIdxF - F.Nleft]
                                                          : F.mvKeys[bestIdxF];

                            //计算匹配点旋转角度差所在的直方图
                            float rot = kp.angle-Fkp.angle;
                            if(rot<0.0)
                                rot+=360.0f;
                            int bin = round(rot*factor);
                            if(bin==HISTO_LENGTH)
                                bin=0;
                            assert(bin>=0 && bin<HISTO_LENGTH);
                            rotHist[bin].push_back(bestIdxF);
                        }
                        //匹配成功数目+1
                        nmatches++;
                    }

                    //若存在右图像
                    if(bestDist1R<=TH_LOW)
                    {
                        if(static_cast<float>(bestDist1R)<mfNNratio*static_cast<float>(bestDist2R) || true)
                        {
                            vpMapPointMatches[bestIdxFR]=pMP;

                            const cv::KeyPoint &kp =
                                    (!pKF->mpCamera2) ? pKF->mvKeysUn[realIdxKF] :
                                    (realIdxKF >= pKF -> NLeft) ? pKF -> mvKeysRight[realIdxKF - pKF -> NLeft]
                                                                : pKF -> mvKeys[realIdxKF];

                            if(mbCheckOrientation)
                            {
                                cv::KeyPoint &Fkp =
                                        (!F.mpCamera2) ? F.mvKeys[bestIdxFR] :
                                        (bestIdxFR >= F.Nleft) ? F.mvKeysRight[bestIdxFR - F.Nleft]
                                                               : F.mvKeys[bestIdxFR];

                                float rot = kp.angle-Fkp.angle;
                                if(rot<0.0)
                                    rot+=360.0f;
                                int bin = round(rot*factor);
                                if(bin==HISTO_LENGTH)
                                    bin=0;
                                assert(bin>=0 && bin<HISTO_LENGTH);
                                rotHist[bin].push_back(bestIdxFR);
                            }
                            nmatches++;
                        }
                    }
                }

            }

            
            KFit++;
            Fit++;
        }

        //对齐
        else if(KFit->first < Fit->first)
        {
            KFit = vFeatVecKF.lower_bound(Fit->first);
        }
        else
        {
            Fit = F.mFeatVec.lower_bound(KFit->first);
        }
    }


    //根据方向剔除误匹配的特征点
    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;
        
        // 筛选出在旋转角度差落在在直方图区间内数量最多的前三个bin的索引
        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                vpMapPointMatches[rotHist[i][j]]=static_cast<MapPoint*>(NULL);
                nmatches--;
            }
        }
    }

    return nmatches;
}





int ORBmatcher::SearchByProjection(KeyFrame* pKF, cv::Mat Scw, const vector<MapPoint*> &vpPoints,
                                   vector<MapPoint*> &vpMatched, int th, float ratioHamming)
{
    // Get Calibration Parameters for later projection
    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;

    // Decompose Scw
    cv::Mat sRcw = Scw.rowRange(0,3).colRange(0,3);
    const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));
    cv::Mat Rcw = sRcw/scw;
    cv::Mat tcw = Scw.rowRange(0,3).col(3)/scw;
    cv::Mat Ow = -Rcw.t()*tcw;

    // Set of MapPoints already found in the KeyFrame
    set<MapPoint*> spAlreadyFound(vpMatched.begin(), vpMatched.end());
    spAlreadyFound.erase(static_cast<MapPoint*>(NULL));

    int nmatches=0;

    // For each Candidate MapPoint Project and Match
    for(int iMP=0, iendMP=vpPoints.size(); iMP<iendMP; iMP++)
    {
        MapPoint* pMP = vpPoints[iMP];

        // Discard Bad MapPoints and already found
        if(pMP->isBad() || spAlreadyFound.count(pMP))
            continue;

        // Get 3D Coords.
        cv::Mat p3Dw = pMP->GetWorldPos();

        // Transform into Camera Coords.
        cv::Mat p3Dc = Rcw*p3Dw+tcw;

        // Depth must be positive
        if(p3Dc.at<float>(2)<0.0)
            continue;

        // Project into Image
        const float x = p3Dc.at<float>(0);
        const float y = p3Dc.at<float>(1);
        const float z = p3Dc.at<float>(2);

        const cv::Point2f uv = pKF->mpCamera->project(cv::Point3f(x,y,z));

        // Point must be inside the image
        if(!pKF->IsInImage(uv.x,uv.y))
            continue;

        // Depth must be inside the scale invariance region of the point
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        cv::Mat PO = p3Dw-Ow;
        const float dist = cv::norm(PO);

        if(dist<minDistance || dist>maxDistance)
            continue;

        // Viewing angle must be less than 60 deg
        cv::Mat Pn = pMP->GetNormal();

        if(PO.dot(Pn)<0.5*dist)
            continue;

        int nPredictedLevel = pMP->PredictScale(dist,pKF);

        // Search in a radius
        const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF->GetFeaturesInArea(uv.x,uv.y,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = 256;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;
            if(vpMatched[idx])
                continue;

            const int &kpLevel= pKF->mvKeysUn[idx].octave;

            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF->mDescriptors.row(idx);

            const int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if(bestDist<=TH_LOW*ratioHamming)
        {
            vpMatched[bestIdx]=pMP;
            nmatches++;
        }

    }

    return nmatches;
}

int ORBmatcher::SearchByProjection(KeyFrame* pKF, cv::Mat Scw, const std::vector<MapPoint*> &vpPoints, const std::vector<KeyFrame*> &vpPointsKFs,
                       std::vector<MapPoint*> &vpMatched, std::vector<KeyFrame*> &vpMatchedKF, int th, float ratioHamming)
{
    // Get Calibration Parameters for later projection
    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;

    // Decompose Scw
    cv::Mat sRcw = Scw.rowRange(0,3).colRange(0,3);
    const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));
    cv::Mat Rcw = sRcw/scw;
    cv::Mat tcw = Scw.rowRange(0,3).col(3)/scw;
    cv::Mat Ow = -Rcw.t()*tcw;

    // Set of MapPoints already found in the KeyFrame
    set<MapPoint*> spAlreadyFound(vpMatched.begin(), vpMatched.end());
    spAlreadyFound.erase(static_cast<MapPoint*>(NULL));

    int nmatches=0;

    // For each Candidate MapPoint Project and Match
    for(int iMP=0, iendMP=vpPoints.size(); iMP<iendMP; iMP++)
    {
        MapPoint* pMP = vpPoints[iMP];
        KeyFrame* pKFi = vpPointsKFs[iMP];

        // Discard Bad MapPoints and already found
        if(pMP->isBad() || spAlreadyFound.count(pMP))
            continue;

        // Get 3D Coords.
        cv::Mat p3Dw = pMP->GetWorldPos();

        // Transform into Camera Coords.
        cv::Mat p3Dc = Rcw*p3Dw+tcw;

        // Depth must be positive
        if(p3Dc.at<float>(2)<0.0)
            continue;

        // Project into Image
        const float invz = 1/p3Dc.at<float>(2);
        const float x = p3Dc.at<float>(0)*invz;
        const float y = p3Dc.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF->IsInImage(u,v))
            continue;

        // Depth must be inside the scale invariance region of the point
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        cv::Mat PO = p3Dw-Ow;
        const float dist = cv::norm(PO);

        if(dist<minDistance || dist>maxDistance)
            continue;

        // Viewing angle must be less than 60 deg
        cv::Mat Pn = pMP->GetNormal();

        if(PO.dot(Pn)<0.5*dist)
            continue;

        int nPredictedLevel = pMP->PredictScale(dist,pKF);

        // Search in a radius
        const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = 256;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;
            if(vpMatched[idx])
                continue;

            const int &kpLevel= pKF->mvKeysUn[idx].octave;

            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF->mDescriptors.row(idx);

            const int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if(bestDist<=TH_LOW*ratioHamming)
        {
            vpMatched[bestIdx] = pMP;
            vpMatchedKF[bestIdx] = pKFi;
            nmatches++;
        }

    }

    return nmatches;
}


/***********初始匹配**********************
 * F1：参考帧
 * F2：当前帧
 * vbPrevMatched:参考帧的关键点
 * vnMatches12:存储F1和F2之间匹配的特征点
 * 
 * 方法：遍历参考图像中所有的特征点，在当前图像帧中
 * 以参考图像帧特征点的坐标为中心，在其搜索范围内
 * 搜索候选特征点，通过计算特征点之间的描述子距离
 * 实现匹配
***************************************/
int ORBmatcher::SearchForInitialization(Frame &F1, Frame &F2, vector<cv::Point2f> &vbPrevMatched, vector<int> &vnMatches12, int windowSize)
{
    int nmatches=0;
    //容量与参考帧关键点数量相等 数值为-1，表示未匹配（大小与F1中的前者相同）
    //从F1到F2的匹配
    vnMatches12 = vector<int>(F1.mvKeysUn.size(),-1);

    //构建旋转直方图 HISTO_LENGTH=30
    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);

    //! 原作者代码是 const float factor = 1.0f/HISTO_LENGTH; 是错误的，更改为下面代码   
    const float factor = HISTO_LENGTH/360.0f;
    
    //存储匹配点对的距离，按照F2的特征点数目分配空间
    vector<int> vMatchedDistance(F2.mvKeysUn.size(),INT_MAX);//容量为当前帧的关键点数量，
    //从F2到F1的匹配，分配F2的特征点数目
    vector<int> vnMatches21(F2.mvKeysUn.size(),-1); //容量为当前帧的关键点数量，大小为-1

    //遍历参考帧F1中的关键点
    for(size_t i1=0, iend1=F1.mvKeysUn.size(); i1<iend1; i1++)
    {
        cv::KeyPoint kp1 = F1.mvKeysUn[i1];
        //octave:特征点提取时所在的金字塔层数
        int level1 = kp1.octave;
        // 只使用原始图像上提取的特征点
        if(level1>0)
            continue;

        //在半径窗口内搜索当前帧F2中所有的候选匹配特征点
        /****************************************
         * vbPrevMatched:存储的是参考帧F1的特征点
         * windowSize：搜索大小
         * level1：
        **************************************/
        vector<size_t> vIndices2 = F2.GetFeaturesInArea(vbPrevMatched[i1].x,
                                                        vbPrevMatched[i1].y, 
                                                        windowSize,
                                                        level1,
                                                        level1);

        if(vIndices2.empty())
            continue;

        // 取出参考帧F1中当前遍历特征点对应的描述子
        cv::Mat d1 = F1.mDescriptors.row(i1);

        int bestDist = INT_MAX;  //最佳描述子匹配距离，越小越好
        int bestDist2 = INT_MAX;    //次佳描述子匹配距离
        int bestIdx2 = -1;          //最佳候选特征点在F2中的index

        // 遍历搜索搜索窗口中的所有潜在的匹配候选点，找到最优的和次优的
        for(vector<size_t>::iterator vit=vIndices2.begin(); vit!=vIndices2.end(); vit++)
        {
            size_t i2 = *vit;
            // 取出候选特征点对应的描述子
            cv::Mat d2 = F2.mDescriptors.row(i2);
            // 计算两个特征点描述子距离
            int dist = DescriptorDistance(d1,d2);

            if(vMatchedDistance[i2]<=dist)
                continue;
            // 如果当前匹配距离更小，更新最佳次佳距离
            if(dist<bestDist)
            {
                bestDist2=bestDist;
                bestDist=dist;
                bestIdx2=i2;
            }
            else if(dist<bestDist2)
            {
                bestDist2=dist;
            }
        }
        if(bestDist<=TH_LOW)
        {
            if(bestDist<(float)bestDist2*mfNNratio)
            {
                if(vnMatches21[bestIdx2]>=0)
                {
                    vnMatches12[vnMatches21[bestIdx2]]=-1;
                    nmatches--;
                }
                //vnMatches12中存储的是F1中,关键点index为i1所对应的F2中关键点的index
                //i1:F1的关键点的序号
                //vnMatches12[i1]：F2中与i1匹配的关键点的序号
                vnMatches12[i1]=bestIdx2;
                //bestIdx2：F2中关键点的序号
                //i1：F1中与bestIdx2匹配的关键点的序号
                vnMatches21[bestIdx2]=i1;
                vMatchedDistance[bestIdx2]=bestDist;
                nmatches++;//匹配的关键点的个数

                if(mbCheckOrientation)
                {
                    float rot = F1.mvKeysUn[i1].angle-F2.mvKeysUn[bestIdx2].angle;
                    if(rot<0.0)
                        rot+=360.0f;
                    int bin = round(rot*factor);
                    if(bin==HISTO_LENGTH)
                        bin=0;
                    assert(bin>=0 && bin<HISTO_LENGTH);
                    rotHist[bin].push_back(i1);
                }
            }
        }

    }

    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                int idx1 = rotHist[i][j];
                if(vnMatches12[idx1]>=0)
                {
                    vnMatches12[idx1]=-1;
                    nmatches--;
                }
            }
        }

    }

    //Update prev matched
    for(size_t i1=0, iend1=vnMatches12.size(); i1<iend1; i1++)
        if(vnMatches12[i1]>=0)
            vbPrevMatched[i1]=F2.mvKeysUn[vnMatches12[i1]].pt;//i1：F1中的特征点序号  vbPrevMatched：匹配的F2的特征点坐标值

    return nmatches;
}


/*********************回环检测中使用********************
 * **************************************************/
int ORBmatcher::SearchByBoW(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches12)
{
    const vector<cv::KeyPoint> &vKeysUn1 = pKF1->mvKeysUn;
    const DBoW2::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
    const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
    const cv::Mat &Descriptors1 = pKF1->mDescriptors;

    const vector<cv::KeyPoint> &vKeysUn2 = pKF2->mvKeysUn;
    const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec;
    const vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();
    const cv::Mat &Descriptors2 = pKF2->mDescriptors;

    vpMatches12 = vector<MapPoint*>(vpMapPoints1.size(),static_cast<MapPoint*>(NULL));
    vector<bool> vbMatched2(vpMapPoints2.size(),false);

    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);

    const float factor = 1.0f/HISTO_LENGTH;

    int nmatches = 0;

    DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
    DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
    DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
    DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();

    while(f1it != f1end && f2it != f2end)
    {
        if(f1it->first == f2it->first)
        {
            for(size_t i1=0, iend1=f1it->second.size(); i1<iend1; i1++)
            {
                const size_t idx1 = f1it->second[i1];
                if(pKF1 -> NLeft != -1 && idx1 >= pKF1 -> mvKeysUn.size()){
                    continue;
                }

                MapPoint* pMP1 = vpMapPoints1[idx1];
                if(!pMP1)
                    continue;
                if(pMP1->isBad())
                    continue;

                const cv::Mat &d1 = Descriptors1.row(idx1);

                int bestDist1=256;
                int bestIdx2 =-1 ;
                int bestDist2=256;

                for(size_t i2=0, iend2=f2it->second.size(); i2<iend2; i2++)
                {
                    const size_t idx2 = f2it->second[i2];

                    if(pKF2 -> NLeft != -1 && idx2 >= pKF2 -> mvKeysUn.size()){
                        continue;
                    }

                    MapPoint* pMP2 = vpMapPoints2[idx2];

                    if(vbMatched2[idx2] || !pMP2)
                        continue;

                    if(pMP2->isBad())
                        continue;

                    const cv::Mat &d2 = Descriptors2.row(idx2);

                    int dist = DescriptorDistance(d1,d2);

                    if(dist<bestDist1)
                    {
                        bestDist2=bestDist1;
                        bestDist1=dist;
                        bestIdx2=idx2;
                    }
                    else if(dist<bestDist2)
                    {
                        bestDist2=dist;
                    }
                }

                if(bestDist1<TH_LOW)
                {
                    if(static_cast<float>(bestDist1)<mfNNratio*static_cast<float>(bestDist2))
                    {
                        vpMatches12[idx1]=vpMapPoints2[bestIdx2];
                        vbMatched2[bestIdx2]=true;

                        if(mbCheckOrientation)
                        {
                            float rot = vKeysUn1[idx1].angle-vKeysUn2[bestIdx2].angle;
                            if(rot<0.0)
                                rot+=360.0f;
                            int bin = round(rot*factor);
                            if(bin==HISTO_LENGTH)
                                bin=0;
                            assert(bin>=0 && bin<HISTO_LENGTH);
                            rotHist[bin].push_back(idx1);
                        }
                        nmatches++;
                    }
                }
            }

            f1it++;
            f2it++;
        }
        else if(f1it->first < f2it->first)
        {
            f1it = vFeatVec1.lower_bound(f2it->first);
        }
        else
        {
            f2it = vFeatVec2.lower_bound(f1it->first);
        }
    }

    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                vpMatches12[rotHist[i][j]]=static_cast<MapPoint*>(NULL);
                nmatches--;
            }
        }
    }

    return nmatches;
}



/****************匹配当前关键帧和周围关键帧****************************
 * 输入：PK1：当前关键帧
 *      PK2：当前关键帧的周围关键帧
 *      F12：关键帧之间的本质矩阵F(2->1)
 *      vMatchedPairs:用于存储匹配关系，初始为空，大小是匹配成功的数量，
 *                    fisrt表示当前关键帧pKF1中特征点的ID
 *                    second表示周围关键帧pKF2中特征点ID
 *      bOnlyStereo：FALSE
 *      bCoarse:标志位，如果是非IMU,则为false，有IMU传入前函数还要进行判断
********************************************************************/   
int ORBmatcher::SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2, cv::Mat F12,
                                       vector<pair<size_t, size_t> > &vMatchedPairs, const bool bOnlyStereo, const bool bCoarse)
{    
    //BOW特征向量
    const DBoW2::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
    const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec;

    //Compute epipole in second image
    //PKF1的相机光心在世界坐标系下的位置
    cv::Mat Cw = pKF1->GetCameraCenter();
    //世界坐标系到关键帧2的相机坐标系下的旋转
    cv::Mat R2w = pKF2->GetRotation();
    //关键帧2坐标系下，世界坐标系的位置
    cv::Mat t2w = pKF2->GetTranslation();
    //KF2坐标系下，KF1相机光心的位置
    cv::Mat C2 = R2w*Cw+t2w;


    // KF1的相机光心在PK2的像素平面的投影
    cv::Point2f ep = pKF2->mpCamera->project(C2);


    //世界坐标系到PK1坐标系的旋转
    cv::Mat R1w = pKF1->GetRotation();
    //PK1坐标系下，世界坐标系的位置
    cv::Mat t1w = pKF1->GetTranslation();

    //PK2到PK1的旋转
    cv::Mat R12;
    //PK1坐标系下，PK2的位置
    cv::Mat t12;

    cv::Mat Rll,Rlr,Rrl,Rrr;
    cv::Mat tll,tlr,trl,trr;

    //相机模型
    GeometricCamera* pCamera1 = pKF1->mpCamera, *pCamera2 = pKF2->mpCamera;

    //如果是单目情况，计算R12和t12
    if(!pKF1->mpCamera2 && !pKF2->mpCamera2)
    {
        R12 = R1w*R2w.t();
        t12 = -R1w*R2w.t()*t2w+t1w;
    }
    else
    {
        Rll = pKF1->GetRotation() * pKF2->GetRotation().t();
        Rlr = pKF1->GetRotation() * pKF2->GetRightRotation().t();
        Rrl = pKF1->GetRightRotation() * pKF2->GetRotation().t();
        Rrr = pKF1->GetRightRotation() * pKF2->GetRightRotation().t();

        tll = pKF1->GetRotation() * (-pKF2->GetRotation().t() * pKF2->GetTranslation()) + pKF1->GetTranslation();
        tlr = pKF1->GetRotation() * (-pKF2->GetRightRotation().t() * pKF2->GetRightTranslation()) + pKF1->GetTranslation();
        trl = pKF1->GetRightRotation() * (-pKF2->GetRotation().t() * pKF2->GetTranslation()) + pKF1->GetRightTranslation();
        trr = pKF1->GetRightRotation() * (-pKF2->GetRightRotation().t() * pKF2->GetRightTranslation()) + pKF1->GetRightTranslation();
    }

    // Find matches between not tracked keypoints
    // Matching speed-up by ORB Vocabulary
    // Compare only ORB that share the same node

    int nmatches=0;
    //存储PK2特征点的匹配情况（PK2是周围关键帧）
    vector<bool> vbMatched2(pKF2->N,false);
    //存储PK1特征点与PK2之间的匹配情况，容量为PK1的特征点
    //数量，成功为PK2的特征点ID，不成功为-1（PK1是当前关键帧）
    vector<int> vMatches12(pKF1->N,-1);

    //直方图
    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);

    //尺度因子
    const float factor = 1.0f/HISTO_LENGTH; 

    // f1it->first对应node编号，f1it->second对应属于该node的所有特特征点编号
    DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
    DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
    DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
    DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();

    //遍历特征向量
    while(f1it!=f1end && f2it!=f2end)
    {
        //当两个关键帧的节点相同时（取相同节点）
        if(f1it->first == f2it->first)
        {
            //遍历当前关键帧中此节点下的特征点
            for(size_t i1=0, iend1=f1it->second.size(); i1<iend1; i1++)
            {
                //当前关键帧的特征点ID
                const size_t idx1 = f1it->second[i1];
                
                //当前关键帧的第idx1号特征点的地图点
                MapPoint* pMP1 = pKF1->GetMapPoint(idx1);
                
                // If there is already a MapPoint skip
                //如果当前地图点存在，说明不需要再匹配，当前操作是构造更多的地图点
                if(pMP1)
                {
                    continue;
                }

                const bool bStereo1 = (!pKF1->mpCamera2 && pKF1->mvuRight[idx1]>=0);

                if(bOnlyStereo)
                    if(!bStereo1)
                        continue;


                //提取当前关键帧的特征点
                const cv::KeyPoint &kp1 = (pKF1 -> NLeft == -1) ? pKF1->mvKeysUn[idx1]
                                                                : (idx1 < pKF1 -> NLeft) ? pKF1 -> mvKeys[idx1]
                                                                                         : pKF1 -> mvKeysRight[idx1 - pKF1 -> NLeft];

                const bool bRight1 = (pKF1 -> NLeft == -1 || idx1 < pKF1 -> NLeft) ? false
                                                                                   : true;
                //if(bRight1) continue;
                //获取当前特征点的描述子
                const cv::Mat &d1 = pKF1->mDescriptors.row(idx1);
                
                int bestDist = TH_LOW;
                int bestIdx2 = -1;
                
                //遍历此节点下周围关键帧的特征点集合
                for(size_t i2=0, iend2=f2it->second.size(); i2<iend2; i2++)
                {
                    //周围关键帧的特征点ID
                    size_t idx2 = f2it->second[i2];
                    
                    //周围关键帧的地图点
                    MapPoint* pMP2 = pKF2->GetMapPoint(idx2);
                    

                    // If we have already matched or there is a MapPoint skip
                    //如果此特征点已匹配或已存在地图点，跳过
                    if(vbMatched2[idx2] || pMP2)
                        continue;

                    const bool bStereo2 = (!pKF2->mpCamera2 &&  pKF2->mvuRight[idx2]>=0);

                    if(bOnlyStereo)
                        if(!bStereo2)
                            continue;
                    
                    //周围关键帧中此特征点对应的描述子
                    const cv::Mat &d2 = pKF2->mDescriptors.row(idx2);
                    
                    // 描述子距离
                    const int dist = DescriptorDistance(d1,d2);
                    
                    if(dist>TH_LOW || dist>bestDist)
                        continue;

                    //周围关键帧中的此特征点
                    const cv::KeyPoint &kp2 = (pKF2 -> NLeft == -1) ? pKF2->mvKeysUn[idx2]
                                                                    : (idx2 < pKF2 -> NLeft) ? pKF2 -> mvKeys[idx2]
                                                                                             : pKF2 -> mvKeysRight[idx2 - pKF2 -> NLeft];
                    const bool bRight2 = (pKF2 -> NLeft == -1 || idx2 < pKF2 -> NLeft) ? false
                                                                                       : true;

                    //单目进行像素点到极点的判断
                    if(!bStereo1 && !bStereo2 && !pKF1->mpCamera2)
                    {
                        // ep：KF1的相机光心在PK2的像素平面的投影
                        const float distex = ep.x-kp2.pt.x;
                        const float distey = ep.y-kp2.pt.y;
                        // 极点e2到kp2的像素距离如果小于阈值th,认为kp2对应的MapPoint距离pKF1相机太近，跳过该匹配点对
                        if(distex*distex+distey*distey<100*pKF2->mvScaleFactors[kp2.octave])
                        {
                            continue;
                        }
                    }

                    if(pKF1->mpCamera2 && pKF2->mpCamera2)
                    {
                        if(bRight1 && bRight2){
                            R12 = Rrr;
                            t12 = trr;

                            pCamera1 = pKF1->mpCamera2;
                            pCamera2 = pKF2->mpCamera2;
                        }
                        else if(bRight1 && !bRight2){
                            R12 = Rrl;
                            t12 = trl;

                            pCamera1 = pKF1->mpCamera2;
                            pCamera2 = pKF2->mpCamera;
                        }
                        else if(!bRight1 && bRight2){
                            R12 = Rlr;
                            t12 = tlr;

                            pCamera1 = pKF1->mpCamera;
                            pCamera2 = pKF2->mpCamera2;
                        }
                        else{
                            R12 = Rll;
                            t12 = tll;

                            pCamera1 = pKF1->mpCamera;
                            pCamera2 = pKF2->mpCamera;
                        }

                    }

                    //计算特征点kp2到kp1对应极线的距离是不是小于预知
                    if(pCamera1->epipolarConstrain(pCamera2,kp1,kp2,R12,t12,pKF1->mvLevelSigma2[kp1.octave],pKF2->mvLevelSigma2[kp2.octave])||bCoarse) // MODIFICATION_2
                    {
                        bestIdx2 = idx2; //周围关键帧的特征点ID
                        bestDist = dist;
                    }
                }
                
                if(bestIdx2>=0)
                {
                    //周围关键帧最佳匹配特征点的关键点
                    const cv::KeyPoint &kp2 = (pKF2 -> NLeft == -1) ? pKF2->mvKeysUn[bestIdx2]
                                                                    : (bestIdx2 < pKF2 -> NLeft) ? pKF2 -> mvKeys[bestIdx2]
                                                                    : pKF2 -> mvKeysRight[bestIdx2 - pKF2 -> NLeft];
                    //当前关键帧中的idx1号特征点对应的周围关键帧bestIdx2号特征点的匹配关系
                    
                    
                    vMatches12[idx1]=bestIdx2;
                    nmatches++;
                    
                    //如果进行旋转检查
                    if(mbCheckOrientation)
                    {
                        float rot = kp1.angle-kp2.angle;
                        if(rot<0.0)
                            rot+=360.0f;
                        int bin = round(rot*factor);
                        if(bin==HISTO_LENGTH)
                            bin=0;
                        assert(bin>=0 && bin<HISTO_LENGTH);
                        rotHist[bin].push_back(idx1);
                    }
                }
            }

            f1it++;
            f2it++;
        }
        else if(f1it->first < f2it->first)
        {
            f1it = vFeatVec1.lower_bound(f2it->first);
        }
        else
        {
            f2it = vFeatVec2.lower_bound(f1it->first);
        }
    }

    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                vMatches12[rotHist[i][j]]=-1;
                nmatches--;
            }
        }

    }

    vMatchedPairs.clear();
    vMatchedPairs.reserve(nmatches);

    //记录匹配关系
    for(size_t i=0, iend=vMatches12.size(); i<iend; i++)
    {
        if(vMatches12[i]<0)
            continue;
        //first是当前关键帧匹配成功的特征点ID，second是周围关键帧中匹配成功的特征点ID
        vMatchedPairs.push_back(make_pair(i,vMatches12[i]));
    }

    return nmatches;
}





int ORBmatcher::SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2, cv::Mat F12,
                                        vector<pair<size_t, size_t> > &vMatchedPairs, const bool bOnlyStereo, vector<cv::Mat> &vMatchedPoints)
{
    const DBoW2::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
    const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec;

    //Compute epipole in second image
    cv::Mat Cw = pKF1->GetCameraCenter();
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();
    cv::Mat C2 = R2w*Cw+t2w;

    cv::Point2f ep = pKF2->mpCamera->project(C2);

    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();

    GeometricCamera* pCamera1 = pKF1->mpCamera, *pCamera2 = pKF2->mpCamera;
    cv::Mat Tcw1,Tcw2;

    // Find matches between not tracked keypoints
    // Matching speed-up by ORB Vocabulary
    // Compare only ORB that share the same node

    int nmatches=0;
    vector<bool> vbMatched2(pKF2->N,false);
    vector<int> vMatches12(pKF1->N,-1);

    vector<cv::Mat> vMatchesPoints12(pKF1 -> N);

    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);

    const float factor = 1.0f/HISTO_LENGTH;

    DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
    DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
    DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
    DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();
    int right = 0;
    while(f1it!=f1end && f2it!=f2end)
    {
        if(f1it->first == f2it->first)
        {
            for(size_t i1=0, iend1=f1it->second.size(); i1<iend1; i1++)
            {
                const size_t idx1 = f1it->second[i1];

                MapPoint* pMP1 = pKF1->GetMapPoint(idx1);

                // If there is already a MapPoint skip
                if(pMP1)
                    continue;

                const cv::KeyPoint &kp1 = (pKF1 -> NLeft == -1) ? pKF1->mvKeysUn[idx1]
                                                                : (idx1 < pKF1 -> NLeft) ? pKF1 -> mvKeys[idx1]
                                                                                            : pKF1 -> mvKeysRight[idx1 - pKF1 -> NLeft];

                const bool bRight1 = (pKF1 -> NLeft == -1 || idx1 < pKF1 -> NLeft) ? false
                                                                                    : true;


                const cv::Mat &d1 = pKF1->mDescriptors.row(idx1);

                int bestDist = TH_LOW;
                int bestIdx2 = -1;

                cv::Mat bestPoint;

                for(size_t i2=0, iend2=f2it->second.size(); i2<iend2; i2++)
                {
                    size_t idx2 = f2it->second[i2];

                    MapPoint* pMP2 = pKF2->GetMapPoint(idx2);

                    // If we have already matched or there is a MapPoint skip
                    if(vbMatched2[idx2] || pMP2)
                        continue;

                    const cv::Mat &d2 = pKF2->mDescriptors.row(idx2);

                    const int dist = DescriptorDistance(d1,d2);

                    if(dist>TH_LOW || dist>bestDist){
                        continue;
                    }


                    const cv::KeyPoint &kp2 = (pKF2 -> NLeft == -1) ? pKF2->mvKeysUn[idx2]
                                                                    : (idx2 < pKF2 -> NLeft) ? pKF2 -> mvKeys[idx2]
                                                                                                : pKF2 -> mvKeysRight[idx2 - pKF2 -> NLeft];
                    const bool bRight2 = (pKF2 -> NLeft == -1 || idx2 < pKF2 -> NLeft) ? false
                                                                                        : true;

                    if(bRight1){
                        Tcw1 = pKF1->GetRightPose();
                        pCamera1 = pKF1->mpCamera2;
                    } else{
                        Tcw1 = pKF1->GetPose();
                        pCamera1 = pKF1->mpCamera;
                    }

                    if(bRight2){
                        Tcw2 = pKF2->GetRightPose();
                        pCamera2 = pKF2->mpCamera2;
                    } else{
                        Tcw2 = pKF2->GetPose();
                        pCamera2 = pKF2->mpCamera;
                    }

                    cv::Mat x3D;
                    if(pCamera1->matchAndtriangulate(kp1,kp2,pCamera2,Tcw1,Tcw2,pKF1->mvLevelSigma2[kp1.octave],pKF2->mvLevelSigma2[kp2.octave],x3D)){
                        bestIdx2 = idx2;
                        bestDist = dist;
                        bestPoint = x3D;
                    }

                }

                if(bestIdx2>=0)
                {
                    const cv::KeyPoint &kp2 = (pKF2 -> NLeft == -1) ? pKF2->mvKeysUn[bestIdx2]
                                                                    : (bestIdx2 < pKF2 -> NLeft) ? pKF2 -> mvKeys[bestIdx2]
                                                                                                    : pKF2 -> mvKeysRight[bestIdx2 - pKF2 -> NLeft];
                    vMatches12[idx1]=bestIdx2;
                    vMatchesPoints12[idx1] = bestPoint;
                    nmatches++;
                    if(bRight1) right++;

                    if(mbCheckOrientation)
                    {
                        float rot = kp1.angle-kp2.angle;
                        if(rot<0.0)
                            rot+=360.0f;
                        int bin = round(rot*factor);
                        if(bin==HISTO_LENGTH)
                            bin=0;
                        assert(bin>=0 && bin<HISTO_LENGTH);
                        rotHist[bin].push_back(idx1);
                    }
                }
            }

            f1it++;
            f2it++;
        }
        else if(f1it->first < f2it->first)
        {
            f1it = vFeatVec1.lower_bound(f2it->first);
        }
        else
        {
            f2it = vFeatVec2.lower_bound(f1it->first);
        }
    }

    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                vMatches12[rotHist[i][j]]=-1;
                nmatches--;
            }
        }

    }

    vMatchedPairs.clear();
    vMatchedPairs.reserve(nmatches);

    for(size_t i=0, iend=vMatches12.size(); i<iend; i++)
    {
        if(vMatches12[i]<0)
            continue;
        vMatchedPairs.push_back(make_pair(i,vMatches12[i]));
        vMatchedPoints.push_back(vMatchesPoints12[i]);
    }
    return nmatches;
}


/***************匹配判断vmMapPoints中的特征点和pKF中的特征点是否重合
//pKF：待匹配的关键帧
//vpMapPoints:当前关键帧的地图点
********************************************************************/   
int ORBmatcher::Fuse(KeyFrame *pKF, const vector<MapPoint *> &vpMapPoints, const float th, const bool bRight)
{
    cv::Mat Rcw,tcw, Ow;
    GeometricCamera* pCamera;

    if(bRight)
    {
        Rcw = pKF->GetRightRotation();
        tcw = pKF->GetRightTranslation();
        Ow = pKF->GetRightCameraCenter();

        pCamera = pKF->mpCamera2;
    }
    else  //单目情况
    {
        //世界坐标系到待匹配的关键帧的旋转
        Rcw = pKF->GetRotation();
        //待匹配关键帧坐标系下世界坐标系的位置
        tcw = pKF->GetTranslation();
        //世界坐标系的位置
        Ow = pKF->GetCameraCenter();

        pCamera = pKF->mpCamera;
    }

    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;
    const float &bf = pKF->mbf;

    int nFused=0;

    //地图点数量
    const int nMPs = vpMapPoints.size();

    // For debbuging
    int count_notMP = 0, count_bad=0, count_isinKF = 0, count_negdepth = 0, 
    count_notinim = 0, count_dist = 0, count_normal=0, count_notidx = 0, count_thcheck = 0;
    
    //遍历所有当前关键帧的地图点
    for(int i=0; i<nMPs; i++)
    {
        //提取地图点
        MapPoint* pMP = vpMapPoints[i];

        //如果地图点不存在，跳过
        if(!pMP)
        {
            count_notMP++;
            continue;
        }

        /*if(pMP->isBad() || pMP->IsInKeyFrame(pKF))
            continue;*/
        //如果是坏点或者这个地图点就是此关键帧的地图点，那就不需要融合
        //需要融合的是，来自不同关键帧，其实是同一个地图点的地图点
        if(pMP->isBad())
        {
            count_bad++;
            continue;
        }
        else if(pMP->IsInKeyFrame(pKF))
        {
            count_isinKF++;
            continue;
        }


        //获得当前关键帧的地图点的世界坐标
        cv::Mat p3Dw = pMP->GetWorldPos();
        //当前地图点在此待匹配关键帧坐标系下的位置
        cv::Mat p3Dc = Rcw*p3Dw + tcw;

        // Depth must be positive
        //跳过深度值为负的3D点
        if(p3Dc.at<float>(2)<0.0f)
        {
            count_negdepth++;
            continue;
        }


        //此地图点在待匹配关键帧中的图像坐标
        const float invz = 1/p3Dc.at<float>(2);
        const float x = p3Dc.at<float>(0);
        const float y = p3Dc.at<float>(1);
        const float z = p3Dc.at<float>(2);

        //此地图点在待匹配关键帧中的像素坐标
        const cv::Point2f uv = pCamera->project(cv::Point3f(x,y,z));

        // Point must be inside the image
        //是否此坐标在待匹配关键帧的图像范围内,若不在此图像范围内则不需要融合，跳过此地图点
        if(!pKF->IsInImage(uv.x,uv.y))
        {
            count_notinim++;
            continue;
        }
        
        //没有用到
        const float ur = uv.x-bf*invz;


        // 检验地图点到关键帧相机光心距离需满足在有效范围内
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        cv::Mat PO = p3Dw-Ow;
        const float dist3D = cv::norm(PO);
        // Depth must be inside the scale pyramid of the image
        if(dist3D<minDistance || dist3D>maxDistance)
        {
            count_dist++;
            continue;
        }


        //此地图点的平均观测方向小于60度
        // Viewing angle must be less than 60 deg
        cv::Mat Pn = pMP->GetNormal();
        if(PO.dot(Pn)<0.5*dist3D)
        {
            count_normal++;
            continue;
        }

        //根据地图点的深度确定尺度，从而确定搜索范围
        int nPredictedLevel = pMP->PredictScale(dist3D,pKF);

        // Search in a radius
        //确定搜索半径
        const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

        //在当前待匹配关键帧中的投影坐标的半径范围内寻找特征点
        //ux,uy是地图点在待匹配关键帧中的投影像素坐标
        const vector<size_t> vIndices = pKF->GetFeaturesInArea(uv.x,uv.y,radius,bRight);

        if(vIndices.empty())
        {
            count_notidx++;
            continue;
        }

        // Match to the most similar keypoint in the radius
        // 获得此地图点的描述子
        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = 256;
        int bestIdx = -1;
        //遍历待匹配关键帧中选出的特征点
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            size_t idx = *vit;
            //待匹配的特征点
            const cv::KeyPoint &kp = (pKF -> NLeft == -1) ? pKF->mvKeysUn[idx]
                                                          : (!bRight) ? pKF -> mvKeys[idx]
                                                                      : pKF -> mvKeysRight[idx];

            const int &kpLevel= kp.octave;

            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                continue;

            if(pKF->mvuRight[idx]>=0)
            {
                // Check reprojection error in stereo
                const float &kpx = kp.pt.x;
                const float &kpy = kp.pt.y;
                const float &kpr = pKF->mvuRight[idx];
                const float ex = uv.x-kpx;
                const float ey = uv.y-kpy;
                const float er = ur-kpr;
                const float e2 = ex*ex+ey*ey+er*er;

                if(e2*pKF->mvInvLevelSigma2[kpLevel]>7.8)
                    continue;
            }
            else
            {
                const float &kpx = kp.pt.x;
                const float &kpy = kp.pt.y;
                const float ex = uv.x-kpx;
                const float ey = uv.y-kpy;
                const float e2 = ex*ex+ey*ey;

                if(e2*pKF->mvInvLevelSigma2[kpLevel]>5.99)
                    continue;
            }

            if(bRight) idx += pKF->NLeft;

            //待匹配关键帧中此特征点的描述子
            const cv::Mat &dKF = pKF->mDescriptors.row(idx);

            //描述子距离
            const int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;  //当前地图点在PKF中的最佳匹配特征点ID
            }
        }

        // If there is already a MapPoint replace otherwise add new measurement
        if(bestDist<=TH_LOW)
        {   
            //如果最佳匹配距离小于阈值，有效匹配


            // 获取待匹配关键帧中此特征点对应的地图点
            MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx);


            if(pMPinKF)//如果此地图点存在
            {
                if(!pMPinKF->isBad())
                {   
                    //如果pKF中的地图点观测数量大于当前地图点的观测
                    if(pMPinKF->Observations()>pMP->Observations())
                        pMP->Replace(pMPinKF);  //替换此地图点
                    else
                        pMPinKF->Replace(pMP);
                }
            }
            else
            {   
                // 如果此最佳特征点不存在地图点，添加观测信息
                pMP->AddObservation(pKF,bestIdx);
                pKF->AddMapPoint(pMP,bestIdx);
            }
            nFused++;
        }
        else
            count_thcheck++;

    }

    /*cout << "count_notMP = " << count_notMP << endl;
    cout << "count_bad = " << count_bad << endl;
    cout << "count_isinKF = " << count_isinKF << endl;
    cout << "count_negdepth = " << count_negdepth << endl;
    cout << "count_notinim = " << count_notinim << endl;
    cout << "count_dist = " << count_dist << endl;
    cout << "count_normal = " << count_normal << endl;
    cout << "count_notidx = " << count_notidx << endl;
    cout << "count_thcheck = " << count_thcheck << endl;
    cout << "tot fused points: " << nFused << endl;*/
    return nFused;
}




int ORBmatcher::Fuse(KeyFrame *pKF, cv::Mat Scw, const vector<MapPoint *> &vpPoints, float th, vector<MapPoint *> &vpReplacePoint)
{
    // Get Calibration Parameters for later projection
    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;

    // Decompose Scw
    cv::Mat sRcw = Scw.rowRange(0,3).colRange(0,3);
    const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));
    cv::Mat Rcw = sRcw/scw;
    cv::Mat tcw = Scw.rowRange(0,3).col(3)/scw;
    cv::Mat Ow = -Rcw.t()*tcw;

    // Set of MapPoints already found in the KeyFrame
    const set<MapPoint*> spAlreadyFound = pKF->GetMapPoints();

    int nFused=0;

    const int nPoints = vpPoints.size();

    // For each candidate MapPoint project and match
    for(int iMP=0; iMP<nPoints; iMP++)
    {
        MapPoint* pMP = vpPoints[iMP];

        // Discard Bad MapPoints and already found
        if(pMP->isBad() || spAlreadyFound.count(pMP))
            continue;

        // Get 3D Coords.
        cv::Mat p3Dw = pMP->GetWorldPos();

        // Transform into Camera Coords.
        cv::Mat p3Dc = Rcw*p3Dw+tcw;

        // Depth must be positive
        if(p3Dc.at<float>(2)<0.0f)
            continue;

        // Project into Image
        const float x = p3Dc.at<float>(0);
        const float y = p3Dc.at<float>(1);
        const float z = p3Dc.at<float>(2);

        const cv::Point2f uv = pKF->mpCamera->project(cv::Point3f(x,y,z));

        // Point must be inside the image
        if(!pKF->IsInImage(uv.x,uv.y))
            continue;

        // Depth must be inside the scale pyramid of the image
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        cv::Mat PO = p3Dw-Ow;
        const float dist3D = cv::norm(PO);

        if(dist3D<minDistance || dist3D>maxDistance)
            continue;

        // Viewing angle must be less than 60 deg
        cv::Mat Pn = pMP->GetNormal();

        if(PO.dot(Pn)<0.5*dist3D)
            continue;

        // Compute predicted scale level
        const int nPredictedLevel = pMP->PredictScale(dist3D,pKF);

        // Search in a radius
        const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF->GetFeaturesInArea(uv.x,uv.y,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius

        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(); vit!=vIndices.end(); vit++)
        {
            const size_t idx = *vit;
            const int &kpLevel = pKF->mvKeysUn[idx].octave;

            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF->mDescriptors.row(idx);

            int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        // If there is already a MapPoint replace otherwise add new measurement
        if(bestDist<=TH_LOW)
        {
            MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx);
            if(pMPinKF)
            {
                if(!pMPinKF->isBad())
                    vpReplacePoint[iMP] = pMPinKF;
            }
            else
            {
                pMP->AddObservation(pKF,bestIdx);
                pKF->AddMapPoint(pMP,bestIdx);
            }
            nFused++;
        }
    }

    return nFused;
}

int ORBmatcher::SearchBySim3(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint*> &vpMatches12,
                             const float &s12, const cv::Mat &R12, const cv::Mat &t12, const float th)
{
    const float &fx = pKF1->fx;
    const float &fy = pKF1->fy;
    const float &cx = pKF1->cx;
    const float &cy = pKF1->cy;

    // Camera 1 from world
    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();

    //Camera 2 from world
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();

    //Transformation between cameras
    cv::Mat sR12 = s12*R12;
    cv::Mat sR21 = (1.0/s12)*R12.t();
    cv::Mat t21 = -sR21*t12;

    const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
    const int N1 = vpMapPoints1.size();

    const vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();
    const int N2 = vpMapPoints2.size();

    vector<bool> vbAlreadyMatched1(N1,false);
    vector<bool> vbAlreadyMatched2(N2,false);

    for(int i=0; i<N1; i++)
    {
        MapPoint* pMP = vpMatches12[i];
        if(pMP)
        {
            vbAlreadyMatched1[i]=true;
            int idx2 = get<0>(pMP->GetIndexInKeyFrame(pKF2));
            if(idx2>=0 && idx2<N2)
                vbAlreadyMatched2[idx2]=true;
        }
    }

    vector<int> vnMatch1(N1,-1);
    vector<int> vnMatch2(N2,-1);

    // Transform from KF1 to KF2 and search
    for(int i1=0; i1<N1; i1++)
    {
        MapPoint* pMP = vpMapPoints1[i1];

        if(!pMP || vbAlreadyMatched1[i1])
            continue;

        if(pMP->isBad())
            continue;

        cv::Mat p3Dw = pMP->GetWorldPos();
        cv::Mat p3Dc1 = R1w*p3Dw + t1w;
        cv::Mat p3Dc2 = sR21*p3Dc1 + t21;

        // Depth must be positive
        if(p3Dc2.at<float>(2)<0.0)
            continue;

        const float invz = 1.0/p3Dc2.at<float>(2);
        const float x = p3Dc2.at<float>(0)*invz;
        const float y = p3Dc2.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF2->IsInImage(u,v))
            continue;

        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        const float dist3D = cv::norm(p3Dc2);

        // Depth must be inside the scale invariance region
        if(dist3D<minDistance || dist3D>maxDistance )
            continue;

        // Compute predicted octave
        const int nPredictedLevel = pMP->PredictScale(dist3D,pKF2);

        // Search in a radius
        const float radius = th*pKF2->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF2->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;

            const cv::KeyPoint &kp = pKF2->mvKeysUn[idx];

            if(kp.octave<nPredictedLevel-1 || kp.octave>nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF2->mDescriptors.row(idx);

            const int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if(bestDist<=TH_HIGH)
        {
            vnMatch1[i1]=bestIdx;
        }
    }

    // Transform from KF2 to KF2 and search
    for(int i2=0; i2<N2; i2++)
    {
        MapPoint* pMP = vpMapPoints2[i2];

        if(!pMP || vbAlreadyMatched2[i2])
            continue;

        if(pMP->isBad())
            continue;

        cv::Mat p3Dw = pMP->GetWorldPos();
        cv::Mat p3Dc2 = R2w*p3Dw + t2w;
        cv::Mat p3Dc1 = sR12*p3Dc2 + t12;

        // Depth must be positive
        if(p3Dc1.at<float>(2)<0.0)
            continue;

        const float invz = 1.0/p3Dc1.at<float>(2);
        const float x = p3Dc1.at<float>(0)*invz;
        const float y = p3Dc1.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF1->IsInImage(u,v))
            continue;

        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        const float dist3D = cv::norm(p3Dc1);

        // Depth must be inside the scale pyramid of the image
        if(dist3D<minDistance || dist3D>maxDistance)
            continue;

        // Compute predicted octave
        const int nPredictedLevel = pMP->PredictScale(dist3D,pKF1);

        // Search in a radius of 2.5*sigma(ScaleLevel)
        const float radius = th*pKF1->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF1->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;

            const cv::KeyPoint &kp = pKF1->mvKeysUn[idx];

            if(kp.octave<nPredictedLevel-1 || kp.octave>nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF1->mDescriptors.row(idx);

            const int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if(bestDist<=TH_HIGH)
        {
            vnMatch2[i2]=bestIdx;
        }
    }

    // Check agreement
    int nFound = 0;

    for(int i1=0; i1<N1; i1++)
    {
        int idx2 = vnMatch1[i1];

        if(idx2>=0)
        {
            int idx1 = vnMatch2[idx2];
            if(idx1==i1)
            {
                vpMatches12[i1] = vpMapPoints2[idx2];
                nFound++;
            }
        }
    }

    return nFound;
}


/***********仅恒速模型中使用，利用特征点投影进行匹配**********************
 * 输入：CurrentFrame：当前图像帧
 *      LastFrame：上一图像帧
 *      th：搜索范围阈值 默认单目为7 (匹配数量不够时进行扩大搜索范围)
 *      bMono  是否为单目
 * 返回：匹配成功数目
 * 原理：将上一图像帧的地图点根据3D-2D投影模型，投影至当前图像帧的像素坐标系中，
 *      在其周围进行搜索，寻找描述子距离最小的点作为匹配结果
*********************************************************************/
int ORBmatcher::SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, const float th, const bool bMono)
{   
    
    //设置初始匹配数目为0
    int nmatches = 0;

    //方向直方图，用于检测特征点的旋转一致性
    // Rotation Histogram (to check rotation consistency)
    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f/HISTO_LENGTH;
    
    //当前图像的相机位姿，世界到当前相机坐标系的位姿
    const cv::Mat Rcw = CurrentFrame.mTcw.rowRange(0,3).colRange(0,3);
    //当前相机坐标系下世界坐标系的位置
    const cv::Mat tcw = CurrentFrame.mTcw.rowRange(0,3).col(3);

    //世界坐标系下当前图像坐标系的位置
    const cv::Mat twc = -Rcw.t()*tcw;

    //上一个图像帧的相机位姿，世界坐标系到上一图像坐标系的位姿
    const cv::Mat Rlw = LastFrame.mTcw.rowRange(0,3).colRange(0,3);
    const cv::Mat tlw = LastFrame.mTcw.rowRange(0,3).col(3);

    //上一图像帧坐标系下，当前图像坐标系的位置
    const cv::Mat tlc = Rlw*twc+tlw;

    //判断前进/后退方向(用于非单目情况，单目情况，前进后退均为false)
    //bMono = true 单目
    //!Mono = false
    //单目时，CurrentFrame.mb = 0
    const bool bForward = tlc.at<float>(2)>CurrentFrame.mb && !bMono;
    const bool bBackward = -tlc.at<float>(2)>CurrentFrame.mb && !bMono;

    //遍历上一图像帧中的特征点
    for(int i=0; i<LastFrame.N; i++)
    {   
        //提取此特征点对应的地图点
        MapPoint* pMP = LastFrame.mvpMapPoints[i];
        if(pMP) //若地图点存在
        {   
            //若此特征点不是外点
            if(!LastFrame.mvbOutlier[i])
            {   
                
                // Project
                //获得此地图点的世界坐标
                cv::Mat x3Dw = pMP->GetWorldPos();
                //计算此地图点在当前图像帧坐标系下的3D位置
                cv::Mat x3Dc = Rcw*x3Dw+tcw;


                const float xc = x3Dc.at<float>(0);
                const float yc = x3Dc.at<float>(1);
                //逆深度
                const float invzc = 1.0/x3Dc.at<float>(2);

                //若逆深度小于0,跳过
                if(invzc<0)
                    continue;

                //投影至当前图像帧的像素坐标系下
                cv::Point2f uv = CurrentFrame.mpCamera->project(x3Dc);

                if(uv.x<CurrentFrame.mnMinX || uv.x>CurrentFrame.mnMaxX)
                    continue;
                if(uv.y<CurrentFrame.mnMinY || uv.y>CurrentFrame.mnMaxY)
                    continue;

                //假定投影前后地图点的尺度信息不变(0-7)
                int nLastOctave = (LastFrame.Nleft == -1 || i < LastFrame.Nleft) ? LastFrame.mvKeys[i].octave
                                                                                    : LastFrame.mvKeysRight[i - LastFrame.Nleft].octave;
                
                // Search in a window. Size depends on scale
                //根据尺度因子计算搜索半径
                float radius = th*CurrentFrame.mvScaleFactors[nLastOctave];

                //记录候选匹配点的ID
                vector<size_t> vIndices2;

                //单目时bForward和bBackward均为false
                if(bForward)
                    vIndices2 = CurrentFrame.GetFeaturesInArea(uv.x,uv.y, radius, nLastOctave);
                else if(bBackward)
                    vIndices2 = CurrentFrame.GetFeaturesInArea(uv.x,uv.y, radius, 0, nLastOctave);
                else// 在[nLastOctave-1, nLastOctave+1]中搜索
                    vIndices2 = CurrentFrame.GetFeaturesInArea(uv.x,uv.y, radius, nLastOctave-1, nLastOctave+1);

                if(vIndices2.empty())
                    continue;
                
                //获得此地图点对应的描述子
                const cv::Mat dMP = pMP->GetDescriptor();

                
                int bestDist = 256; //最佳匹配
                int bestIdx2 = -1;  //匹配最佳对应的ID

                //遍历当前地图点对应的所有的候选匹配点，寻找匹配距离最佳的匹配点
                for(vector<size_t>::const_iterator vit=vIndices2.begin(), vend=vIndices2.end(); vit!=vend; vit++)
                {   
                    //候选匹配点ID
                    const size_t i2 = *vit;

                    //若当前图像帧中的此地图点已匹配过且观测数量大于0,则不进行匹配
                    if(CurrentFrame.mvpMapPoints[i2])
                        if(CurrentFrame.mvpMapPoints[i2]->Observations()>0)
                            continue;

                    // 双目和rgbd的情况
                    if(CurrentFrame.Nleft == -1 && CurrentFrame.mvuRight[i2]>0)
                    {
                        const float ur = uv.x - CurrentFrame.mbf*invzc;
                        const float er = fabs(ur - CurrentFrame.mvuRight[i2]);
                        if(er>radius)
                            continue;
                    }

                    //获得此当前图像的匹配点的描述子
                    const cv::Mat &d = CurrentFrame.mDescriptors.row(i2);

                    //计算描述子距离
                    const int dist = DescriptorDistance(dMP,d);

                    //选出最小距离的描述子和ID
                    if(dist<bestDist)
                    {
                        bestDist=dist;
                        bestIdx2=i2;
                    }
                }


                //最佳匹配要小于阈值，则记录此匹配点且匹配数目增加
                if(bestDist<=TH_HIGH)
                {
                    CurrentFrame.mvpMapPoints[bestIdx2]=pMP;
                    nmatches++;

                    

                    //若检查方向，则计算匹配点旋转角度所在的直方图
                    if(mbCheckOrientation)
                    {
                        cv::KeyPoint kpLF = (LastFrame.Nleft == -1) ? LastFrame.mvKeysUn[i]
                                                                    : (i < LastFrame.Nleft) ? LastFrame.mvKeys[i]
                                                                                            : LastFrame.mvKeysRight[i - LastFrame.Nleft];

                        cv::KeyPoint kpCF = (CurrentFrame.Nleft == -1) ? CurrentFrame.mvKeysUn[bestIdx2]
                                                                        : (bestIdx2 < CurrentFrame.Nleft) ? CurrentFrame.mvKeys[bestIdx2]
                                                                                                            : CurrentFrame.mvKeysRight[bestIdx2 - CurrentFrame.Nleft];
                        float rot = kpLF.angle-kpCF.angle;
                        if(rot<0.0)
                            rot+=360.0f;
                        int bin = round(rot*factor);
                        if(bin==HISTO_LENGTH)
                            bin=0;
                        assert(bin>=0 && bin<HISTO_LENGTH);
                        rotHist[bin].push_back(bestIdx2);
                    }
                
                }
                
                //一般均为-1
                if(CurrentFrame.Nleft != -1)
                {
                    cv::Mat x3Dr = CurrentFrame.mTrl.colRange(0,3).rowRange(0,3) * x3Dc + CurrentFrame.mTrl.col(3);

                    cv::Point2f uv = CurrentFrame.mpCamera->project(x3Dr);

                    int nLastOctave = (LastFrame.Nleft == -1 || i < LastFrame.Nleft) ? LastFrame.mvKeys[i].octave
                                                                                        : LastFrame.mvKeysRight[i - LastFrame.Nleft].octave;

                    // Search in a window. Size depends on scale
                    float radius = th*CurrentFrame.mvScaleFactors[nLastOctave];

                    vector<size_t> vIndices2;

                    if(bForward)
                        vIndices2 = CurrentFrame.GetFeaturesInArea(uv.x,uv.y, radius, nLastOctave, -1,true);
                    else if(bBackward)
                        vIndices2 = CurrentFrame.GetFeaturesInArea(uv.x,uv.y, radius, 0, nLastOctave, true);
                    else
                        vIndices2 = CurrentFrame.GetFeaturesInArea(uv.x,uv.y, radius, nLastOctave-1, nLastOctave+1, true);

                    const cv::Mat dMP = pMP->GetDescriptor();

                    int bestDist = 256;
                    int bestIdx2 = -1;

                    for(vector<size_t>::const_iterator vit=vIndices2.begin(), vend=vIndices2.end(); vit!=vend; vit++)
                    {
                        const size_t i2 = *vit;
                        if(CurrentFrame.mvpMapPoints[i2 + CurrentFrame.Nleft])
                            if(CurrentFrame.mvpMapPoints[i2 + CurrentFrame.Nleft]->Observations()>0)
                                continue;

                        const cv::Mat &d = CurrentFrame.mDescriptors.row(i2 + CurrentFrame.Nleft);

                        const int dist = DescriptorDistance(dMP,d);

                        if(dist<bestDist)
                        {
                            bestDist=dist;
                            bestIdx2=i2;
                        }
                    }

                    if(bestDist<=TH_HIGH)
                    {
                        CurrentFrame.mvpMapPoints[bestIdx2 + CurrentFrame.Nleft]=pMP;
                        nmatches++;
                        if(mbCheckOrientation)
                        {
                            cv::KeyPoint kpLF = (LastFrame.Nleft == -1) ? LastFrame.mvKeysUn[i]
                                                                        : (i < LastFrame.Nleft) ? LastFrame.mvKeys[i]
                                                                                                : LastFrame.mvKeysRight[i - LastFrame.Nleft];

                            cv::KeyPoint kpCF = CurrentFrame.mvKeysRight[bestIdx2];

                            float rot = kpLF.angle-kpCF.angle;
                            if(rot<0.0)
                                rot+=360.0f;
                            int bin = round(rot*factor);
                            if(bin==HISTO_LENGTH)
                                bin=0;
                            assert(bin>=0 && bin<HISTO_LENGTH);
                            rotHist[bin].push_back(bestIdx2  + CurrentFrame.Nleft);
                        }
                    }

                }

            }
        }
    }

    //Apply rotation consistency
    //旋转一致性检查
    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i!=ind1 && i!=ind2 && i!=ind3)
            {
                for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
                {
                    CurrentFrame.mvpMapPoints[rotHist[i][j]]=static_cast<MapPoint*>(NULL);
                    nmatches--;
                }
            }
        }
    }


    

    return nmatches;
}



/***********重定位中使用，3D-2D匹配**********************
 * CurrentFrame:当前图像帧
 * pKF:候选关键帧
 * sAlreadyFound  :已经匹配成功的地图点
 * th：匹配时搜索范围，会乘以金字塔尺度
 * ORBdist：匹配的ORB描述子距离应该小于这个阈值    
 * ***************************************************/
int ORBmatcher::SearchByProjection(Frame &CurrentFrame, KeyFrame *pKF, 
                                    const set<MapPoint*> &sAlreadyFound,
                                     const float th , const int ORBdist)
{
    int nmatches = 0;

    const cv::Mat Rcw = CurrentFrame.mTcw.rowRange(0,3).colRange(0,3);
    const cv::Mat tcw = CurrentFrame.mTcw.rowRange(0,3).col(3);
    const cv::Mat Ow = -Rcw.t()*tcw;

    // Rotation Histogram (to check rotation consistency)
    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f/HISTO_LENGTH;

    //得到候选关键帧的地图点
    const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();


    //遍历地图点
    for(size_t i=0, iend=vpMPs.size(); i<iend; i++)
    {
        MapPoint* pMP = vpMPs[i];

        if(pMP)
        {
            if(!pMP->isBad() && !sAlreadyFound.count(pMP))
            {
                
                //Project
                //候选帧中的地图点的世界坐标
                cv::Mat x3Dw = pMP->GetWorldPos();
                //地图点在当前坐标系中的坐标
                cv::Mat x3Dc = Rcw*x3Dw+tcw;

                //投影至当前图像帧的像素坐标系
                const cv::Point2f uv = CurrentFrame.mpCamera->project(x3Dc);

                if(uv.x<CurrentFrame.mnMinX || uv.x>CurrentFrame.mnMaxX)
                    continue;
                if(uv.y<CurrentFrame.mnMinY || uv.y>CurrentFrame.mnMaxY)
                    continue;

                // Compute predicted scale level
                cv::Mat PO = x3Dw-Ow;
                float dist3D = cv::norm(PO);

                const float maxDistance = pMP->GetMaxDistanceInvariance();
                const float minDistance = pMP->GetMinDistanceInvariance();

                // Depth must be inside the scale pyramid of the image
                if(dist3D<minDistance || dist3D>maxDistance)
                    continue;

                int nPredictedLevel = pMP->PredictScale(dist3D,&CurrentFrame);

                // Search in a window
                const float radius = th*CurrentFrame.mvScaleFactors[nPredictedLevel];

                //在当前图像中，投影点的半径搜索范围内搜索特征点
                const vector<size_t> vIndices2 = CurrentFrame.GetFeaturesInArea(uv.x, uv.y, radius, nPredictedLevel-1, nPredictedLevel+1);

                if(vIndices2.empty())
                    continue;

                const cv::Mat dMP = pMP->GetDescriptor();

                int bestDist = 256;
                int bestIdx2 = -1;

                for(vector<size_t>::const_iterator vit=vIndices2.begin(); vit!=vIndices2.end(); vit++)
                {
                    const size_t i2 = *vit;
                    if(CurrentFrame.mvpMapPoints[i2])
                        continue;

                    const cv::Mat &d = CurrentFrame.mDescriptors.row(i2);

                    const int dist = DescriptorDistance(dMP,d);

                    if(dist<bestDist)
                    {
                        bestDist=dist;
                        bestIdx2=i2;
                    }
                }

                if(bestDist<=ORBdist)
                {
                    CurrentFrame.mvpMapPoints[bestIdx2]=pMP;
                    nmatches++;

                    if(mbCheckOrientation)
                    {
                        float rot = pKF->mvKeysUn[i].angle-CurrentFrame.mvKeysUn[bestIdx2].angle;
                        if(rot<0.0)
                            rot+=360.0f;
                        int bin = round(rot*factor);
                        if(bin==HISTO_LENGTH)
                            bin=0;
                        assert(bin>=0 && bin<HISTO_LENGTH);
                        rotHist[bin].push_back(bestIdx2);
                    }
                }

            }
        }
    }

    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i!=ind1 && i!=ind2 && i!=ind3)
            {
                for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
                {
                    CurrentFrame.mvpMapPoints[rotHist[i][j]]=NULL;
                    nmatches--;
                }
            }
        }
    }

    return nmatches;
}

void ORBmatcher::ComputeThreeMaxima(vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3)
{
    int max1=0;
    int max2=0;
    int max3=0;

    for(int i=0; i<L; i++)
    {
        const int s = histo[i].size();
        if(s>max1)
        {
            max3=max2;
            max2=max1;
            max1=s;
            ind3=ind2;
            ind2=ind1;
            ind1=i;
        }
        else if(s>max2)
        {
            max3=max2;
            max2=s;
            ind3=ind2;
            ind2=i;
        }
        else if(s>max3)
        {
            max3=s;
            ind3=i;
        }
    }

    if(max2<0.1f*(float)max1)
    {
        ind2=-1;
        ind3=-1;
    }
    else if(max3<0.1f*(float)max1)
    {
        ind3=-1;
    }
}


// Bit set count operation from
// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
int ORBmatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
{
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist=0;

    for(int i=0; i<8; i++, pa++, pb++)
    {
        unsigned  int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}

} //namespace ORB_SLAM
