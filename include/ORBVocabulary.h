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


#ifndef ORBVOCABULARY_H
#define ORBVOCABULARY_H

#include"Thirdparty/DBoW2/DBoW2/FORB.h"
#include"Thirdparty/DBoW2/DBoW2/TemplatedVocabulary.h"

namespace ORB_SLAM3
{

/******************************************************************
 * typedef：在任何声明变量的语句前加上typedef后，原来的变量都变成了一种类型
 * ORBVocabulary指代DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB>类型 
 * DBoW2::TemplatedVocabulary是一个字典模板类
 * DBoW2::FORB::TDescriptor  描述子（实际是cv::Mat类）
 * DBoW2::FORB 定义了对描述子操作的函数，主要有计算描述子集合中值的meanValue(),计算描述子距离的distance()
******************************************************************/
typedef DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB>
  ORBVocabulary;

} //namespace ORB_SLAM

#endif // ORBVOCABULARY_H
