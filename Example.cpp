/*=========================================================================
 *
 *  Copyright David Doria 2012 daviddoria@gmail.com
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#include "XMeansClustering.h"

#include <iostream>

XMeansClustering::VectorOfPoints GenerateData();

int main(int, char *[])
{
  XMeansClustering::VectorOfPoints points = GenerateData();
  XMeansClustering xmeans;
  xmeans.SetMaxK(2);
  xmeans.SetPoints(points);
  xmeans.Cluster();

  std::vector<unsigned int> labels = xmeans.GetLabels();

  std::cout << "Resulting cluster ids:" << std::endl;
  for(unsigned int i = 0; i < labels.size(); ++i)
  {
    std::cout << labels[i] << std::endl;
  }
  
  return EXIT_SUCCESS;
}

XMeansClustering::VectorOfPoints GenerateData()
{
  XMeansClustering::VectorOfPoints points;

  XMeansClustering::PointType p = XMeansClustering::PointType::Zero(2);
  p[0] = 10; p[1] = 10;
  points.push_back(p);
  p[0] = 10.1; p[1] = 10.1;
  points.push_back(p);
  p[0] = 10.2; p[1] = 10.2;
  points.push_back(p);

  p[0] = 5; p[1] = 5;
  points.push_back(p);
  p[0] = 5.1; p[1] = 5.1;
  points.push_back(p);
  p[0] = 5.2; p[1] = 5.2;
  points.push_back(p);

  return points;
}
