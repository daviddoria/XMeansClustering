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

Eigen::MatrixXd GenerateData();

int main(int, char *[])
{
  Eigen::MatrixXd points = GenerateData();
  XMeansClustering xmeans;
  xmeans.SetMinK(1);
  xmeans.SetMaxK(3);
  xmeans.SetPoints(points);
  xmeans.Cluster();

  std::vector<unsigned int> labels = xmeans.GetLabels();

  std::cout << "Best number of clusters:" << xmeans.GetClusterCenters().cols() << std::endl;

  std::cout << "Resulting cluster ids:" << std::endl;
  for(unsigned int i = 0; i < labels.size(); ++i)
  {
    std::cout << labels[i] << std::endl;
  }
  
  return EXIT_SUCCESS;
}

Eigen::MatrixXd GenerateData()
{
  // This data is 2D.
  // Here there are two obvious clusters (near (10,10), and (5,5))
  Eigen::MatrixXd points(2, 6);

  Eigen::Vector2d p;
  p[0] = 10; p[1] = 10;
  points.col(0) = p;
  p[0] = 10.1; p[1] = 10.1;
  points.col(1) = p;
  p[0] = 10.2; p[1] = 10.2;
  points.col(2) = p;

  p[0] = 5; p[1] = 5;
  points.col(3) = p;
  p[0] = 5.1; p[1] = 5.1;
  points.col(4) = p;
  p[0] = 5.2; p[1] = 5.2;
  points.col(5) = p;

  return points;
}
