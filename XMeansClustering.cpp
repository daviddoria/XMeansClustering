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

// STL
#include <iostream>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>

// Submodules
#include "KMeansClustering/EigenHelpers/EigenHelpers.h"
#include "KMeansClustering/Helpers/Helpers.h"
#include "../../Examples/c++/Templates/ClassTemplateSpecialization/Point.h"

XMeansClustering::XMeansClustering() : MaxK(3)
{

}

void XMeansClustering::Cluster()
{
  if(this->Points.size() < this->MaxK)
  {
    std::stringstream ss;
    ss << "The number of points (" << this->Points.size()
       << " must be larger than the number of clusters (" << this->MaxK << ")";
    throw std::runtime_error(ss.str());
  }

  // We must store the labels at the previous iteration to determine whether any labels changed at each iteration.
  std::vector<unsigned int> oldLabels(this->Points.size(), 0); // initialize to all zeros

  // Initialize the labels array
  this->Labels.resize(this->Points.size());

  // The current iteration number
  int iter = 0;

  // Track whether any labels changed in the last iteration
  bool changed = true;
  do
    {

    // Save the old labels
    oldLabels = this->Labels;
    iter++;
    }while(changed);
    //}while(iter < 100); // You could use this stopping criteria to make kmeans run for a specified number of iterations

  std::cout << "KMeans finished in " << iter << " iterations." << std::endl;
}

std::vector<unsigned int> XMeansClustering::GetIndicesWithLabel(unsigned int label)
{
  std::vector<unsigned int> pointsWithLabel;
  for(unsigned int i = 0; i < this->Labels.size(); i++)
    {
    if(this->Labels[i] == label)
      {
      pointsWithLabel.push_back(i);
      }
    }

  return pointsWithLabel;
}

XMeansClustering::VectorOfPoints XMeansClustering::GetPointsWithLabel(const unsigned int label)
{
  VectorOfPoints points;
  
  std::vector<unsigned int> indicesWithLabel = GetIndicesWithLabel(label);

  for(unsigned int i = 0; i < indicesWithLabel.size(); i++)
    {
    points.push_back(this->Points[indicesWithLabel[i]]);
    }

  return points;
}


void XMeansClustering::SetMaxK(const unsigned int maxk)
{
  this->MaxK = maxk;
}

unsigned int XMeansClustering::GetMaxK()
{
  return this->MaxK;
}

void XMeansClustering::SetPoints(const VectorOfPoints& points)
{
  this->Points = points;
}

std::vector<unsigned int> XMeansClustering::GetLabels()
{
  return this->Labels;
}

void XMeansClustering::OutputClusterCenters()
{
  std::cout << std::endl << "Cluster centers: " << std::endl;
  
  for(unsigned int i = 0; i < ClusterCenters.size(); ++i)
    {
    std::cout << ClusterCenters[i] << " ";
    }
  std::cout << std::endl;
}
