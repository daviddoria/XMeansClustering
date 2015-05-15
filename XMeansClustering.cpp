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
#include "EigenHelpers/EigenHelpers.h"

XMeansClustering::XMeansClustering() : MinK(1), MaxK(3)
{

}

// This step (so-named in the original paper) is simply a conventional K-means run
void XMeansClustering::ImproveParams()
{

}

// This step (so-named in the original paper) determines where the new centroids should appear
void XMeansClustering::ImproveStructure()
{

}

float XMeansClustering::ComputeBIC()
{
    //BIC(M_j) = \hat{l}_j(D) - \frac{p_j}{2} log(R) % p_j (the number of free parameters of model j)
    // is the sum of K-1 class probabilities (because the Kth is exactly determined as 1 - the rest ?
    // then why isn't this sum always just 1?), MK centroid coordinates, and one variance estimate.

    // \hat{\sigma}^2 = \frac{1}{R-K}\sum_i (x_i - \mu_{(i)})^2 % why is this R-K instead of just R?

    // \hat{P}(x_i) = \frac{R_(i)}{R} \frac{1}{\sqrt{2\pi}\hat{\sigma}^M} \exp(-\frac{1}{2\hat{\sigma}^2 \|x_i - \mu_{(i)}\|^2)

    // l(D) = \sum_i (\log \frac{1}{\sqrt{2\pi}\sigma^M} - \frac{1}{2\sigma^2} \|x_i - \mu_{(i)}\|^2 + log\frac{R_{(i)}{R})

    // \hat{l}(D_n) = -\frac{R_n}{2}log(2\pi) - \frac{R_n M}{2} log(\hat{\sigma}^2) - \frac{R_n - K}{2} + R_n log(R_n) - R_n log(R)
}

void XMeansClustering::Cluster()
{
  if(this->Points.size() < this->MaxK)
  {
    std::stringstream ss;
    ss << "The number of points (" << this->Points.size()
       << " must be larger than the maximum number of clusters (" << this->MaxK << ")";
    throw std::runtime_error(ss.str());
  }

  // We must store the labels at the previous iteration to determine whether any labels changed at each iteration.
  std::vector<unsigned int> oldLabels(this->Points.size(), 0); // initialize to all zeros

  // Initialize the labels array
  this->Labels.resize(this->Points.size());

  do
  {
    // Save the old labels
    oldLabels = this->Labels;

    ImproveParams();
    ImproveStructure();

  }while(this->ClusterCenters.size() < this->MaxK);

}

void XMeansClustering::SplitClusters()
{
  assert(this->Points.size() > 0);

  VectorOfPoints newClusterCenters;

  for(unsigned int clusterId = 0; clusterId < this->ClusterCenters.size(); ++clusterId)
  {
    // Generate a random direction
    PointType randomUnitVector = EigenHelpers::RandomUnitVector<PointType>(this->Points[0].size());

    // Get the bounding box of the points that belong to this cluster
    PointType minCorner;
    PointType maxCorner;
    EigenHelpers::GetBoundingBox(this->Points, minCorner, maxCorner);

    // Scale the unit vector by the size of the region
    PointType splitVector = randomUnitVector * (maxCorner - minCorner) / 2.0f;
    PointType childCenter1 = this->ClusterCenters[clusterId] + splitVector;
    PointType childCenter2 = this->ClusterCenters[clusterId] + splitVector;

    // Compute the Bayesian Information Criterion (BIC) of the original model
    float BIC_parent = ComputeBIC();

    // Compute the BIC of the new (split) model
    float BIC_children = ComputeBIC();

    // If the split was useful, keep it
    if(BIC_children < BIC_parent)
    {
      newClusterCenters.push_back(childCenter1);
      newClusterCenters.push_back(childCenter2);
    }
    else
    {
      newClusterCenters.push_back(this->ClusterCenters[clusterId]);
    }
  }

  this->ClusterCenters = newClusterCenters;
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

void XMeansClustering::SetMinK(const unsigned int mink)
{
  this->MinK = mink;
}

unsigned int XMeansClustering::GetMinK()
{
  return this->MinK;
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
