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
#include <algorithm>
#include <iostream>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>

// Submodules
#include "EigenHelpers/EigenHelpers.h"
#include "KMeansClustering/KMeansClustering.h"

void XMeansClustering::Initialize()
{
    this->KMeans.SetK(this->MinK);
    this->KMeans.SetPoints(this->Points);
    this->KMeans.SetInitMethod(KMeansClustering::RANDOM);
    this->KMeans.Cluster();

    this->ClusterCenters = this->KMeans.GetClusterCenters();
    this->Labels = this->KMeans.GetLabels();
}

// This step (so-named in the original paper) is simply a conventional K-means run
void XMeansClustering::ImproveParams()
{
    this->KMeans.SetK(this->ClusterCenters.cols());
    this->KMeans.SetClusterCenters(this->ClusterCenters);
    this->KMeans.SetPoints(this->Points);
    this->KMeans.SetInitMethod(KMeansClustering::MANUAL);
    this->KMeans.Cluster();

    this->ClusterCenters = this->KMeans.GetClusterCenters();
    this->Labels = this->KMeans.GetLabels();
}

// This step (so-named in the original paper) determines if and where the new (child) centroids should appear
void XMeansClustering::ImproveStructure()
{
    // We are going to build this from scratch
    Eigen::MatrixXd newClusterCenters;

    for(unsigned int clusterId = 0; clusterId < this->ClusterCenters.cols(); ++clusterId)
    {
       Eigen::MatrixXd oldClusterCenters = newClusterCenters;
       Eigen::MatrixXd splitCenters = TryToSplitCluster(clusterId);
       newClusterCenters.conservativeResize(this->Points.rows(), oldClusterCenters.cols() + splitCenters.cols());

       if(oldClusterCenters.cols() > 0)
       {
         newClusterCenters << oldClusterCenters, splitCenters;
       }
       else
       {
         newClusterCenters << splitCenters;
       }
    }

    std::cout << "There are " << newClusterCenters.cols() << " clusters after ImproveStructure()." << std::endl;
    this->ClusterCenters = newClusterCenters;
}

/*
The description of BIC in the XMeans paper seems overly convoluted.
float XMeansClustering::ComputeBIC(KMeansClustering* const kmeansModel)
{
    // See another implementation: https://github.com/mynameisfiber/pyxmeans/blob/master/pyxmeans/xmeans.py
    // BIC(M_j) = \hat{l}_j(D) - \frac{p_j}{2} log(R)

    float M = this->GetDimensionality();
    float K = kmeansModel->GetK();
    float R = this->GetNumberOfPoints();

    // p_j = (K-1) + (M*K) + 1 (we really just need p, because "model j" is implied by the model that we've passed)
    float p = (K - 1) + (M*K) + 1;

    // \hat{l}_j(D) is the log-likelihood of the data according to the j^{th} model at the maximum likelihood point


    // (K-1) class probabilities because the Kth is exactly determined as 1 - the rest
    // M*K centroid coordinates
    // 1 variance estimate.

    // R is the number of points
    // R_i is the number of points that belong to class i

    // \hat{\sigma}^2 = \frac{1}{R-K}\sum_i (x_i - \mu_{(i)})^2 % why is this R-K instead of just R?

    // \hat{P}(x_i) = \frac{R_(i)}{R} \frac{1}{\sqrt{2\pi}\hat{\sigma}^M} \exp(-\frac{1}{2\hat{\sigma}^2 \|x_i - \mu_{(i)}\|^2)

    // l(D) = \sum_i (\log \frac{1}{\sqrt{2\pi}\sigma^M} - \frac{1}{2\sigma^2} \|x_i - \mu_{(i)}\|^2 + log\frac{R_{(i)}{R})

    // \hat{l}(D_n) = -\frac{R_n}{2}log(2\pi) - \frac{R_n M}{2} log(\hat{\sigma}^2) - \frac{R_n - K}{2} + R_n log(R_n) - R_n log(R)
}
*/

void XMeansClustering::Cluster()
{
   assert(this->Points.size() > 0);
//  if(this->Points.size() < this->MaxK)
//  {
//    std::stringstream ss;
//    ss << "The number of points (" << this->Points.size()
//       << " must be larger than the maximum number of clusters (" << this->MaxK << ")";
//    throw std::runtime_error(ss.str());
//  }

  // Initialize the labels array
  this->Labels.resize(this->Points.cols());
  std::fill(this->Labels.begin(),this->Labels.end(), 0);

  // We must store the labels at the previous iteration to determine whether any labels changed at each iteration.
  std::vector<unsigned int> oldLabels(this->Points.cols());

  Initialize();

  unsigned int iterationCounter = 0;
  do
  {
    std::cout << "XMeans iteration " << iterationCounter << std::endl;

    ImproveParams();
    ImproveStructure();

    iterationCounter++;
  }while(this->ClusterCenters.cols() < this->MaxK);

  // Do the last parameter estimation with the latest structure
  ImproveParams();

}

Eigen::MatrixXd XMeansClustering::TryToSplitCluster(const unsigned int clusterId)
{
    Eigen::MatrixXd pointsInCluster = this->KMeans.GetPointsWithLabel(clusterId);

    // Generate a random direction
    Eigen::VectorXd randomUnitVector = EigenHelpers::RandomUnitVector<Eigen::VectorXd>(this->GetDimensionality());

    // Scale the random vector by the size of the region
    Eigen::VectorXd minCorner;
    Eigen::VectorXd maxCorner;
    EigenHelpers::GetBoundingBox(this->Points, minCorner, maxCorner);

    Eigen::VectorXd splitVector = randomUnitVector * (maxCorner - minCorner).norm() / 2.0f;
    Eigen::VectorXd childCenter1 = this->ClusterCenters.col(clusterId) + splitVector;
    Eigen::VectorXd childCenter2 = this->ClusterCenters.col(clusterId) - splitVector;
    Eigen::MatrixXd childCenters(this->GetDimensionality(), 2);
    childCenters << childCenter1, childCenter2;

    // Perform a K=2 clustering
    KMeansClustering childKMeans;
    childKMeans.SetK(2);
    childKMeans.SetPoints(pointsInCluster);
    childKMeans.SetClusterCenters(childCenters);
    childKMeans.SetInitMethod(KMeansClustering::MANUAL);
    childKMeans.Cluster();

    childCenters = childKMeans.GetClusterCenters();

    // Compute the Bayesian Information Criterion (BIC) of the original model
    float BIC_parent = this->KMeans.ComputeBIC(clusterId);

    // Compute the BIC of the new (split) model
    float BIC_children = childKMeans.ComputeBIC();

    // If the split was useful, keep it
    if(BIC_children < BIC_parent)
    {
      return childCenters;
    }
    else
    {
      // The split should not be kept, so return the old cluster center
      Eigen::MatrixXd newClusterCenters = Eigen::MatrixXd(this->GetDimensionality(), 1);
      newClusterCenters << this->ClusterCenters.col(clusterId);
      return newClusterCenters;
    }

}

std::vector<unsigned int> XMeansClustering::GetIndicesWithLabel(unsigned int label)const
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

Eigen::MatrixXd XMeansClustering::GetPointsWithLabel(const unsigned int label) const
{
  std::vector<unsigned int> indicesWithLabel = GetIndicesWithLabel(label);

  Eigen::MatrixXd points;
  points.resize(this->GetDimensionality(), indicesWithLabel.size());

  for(unsigned int i = 0; i < indicesWithLabel.size(); i++)
  {
    points.col(i) = this->Points.col(indicesWithLabel[i]);
  }

  return points;
}

void XMeansClustering::SetMinK(const unsigned int mink)
{
  this->MinK = mink;
}

unsigned int XMeansClustering::GetMinK() const
{
  return this->MinK;
}

void XMeansClustering::SetMaxK(const unsigned int maxk)
{
  this->MaxK = maxk;
}

unsigned int XMeansClustering::GetMaxK() const
{
  return this->MaxK;
}

void XMeansClustering::SetPoints(const Eigen::MatrixXd& points)
{
  this->Points = points;
}

std::vector<unsigned int> XMeansClustering::GetLabels() const
{
  return this->Labels;
}

void XMeansClustering::OutputClusterCenters()
{
  std::cout << std::endl << "Cluster centers: " << std::endl;

  for(unsigned int i = 0; i < this->ClusterCenters.cols(); ++i)
  {
    std::cout << this->ClusterCenters.col(i) << " ";
  }
  std::cout << std::endl;
}

unsigned int XMeansClustering::GetNumberOfPoints() const
{
    return this->Points.cols();
}

unsigned int XMeansClustering::GetDimensionality() const
{
    return this->Points.rows();
}

Eigen::MatrixXd XMeansClustering::GetClusterCenters() const
{
    return this->ClusterCenters;
}
