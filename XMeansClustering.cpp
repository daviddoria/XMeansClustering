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

  KMeansPPInit();

  // Output cluster centers
//   std::cout << "Initial cluster centers: " << std::endl;
//   for(unsigned int i = 0; i < ClusterCenters.size(); i++)
//     {
//     std::cout << "Cluster center " << i << " : " << ClusterCenters[i] << std::endl;
//     }

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
    AssignLabels();

    EstimateClusterCenters();

    changed = CheckChanged(this->Labels, oldLabels);

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

unsigned int XMeansClustering::SelectWeightedIndex(std::vector<double> weights)
{
  // Ensure all weights are positive
  for(unsigned int i = 0; i < weights.size(); i++)
    {
    if(weights[i] < 0)
      {
      std::stringstream ss;
      ss << "weights[" << i << "] is " << weights[i] << " (must be positive!)";
      throw std::runtime_error(ss.str());
      }
    }

  //Helpers::Output(weights);
  
  // Sum
  double sum = std::accumulate(weights.begin(), weights.end(), 0.0f);
  //std::cout << "sum: " << sum << std::endl;
  if(sum <= 0)
    {
    std::stringstream ss;
    ss << "Sum must be positive, but it is " << sum << "!";
    throw std::runtime_error(ss.str());
    }

  // Normalize
  for(unsigned int i = 0; i < weights.size(); i++)
    {
    weights[i] /= sum;
    }

  double randomValue = drand48();

  double runningTotal = 0.0;
  for(unsigned int i = 0; i < weights.size(); i++)
    {
    runningTotal += weights[i];
    if(randomValue < runningTotal)
      {
      return i;
      }
    }

  std::cerr << "runningTotal: " << runningTotal << std::endl;
  std::cerr << "randomValue: " << randomValue << std::endl;
  throw std::runtime_error("KMeansClustering::SelectWeightedIndex() reached end, we should never get here.");

  return 0;
}

XMeansClustering::PointType XMeansClustering::GetRandomPointInBounds()
{
  PointType minVector = EigenHelpers::ComputeMinVector(Points);
  PointType maxVector = EigenHelpers::ComputeMaxVector(Points);

  PointType randomVector = PointType::Zero(minVector.size());

  for(int i = 0; i < randomVector.size(); ++i)
    {
    float range = maxVector(i) - minVector(i);
    float randomValue = drand48() * range + minVector(i);
    randomVector(i) = randomValue;
    }

  return randomVector;
}

bool XMeansClustering::CheckChanged(std::vector<unsigned int> labels, std::vector<unsigned int> oldLabels)
{
  bool changed = false;
  for(unsigned int i = 0; i < labels.size(); i++)
    {
    if(labels[i] != oldLabels[i]) //if something changed
      {
      changed = true;
      break;
      }
    }
  return changed;
}

void XMeansClustering::AssignLabels()
{
  // Assign each point to the closest cluster
  for(unsigned int point = 0; point < Points.size(); ++point)
    {
    unsigned int closestCluster = ClosestCluster(Points[point]);
    this->Labels[point] = closestCluster;
    }
}

void XMeansClustering::EstimateClusterCenters()
{
  VectorOfPoints oldCenters = this->ClusterCenters;

  for(unsigned int cluster = 0; cluster < this->MaxK; ++cluster)
    {
    VectorOfPoints classPoints;
    for(unsigned int point = 0; point < Points.size(); point++)
      {
      if(this->Labels[point] == cluster)
        {
        classPoints.push_back(Points[point]);
        }
      }
    PointType center;
    if(classPoints.size() == 0)
      {
      center = oldCenters[cluster];
      }
    else
      {
      center = EigenHelpers::ComputeMeanVector(classPoints);
      }

    ClusterCenters[cluster] = center;
    }
}

unsigned int XMeansClustering::ClosestCluster(const PointType& queryPoint)
{
  // Should NOT use the KDTree here, as the clusters are always changing (would have to rebuild the tree each iteration)!
  unsigned int closestCluster = 0;
  double minDist = std::numeric_limits<double>::max();
  for(unsigned int i = 0; i < ClusterCenters.size(); ++i)
    {

    double dist = (ClusterCenters[i] - queryPoint).norm();
    if(dist < minDist)
      {
      minDist = dist;
      closestCluster = i;
      }
    }

  return closestCluster;
}

unsigned int XMeansClustering::ClosestPointIndex(const PointType& queryPoint)
{
  // Should use the KDTree here!
  unsigned int closestPoint = 0;
  double minDist = std::numeric_limits<double>::max();
  for(unsigned int i = 0; i < Points.size(); i++)
    {
    //double dist = sqrt(vtkMath::Distance2BetweenPoints(points->GetPoint(i), queryPoint));
    double dist = (Points[i] - queryPoint).norm();
    if(dist < minDist)
      {
      minDist = dist;
      closestPoint = i;
      }
    }

  return closestPoint;
}

double XMeansClustering::ClosestPointDistanceExcludingId(const PointType& queryPoint, const unsigned int excludedId)
{
  std::vector<unsigned int> excludedIds;
  excludedIds.push_back(excludedId);
  return ClosestPointDistanceExcludingIds(queryPoint, excludedIds);
}

double XMeansClustering::ClosestPointDistanceExcludingIds(const PointType& queryPoint, const std::vector<unsigned int> excludedIds)
{
  double minDist = std::numeric_limits<double>::infinity();
  for(unsigned int pointId = 0; pointId < Points.size(); ++pointId)
    {
    if(Helpers::Contains(excludedIds, pointId))
      {
      continue;
      }
    double dist = (Points[pointId] - queryPoint).norm();

    if(dist < minDist)
      {
      minDist = dist;
      }
    }
  return minDist;
}

double XMeansClustering::ClosestPointDistance(const PointType& queryPoint)
{
  std::vector<unsigned int> excludedIds; // none
  return ClosestPointDistanceExcludingIds(queryPoint, excludedIds);
}

void XMeansClustering::KMeansPPInit()
{
  // Assign one center at random
  unsigned int randomId = rand() % this->Points.size();
  PointType p = this->Points[randomId];
  this->ClusterCenters.push_back(p);

  // Assign the rest of the initial centers using a weighted probability of the distance to the nearest center
  std::vector<double> weights(this->Points.size());
  for(unsigned int cluster = 1; cluster < this->MaxK; ++cluster) // Start at 1 because cluster 0 is already set
    {
    // Create weight vector
    for(unsigned int i = 0; i < this->Points.size(); i++)
      {
      PointType currentPoint = this->Points[i];
      unsigned int closestCluster = ClosestCluster(currentPoint);
      weights[i] = (ClusterCenters[closestCluster] - currentPoint).norm();
      }

    unsigned int selectedPointId = SelectWeightedIndex(weights);
    p = this->Points[selectedPointId];
    this->ClusterCenters.push_back(p);
    }
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
