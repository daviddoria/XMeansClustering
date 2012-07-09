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

/** XMeans clustering is a method that creates K clusters of points from
  * an unorganized set of input points. It is an extension of KMeans clustering
  * that attempts to determine K during the algorithm.
  */

#ifndef XMeansClustering_h
#define XMeansClustering_h

// STL
#include <vector>

// Eigen
#include <Eigen/Dense>

class XMeansClustering
{
public:
  typedef Eigen::VectorXf PointType;
  typedef std::vector<PointType, Eigen::aligned_allocator<PointType> > VectorOfPoints;

  /** Constructor. */
  XMeansClustering();

  /** Set the maximum number of clusters to find. */
  void SetMaxK(const unsigned int maxk);

  /** Get the maximum number of clusters to find. */
  unsigned int GetMaxK();

  /** Get the ids of the points that belong to class 'label'. */
  std::vector<unsigned int> GetIndicesWithLabel(const unsigned int label);

  /** Get the coordinates of the points that belong to class 'label'. */
  VectorOfPoints GetPointsWithLabel(const unsigned int label);

  /** Set the points to cluster. */
  void SetPoints(const VectorOfPoints& points);

  /** Get the resulting cluster id for each point. */
  std::vector<unsigned int> GetLabels();

  /** Actually perform the clustering. */
  void Cluster();

  /** Write the cluster centers to the standard output. */
  void OutputClusterCenters();

private:

  /** Split every cluster into two clusters if that helps the description of the data. */
  void SplitClusters();
  
  /** The label (cluster ID) of each point. */
  std::vector<unsigned int> Labels;

  /** The maximum number of clusters to find */
  unsigned int MaxK;

  /** The points to cluster. */
  VectorOfPoints Points;

  /** The current cluster centers. */
  VectorOfPoints ClusterCenters;
};

#endif
