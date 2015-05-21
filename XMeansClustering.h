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

// Submodules
#include "KMeansClustering/KMeansClustering.h"

class XMeansClustering
{
public:
  /** Initialize by clustering with MinK clusters. */
  void Initialize();

  /** Set the minimum number of clusters to find. */
  void SetMinK(const unsigned int mink);

  /** Get the minimum number of clusters to find. */
  unsigned int GetMinK() const;

  /** Set the maximum number of clusters to find. */
  void SetMaxK(const unsigned int maxk);

  /** Get the maximum number of clusters to find. */
  unsigned int GetMaxK() const;

  /** Get the ids of the points that belong to class 'label'. */
  std::vector<unsigned int> GetIndicesWithLabel(const unsigned int label) const;

  /** Get the coordinates of the points that belong to class 'label'. */
  Eigen::MatrixXd GetPointsWithLabel(const unsigned int label) const;

  /** Get the resulting cluster centers. */
  Eigen::MatrixXd GetClusterCenters() const;

  /** Set the points to cluster. */
  void SetPoints(const Eigen::MatrixXd& points);

  /** Get the resulting cluster id for each point. */
  std::vector<unsigned int> GetLabels() const;

  /** Actually perform the clustering (the main driver of the algorithm). */
  void Cluster();

  /** Write the cluster centers to the standard output. */
  void OutputClusterCenters();

  /** Get the number of points in the data set. */
  unsigned int GetNumberOfPoints() const;

  /** Get the dimensionality of the data set. */
  unsigned int GetDimensionality() const;

private:

  /** Run a normal KMeans clustering. */
  void ImproveParams();

  /** Split every cluster into two clusters if that helps the description of the data. */
  void ImproveStructure();

  /** Try to split cluster 'clusterId' into two clusters if that helps the description of the data.
      Return a matrix of the best cluster centers (1 column indicates that the cluster did not split and is the
      cluster center of the parent, while two columns indicates the the cluster was split and the columns are
      the cluster centers of the children. */
  Eigen::MatrixXd TryToSplitCluster(const unsigned int clusterId);
  
  /** The label (cluster ID) of each point. */
  std::vector<unsigned int> Labels;

  /** The minimum number of clusters to find */
  unsigned int MinK = 1;

  /** The maximum number of clusters to find */
  unsigned int MaxK = 5;

  /** The points to cluster. Data in this class is stored as an Eigen matrix, where the data points are column vectors.
      That is, if we have P N-D points, the matrix is N rows by P columns.*/
  Eigen::MatrixXd Points;

  /** The current cluster centers. */
  Eigen::MatrixXd ClusterCenters;

  /** This object will get updated during the ImproveParams() step. */
  KMeansClustering KMeans;
};

#endif
