#define REGULARIZATION 1e-8  // covariance += REGULARIZATION*Identity - this is necessary to stabilize the matrix decomposition used for the Mahalanobis distance calculation
/**!
 * findKMostLikely
 *
 * given an input descriptor and mean and covariance for each label's descriptor vectors, find the
 * top K best matches based using a Mahalanobis distance metric
 *
 */
bool findKMostLikely(dlib::matrix<float, 0, 1> const& faceDescriptorQuery,
    std::map<int, dlib::matrix<float, 0, 1>> const& meanLabeledDescriptors,
    std::map<int, dlib::matrix<float, 0, 0>> const& covarianceLabeledDescriptors,
    const size_t& k, std::vector<std::pair<int, double>>& matches) {
  // check that input vector sizes match, mark failure and exit if they don't
  if (meanLabeledDescriptors.size() != covarianceLabeledDescriptors.size()) {
    std::cout << "Size mismatch.  Exiting." << std::endl;
    return false;
  }

  // loop over all sets of mean/covariance pairs
  std::vector<std::pair<int, double>> mahalanobisVec = {};
  for (int i = 0; i < meanLabeledDescriptors.size(); ++i) {
    auto covariance = covarianceLabeledDescriptors.at(i);

    // add some noise to the primary diagonal of the covariance matrix to regularize it
    // and improve the numerical stability of the subsequent solver
    auto transCov = covariance + REGULARIZATION * dlib::identity_matrix<float>(covariance.nr());
    auto luDecomp = dlib::lu_decomposition<dlib::matrix<float, 0, 0>>(transCov);
    
    // check if the object indicates a system that is not full-rank
    if (!luDecomp.is_singular()) {
      // there's nothing further to be done if the starting problem is singular, so go
      // to the next loop iteration
      std::cout << "Starting matrix is singular" << std::endl;
      continue;
    }

    // compute residual of query descriptor with the current mean
    auto residual = faceDescriptorQuery - meanLabeledDescriptors.at(i);
    
    // solve the linear system residual = S*y to get a more numerically-stable
    // representation of S^{-1}*residual in the Mahalanobis calculation 
    auto y = luDecomp.solve(residual);

    // compute Mahalanobis distance given mean, m, and covariance, S:
    // d(v1, m, S) = std::sqrt((v1-m)^T * S^{-1} * (v1-m));
    double mahalanobisDistance = std::sqrt(dlib::trans(residual) * y);

    // add result to full vector
    mahalanobisVec.emplace_back(std::make_pair(i, mahalanobisDistance));
  }
  
  // do the sort (smallest mahalanobis distance -> largest)
  std::sort(mahalanobisVec.begin(), mahalanobisVec.end(),
      [](std::pair<int, double> const& p1, std::pair<int, double> const& p2){ return p1.second < p2.second; });
  
  // get k matches that have smallest mahalanobis distance
  matches.clear();
  std::copy_n(mahalanobisVec.begin(), k, std::back_inserter(matches));
  return true;
}