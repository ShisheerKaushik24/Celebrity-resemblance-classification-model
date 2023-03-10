/**!
 * computeStatsPerLabel
 *
 * given a set of labels and associated descriptors, FOR EACH LABEL i: compute the mean and covariance of descriptor vectors that have
 * label i
 */
bool computeStatsPerLabel(std::vector<int> const& faceLabels, std::vector<dlib::matrix<float, 0, 1>> const& faceDescriptors,
    std::map<int, dlib::matrix<float, 0, 1>>& meanLabeledDescriptors,
    std::map<int, dlib::matrix<float, 0, 0>>& covarianceLabeledDescriptors) {
  // check that input vector sizes match, mark failure and exit if they don't
  if (faceLabels.size() != faceDescriptors.size()) {
    std::cout << "Size mismatch.  Exiting" << std::endl;
    return false;
  }

  // empty containers
  meanLabeledDescriptors.clear();
  covarianceLabeledDescriptors.clear();

  // setup associative container for labeled descriptors and populate it
  std::map<int, std::vector<dlib::matrix<float, 0, 1>>> labeledDescriptors = {};
  for (int i = 0; i < faceLabels.size(); ++i) {
    // if we haven't seen any descriptors for the present label, initialize
    // the vector for this label
    if (labeledDescriptors.find(faceLabels[i]) == labeledDescriptors.end()) {
      labeledDescriptors[faceLabels[i]] = { faceDescriptors[i] };
    } else {
      // if we have already have descriptors for this label, append the current descriptor
      labeledDescriptors[faceLabels[i]].emplace_back(faceDescriptors[i]);
    }
  }

  // for each key-value pair in the labeledDescriptors container
  for (auto &pr : labeledDescriptors) {
    // compute mean and covariance
    auto &descriptors = pr.second;
    dlib::matrix<float, 0, 1> mean;
    dlib::matrix<float, 0, 0> covariance;
    computeNormalParameters(descriptors, mean, covariance);
    auto label = pr.first;
    // add to output data containers
    meanLabeledDescriptors[label] = mean;
    covarianceLabeledDescriptors[label] = covariance;
  }

  // mark successful execution
  return true;
}