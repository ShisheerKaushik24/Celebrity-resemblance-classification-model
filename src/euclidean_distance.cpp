/**!
 * findKNearestNeighbors
 *
 * given an input descriptor and database of labeled descriptors, find the
 * top K best matches based using a Euclidean distance metric
 *
 */
bool findKNearestNeighbors(dlib::matrix<float, 0, 1> const& faceDescriptorQuery,
    std::vector<dlib::matrix<float, 0, 1>> const& faceDescriptors,
    std::vector<int> const& faceLabels, int const& k,
    std::vector<std::pair<int, double>>& matches) {
  // check that input vector sizes match, mark failure and exit if they don't
  if (faceLabels.size() != faceDescriptors.size()) {
    std::cout << "Size mismatch.  Exiting" << std::endl;
    return false;
  }

  // loop over all descriptors and compute Euclidean distance with query descriptor
  std::vector<std::pair<int, double>> neighbors = {};
  for (int i = 0; i < faceDescriptors.size(); i++) {
    // compute distance between descriptors v1 and v2:
    // d(v1, v2) = std::sqrt((v1-v2)^T * (v1-v2));
    // - this is implemented in dlib with the `length` function
    double distance = dlib::length(faceDescriptorQuery - faceDescriptors[i]);
    // check if a distance for this label has already been determined
    auto it = std::find_if(neighbors.begin(), neighbors.end(),
          [&](std::pair<int, double> const& p) { return p.first == faceLabels[i]; });
    if (it != neighbors.end()) {
      // if there has already been a distance found for this label, check if the current distance is less
      // than the one previously computed
      if (distance < it->second) {
        // if the current distance is less than the one previously recorded for the label, update it
        it->second = distance;
      }
    } else {
      // this is the first time encountering this label, so add the (label, distance) pair to neighbors
      neighbors.emplace_back(std::make_pair(faceLabels[i], distance));
    }
  }
  
  // do the sort (closest to -> furthest away)
  std::sort(neighbors.begin(), neighbors.end(),
      [](std::pair<int, double> const& p1, std::pair<int, double> const& p2){ return p1.second < p2.second; });
    
  // get k closest
  matches.clear();
  std::copy_n(neighbors.begin(), k, std::back_inserter(matches));
  return true;
}