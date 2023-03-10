/**!
 * computeNormalParameters
 *
 * given a set of input descriptor vectors, compute the mean and covariance of that set
 *
 */
void computeNormalParameters(std::vector<dlib::matrix<float, 0, 1>> const& vecs,
    dlib::matrix<float, 0, 1>& mean, dlib::matrix<float, 0, 0>& covariance) {
  // if the input vector is empty, just exit
  if (vecs.size() == 0) {
    std::cout << "Nothing to do" << std::endl;
    return;
  }

  // shorthand for vector size
  auto const& N = vecs.size();

  // compute the mean = sum(v in vecs) / N
  mean.set_size(vecs[0].nr());
  dlib::set_all_elements(mean, 0);
  for (auto &v : vecs) {
    mean += v;
  }
  mean /= static_cast<float>(N);

  // compute the covariance = sum( (v-mean)*(v-mean)^T ) / N
  covariance.set_size(mean.nr(), mean.nr());
  dlib::set_all_elements(covariance, 0);
  for (auto &v : vecs) {
    covariance += (v - mean) * dlib::trans(v - mean);
  }
  covariance /= static_cast<float>(N);
  return;
}