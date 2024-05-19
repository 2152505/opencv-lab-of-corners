#include "corner_detector.h"

CornerDetector::CornerDetector(CornerMetric metric,
                               bool do_visualize,
                               float quality_level,
                               float gradient_sigma,
                               float window_sigma)
    : metric_type_{metric}
    , do_visualize_{do_visualize}
    , quality_level_{quality_level}
    , window_sigma_{window_sigma}
    , g_kernel_{create1DGaussianKernel(gradient_sigma)}
    , dg_kernel_{create1DDerivatedGaussianKernel(gradient_sigma)}
    , win_kernel_{create1DGaussianKernel(window_sigma_)}
{ }

std::vector<cv::KeyPoint> CornerDetector::detect(cv::Mat image) const
{
  // Estimate image gradients Ix and Iy using g_kernel_ and dg_kernel.
  cv::Mat Ix;
  cv::Mat Iy;
  cv::filter2D(image, Ix, -1, g_kernel_);
  cv::filter2D(image, Iy, -1, dg_kernel_);

  // Compute the elements of M; A, B and C from Ix and Iy.
  cv::Mat A = Ix.mul(Ix);
  cv::Mat B = Ix.mul(Iy);
  cv::Mat C = Iy.mul(Iy);

  // Apply the windowing gaussian win_kernel_ on A, B and C.
  cv::filter2D(A, A, -1, win_kernel_);
  cv::filter2D(B, B, -1, win_kernel_);
  cv::filter2D(C, C, -1, win_kernel_);

  // Compute corner response.
  cv::Mat response;
  switch (metric_type_)
  {
  case CornerMetric::harris:
    response = harrisMetric(A, B, C);
    break;

  case CornerMetric::harmonic_mean:
    response = harmonicMeanMetric(A, B, C);
    break;

  case CornerMetric::min_eigen:
    response = minEigenMetric(A, B, C);
    break;
  }

  // Dilate image to make each pixel equal to the maximum in the neighborhood.
  cv::Mat local_max;
  cv::dilate(response, local_max, cv::Mat());

  // Compute the threshold.
  double max_val;
  cv::minMaxLoc(response, nullptr, &max_val);
  double threshold = quality_level_ * max_val;

  // Extract local maxima above threshold.
  cv::Mat is_strong_and_local_max = (response > threshold) & (response == local_max);
  std::vector<cv::Point> max_locations;
  cv::findNonZero(is_strong_and_local_max, max_locations);

  // Add all strong local maxima as keypoints.
  const float keypoint_size = 3.0f * window_sigma_;
  std::vector<cv::KeyPoint> key_points;
  for (const auto& point : max_locations)
  {
    key_points.emplace_back(cv::KeyPoint{point, keypoint_size, -1, response.at<float>(point)});
  }

  // Show additional debug/educational figures.
  if (do_visualize_)
  {
    if (!Ix.empty()) { cv::imshow("Gradient Ix", Ix); }
    if (!Iy.empty()) { cv::imshow("Gradient Iy", Iy); }
    if (!A.empty()) { cv::imshow("Image A", A); }
    if (!B.empty()) { cv::imshow("Image B", B); }
    if (!C.empty()) { cv::imshow("Image C", C); }
    if (!response.empty()) { cv::imshow("Response", response / (0.9 * max_val)); }
    if (!is_strong_and_local_max.empty()) { cv::imshow("Local max", is_strong_and_local_max); }
  }

  return key_points;
}

cv::Mat CornerDetector::harrisMetric(cv::Mat& A, cv::Mat& B, cv::Mat& C) const
{
  // Compute the Harris metric for each pixel.
  const float alpha = 0.06f;
  cv::Mat response;

  // Calculate the determinant and trace of the matrix M
  cv::Mat det_M = A.mul(C) - B.mul(B);
  cv::Mat trace_M = A + C;

  // Compute the Harris response
  response = det_M - alpha * trace_M.mul(trace_M);

  // Set negative values to zero
  response = cv::max(response, 0.0);

  return response;
}

cv::Mat CornerDetector::harmonicMeanMetric(cv::Mat& A, cv::Mat& B, cv::Mat& C) const
{
  // Compute the Harmonic mean metric for each pixel.
  cv::Mat response;

  // Compute the Harmonic mean response
  response = (A.mul(C)) / (A + C);

  return response;
}

cv::Mat CornerDetector::minEigenMetric(cv::Mat& A, cv::Mat& B, cv::Mat& C) const
{
  // Compute the Min. Eigen metric for each pixel.
  cv::Mat response;

  // Compute eigenvalues of M
  cv::Mat M = cv::Mat::zeros(A.size(), CV_32FC3);
  M.at<cv::Vec3f>(0) = A;
  M.at<cv::Vec3f>(1) = B;
  M.at<cv::Vec3f>(2) = C;

  cv::Mat eigenvalues;
  cv::eigen(M, eigenvalues);

  // Compute the minimum eigenvalue response
  cv::Mat lambda1 = eigenvalues.row(0);
  cv::Mat lambda2 = eigenvalues.row(1);
  response = cv::min(lambda1, lambda2);

  return response;
}

