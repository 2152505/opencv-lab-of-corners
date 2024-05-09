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
  // Todo: Step 2: Estimate image gradients Ix and Iy using g_kernel_ and dg_kernel_.
  cv::Mat Ix;
  cv::Mat Iy;

  // Compute the elements of M; A, B and C from Ix and Iy.
  // Todo: Step 3: Compute the elements of M; A, B and C from Ix and Iy.
  cv::Mat A;
  cv::Mat B;
  cv::Mat C;

  // Apply the windowing gaussian win_kernel_ on A, B and C.
  // Todo: Step 3: Apply the windowing gaussian.

  // Compute corner response.
  // Todo: Step 4: Finish all the corner response functions.
  cv::Mat response;
  switch (metric_type_)
  {
  case CornerMetric::harris:
    response = harrisMetric(A, B, C); break;

  case CornerMetric::harmonic_mean:
    response = harmonicMeanMetric(A, B, C); break;

  case CornerMetric::min_eigen:
    response = minEigenMetric(A, B, C); break;
  }

  // Todo: Step 5: Dilate image to make each pixel equal to the maximum in the neighborhood.
  cv::Mat local_max;

  // Todo: Step 6: Compute the threshold.
  // Compute the threshold by using quality_level_ on the maximum response.
  double max_val(0.0);

  // Todo: Step 7: Extract local maxima above threshold.
  cv::Mat is_strong_and_local_max; // = response > threshold and response == local_max
  std::vector<cv::Point> max_locations;

  // ----- Now detect() is finished! -----
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
    if (!Ix.empty()) { cv::imshow("Gradient Ix", Ix); };
    if (!Iy.empty()) { cv::imshow("Gradient Iy", Iy); };
    if (!A.empty()) { cv::imshow("Image A", A); };
    if (!B.empty()) { cv::imshow("Image B", B); };
    if (!C.empty()) { cv::imshow("Image C", C); };
    if (!response.empty()) { cv::imshow("Response", response/(0.9*max_val)); };
    if (!is_strong_and_local_max.empty()) { cv::imshow("Local max", is_strong_and_local_max); };
  }

  return key_points;
}

cv::Mat CornerDetector::harrisMetric(cv::Mat& A, cv::Mat& B, cv::Mat& C) const
{
  // Compute the Harris metric for each pixel.
  // Todo: Step 4: Finish all the corner response functions.
  const float alpha = 0.06f;

  return cv::Mat();
}

cv::Mat CornerDetector::harmonicMeanMetric(cv::Mat& A, cv::Mat& B, cv::Mat& C) const
{
  // Compute the Harmonic mean metric for each pixel.
  // Todo: Step 4: Finish all the corner response functions.
  return cv::Mat();
}

cv::Mat CornerDetector::minEigenMetric(cv::Mat& A, cv::Mat& B, cv::Mat& C) const
{
  // Compute the Min. Eigen metric for each pixel.
  // Todo: Step 4: Finish all the corner response functions.
  return cv::Mat();
}

