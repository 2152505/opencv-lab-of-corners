#include "filters.h"

cv::Mat create1DGaussianKernel(float sigma, int radius)
{
  if (radius <= 0) radius = static_cast<int>(std::ceil(3.5f * sigma));

  const int length = 2*radius + 1;
  cv::Mat kernel(length, 1, CV_32F);

  const float scale(-0.5f / (sigma*sigma));
  double sum(0);

  for (int i = 0; i < length; ++i)
  {
    const int x = i - radius;
    const float kernel_element = std::exp(x * x * scale);
    kernel.at<float>(i) = kernel_element;
    sum += kernel_element;
  }

  kernel /= sum;
  return kernel;
}

cv::Mat create1DDerivatedGaussianKernel(float sigma, int radius)
{
  if (radius <= 0) radius = static_cast<int>(std::ceil(3.5f * sigma));

  // Todo: Step 1: Use the function above and finish this function.
  cv::Mat kernel;

  return kernel;
}
