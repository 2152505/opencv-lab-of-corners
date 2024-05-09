#ifndef LAB_3_CORNERS_FILTERS_H
#define LAB_3_CORNERS_FILTERS_H

#include "opencv2/core.hpp"

/// \brief Creates a Nx1 gaussian filter kernel.
/// \param sigma The sigma (standard deviation) parameter for the gaussian.
/// \param radius
/// \parblock
/// The filter radius, so that N = 2*radius + 1.
///
/// If set to 0, the radius will be computed so that radius = 3.5 * sigma.
/// \endparblock
cv::Mat create1DGaussianKernel(float sigma, int radius = 0);

/// \brief Creates a Nx1 derivated gaussian filter kernel.
/// \param sigma The sigma (standard deviation) parameter for the gaussian.
/// \param radius
/// \parblock
/// The filter radius, so that N = 2*radius + 1.
///
/// If set to 0, the radius will be computed so that radius = 3.5 * sigma.
/// \endparblock
cv::Mat create1DDerivatedGaussianKernel(float sigma, int radius = 0);

#endif //LAB_3_CORNERS_FILTERS_H
