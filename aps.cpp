#include <unistd.h>
#include <Eigen/Dense>
#include <string>
#include <fstream>
#include "common.h"

#define TINYEXR_IMPLEMENTATION
#include "tinyexr.h"

using namespace std;


float L(float cosTheta, float rough, float f0, int s)
{
	if (rough == 0) return schlick(cosTheta, f0);
	cosTheta = std::max(cosTheta, 0.001f); // avoid issues at exact grazing

	float sinTheta = std::sqrt(1 - sqr(cosTheta));
	Vector3 incoming(sinTheta, 0, cosTheta);
	float alpha = sqr(rough);
	MicrofacetLobe lobe(alpha, f0, incoming);
	float total = 0;

	for (int i = 0; i < s; i++)
		for (int j = 0; j < s; j++)
		{
			// stratified sampling
			float r1 = (randf() + i) / s;
			float r2 = (randf() + j) / s;
			Vector3 dir;
			float weight;
			bool ok = lobe.sample(r1, r2, dir, weight);
			if (!ok) continue;
			total += weight;
		}

	total /= s * s;
	return total;
}

float T(int i, const FloatImage& img)
{
	// computing the hemispherical integral in solid angle measure:
	// I = 1/pi * int[f(cos theta) * cos theta d omega]

	// this is turned into spherical coordinates, resulting in:
	// I = 2 int[f(cos theta) * cos theta * sin theta d theta] from 0 to pi/2

	// substitute t := cos theta, obtaining
	// I = 2 int[f(t) * t dt]  from 0 to 1
	// compute the latter using trapezoid rule

	int n = img.cols();
	float total = 0;

	for (int j = 0; j < n; j++)
	{
		float t = float(j) / (n - 1);
		float w = j == 0 || j == n-1 ? 0.5f : 1;
		total += 2 * t * img(i, j) * w / (n - 1);
	}

	return total;
}

void writeExr(FloatImage& img, string filename)
{
	int m = img.rows(), n = img.cols();
	ColorImage tmp(m, n);
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
			tmp(i, j) = Color::Constant(img(i, j));
	SaveEXR((float*) &tmp(0,0), n, m, 3, 1, filename.c_str(), 0);
}

int main(int argc, char** argv)
{
	int s = 100;  // use s^2 stratified samples
	int n = 256;  // result table size n x n
	float f0 = 0; // Schlick Fresnel normal incidence

	int tmp;
	while ((tmp = getopt(argc, argv, "n:s:f:")) != -1)
	{
		switch (tmp)
		{
			case 'n': n = atoi(optarg); break;
			case 's': s = atoi(optarg); break;
			case 'f': f0 = atof(optarg); break;
		}
	}

	FloatImage Limg(n, n);

	#pragma omp parallel for
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
		{
			float rough = float(i) / (n - 1);
			float cosTheta = float(j) / (n - 1);
			Limg(i, j) = L(cosTheta, rough, f0, s);
		}

	FloatImage Timg(1, n);
	for (int i = 0; i < n; i++) Timg(0, i) = T(i, Limg);

	writeExr(Limg, "L.exr");
	writeExr(Timg, "T.exr");
	return 0;
}

