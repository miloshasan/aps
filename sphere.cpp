#include <unistd.h>
#include <string>
#include <fstream>
#include "common.h"

#define TINYEXR_IMPLEMENTATION
#include "tinyexr.h"

using namespace std;

int n = 256;  // result image size n x n
int spp = 16; // spp
float f0 = 0.04f; // Schlick Fresnel normal incidence
float rough = 0.1f; // roughness
Color rho(0.8f, 0.1f, 0.1f); // diffuse albedo

FloatImage L0, T0, L1, T1;

float lookup1(FloatImage& A, float x)
{
	int n = A.cols();
	x *= n - 1;
	int i = int(std::floor(x));
	int i1 = std::min(i+1, n-1);
	float xf = x - i;
	return lerp(xf, A(0, i), A(0, i1));
}

float lookup2(FloatImage& A, float x, float y)
{
	// note: x <-> i, y <-> j
	int n = A.cols();
	x *= n - 1;
	y *= n - 1;
	int i0 = int(std::floor(x));
	int i1 = std::min(i0+1, n-1);
	int j0 = int(std::floor(y));
	int j1 = std::min(j0+1, n-1);
	float xf = x - i0;
	float yf = y - j0;
	return lerp(xf,
		lerp(yf, A(i0, j0), A(i0, j1)),
		lerp(yf, A(i1, j0), A(i1, j1)));
}

class DiffuseLobe
{
	Color m_cached;
	float m_rough, m_f0;

	inline float L(float cosTheta)
	{
		return (1 - m_f0) * lookup2(L0, m_rough, cosTheta) + m_f0 * lookup2(L1, m_rough, cosTheta);
	}

	inline float T()
	{
		return (1 - m_f0) * lookup1(T0, m_rough) + m_f0 * lookup1(T1, m_rough);
	}

	inline Vector3 sampleCosine(float r1, float r2)
	{
		float theta = 2 * pi * r1;
		float r = std::sqrt(r2);
		float x, y;
		__sincosf(theta, &y, &x);
		x *= r; y *= r;
		float z = std::sqrt(1 - r2); // note: x^2 + y^2 = r^2 = r2
		return Vector3(x, y, z);
	}

public:

	DiffuseLobe(Color rho, float rough, float f0, Vector3 incoming):
		m_rough(rough), m_f0(f0)
	{
		float cosIn = incoming[2];
		m_cached = rho * (1 - L(cosIn));
		m_cached /= pi * (1 - T());
	}

	bool sample(float r1, float r2, Vector3& direction, Color& weight)
	{
		direction = sampleCosine(r1, r2);
		float cosOut = direction[2];
		weight = m_cached * (1 - L(cosOut)) * pi; // the pdf is 1/pi
		return true;
	}
};


float solveQuadratic(float a, float b, float c)
{
	float D = sqr(b) - 4*a*c;
	if (D <= 0) return 0;
	return (-b - std::sqrt(D)) / (2*a);
}

// solve |p + tx|^2 = 1
float traceSphere(Vector3 p, Vector3 d)
{
	float a = d.squaredNorm();
	float b = 2 * p.dot(d);
	float c = p.squaredNorm() - 1;
	return solveQuadratic(a, b, c);
}

/// make orthogonal frame from unit normal
void constructFromZ(Vector3 a, Vector3 &b, Vector3 &c)
{
	float x = a[0], y = a[1], z = a[2];
	if (std::abs(x) > std::abs(y)) c = Vector3(z, 0, -x).normalized();
	else c = Vector3(0, z, -y).normalized();
	b = c.cross(a);
}

void writeExr(ColorImage& img, string filename)
{
	int m = img.rows(), n = img.cols();
	SaveEXR((float*) &img(0,0), n, m, 3, 1, filename.c_str(), 0);
}

void readExr(string filename, FloatImage& out)
{
	int w, h;
	float* data;
	LoadEXR(&data, &w, &h, filename.c_str(), 0);
	float* ptr = data;
	out.resize(h, w);
	for (int i = 0; i < h; i++)
		for (int j = 0; j < w; j++) { out(i, j) = *ptr; ptr += 4; }
	delete[] data;
}

Color shade(Vector3 p, Vector3 incoming)
{
	Vector3 x, y;
	constructFromZ(p, x, y);
	Vector3 local(incoming.dot(x), incoming.dot(y), incoming.dot(p));
	Color c = Color::Zero();

	MicrofacetLobe mf(sqr(rough), f0, local);
	Vector3 dir; float weight;
	bool ok = mf.sample(randf(), randf(), dir, weight);
	if (ok) c += weight;

	Color weight2;
	DiffuseLobe diff(rho, rough, f0, local);
	diff.sample(randf(), randf(), dir, weight2);
	c += weight2;

	return c;
}

Color computePixel(int i, int j)
{
	Color total;
	total.setZero();
	float invspp = 1.0f / spp;

	for (int k = 0; k < spp; k++)
	{
		float x = (randf() + i) / n;
		float y = (randf() + j) / n;
		x = 2*x - 1; y = 2*y - 1;

		Vector3 p(x, y, -2);
		Vector3 d(0, 0, 1);

		float t = traceSphere(p, d);
		if (t <= 0) { total += invspp; continue; }

		p += t * d;
		p.normalize(); // should be already normalized but just in case

		total += invspp * shade(p, -d);
	}

	return total;
}

int main(int argc, char** argv)
{
	int tmp;
	while ((tmp = getopt(argc, argv, "n:s:f:r:a:")) != -1)
	{
		switch (tmp)
		{
			case 'n': n = atoi(optarg); break;
			case 's': spp = atoi(optarg); break;
			case 'f': f0 = atof(optarg); break;
			case 'r': rough = atof(optarg); break;
			case 'a': rho.setConstant(atof(optarg)); break;
		}
	}

	readExr("L0.exr", L0);
	readExr("L1.exr", L1);
	readExr("T0.exr", T0);
	readExr("T1.exr", T1);

	ColorImage img(n, n);

	#pragma omp parallel for
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
			img(i, j) = computePixel(i, j);

	writeExr(img, "img.exr");
	return 0;
}
