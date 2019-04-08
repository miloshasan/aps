#pragma once

#include <Eigen/Dense>

typedef Eigen::Array<float, 3, 1> Color;
typedef Eigen::Matrix<float, 3, 1> Vector3;
typedef Eigen::Array<Color, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> ColorImage;
typedef Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> FloatImage;

inline float sqr(float x) { return x * x; }
inline float lerp(float t, float x, float y) { return (1-t) * x + t * y; }
const float pi = float(M_PI);

/// Schlick Fresnel term
inline float schlick(float cosTheta, float f0)
{
	return f0 + (1 - f0) * std::pow(1 - cosTheta, 5);
}

class MicrofacetLobe
{
	float m_alpha;
	float m_f0;
	Vector3 m_incoming;

	// GGX NDF importance sampling
	inline Vector3 sampleGGX(float r1, float r2, float alpha)
	{
		float theta = std::atan(alpha * std::sqrt(r1) / std::sqrt(1 - r1));
		float phi = 2 * pi * r2;
		float ct, cp, st, sp;
		__sincosf(theta, &st, &ct);
		__sincosf(phi, &sp, &cp);
		return Vector3(st*cp, st*sp, ct);
	}

	/// Smith shadowing-masking
	inline float smithG(float cosTheta, float alpha)
	{
		if (cosTheta <= 0) return 0;
		float c2 = sqr(cosTheta);
		float t2 = (1 - c2) / c2;
		float a2 = sqr(alpha);
		float tmp = std::sqrt(1 + a2 * t2);
		return 2 / (1 + tmp);
	}

public:
	MicrofacetLobe(float alpha, float f0, Vector3 incoming):
		m_alpha(alpha), m_f0(f0), m_incoming(incoming) {}

	bool sample(float r1, float r2, Vector3& direction, float& weight)
	{
		// choose microfacet normal
		Vector3 half = sampleGGX(r1, r2, m_alpha);
		float hdotn = half[2];

		// check if incoming above microsurface
		float idoth = m_incoming.dot(half);
		if (idoth <= 0) return false;

		// reflect incoming around half
		direction = -m_incoming + 2 * idoth * half;

		// check if reflection above horizon
		float odotn = direction[2];
		if (odotn <= 0) return false;

		float idotn = m_incoming[2];
		float F = schlick(idoth, m_f0);
		float G = smithG(idotn, m_alpha) * smithG(odotn, m_alpha);
		weight = idoth / (idotn * hdotn) * F * G;
		return true;
	}
};

inline float randf()
{
	return float(rand()) / RAND_MAX;
}
