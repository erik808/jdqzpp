#include "gtest/gtest.h" // google test framework
#include "jdqz.H"
#include <vector>
#include <complex>
#include <iostream>
#include <assert.h>

#include "test.H"

std::vector<double>  TestVector::norms = std::vector<double>();
std::vector<complex> TestVector::alphas = std::vector<complex>();
std::vector<complex> TestVector::betas = std::vector<complex>();
std::vector<complex> TestVector::dotresults = std::vector<complex>();

//------------------------------------------------------------------
TEST(Vector, General)
{
	TestVector vec1(4, 0.0);
	TestVector vec2(4, 0.0);
	vec1[0].real(1.0); vec1[0].imag(2.0);
	vec1[1].real(1.0); vec1[1].imag(0.0);
	vec1[2].real(1.0); vec1[2].imag(0.0);
	vec1[3].real(2.0); vec1[3].imag(0.0);

	vec2[0].real(0.0); vec2[0].imag(2.0);
	vec2[1].real(0.0); vec2[1].imag(1.0);
	vec2[2].real(1.0); vec2[2].imag(-1.0);
	vec2[3].real(1.0); vec2[3].imag(2.0);

	std::complex<double> result = vec1.dot(vec2);
	
	EXPECT_EQ(result.real(), 7);
	EXPECT_EQ(result.imag(), 6);

	std::complex<double> a(4,3);
	vec1.axpy(a, vec2);
	EXPECT_NEAR(vec1.norm(), 18.1934, 1e-4);
	
	std::complex<double> b(2,1);
	vec1.axpby(a, vec2, b);
	EXPECT_NEAR(vec1.norm(), 57.3149, 1e-4);
}

//------------------------------------------------------------------
TEST(Matrix, Operators)
{
	size_t n = 100;
	TestVector vec1(n, 1.0);
	TestVector vec2(n, 0.0);
	TestVector vec3(n, 0.0);
		
	TestMatrix testmat(n);
	testmat.AMUL(vec1, vec2);
	EXPECT_EQ(vec2[49].real(), 50);
	EXPECT_EQ(vec2[77].real(), 78);
	testmat.BMUL(vec2, vec3);
	EXPECT_EQ(vec1 == vec3, true);
	testmat.PRECON(vec1);
	EXPECT_EQ(vec1[9], (double) 10 / (100-31));
}

//------------------------------------------------------------------
TEST(JDQZ, Run)
{
	TestVector::norms.clear();
	TestVector::alphas.clear();
	TestVector::betas.clear();
	TestVector::dotresults.clear();

	size_t size = 100;
	TestMatrix testmat(size);
	JDQZ<TestMatrix> jdqz(testmat);

	
	ParameterList params;
	jdqz.setParameters(params);
	
	jdqz.solve();	
	
	// testing rnrm in the inner "attempt" loop
	EXPECT_NEAR(TestVector::norms[7],   7.370188617854, 1e-9 );
	EXPECT_NEAR(TestVector::norms[20],  0.808903713729, 1e-9 );
	EXPECT_NEAR(TestVector::norms[35],  0.671927897667, 1e-9 );
	EXPECT_NEAR(TestVector::norms[52],  0.467505198259, 1e-9 );
	EXPECT_NEAR(TestVector::norms[196], 0.777032215617, 1e-9 );
	EXPECT_NEAR(TestVector::norms[238], 0.158442752163, 1e-9 );
	EXPECT_NEAR(TestVector::norms[282], 1.1112136E-004, 1e-9 );

	// testing rnrm in gmres
	EXPECT_NEAR(TestVector::norms[10],  1.625402443487, 1e-9 );
	EXPECT_NEAR(TestVector::norms[23],  1.1267987E-002, 1e-9 );
	EXPECT_NEAR(TestVector::norms[38],  3.3277400E-002, 1e-9 );
	
	EXPECT_NEAR(TestVector::norms[297], 9.6501368E-011, 1e-13);
	EXPECT_NEAR(TestVector::norms[330], 1.0632243E-010, 1e-13);
	EXPECT_NEAR(TestVector::norms[329], 2.7895008E-009, 1e-13);
	EXPECT_NEAR(TestVector::norms[344], 2.8189689E-015, 1e-16);
	EXPECT_NEAR(TestVector::norms[385], 8.7597752E-004, 1e-9 );
	EXPECT_NEAR(TestVector::norms[386], 9.9515335E-005, 1e-9 );
	EXPECT_NEAR(TestVector::norms[388], 1.0223998E-005, 1e-9 );
	EXPECT_NEAR(TestVector::norms[402], 1.1491491E-004, 1e-9 );

	EXPECT_NEAR(TestVector::norms[418], 14.00795121132, 1e-6 );
	EXPECT_NEAR(TestVector::norms[445], 1.0260044E-005, 1e-8 );
	EXPECT_NEAR(TestVector::norms[462], 14.66584688628, 1e-6 );
	EXPECT_NEAR(TestVector::norms[474], 3.9779726E-008, 1e-14);
	EXPECT_NEAR(TestVector::norms[521], 1.3973863E-010, 1e-15);
	EXPECT_NEAR(TestVector::norms[632], 1.4030857E-010, 1e-15);
	EXPECT_NEAR(TestVector::norms[773], 6.1466513E-013, 1e-15);
	
	// testing shifts used in jdqzmv, accessible through axpby
	EXPECT_NEAR(TestVector::alphas[67].real() /
				TestVector::betas[67].real(), 1.00000001236595, 1e-9);

	EXPECT_NEAR(TestVector::alphas[83].real() /
				TestVector::betas[83].real(), 0.99999999999999, 1e-9);

	EXPECT_NEAR(TestVector::alphas[117].real() /
				TestVector::betas[117].real(), 4, 1e-9);

	EXPECT_NEAR(TestVector::alphas[126].real() /
				TestVector::betas[126].real(), 9, 1e-9);

	EXPECT_NEAR(TestVector::alphas[131].real() /
				TestVector::betas[131].real(), 16, 1e-9);

	EXPECT_NEAR(TestVector::alphas[132].real() /
				TestVector::betas[132].real(), 25, 1e-9);
}

//------------------------------------------------------------------
TEST(JDQZ, Results)
{
	size_t size = 100;
	TestMatrix testmat(size);
	JDQZ<TestMatrix> jdqz(testmat);
	
	ParameterList params;
 	jdqz.setParameters(params);

	jdqz.solve();
	
	std::vector<TestVector>            eivec = jdqz.getEigenVectors();
	std::vector<std::complex<double> > alpha = jdqz.getAlpha();
	std::vector<std::complex<double> > beta  = jdqz.getBeta();

	EXPECT_EQ(alpha[0].real() / beta[0].real(), 1);
	EXPECT_EQ(alpha[1].real() / beta[1].real(), 4);
	EXPECT_EQ(alpha[2].real() / beta[2].real(), 9);
	EXPECT_EQ(alpha[3].real() / beta[3].real(), 16);
	EXPECT_EQ(alpha[4].real() / beta[4].real(), 25);

	std::vector<double> example_norms = { 0.607317e-13,
										  0.129407e-08,
										  0.205962e-08,
										  0.686810e-09,
										  0.614660e-12 };
	
	for (int j = 0; j != jdqz.kmax(); ++j)
	{
		TestVector residue(size, 0.0);
		TestVector tmp(size, 0.0);
		testmat.AMUL(eivec[j], residue);
		residue.scale(beta[j]);
		testmat.BMUL(eivec[j], tmp);
		residue.axpy(-alpha[j], tmp);
		EXPECT_NEAR(residue.norm(), example_norms[j], 1e-14);
	}	
}

//------------------------------------------------------------------
int main(int argc, char **argv)
{
	::testing::InitGoogleTest(&argc, argv);
	int out = RUN_ALL_TESTS();
	std::cout << "TEST exit code " << out << std::endl;
	return out;
}
