#include "gtest/gtest.h" // google test framework
#include "jdqz.hpp"
#include <vector>
#include <complex>
#include <iostream>
#include <assert.h>

#include "test.hpp"

std::vector<double>  TestVector::norms = std::vector<double>();
std::vector<complex> TestVector::alphas = std::vector<complex>();
std::vector<complex> TestVector::betas = std::vector<complex>();
std::vector<complex> TestVector::dotresults = std::vector<complex>();

//------------------------------------------------------------------
TEST(JDQZ, Results)
{
	size_t size = 50;
	TestMatrix2 testmat(size);
	TestVector vec(size, 0.0);
	JDQZ<TestMatrix2> jdqz(testmat, vec);

	std::map<std::string, double> list;
	
	list["Shift (real part)"]         = 0.5;
	list["Number of eigenvalues"]     = 5;
	list["Max size search space"]     = 20;
	list["Min size search space"]     = 10;
	list["Max JD iterations"]         = 200;
	list["Tracking parameter"]        = 1e-8;
	list["Criterion for Ritz values"] = 0;
	list["Linear solver"]             = 1;
	list["GMRES search space"]        = 20;
	list["Verbosity"]                 = 5;
    list["Profile"]                   = 1;
		
	ParameterList params(list);
	
	jdqz.setParameters(params);
	jdqz.printParameters();
	jdqz.solve();
    jdqz.printProfile("jdqz_profile");
	
	std::vector<TestVector>            eivec = jdqz.getEigenVectors();
	std::vector<std::complex<double> > alpha = jdqz.getAlpha();
	std::vector<std::complex<double> > beta  = jdqz.getBeta();

	std::cout << jdqz.kmax() << " converged eigenvalues\n\n";

	EXPECT_EQ(jdqz.kmax(), 5);
	EXPECT_LT(jdqz.iterations(), 120);
	
	for (int j = 0; j != jdqz.kmax(); ++j)
	{
		TestVector residue(size, 0.0);
		TestVector tmp(size, 0.0);
		testmat.AMUL(eivec[j], residue);
		residue.scale(beta[j]);
		testmat.BMUL(eivec[j], tmp);
		residue.axpy(-alpha[j], tmp);
		std::cout << "alpha: " << std::setw(20) << alpha[j]
				  << " beta: " << std::setw(20) << beta[j]
				  << " " << residue.norm() << std::endl;
		
		EXPECT_NEAR(residue.norm(), 0, 1e-8);
	}	
}
