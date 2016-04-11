#ifndef JDQZ_DECL_H
#define JDQZ_DECL_H

#include <vector>
#include <complex>

// This is (going to be) a templated port of
// the original JDQZ code by Fokkema and van Gijzen.

// Matrix members:
//   AMUL(N, Vector, Vector)
//   BMUL(N, Vector, Vector)
//   PRECON(N, Vector, Vector)

// Vector members:
//   TODO

template<typename Matrix, typename Vector>
class JDQZ
{
	// Size of problem
	int n_;

	// The value near which the eigenvalues are sought
	std::complex<double> target_;
	
	double tol_;

	// Tracking parameter	
	double lock_;
	
	int kmax_;
	int jmax_;
	int jmin_;
	int maxstep_;
	int testspace;
	int j_;

	// order =  0: nearest to target
	// order = -1: smallest real part
	// order =  1: largest real part
	// order = -2: smallest complex part
	// order =  2: largest complex part
	int order_;

	// method = 1: gmres(m)
	// method = 2: cgstab(l)
	int method_;

    // Searchspace gmres(m):
	int m_;

	// Degree polynomial in cgstab(l):
	int l_;

	// Maximum number of matvecs in cgstab or gmres
	int maxnmv_;

	// Testspace 1: w = "Standard Petrov" * v (Section 3.1.1)
	// Testspace 2: w = "Standard 'variable' Petrov" * v (Section 3.1.2)
	// Testspace 3: w = "Harmonic Petrov" * v (Section 3.5.1)
	int testspace_;

	// Compute the converged eigenvectors
	bool wanted_;
	
	// Eigenvalue pairs
	std::vector<std::complex<double> > alpha_,;
	int beta_;

	// Size of workspace
	int lwork_;

	// Workspace: two dimensional complex array of size n x lwork
	std::vector<Vector> zwork_;

	// Converged eigenvectors if wanted = true, else converged Schur vectors.
	std::vector<Vector> eivec_;

	
public:
	JDQZ(Matrix matrix);
	void setParameters();	
};

#endif
