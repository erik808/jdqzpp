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

template<typename Matrix>
class JDQZ
{
	// Get Vector type from matrix
	using Vector = typename Matrix::Vector;

	// Size of problem
	int n_;

	// The shift / value near which the eigenvalues are sought
	std::complex<double> shift_;
	
	double tol_;

	// Tracking parameter	
	double lock_;

	// Number of wanted eigensolutions, on output: number of converged eigenpairs
	int kmax_;

	// Maximum size of the search space
	int jmax_;

	// Minimum size of the search space
	int jmin_;

	// Maximum number of Jacobi-Davidson iterations
	int maxstep_;

	// Determines how to expand the testspace W:
	//     Testspace 1: w = "Standard Petrov" * v            (Section 3.1.1)
	//     Testspace 2: w = "Standard 'variable' Petrov" * v (Section 3.1.2)
	//     Testspace 3: w = "Harmonic Petrov" * v            (Section 3.5.1)
	int testspace_;

	// Selection criterion for Ritz values:
	//   order =  0: nearest to target
	//   order = -1: smallest real part
	//   order =  1: largest real part
	//   order = -2: smallest complex part
	//   order =  2: largest complex part
	int order_;

	// method = 1: gmres(m)
	// method = 2: cgstab(l)
 	int method_;

	enum Method {GMRES = 1, CGSTAB};

    // Searchspace gmres(m):
	int m_;

	// Degree polynomial in cgstab(l):
	int l_;

	// Maximum number of matvecs in cgstab or gmres
	int maxnmv_;

	// Compute the converged eigenvectors
	bool wanted_;
	
	// Eigenvalue pairs
	std::vector<std::complex<double> > alpha_, beta_;

	// Size of workspace
	int lwork_;
	
	// Workspace: two dimensional complex array of size n x lwork
	std::vector<Vector> work_;

	// Indices referring to columns in the workspace, see setIndices()
	int d_, tp_, u_, v_, w_, av_, bv_, aux_, q_, z_, kz_;

	// Converged eigenvectors if wanted = true, else converged Schur vectors.
	std::vector<Vector> eivec_;

	// Matrix wrapper
	Matrix mat_;
	
public:
	JDQZ(Matrix &matrix);

	void solve();
	int  size() { return n_; }

private:
	void setParameters();
	void setIndices();
	void gmres();
	void mgs(int n, int k, int v, int w, int job);
	void ortho(int n, int v, int w,
			   double &s0, double &s1,
			   std::complex<double> &znrm);
};

#endif
