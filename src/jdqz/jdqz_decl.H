#ifndef JDQZ_DECL_H
#define JDQZ_DECL_H

#include <vector>
#include <complex>
#include <map>
#include <array>
#include <stack>
#include <ctime>
#include "jdqz_macros.H"

typedef std::complex<double> complex;

class Complex1D;
class Complex2D;
class JDQZTimer;

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
    using ProfileType =
        typename std::map<std::string,
                          std::array<double, PROFILE_ENTRIES> >;

	// Size of problem
	int n_;

	// The shift / value near which the eigenvalues are sought
	std::complex<double> shift_;

	// Tolerance of the eigensolutions,
	// $\| \beta \BA\Bx - \alpha \BB\Bx \| / | \alpha/\beta | < \epsilon$
	double eps_;

	// Tracking parameter:
	//  take it small to avoid missing eigensolutions (~1e-9)
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
	std::vector<complex> alpha_;
	std::vector<complex> beta_;

	// Size of workspace
	int lwork_;
	
	// The workspace: a std::vector of templated Vectors.
	std::vector<Vector> work_;

	// Indices referring to Vectors in the workspace, see setIndices()
	int D_, Tp_, U_, V_, W_, Av_, Bv_, Aux_, Q_, Z_, Kz_;

	// Converged eigenvectors if wanted = true, else converged Schur vectors.
	std::vector<Vector> eivec_;

	// Matrix wrapper
	Matrix mat_;

	// Initial vector
	Vector initial_;
	
	bool initialized_;

	// output level
	int verbosity_;

    //-------------------------------------------------------
    // Profiling: 
    // enable profile
	bool profile_;

    // profile data
    ProfileType profileData_;
    
    // timer stack to enable nested timings
    std::stack<JDQZTimer> timerStack_;
    //-------------------------------------------------------

public:
	// constructor
	JDQZ(Matrix &matrix, Vector &initial);

	~JDQZ();
	
	void solve();

	template<typename PList>
	void setParameters(PList &params);
	void printParameters();	
	
	int kmax() {return kmax_;}
	std::vector<Vector> getEigenVectors() {return eivec_;}
	std::vector<std::complex<double> > getAlpha() {return alpha_;}
	std::vector<std::complex<double> > getBeta() {return beta_;}

    void printProfile(std::string const &filename = "jdqz_profile");

private:

	void setup();
	
	void gmres(int n, int x, int r, int mxm, double &eps, int &mxmv,
			   complex alpha, complex beta, int k, int kz, int q,
			   Complex2D &invqkz, int ldqkz, std::vector<int> &ipivqkz,
			   Complex1D &f, int u, int tp);
	
	void psolve(int n, int x, int nq, int q, int kz,
				Complex2D &invqkz, int ldqkz,
				std::vector<int> ipiv, Complex1D &f);
	
	void mgs(int n, int k, int v, int w, int job);
	
	void ortho(int n, int v, int w,
			   double &s0, double &s1,
			   std::complex<double> &znrm);
	
	void jdqzmv(int x, int y, int tmp,
				std::complex<double> alpha,
				std::complex<double> beta);

	void makemm(int n, int k, int w, int v,
				Complex2D &m, Complex2D &zm, int ldm);

	// Our wrapper for lapack's zgegs_
	void gegs(int N, Complex2D &A, Complex2D &B,
			  Complex1D &alpha, Complex1D &beta,
			  Complex2D &VSL, Complex2D &VSR,
			  Complex1D &work, std::vector<double> &rwork);

	void qzsort(complex ta, complex tb, int k,
				Complex2D &s, Complex2D &t, Complex2D &z,
				Complex2D &q, int ldz, Complex1D &alpha,
				Complex1D &beta, int order);		

	// Matrix-Vector multiplication, type TRANS = N
	// Here we assume A and Y are in the workspace and use
	// members of the templated Vector type.
	// Y = alpha*A*X + beta*Y
	void gemv(int m, int n, complex alpha, int A,
			  complex *X, complex beta, int Y);

	// Matrix-Vector multiplication, type TRANS = C
	// Here we assume A and X are in the workspace.
	// Y = alpha*A**H*X + beta*Y
	void gemv(int m, int n, complex alpha, int A,
			  int X, complex beta, Complex1D &Y);
	
	// Matrix-Vector multiplication, type TRANS = C
	// Here we assume A and X are in the workspace.
	// Instead of a Complex1D we accept a complex *.
	// Y = alpha*A**H*X + beta*Y 
	void gemv(int m, int n, complex alpha, int A,
			  int X, complex beta, complex *Y); 
	
	// Matrix-Matrix multiplication
	// Here we assume A and C are in the workspace.
	// C = alpha*A*B + beta*C
	void gemm(int m, int n, int k, complex alpha, int A,
			  complex *B, int ldb, complex beta, int C);

	// Matrix-Matrix multiplication
	// Here we assume A and C are in the workspace.
	// Instead of an index we expect an entire
	// std::vector<Vector>
	// C = alpha*A*B + beta*C
	void gemm(int m, int n, int k, complex alpha, int A,
			  complex *B, int ldb, complex beta, std::vector<Vector> &C);
	
	void makeqkz(int n, int k, int Q, int Kq, Complex2D &qkz,
				 Complex2D &invqkz, int ldqkz,
				 std::vector<int> &ipivqkz);

	void setIndices();

    // Profile things
    void timerStart(std::string const &msg);
    void timerStop(std::string const &msg);
};

// ==================================================================
// Our own inherited complex vector class
//  This will be based around a 1D std::vector<std::complex<double> >.
class Complex1D : public std::vector<std::complex<double> >
{
public: 
	// constructor
	Complex1D(size_t n)
		:
		std::vector<std::complex<double> >(n, 0)
		{}

	void scale(complex za)
		{
			for (auto &el: *this)
				el *= za;
		}
};

// ==================================================================
// Our own complex matrix class.
//  This will be based around a 1D std::vector<std::complex<double> >.
//  We will overload operator()(int, int) to mimick double indexing
//  in fortran. The data will be stored in column major form, which
//  simplifies the coupling to lapack: if array is a Complex2D object
//  you can pass &array[0] to a lapack routine expecting a matrix.

class Complex2D : public std::vector<std::complex<double> >
	// row dimension
{
	size_t m_;
	
	// column dimension
	size_t n_;
	
public:

	// constructor
	Complex2D(size_t m, size_t n)
		:
		std::vector<std::complex<double> >(m*n, 0),
		m_(m),
		n_(n)
		{}

	std::complex<double> &operator()(size_t i, size_t j)
		{ return (*this)[i+j*m_];}

	std::complex<double> const &operator()(size_t i, size_t j) const
		{ return (*this)[i+j*m_];}

	size_t rows() {return m_;}
	size_t cols() {return n_;}
};

//==================================================================
// Timer class
class JDQZTimer
{
	double startTime_;
    std::string label_;
public:
	JDQZTimer(std::string const &label)
		:
		startTime_(0.0),
        label_(label)
		{}
	
	double wallTime()
		{
#ifdef HAVE_MPI 
			int mpiInit;
			MPI_Initialized(&mpiInit);
			if (mpiInit)
				return MPI_Wtime();
			else
				return (double) std::clock() / CLOCKS_PER_SEC;
#else
            return (double) std::clock() / CLOCKS_PER_SEC;
#endif 
		}
	
	void resetStartTime()
		{ startTime_ = wallTime();}
	
	double elapsedTime()
		{ return (double) (wallTime() - startTime_); }

    void printLabel()
        { WRITE(" Timer: " << label_); }

    std::string label() { return label_;}
};

#endif
