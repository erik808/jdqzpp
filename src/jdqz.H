#ifndef JDQZ_H
#define JDQZ_H

#include "jdqz_decl.H"
#include <math.h>



//==================================================================
// constructor
template<typename Matrix>
JDQZ<Matrix>::
JDQZ(Matrix &matrix)
	:
	mat_(matrix)
{
	// set parameters
	setParameters();

	// setup workspace
	work_ = std::vector<Vector>(lwork_, Vector(n_));

	setIndices();
}


//==================================================================
template<typename Matrix>
void JDQZ<Matrix>::solve()
{
	//-------------------------------------------------------
	// some initializations
	int step      = 0;
	int solvestep = 0;
	int k         = 0;
	int j         = 0;
	int mxmv      = 0;
	int ldvs      = 50;
	int ldqkz     = ldvs;

	double deps   = 1.0;
	double rnrm   = 0.0;

	complex evcond( sqrt(pow(abs(shift_),2) + 1.0));
	complex shifta  = shift_ / evcond;
	complex shiftb  = 1.0    / evcond;
	
	complex targeta = shifta;
	complex targetb = shiftb;
	complex zalpha  = shifta;
	complex zbeta   = shiftb;

	Complex1D f(ldqkz);
	Complex1D alpha(ldvs);
	Complex1D beta(ldvs);	
	Complex1D zwork(4*ldvs);
	
	Complex2D ma(ldvs, ldvs);
	Complex2D mb(ldvs, ldvs);
	Complex2D zma(ldvs, ldvs);
	Complex2D zmb(ldvs, ldvs);
	Complex2D vsl(ldvs, ldvs);
	Complex2D vsr(ldvs, ldvs);

	Complex2D mqkz(ldqkz, ldqkz);
	Complex2D invqkz(ldqkz, ldqkz);

	std::vector<double> rwork(3*ldvs);

	std::vector<int> ipivqkz(ldqkz);

	// a few flags...
	bool ok    = true;
	bool found = false;
	
	while (k < kmax_ && step < maxstep_)
	{
		step++;
		solvestep++;
		if (j == 0)
		{
			// Initialize search and test space with random real part
			// and zero imaginary part. Values uniform in [-1,1].

			//--> right now we use ones in the example code for comparison
			work_[V_].random();
		}
		else 
		{
			mxmv = maxnmv_;
			deps = pow(2.0, -solvestep);
			if (j < jmin_)
			{
				mxmv = 1;
				gmres(n_, V_+j, D_, m_, deps, mxmv, zalpha, zbeta, k+1,
					  Kz_, Q_, invqkz, ldqkz, ipivqkz, f, U_, Tp_);
				// -- > continue here
			}
			else if (method_ == GMRES)
			{
			}
 			else if (method_ == CGSTAB)
			{ 
			}
		}
		
		j++;

		mgs(n_, j-1, V_, V_+j-1, 1);
		mgs(n_, k,   Q_, V_+j-1, 1);
		
		if (testspace_ == 1)      // Standard Petrov
		{
			jdqzmv(V_+j-1, W_+j-1, Tp_,
				   -std::conj(shiftb), std::conj(shifta));
		}
		else if (testspace_ == 2) // Standard 'variable' Petrov
		{
			jdqzmv(V_+j-1, W_+j-1, Tp_,
				   -std::conj(zbeta), std::conj(zalpha));
		}
		else if (testspace_ == 3) // Harmonic Petrov
		{
			jdqzmv(V_+j-1, W_+j-1, Tp_,
				   shifta, shiftb);
		}

		mgs(n_, j-1, W_, W_+j-1, 1);
		mgs(n_, k  , Z_, W_+j-1, 1);

		mat_.AMUL(work_[V_+j-1], work_[Av_+j-1]);
		mat_.BMUL(work_[V_+j-1], work_[Bv_+j-1]);
		
		makemm(n_, j, W_, Av_, ma, zma, ldvs);
		makemm(n_, j, W_, Bv_, mb, zmb, ldvs);

		gegs(j, zma, zmb, alpha, beta, vsl, vsr, zwork, rwork);

		bool attempt = true;
		while (attempt)
		{
			// --- Sort the Petrov pairs ---
			qzsort(targeta, targetb, j, zma, zmb,
				   vsl, vsr, ldvs, alpha, beta, order_);

			zalpha = zma(0,0);
            zbeta  = zmb(0,0);

			evcond = sqrt(pow(abs(zalpha),2) + pow(abs(zbeta),2));

			// --- compute new q ---
			// Q(:,k) = V_ * VSR(:,0)
			gemv(n_, j, 1.0, V_, &vsr(0,0), 0.0, Q_+k);			

			// --- orthogonalize new q ---
			mgs(n_, k, Q_, Q_+k, 1);

			// --- compute new z ---
			// Z(:,k) = W_ * VSL(:,0)
			gemv(n_, j, 1.0, W_, &vsl(0,0), 0.0, Z_+k);

			// --- orthogonalize new z ---
			mgs(n_, k, Z_, Z_+k, 1);

			// --- make new qkz ---
			work_[Kz_+k] = work_[Z_+k];
			mat_.PRECON(work_[Kz_+k]);
			makeqkz(n_, k+1, Q_, Kz_, mqkz, invqkz, ldqkz, ipivqkz);

			// --- compute new (right) residual= beta Aq - alpha Bq ---
			jdqzmv(Q_+k, D_, Tp_, zalpha, zbeta);
			
			// --- orthogonalize this vector to Z ---
			mgs(n_, k, Z_, D_, 0);

			rnrm = work_[D_].norm() / evcond.real();

			if (rnrm < lock_ && ok)
			{
				targeta = zalpha;
				targetb = zbeta;
				ok = false;
			}

			found   = ((rnrm < eps_ && j > 1) || k == kmax_ - 1);
			attempt = found;

			if (found)
			{
				// do stuff
				return;
			}
			else if (j == jmax_)
			{
				// do other stuff
				return;
			}			
		}
	}
}
//==================================================================
extern "C" void zgetrf_(int* M, int *N, complex *A,
						int *LDA, int *IPIV, int *INFO);

//------------------------------------------------------------------
template<typename Matrix>
void JDQZ<Matrix>::makeqkz(int n, int k, int Q, int Kq, Complex2D &qkz,
						   Complex2D &invqkz, int ldqkz,
						   std::vector<int> &ipiv)
{
	for (int i = 0; i != k; ++i)
		for (int j = 0; j != k; ++j)
		{
			if (i == k-1 || j == k-1)
			{
				qkz(i,j) = work_[Q+i].dot(work_[Kq+j]);
			}
			invqkz(i,j) = qkz(i,j);			
		}

	int info = 0;
	zgetrf_(&k, &k, &invqkz[0], &ldqkz, &ipiv[0], &info);
}

//==================================================================
// Y = alpha*A*X + beta*Y
template<typename Matrix>
void JDQZ<Matrix>::gemv(int m, int n, complex alpha, int A,
						complex *X, complex beta, int Y)
{
	work_[Y].scale(beta);
	for (int i = 0; i != n; ++i)
		work_[Y].axpy(alpha * X[i], work_[A+i]);
}

//==================================================================
// Y = alpha*A**H*X + beta*Y
template<typename Matrix>
void JDQZ<Matrix>::gemv(int m, int n, complex alpha, int A,
						int X, complex beta, Complex1D &Y)
{
	Y.scale(beta);
	for (int i = 0; i != n; ++i)
		Y[i] = work_[A].dot(work_[X]);
}

//==================================================================

extern "C" int select_(int *n, complex *sa, complex *sb,
					   complex *a, complex *b, int *order);

extern "C" void myexc_(int *n, complex *s, complex *t, complex *z,
					   complex *q, int *ldz, int *ifst, int *ilst);

//------------------------------------------------------------------
template<typename Matrix>
void JDQZ<Matrix>::qzsort(complex ta, complex tb, int k,
						  Complex2D &s, Complex2D &t, Complex2D &z,
						  Complex2D &q, int ldz, Complex1D &alpha,
						  Complex1D &beta, int order)
{
	// In this interface we try to be careful
	// with 1 and 0 based calls...
	int j = 0;
	for (int i = 1; i <= k; ++i)
	{
		for (j = 1; j <= k; ++j)
		{
			alpha[j-1] = s(j-1,j-1);
			beta[j-1] = t(j-1,j-1);
		}
		
		int N = k-i-1;		
		j = select_(&N, &ta, &tb, &alpha[i], &beta[i], &order) + i-1;		
		myexc_(&k, &s[0], &t[0], &z[0], &q[0], &ldz, &j, &i);		
	}
}

//==================================================================
// Lapack stuff:
extern "C" void zrot_(int *N, complex *CX, int *INCX,
					  complex *CY, int *INCY, double *C, complex *S);

extern "C" void zlartg_(complex *F, complex *G, double *CS,
						complex *SN, complex *R);

//------------------------------------------------------------------
template<typename Matrix>
void JDQZ<Matrix>::gmres(int n, int x, int r, int mxm, double eps, int mxmv,
						 complex alpha, complex beta, int k, int kz, int q,
						 Complex2D &invqkz, int ldqkz, std::vector<int> &ipiv,
						 Complex1D &f, int v, int tp)
{
	// -- some initializations --- 
	int maxm = 100;
	int nmv  = 0;
	int m, m1;
	int one  = 1;
	
	assert(mxm < maxm);

	complex ztmp, zdotc;
	complex rcs(0,0);

	std::vector<double> c(maxm);
	
	Complex1D rs(maxm);
	Complex1D  s(maxm);
	Complex1D  y(maxm);

	Complex2D hh(maxm, maxm-1);

	// --- initialize first residue ---
	work_[x].zero();
	work_[r].scale(-1.0);
	psolve(n, r, k, q, kz, invqkz, ldqkz, ipiv, f);

	double rnrm0 = work_[r].norm();
	double rnrm  = rnrm0;
	double eps1  = eps * rnrm;

	work_[v] = work_[r];

	// -- restart loop ---
	while (nmv < mxmv && rnrm > eps1) 
	{
		ztmp = 1.0 / rnrm;
		work_[v].scale(ztmp);
		rs[0] = rnrm;

		// --- inner loop ---
		m = -1;
		while(nmv < mxmv && m < mxm && rnrm > eps1)
		{
			m  = m + 1;
			m1 = m + 1;					
			jdqzmv(v+m, v+m1, tp, alpha, beta);
			psolve(n, v+m1, k, q, kz, invqkz, ldqkz, ipiv, f);

			nmv++;
			for (int i = 0; i <= m; ++i)
			{
				ztmp = work_[v+i].dot(work_[v+m1]);
				hh(i,m) = ztmp;
				work_[v+m1].axpy(-ztmp, work_[v+i]);
			}
			ztmp     = work_[v+m1].norm();
			hh(m1,m) = ztmp;
			work_[v+m1].scale(1.0 / ztmp);

			for (int i = 0; i <= m-1; ++i)
				zrot_(&one, &hh(i,m), &one, &hh(i+1,m), &one, &c[i], &s[i]);

			zlartg_(&hh(m,m), &hh(m1,m), &c[m], &s[m], &rcs);
			hh(m,m)  = rcs;
			hh(m1,m) = 0;
			rs[m1]   = 0;

			zrot_(&one, &rs[m], &one, &rs[m1], &one, &c[m], &s[m]);
			rnrm = abs(rs[m1]);
			return;
		}
		return;
	}
	
}

//==================================================================
extern "C" void zgetrs_(char* TRANS, int *N, int *NRHS, complex *A,
						int *LDA, int *IPIV, complex *B,
						int *LDB, int *INFO);

//------------------------------------------------------------------
template<typename Matrix>
void JDQZ<Matrix>::psolve(int n, int x, int nq, int q, int kz,
						  Complex2D &invqkz, int ldqkz,
						  std::vector<int> ipiv, Complex1D &f)
{
	mat_.PRECON(work_[x]);

	gemv(n, nq, 1.0, q, x, 0.0, f);

	char trans = 'N';
	int  nrhs = 1;
	int  info = 0;
	zgetrs_(&trans, &nq, &nrhs, &invqkz[0], &ldqkz,
			&ipiv[0], &f[0], &ldqkz, &info);

	gemv(n, nq, -1.0, kz, &f[0], 1.0, x);
	
}

//==================================================================
template<typename Matrix>
void JDQZ<Matrix>::mgs(int n, int k, int v, int w, int job)
{
	double s1 = work_[w].norm();
	double s0;
	std::complex<double> nrm;
	for (int i = 0; i <= k-1; ++i)
	{
		s0 = s1;
		ortho(n, v+i, w, s0, s1, nrm);
	}
	if (job != 0)
	{
		nrm = 1.0 / s1;
		work_[w].scale(nrm);
	}
}
//==================================================================
template<typename Matrix>
void JDQZ<Matrix>::jdqzmv(int x, int y, int tmp,
						  std::complex<double> alpha,
						  std::complex<double> beta)
{
	// y = beta * A * x - alpha * B * x
	mat_.AMUL(work_[x], work_[tmp]);
	mat_.BMUL(work_[x], work_[y]);
	work_[y].axpby(beta, work_[tmp], -alpha);
}

//==================================================================
template<typename Matrix>
void JDQZ<Matrix>::ortho(int n, int v, int w,
						  double &s0, double &s1,
						  std::complex<double> &nrm)
{
	nrm = work_[v].dot(work_[w]);
	work_[w].axpy(-nrm, work_[v]);
	s1 = work_[w].norm();
	
	double kappa = 100;
	if (s1 < s0 / kappa) // check for zero vector
	{
		// additional orthogonalization
		s0 = s1;
		std::complex<double>
			tmp = work_[v].dot(work_[w]);
		nrm += tmp;
		work_[w].axpy(-tmp, work_[v]);
		s1 = work_[w].norm();

		assert(s1 < s0 / kappa);
	}
	return;
}

//==================================================================
template<typename Matrix>
void JDQZ<Matrix>::makemm(int n, int k, int w, int v,
						  Complex2D &m, Complex2D &zm, int ldm)
{
	for (int i = 0; i <= k-1; ++i)
		for (int j = 0; j <= k-1; ++j)
		{
			if (i == k-1 || j == k-1)
				m(i,j) = work_[w+i].dot(work_[v+j]);
			zm(i,j) = m(i,j);
		}
}

//==================================================================
// LAPACK zgegs:
//   compute the left and/or right Schur vectors (VSL and VSR)
extern "C" void zgegs_(char *JOBVSL, char *JOBVSR, int *N,
					   complex *A, int *LDA,
					   complex *B, int *LDB,
					   complex *ALPHA, complex *BETA,
					   complex *VSL, int *LDVSL,
					   complex *VSR, int *LDVSR,
					   complex *WORK, int *LWORK,
					   double *RWORK, int *INFO);

//------------------------------------------------------------------
// Our wrapper for zgegs_:
template<typename Matrix>
void JDQZ<Matrix>::gegs(int N, Complex2D &A, Complex2D &B, 
						Complex1D &alpha, Complex1D &beta,
						Complex2D &VSL, Complex2D &VSR,
						Complex1D &work, std::vector<double> &rwork)
{
	int  LDA   = A.rows();
	int  LDB   = B.rows();
	int  LDVSL = VSL.rows();
	int  LDVSR = VSR.rows();
	int  LWORK = work.size();
	int  INFO  = 0;

	assert(LDA == LDB);
	assert(LDA == LDVSL);
	assert(LDA == LDVSR);
	assert(LWORK >= std::max(1, 2*N));
	assert(rwork.size() >= std::max(1, 3*N));
	
	char JOBVSL = 'V'; // compute left  Schur vectors
	char JOBVSR = 'V'; // compute right Schur vectors

	// calling lapack
	zgegs_(&JOBVSL, &JOBVSR, &N, &A[0], &LDA, &B[0], &LDB,
		   &alpha[0], &beta[0], &VSL[0], &LDVSL, &VSR[0], &LDVSR,
		   &work[0], &LWORK, &rwork[0], &INFO);	
}

//==================================================================
template<typename Matrix>
void JDQZ<Matrix>::setParameters()
{
	// obtain problem size from matrix
	n_ = mat_.size();

	//-------------------------------------------------------
	// temporarily hardcoded parameters 
	shift_     = std::complex<double>(0,0);
	eps_       = 1e-9;
	kmax_      = 5;
	jmax_      = 20;
	jmin_      = 10;
	maxstep_   = 1000;
	lock_      = 1e-9;
	order_     = 0;
	method_    = 1;
	m_         = 30;
	l_         = 2;
	maxnmv_    = 100;
	testspace_ = 3; 
	wanted_    = true;
	//-------------------------------------------------------

	if (method_ == 1)
		lwork_ =  4 +  m_  + 5*jmax_ + 3*kmax_;
	else
		lwork_ = 10 + 6*l_ + 5*jmax_ + 3*kmax_;
}

//==================================================================
template<typename Matrix>
void JDQZ<Matrix>::printParameters()
{
	std::cout << "\nJDQZ Parameters:   " << std::endl;
	std::cout << " | n_         " << n_ << std::endl;
	std::cout << " | shift_     " << shift_ << std::endl;
	std::cout << " | eps_       " << eps_ << std::endl;
	std::cout << " | kmax_      " << kmax_ << std::endl;
	std::cout << " | jmax_      " << jmax_ << std::endl;
	std::cout << " | jmin_      " << jmin_ << std::endl;
	std::cout << " | maxstep_   " << maxstep_ << std::endl;
	std::cout << " | lock_      " << lock_ << std::endl;
	std::cout << " | order_     " << order_ << std::endl;
	std::cout << " | method_    " << method_ << std::endl;
	std::cout << " | m_         " << m_ << std::endl;
	std::cout << " | l_         " << l_ << std::endl;
	std::cout << " | maxnmv_    " << maxnmv_ << std::endl;
	std::cout << " | testspace_ " << testspace_ << std::endl;
	std::cout << " | wanted_    " << wanted_ << std::endl;
	std::cout << " | lwork_     " << lwork_ << '\n' << std::endl;	
}

//==================================================================
template<typename Matrix>
void JDQZ<Matrix>::setIndices()
{
	// d   = rhs, these pointers refer to the columns of the workspace
	D_   = 0;
	// Workspace for jdqzmv
	Tp_  = D_ + 1;
	// u   = pointer to Krylov space GMRES(m) or Bi-CSTAB(l)
	U_   = Tp_ + 1; 
	// v   = pointer to search space JDQZ with max dimension jmax
	if (method_ == GMRES)
		V_ = U_ + m_ + 1;
	else if (method_ == CGSTAB)
		V_ = U_ + 2*l_ + 6;
	// w   = pointer to test subspace JDQZ with max dimension jmax
	W_   = V_ + jmax_;
	// av  = pointer to subspace AV with max dimension jmax
	Av_  = W_ + jmax_;
	// bv  = pointer to subspace BV with max dimension jmax
	Bv_  = Av_ + jmax_;
	// aux =
	Aux_ = Bv_ + jmax_;
	// q   = pointer to search Schur basis in JDQZ with max dimension kmax
	Q_   = Aux_ + jmax_;
	// z   = pointer to test Schur basis in JDQZ with max dimension kmax
	Z_   = Q_ + kmax_;
	// kz  = pointer to matrix K^{-1}Z_k
	Kz_  = Z_ + kmax_;
}

#endif
