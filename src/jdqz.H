#ifndef JDQZ_H
#define JDQZ_H

#include "jdqz_decl.H"
#include <math.h>

typedef std::complex<double> complex;

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
	// some initializations
	int step      = 0;
	int solvestep = 0;
	int k         = 0;
	int j         = 0;
	int mxmv      = 0;
	int ldvs      = 50;
	int ldqkz     = ldvs;

	double deps   = 1.0;		

	complex evcond( sqrt(pow(abs(shift_),2) + 1.0));
	complex shifta  = shift_ / evcond;
	complex shiftb  = 1.0    / evcond;
	
	complex targeta = shifta;
	complex targetb = shiftb;
	complex alpha   = shifta;
	complex beta    = shiftb;

	std::vector<std::vector<complex> > invqkz
		(ldqkz, std::vector<complex>(ldqkz));

	std::vector<int> ipivqkz(ldqkz);

	std::vector<complex> f(ldqkz);
	
	while (k < kmax_ && step < maxstep_)
	{
		step++;
		solvestep++;
		if (j == 0)
		{
			// Initialize search and test space with random real part
			// and zero imaginary part. Values uniform in [-1,1].

			//--> right now we use ones in the example code for comparison
			work_[v_].random();
		}
		else 
		{
			mxmv = maxnmv_;
			deps = pow(2.0, -solvestep);
			if (j < jmin_)
			{
				mxmv = 1;
				// --> we should to pass indices instead of arrays
				//  gmres(n_, work_[v_+j], work_[d_], m_, deps,
				//        mxmv, zalpha, zbeta, k+1,
				//        work_[kz_], work_[q_], invqkz, ldqkz,
				//        ipivqkz, f, work_[u_], work_[tp_] );
			}
			else if (method_ == GMRES)
			{
			}
 			else if (method_ == CGSTAB)
			{
			}
		}
		j++;

		mgs(n_, j-1, v_, v_+j-1, 1);
		mgs(n_, k,   q_, v_+j-1, 1);
		
		if (testspace_ == 1)      // Standard Petrov
		{
			jdqzmv(n_, v_+j-1, w_+j-1, tp_,
				   -std::conj(shiftb), std::conj(shifta));
		}
		else if (testspace_ == 2) // Standard 'variable' Petrov
		{
			jdqzmv(n_, v_+j-1, w_+j-1, tp_,
				   -std::conj(beta), std::conj(alpha));
		}
		else if (testspace_ == 3) // Harmonic Petrov
		{
			jdqzmv(n_, v_+j-1, w_+j-1, tp_,
				   -std::conj(shiftb), std::conj(shifta));
		}
		
		return;
	}
}

//==================================================================
template<typename Matrix>
void JDQZ<Matrix>::gmres()
{}

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
void JDQZ<Matrix>::jdqzmv(int n, int x, int y, int tmp,
				std::complex<double> alpha, std::complex<double> beta)
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
void JDQZ<Matrix>::setParameters()
{
	// obtain problem size from matrix
	n_ = mat_.size();

	//-------------------------------------------------------
	// temporarily hardcoded parameters 
	shift_     = std::complex<double>(0,0);
	tol_       = 1e-9;
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
	std::cout << " | tol_       " << tol_ << std::endl;
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
	//     d   = rhs, these pointers refer to the columns of the workspace
	d_   = 1;
	//     Workspace for jdqzmv
	tp_  = d_ + 1;
	//     u   = pointer to Krylov space GMRES(m) or Bi-CSTAB(l)
	u_   = tp_ + 1;
	//     v   = pointer to search space JDQZ with max dimension jmax
	if (method_ == GMRES)
		v_ = u_ + m_ + 1;
	else if (method_ == CGSTAB)
		v_ = u_ + 2*l_ + 6;
	//     w   = pointer to test subspace JDQZ with max dimension jmax
	w_   = v_ + jmax_;
	//     av  = pointer to subspace AV with max dimension jmax
	av_  = w_ + jmax_;
	//     bv  = pointer to subspace BV with max dimension jmax
	bv_  = av_ + jmax_;
	//     aux =
	aux_ = bv_ + jmax_;
	//     q   = pointer to search Schur basis in JDQZ with max dimension kmax
	q_   = aux_ + jmax_;
	//     z   = pointer to test Schur basis in JDQZ with max dimension kmax
	z_   = q_ + kmax_;
	//     kz  = pointer to matrix K^{-1}Z_k
	kz_  = z_ + kmax_;
} 

#endif
