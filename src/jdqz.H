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
	// some initializations
	int step      = 0;
	int solvestep = 0;
	int k         = 0;
	int j         = 0;
	int mxmv     = 0;
	
	double deps   = 1.0;
	std::complex<double> evcond( sqrt(pow(abs(shift_),2)+1.0));

	
    //  complex 
	//  evcond = dcmplx(sqrt(abs(shift)**2+abs(one)**2))
	// 	shifta = shift/evcond
	// 	shiftb = one/evcond

    //  targeta = shifta
    //  targetb = shiftb

    //  zalpha = shifta
    //  zbeta = shiftb

	// 	std::complex<double> zalpha = shifta 
    //  zbeta = shiftb

	
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
			std::cout << work_[v_].norm() << std::endl;
			std::cout << work_[v_].size() << std::endl;
		}
		else 
		{
			mxmv = maxnmv_;
			deps = pow(2.0, -solvestep);
			if (j < jmin_)
			{
				mxmv = 1;
				// zgmres(n_, work_[v_+j], work_[d_], m_, deps,
				// 	   mxmv, zalpha, zbeta, k+1,
				// 	   work(1,kz), work(1,q), invqkz, ldqkz,
				// 	   ipivqkz, f, work(1,u), work(1,tp) )
			}
		}

	}
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
