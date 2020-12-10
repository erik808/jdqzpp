#ifndef JDQZ_HPP
#define JDQZ_HPP

#include "jdqz_decl.hpp"
#include "jdqz_macros.hpp"
#include <ios>
#include <fstream>
#include <math.h>
#include <iomanip>

//==================================================================
// LAPACK dependencies
extern "C"
{
    void zcopy_(int *N, complex *ZX, int *INCX, complex *ZY,	int *INCY);

    void zlacpy_(char const *UPLO, int *M, int *N, complex *A,
                 int *LDA, complex *B, int *LDB);

    void zlaset_(char const *UPLO, int *M, int *N,
                 complex *alpha, complex *beta, complex *A,
                 int *LDA);

    void zggev_(char const *JOBVL, char const *JOBVR, int *N,
                complex *A, int *LDA,
                complex *B, int *LDB,
                complex *ALPHA, complex *BETA,
                complex *VL, int *LDVL,
                complex *VR, int *LDVR,
                complex *WORK, int *LWORK,
                double *RWORK, int *INFO);

    void zgetrf_(int* M, int *N, complex *A,
                 int *LDA, int *IPIV, int *INFO);

    void zrot_(int *N, complex *CX, int *INCX,
               complex *CY, int *INCY, double *C, complex *S);

    void zlartg_(complex *F, complex *G, double *CS,
                 complex *SN, complex *R);

    void ztrsv_(char const *UPLO, char const *TRANS, char const *DIAG,
                int *N, complex *A, int *LDA, complex *X, int *INCX);

    void zgetrs_(char const *TRANS, int *N, int *NRHS, complex *A,
                 int *LDA, int *IPIV, complex *B,
                 int *LDB, int *INFO);

//  compute the left and/or right Schur vectors (VSL and VSR)
    void zgges_(char *JOBVSL, char *JOBVSR,
                char *SORT, int *SELCTG, int *N,
                complex *A, int *LDA, complex *B, int *LDB,
                int *SDIM,
                complex *ALPHA, complex *BETA,
                complex *VSL, int *LDVSL,
                complex *VSR, int *LDVSR,
                complex *WORK, int *LWORK,
                double *RWORK, int *BWORK, int *INFO);

//  reorder the left and/or right Schur vectors (VSL and VSR)
    void ztgexc_(int *WANTQ, int *WANTZ, int *N,
                 complex *A, int *LDA, complex *B, int *LDB,
                 complex *Q, int *LDQ, complex *Z, int *LDZ,
                 int *IFST, int *ILST,
                 int *INFO);
}
//==================================================================
// constructor
template<typename Matrix>
JDQZ<Matrix>::
JDQZ(Matrix &matrix, Vector &initial)
    :
    n_(initial.length()),
    j_(0),
    k_(0),
    mat_(matrix),
    initial_(initial),
    initialized_(false)
{}


//==================================================================
// constructor
template<typename Matrix>
JDQZ<Matrix>::
~JDQZ()
{
    WRITE( "JDQZ: Destructor called..." );
}

//==================================================================
template<typename Matrix>
void JDQZ<Matrix>::setup()
{
    // setup workspace
    initial_.zero();
    work_ = std::vector<Vector>(lwork_, Vector(initial_));

    // setup solution
    eivec_ = std::vector<Vector>(kmax_, Vector(initial_));

    // set the indices for specific components in the workspace
    setIndices();

    initialized_ = true;
}

//==================================================================
template<typename Matrix>
void JDQZ<Matrix>::solve()
{
    timerStart("jdqz solve");

    if (!initialized_)
    {
        WRITE( "JDQZ: ERROR! Parameters are not set, "
               << "setup() not called, RETURNING\n");
        return;
    }

    //-------------------------------------------------------
    // some initializations
    iterations_         = 0;
    int solveIterations = 0;
    int jprev           = j_;
    j_                  = 0;
    k_                  = 0;
    int mxmv            = 0;
    int ldvs            = std::max(50, jmax_);
    int ldzwork         = 4*ldvs;
    int ldqkz           = ldvs;
    int itmp            = 0;
    int one             = 1;

    double deps         = 1.0;
    double rnrm         = 0.0;

    complex zzero(0,0);
    complex zone(1,0);

    complex evcond( sqrt(pow(abs(shift_),2) + 1.0));
    complex shifta  = shift_ / evcond;
    complex shiftb  = 1.0    / evcond;

    complex targeta = shifta;
    complex targetb = shiftb;
    complex zalpha  = shifta;
    complex zbeta   = shiftb;

    Complex1D alpha(ldvs);
    Complex1D beta(ldvs);

    Complex1D f(ldqkz);
    Complex1D zwork(ldzwork);
    Complex1D aconv(ldvs);
    Complex1D bconv(ldvs);

    Complex2D ma(ldvs, ldvs);
    Complex2D mb(ldvs, ldvs);
    Complex2D zma(ldvs, ldvs);
    Complex2D zmb(ldvs, ldvs);
    Complex2D vsl(ldvs, ldvs);
    Complex2D vsr(ldvs, ldvs);
    Complex2D ra(ldvs, ldvs);
    Complex2D rb(ldvs, ldvs);

    Complex2D mqkz(ldqkz, ldqkz);
    Complex2D invqkz(ldqkz, ldqkz);

    std::vector<double> rwork(3*ldvs);

    std::vector<int> ipivqkz(ldqkz);

    // a few flags...
    bool ok    = true;
    bool found = false;

    while (k_ < kmax_ && iterations_ < maxIterations_)
    {
        timerStart("jdqz main iteration");

        iterations_++;
        solveIterations++;
        if (j_ == 0 && reuseBasis_ && jprev > 0)
        {
            j_ = jprev - 1;
            for (int i = 0; i < j_; i++)
            {
                mat_.AMUL(work_[V_+i], work_[Av_+i]);
                mat_.BMUL(work_[V_+i], work_[Bv_+i]);

                makemm(n_, i+1, W_, Av_, ma, zma, ldvs);
                makemm(n_, i+1, W_, Bv_, mb, zmb, ldvs);

                gegs(i+1, zma, zmb, alpha, beta, vsl, vsr, zwork, rwork);
            }
        }
        else if (j_ == 0)
        {
            // Initialize search and test space with random real part
            // and zero imaginary part. Values uniform in [-1,1]. For
            // the test vector we use ones.
            work_[V_].random();
        }
        else
        {
            mxmv = maxnmv_;
            deps = pow(2.0, -solveIterations);
            if (j_ < jmin_)
            {
                mxmv = 1;
                gmres(n_, V_+j_, D_, m_, deps, mxmv, zalpha, zbeta, k_+1,
                      Kz_, Q_, invqkz, ldqkz, ipivqkz, f, U_, Tp_);
            }
            else if (method_ == GMRES)
            {
                mxmv = m_;
                gmres(n_, V_+j_, D_, m_, deps, mxmv, zalpha, zbeta, k_+1,
                      Kz_, Q_, invqkz, ldqkz, ipivqkz, f, U_, Tp_);
            }
            else if (method_ == CGSTAB)
            {
                // TODO
            }
        }
        j_++;

        mgs(n_, j_-1, V_, V_+j_-1, 1);
        mgs(n_, k_,   Q_, V_+j_-1, 1);

        if (testspace_ == 1)      // Standard Petrov
        {
            jdqzmv(V_+j_-1, W_+j_-1, Tp_,
                   -std::conj(shiftb), std::conj(shifta));
        }
        else if (testspace_ == 2) // Standard 'variable' Petrov
        {
            jdqzmv(V_+j_-1, W_+j_-1, Tp_,
                   -std::conj(zbeta), std::conj(zalpha));
        }
        else if (testspace_ == 3) // Harmonic Petrov
        {
            jdqzmv(V_+j_-1, W_+j_-1, Tp_,
                   shifta, shiftb);
        }

        mgs(n_, j_-1, W_, W_+j_-1, 1);
        mgs(n_, k_,   Z_, W_+j_-1, 1);

        mat_.AMUL(work_[V_+j_-1], work_[Av_+j_-1]);
        mat_.BMUL(work_[V_+j_-1], work_[Bv_+j_-1]);

        makemm(n_, j_, W_, Av_, ma, zma, ldvs);
        makemm(n_, j_, W_, Bv_, mb, zmb, ldvs);

        gegs(j_, zma, zmb, alpha, beta, vsl, vsr, zwork, rwork);

        bool attempt = true;
        while (attempt)
        {
            timerStart("jdqz attempt");
            // --- Sort the Petrov pairs ---
            qzsort(targeta, targetb, j_, zma, zmb,
                   vsl, vsr, ldvs, order_);

            zalpha = zma(0,0);
            zbeta  = zmb(0,0);

            evcond = sqrt(pow(abs(zalpha),2) + pow(abs(zbeta),2));

            // --- compute new q ---
            // Q(:,k) = V_ * VSR(:,0)
            gemv(n_, j_, 1.0, V_, &vsr(0,0), 0.0, Q_+k_);

            // --- orthogonalize new q ---
            mgs(n_, k_, Q_, Q_+k_, 1);

            // --- compute new z ---
            // Z(:,k) = W_ * VSL(:,0)
            gemv(n_, j_, 1.0, W_, &vsl(0,0), 0.0, Z_+k_);

            // --- orthogonalize new z ---
            mgs(n_, k_, Z_, Z_+k_, 1);

            // --- make new qkz ---
            work_[Kz_+k_] = work_[Z_+k_];
            mat_.PRECON(work_[Kz_+k_]);
            makeqkz(n_, k_+1, Q_, Kz_, mqkz, invqkz, ldqkz, ipivqkz);

            // --- compute new (right) residual= beta Aq - alpha Bq ---
            jdqzmv(Q_+k_, D_, Tp_, zalpha, zbeta);

            // --- orthogonalize this vector to Z ---
            mgs(n_, k_, Z_, D_, 0);

            rnrm = work_[D_].norm() / evcond.real();

            if (rnrm < lock_ && ok)
            {
                targeta = zalpha;
                targetb = zbeta;
                ok = false;
            }

            found   = (rnrm < eps_ && (j_ > 1 || k_ == kmax_ - 1));
            attempt = found;

            timerStop("jdqz attempt");

            if (verbosity_ > 4)
                WRITE("  iteration = " << std::setw(3) << iterations_ << "  j = "
                      << std::setw(3) << j_ << "  residual = " << rnrm);

            if (found)
            {
                if (verbosity_ > 4)
                    WRITE("\n-- found eigenvalue: k = " << k_ << '\n'
                          << "       alpha = " << zalpha << '\n'
                          << "        beta = " << zbeta << '\n');

                // --- increase the number of found evs ---
                k_++;

                // --- store the eigenvalue ---
                aconv[k_-1] = zalpha;
                bconv[k_-1] = zbeta;

                // --- reset solveIterations
                solveIterations = 0;

                if (k_ == kmax_) break; // break from this while loop

                timerStart("jdqz next ev");


                gemm(n_, j_-1, j_, 1.0, V_, &vsr(0,1), ldvs, 0.0, Aux_);

                // -- swap indices... --
                itmp = V_;
                V_   = Aux_;
                Aux_ = itmp;

                gemm(n_, j_-1, j_, 1.0, Av_, &vsr(0,1), ldvs, 0.0, Aux_);

                // -- swap indices... --
                itmp = Av_;
                Av_  = Aux_;
                Aux_ = itmp;

                gemm(n_, j_-1, j_, 1.0, Bv_, &vsr(0,1), ldvs, 0.0, Aux_);

                // -- swap indices... --
                itmp = Bv_;
                Bv_  = Aux_;
                Aux_ = itmp;

                gemm(n_, j_-1, j_, 1.0, W_, &vsl(0,1), ldvs, 0.0, Aux_);

                // -- swap indices... --
                itmp = W_;
                W_   = Aux_;
                Aux_ = itmp;

                j_ = j_-1;

                zlacpy_("A", &j_, &j_, &zma(1,1), &ldvs, &ma[0],  &ldvs);
                zlacpy_("A", &j_, &j_, &ma[0],    &ldvs, &zma[0], &ldvs);
                zlacpy_("A", &j_, &j_, &zmb(1,1), &ldvs, &mb[0],  &ldvs);
                zlacpy_("A", &j_, &j_, &mb[0],    &ldvs, &zmb[0], &ldvs);

                zlaset_("A", &j_, &j_, &zzero, &zone, &vsr[0], &ldvs);
                zlaset_("A", &j_, &j_, &zzero, &zone, &vsl[0], &ldvs);

                targeta = shifta;
                targetb = shiftb;

                ok = true;
                mxmv = 0;
                deps = 1.0;

                timerStop("jdqz next ev");
            }
            else if (j_ == jmax_)
            {
                timerStart("jdqz reduce j");
                gemm(n_, jmin_, j_, 1.0, V_, &vsr[0], ldvs, 0.0, Aux_);
                itmp  = V_;
                V_    = Aux_;
                Aux_  = itmp;

                gemm(n_, jmin_, j_, 1.0, Av_, &vsr[0], ldvs, 0.0, Aux_);
                itmp  = Av_;
                Av_   = Aux_;
                Aux_  = itmp;

                gemm(n_, jmin_, j_, 1.0, Bv_, &vsr[0], ldvs, 0.0, Aux_);
                itmp  = Bv_;
                Bv_   = Aux_;
                Aux_  = itmp;

                gemm(n_, jmin_, j_, 1.0, W_, &vsl[0], ldvs, 0.0, Aux_);

                itmp = W_;
                W_   = Aux_;
                Aux_ = itmp;
                j_    = jmin_;

                zlacpy_("A", &j_, &j_, &zma[0], &ldvs, &ma[0],  &ldvs);
                zlacpy_("A", &j_, &j_, &zmb[0], &ldvs, &mb[0],  &ldvs);

                zlaset_("A", &j_, &j_, &zzero, &zone, &vsr[0], &ldvs);
                zlaset_("A", &j_, &j_, &zzero, &zone, &vsl[0], &ldvs);
                timerStop("jdqz reduce j");
            }
        }
        timerStop("jdqz main iteration");
    }

    // --- Did enough eigenpairs converge? ---
    if (kmax_ != k_ && verbosity_ > 0)
        WRITE(" --- less eigenvalues have converged than requested --- ");

    if (iterations_ >= maxIterations_  && verbosity_ > 0)
        WRITE(" --- maximum number of iterations reached --- ");

    if (wanted_)
    {
        // --- Compute the Schur matrices if the eigenvectors are ---
        // ---  wanted, work_[Tp_] is used for temporary storage  ---
        // ---  Compute RA:                                       ---
        zlaset_("L", &k_, &k_, &zzero, &zzero, &ra[0], &ldvs);
        for (int i = 1; i <= k_; ++i) // 1-based !!
        {
            mat_.AMUL(work_[Q_+i-1], work_[Tp_]);
            gemv(n_, i, 1.0, Z_, Tp_, 0.0, &ra(0,i-1));
        }
        // --- Compute RB: ---
        zlaset_("L", &k_, &k_, &zzero, &zzero, &rb[0], &ldvs);
        for (int i = 1; i <= k_; ++i) // 1-based !!
        {
            mat_.BMUL(work_[Q_+i-1], work_[Tp_]);
            gemv(n_, i, 1.0, Z_, Tp_, 0.0, &rb(0,i-1));
        }

        // --- The eigenvectors RA and RB  belonging to the found eigenvalues
        // --- are computed. The Schur vectors in VR and VS are replaced by the
        // --- eigenvectors of RA and RB
        int INFO = 0;
        zggev_("N", "V", &k_, &ra[0], &ldvs, &rb[0], &ldvs,
               &alpha[0], &beta[0], &vsl[0], &ldvs, &vsr[0], &ldvs,
               &zwork[0], &ldzwork, &rwork[0], &INFO);

        // --- Compute the eigenvectors belonging to the found eigenvalues
        // --- of A and put them in EIVEC
        gemm(n_, k_, k_, 1.0, Q_, &vsr[0], ldvs, 0.0, eivec_);
    }
    else
    {
        // --- Store the Schurvectors in eivec
        for (int i = 0; i != k_; ++i)
        {
            eivec_[i] = work_[Q_+i];
            zcopy_(&k_, &aconv[0], &one, &alpha[0], &one);
            zcopy_(&k_, &bconv[0], &one, &beta[0], &one);
        }
    }
    alpha_ = alpha;
    beta_  = beta;

    timerStop("jdqz solve");
}

//==================================================================
template<typename Matrix>
void JDQZ<Matrix>::makeqkz(int n, int k, int Q, int Kq, Complex2D &qkz,
                           Complex2D &invqkz, int ldqkz,
                           std::vector<int> &ipiv)
{
    timerStart("jdqz makeqkz");
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
    timerStop("jdqz makeqkz");
}

//==================================================================
// Y = alpha*A*X + beta*Y
template<typename Matrix>
void JDQZ<Matrix>::gemv(int m, int n, complex alpha, int A,
                        complex *X, complex beta, int Y)
{
    timerStart("jdqz gemv");

    work_[Y].scale(beta);
    for (int i = 0; i != n; ++i)
        work_[Y].axpy(alpha * X[i], work_[A+i]);

    timerStop("jdqz gemv");
}

//==================================================================
// Y = alpha*A**H*X + beta*Y
template<typename Matrix>
void JDQZ<Matrix>::gemv(int m, int n, complex alpha, int A,
                        int X, complex beta, Complex1D &Y)
{
    timerStart("jdqz gemv");
    Y.scale(beta);
    for (int i = 0; i != n; ++i)
        Y[i] += alpha * work_[A+i].dot(work_[X]);
    timerStop("jdqz gemv");
}

//==================================================================
// Y = alpha*A**H*X + beta*Y
template<typename Matrix>
void JDQZ<Matrix>::gemv(int m, int n, complex alpha, int A,
                        int X, complex beta, complex *Y)
{
    timerStart("jdqz gemv");
    for (int i = 0; i != n; ++i)
    {
        Y[i] *= beta;
        Y[i] += alpha * work_[A+i].dot(work_[X]);
    }
    timerStop("jdqz gemv");
}

//==================================================================
// C = alpha*A*B + beta*C
template<typename Matrix>
void JDQZ<Matrix>::gemm(int m, int n, int k, complex alpha, int A,
                        complex *B, int ldb, complex beta, int C)
{
    timerStart("jdqz gemm");
    for (int j = 0; j != n; ++j)
    {
        work_[C+j].scale(beta);
        for (int i = 0; i != k; ++i)
            work_[C+j].axpy(alpha * B[i+ldb*j], work_[A+i]);
    }
    timerStop("jdqz gemm");
}

//==================================================================
// C = alpha*A*B + beta*C
template<typename Matrix>
void JDQZ<Matrix>::gemm(int m, int n, int k, complex alpha, int A,
                        complex *B, int ldb, complex beta,
                        std::vector<Vector> &C)
{
    timerStart("jdqz gemm");
    assert((int) C.size() >= n);
    for (int j = 0; j != n; ++j)
    {
        C[j].scale(beta);
        for (int i = 0; i != k; ++i)
            C[j].axpy(alpha * B[i+ldb*j], work_[A+i]);
    }
    timerStop("jdqz gemm");
}

int select(int start, int n, complex ta, complex tb,
           Complex2D &s, Complex2D &t, int order)
{
    std::vector<int> idx(n - start);
    for (int j = 0; j < n - start; ++j)
        idx[j] = j + start;

    switch (order)
    {
    case 0:
        // Nearest to target
        return *std::min_element(
            idx.begin(), idx.end(),
            [s, t, ta, tb](int i1, int i2) {
                return std::abs(s(i1, i1) / t(i1, i1) - ta / tb) < std::abs(s(i2, i2) / t(i2, i2) - ta / tb);
            });
    case -1:
        // Smallest real part
        return *std::min_element(
            idx.begin(), idx.end(),
            [s, t](int i1, int i2){
                return (t(i1, i1).real() == 0.0 and s(i1, i1).real() < 0.0) or
                    (s(i1, i1).real() / t(i1, i1).real() < s(i2, i2).real() / t(i2, i2).real());
            });
    case 1:
        // Largest real part
        return *std::min_element(
            idx.begin(), idx.end(),
            [s, t](int i1, int i2){
                return (t(i1, i1).real() == 0.0 and s(i1, i1).real() > 0.0) or
                    (s(i1, i1).real() / t(i1, i1).real() > s(i2, i2).real() / t(i2, i2).real());
            });
    case -2:
        // Smallest imaginary part
        return *std::min_element(
            idx.begin(), idx.end(),
            [s, t](int i1, int i2){
                return (t(i1, i1).imag() == 0.0 and s(i1, i1).imag() < 0.0) or
                    (s(i1, i1).imag() / t(i1, i1).imag() < s(i2, i2).imag() / t(i2, i2).imag());
            });
    case 2:
        // Largest imaginary part
        return *std::min_element(
            idx.begin(), idx.end(),
            [s, t](int i1, int i2){
                return (t(i1, i1).imag() == 0.0 and s(i1, i1).imag() > 0.0) or
                    (s(i1, i1).imag() / t(i1, i1).imag() > s(i2, i2).imag() / t(i2, i2).imag());
            });
    }

    return -1;
}

//==================================================================
template<typename Matrix>
void JDQZ<Matrix>::qzsort(complex ta, complex tb, int k,
                          Complex2D &s, Complex2D &t, Complex2D &z,
                          Complex2D &q, int ldz, int order)
{
    timerStart("jdqz qzsort");

    int wantq = 1;
    int wantz = 1;
    int n = k;
    int lds = ldz;
    int ldt = ldz;
    int ldq = ldz;
    int info;

    for (int i = 0; i < n; ++i)
    {
        int ifst = select(i, n, ta, tb, s, t, order) + 1;
        int ilst = i + 1;

        if (ifst == ilst)
            continue;

        ztgexc_(&wantq, &wantz, &n, &s[0], &lds, &t[0], &ldt,
                &z[0], &ldz, &q[0], &ldq, &ifst, &ilst, &info);

        assert(info == 0);
    }

    timerStop("jdqz qzsort");
}

//==================================================================
template<typename Matrix>
void JDQZ<Matrix>::gmres(int n, int x, int r, int mxm, double &eps, int &mxmv,
                         complex alpha, complex beta, int k, int kz, int q,
                         Complex2D &invqkz, int ldqkz, std::vector<int> &ipiv,
                         Complex1D &f, int v, int tp)
{
    // -- some initializations ---
    timerStart("jdqz gmres");

    int maxm = std::max(100, mxm + 1);
    int nmv  = 0;
    int m, m1;
    int one  = 1;

    complex ztmp;
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
        while (nmv < mxmv && m < mxm-1 && rnrm > eps1)
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
        }

        // --- compute approximate solution x ---
        int M = m+1; // move to 1-based for lapack and gemv
        zcopy_(&M, &rs[0], &one, &y[0], &one);
        ztrsv_("U", "N", "N", &M, &hh[0], &maxm, &y[0], &one);

        gemv(n, M, 1.0, v, &y[0], 1.0, x);

        //--- compute residual for restart ---
        jdqzmv(x, v+1, tp, alpha, beta);
        psolve(n, v+1, k, q, kz, invqkz, ldqkz, ipiv, f);
        work_[v] = work_[r];
        work_[v].axpy(-1.0, work_[v+1]);

        rnrm = work_[v].norm();

        if (verbosity_ > 6)
            WRITE("  jdqz->gmres: rnrm = " << rnrm << " nmv = " << nmv)
                }
    // --- return ---
    eps  = rnrm / rnrm0;
    mxmv = nmv;

    timerStop("jdqz gmres");
}

//==================================================================
template<typename Matrix>
void JDQZ<Matrix>::psolve(int n, int x, int nq, int q, int kz,
                          Complex2D &invqkz, int ldqkz,
                          std::vector<int> ipiv, Complex1D &f)
{
    timerStart("jdqz psolve");
    mat_.PRECON(work_[x]);

    gemv(n, nq, 1.0, q, x, 0.0, f);

    int  nrhs = 1;
    int  info = 0;
    zgetrs_("N", &nq, &nrhs, &invqkz[0], &ldqkz,
            &ipiv[0], &f[0], &ldqkz, &info);

    gemv(n, nq, -1.0, kz, &f[0], 1.0, x);
    timerStop("jdqz psolve");
}

//==================================================================
template<typename Matrix>
void JDQZ<Matrix>::mgs(int n, int k, int v, int w, int job)
{
    timerStart("jdqz mgs");

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

    timerStop("jdqz mgs");
}

//==================================================================
template<typename Matrix>
void JDQZ<Matrix>::ortho(int n, int v, int w,
                         double &s0, double &s1,
                         std::complex<double> &nrm)
{
    timerStart("jdqz ortho");

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

        if (s1 < s0 / kappa  && verbosity_ > 0)
            WRITE("WARNING: zero vector in mgs...")
                }

    timerStop("jdqz ortho");
}

//==================================================================
template<typename Matrix>
void JDQZ<Matrix>::jdqzmv(int x, int y, int tmp,
                          std::complex<double> alpha,
                          std::complex<double> beta)
{
    timerStart("jdqz jdqzmv");

    // y = beta * A * x - alpha * B * x
    mat_.AMUL(work_[x], work_[tmp]);
    mat_.BMUL(work_[x], work_[y]);
    work_[y].axpby(beta, work_[tmp], -alpha);

    timerStop("jdqz jdqzmv");
}

//==================================================================
template<typename Matrix>
void JDQZ<Matrix>::makemm(int n, int k, int w, int v,
                          Complex2D &m, Complex2D &zm, int ldm)
{
    timerStart("jdqz makemm");

    for (int i = 0; i <= k-1; ++i)
        for (int j = 0; j <= k-1; ++j)
        {
            if (i == k-1 || j == k-1)
                m(i,j) = work_[w+i].dot(work_[v+j]);
            zm(i,j) = m(i,j);

            assert(std::isnan(zm(i,j).real()) == false);
            assert(std::isnan(zm(i,j).imag()) == false);
        }

    timerStop("jdqz makemm");
}

//==================================================================
template<typename Matrix>
void JDQZ<Matrix>::gegs(int N, Complex2D &A, Complex2D &B,
                        Complex1D &alpha, Complex1D &beta,
                        Complex2D &VSL, Complex2D &VSR,
                        Complex1D &work, std::vector<double> &rwork)
{
    timerStart("jdqz gegs");

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
    assert((int) rwork.size() >= std::max(1, 3*N));

    char JOBVSL = 'V'; // compute left  Schur vectors
    char JOBVSR = 'V'; // compute right Schur vectors

    char SORT   = 'N'; // do not sort eigenvalues
    int  SELCTG = 0;   // dummy eigenvalue selection
    int  SDIM   = 0;   // dummy sort argument
    int  BWORK  = 0;   // dummy sort work array

    // calling lapack
    zgges_(&JOBVSL, &JOBVSR,
           &SORT, &SELCTG, &N,
           &A[0], &LDA, &B[0], &LDB,
           &SDIM,
           &alpha[0], &beta[0],
           &VSL[0], &LDVSL,
           &VSR[0], &LDVSR,
           &work[0], &LWORK,
           &rwork[0], &BWORK, &INFO);

    assert(INFO == 0);

    timerStop("jdqz gegs");
}

//==================================================================
template<typename Matrix>
void JDQZ<Matrix>::setIndices()
{
    // These pointers refer to the columns of the workspace

    D_   = 0; 	    // Storage for rhs,
    Tp_  = D_ + 1;  // Workspace for jdqzmv
    U_   = Tp_ + 1; // Krylov space for GMRES(m) or Bi-CSTAB(l)

    // Search space JDQZ with max dimension jmax
    if (method_ == GMRES)
        V_ = U_ + m_ + 1;
    else if (method_ == CGSTAB)
        V_ = U_ + 2*l_ + 6;

    W_   = V_   + jmax_;  // Test space JDQZ with max dimension jmax
    Av_  = W_   + jmax_;  // Subspace AV with max dimension jmax
    Bv_  = Av_  + jmax_;  // Subspace BV with max dimension jmax
    Aux_ = Bv_  + jmax_;  // Auxiliary space for GEMM mults
    Q_   = Aux_ + jmax_;  // Search Schur basis in JDQZ, max dim kmax
    Z_   = Q_   + kmax_;  // Test Schur basis in JDQZ, max dim kmax
    Kz_  = Z_   + kmax_;  // Matrix K^{-1}Z_k
}

//==================================================================
template<typename Matrix>
void JDQZ<Matrix>::timerStart(std::string const &msg)
{
    if (!profile_) return;

    JDQZTimer timer(msg);
    timer.resetStartTime();
    if (profileData_.find(msg) == profileData_.end())
        profileData_[msg] = std::array<double, PROFILE_ENTRIES>();
    timerStack_.push(timer);
}

//==================================================================
template<typename Matrix>
void JDQZ<Matrix>::timerStop(std::string const &msg)
{
    if (!profile_) return;

    double time = timerStack_.top().elapsedTime();

    bool sane = true;;
    if (!(msg == timerStack_.top().label()))
    {
        WRITE("Warning: msg and label not equal!");
        WRITE("   msg = " << msg);
        WRITE(" label = " << timerStack_.top().label());
        sane = false;
    }
    assert(sane);

    timerStack_.pop();
    profileData_[msg][0] += time;
    profileData_[msg][1] += 1;
    profileData_[msg][2] = profileData_[msg][0] / profileData_[msg][1];
}

//==================================================================
template<typename Matrix>
template<typename PList>
void JDQZ<Matrix>::getDefaultParameters(PList &params)
{
    params.set("Shift (real part)", 0.0);
    params.set("Shift (imaginary part)", 0.0);
    params.set("Tolerance", 1e-9);
    params.set("Number of eigenvalues", 5);
    params.set("Max size search space", 20);
    params.set("Min size search space", 10);
    params.set("Max JD iterations", 1000);
    params.set("Tracking parameter", 1e-9);
    params.set("Criterion for Ritz values", 0);
    params.set("Linear solver", 1);
    params.set("GMRES search space", 30);
    params.set("Bi-CGstab polynomial degree", 2);
    params.set("Max mat-vec mults", 100);
    params.set("Testspace expansion", 3);
    params.set("Compute converged eigenvectors", true);
    params.set("Verbosity", 1);
    params.set("Profile", false);
    params.set("Reuse basis", false);
}

//==================================================================
template<typename Matrix>
template<typename PList>
void JDQZ<Matrix>::setParameters(PList &params)
{
    double shiftRe = params.get("Shift (real part)", 0.0);
    double shiftIm = params.get("Shift (imaginary part)", 0.0);
    shift_         = std::complex<double>(shiftRe, shiftIm);

    eps_           = params.get("Tolerance", 1e-9);
    kmax_          = params.get("Number of eigenvalues", 5);
    jmax_          = params.get("Max size search space", 20);
    jmin_          = params.get("Min size search space", 10);
    maxIterations_ = params.get("Max JD iterations", 1000);
    lock_          = params.get("Tracking parameter", 1e-9);
    order_         = params.get("Criterion for Ritz values", 0);
    method_        = params.get("Linear solver", 1);
    m_             = params.get("GMRES search space", 30);
    l_             = params.get("Bi-CGstab polynomial degree", 2);
    maxnmv_        = params.get("Max mat-vec mults", 100);
    testspace_     = params.get("Testspace expansion", 3);
    wanted_        = params.get("Compute converged eigenvectors", true);
    verbosity_     = params.get("Verbosity", 1);
    profile_       = params.get("Profile", false);
    reuseBasis_    = params.get("Reuse basis", false);

    if (method_ == 1)
        lwork_ =  4 +  m_  + 5*jmax_ + 3*kmax_;
    else
        lwork_ = 10 + 6*l_ + 5*jmax_ + 3*kmax_;

    // now we can setup the solver
    setup();
}

//==================================================================
template<typename Matrix>
void JDQZ<Matrix>::printParameters()
{
    WRITE("\nJDQZ Parameters:   ");
    WRITE(" | n_             " << n_ );
    WRITE(" | shift_         " << shift_ );
    WRITE(" | eps_           " << eps_ );
    WRITE(" | kmax_          " << kmax_ );
    WRITE(" | jmax_          " << jmax_ );
    WRITE(" | jmin_          " << jmin_ );
    WRITE(" | maxIterations_ " << maxIterations_ );
    WRITE(" | lock_          " << lock_ );
    WRITE(" | order_         " << order_ );
    WRITE(" | method_        " << method_ );
    WRITE(" | m_             " << m_ );
    WRITE(" | l_             " << l_ );
    WRITE(" | maxnmv_        " << maxnmv_ );
    WRITE(" | testspace_     " << testspace_ );
    WRITE(" | wanted_        " << wanted_ );
    WRITE(" | lwork_         " << lwork_ << '\n' );
}

//==================================================================
template<typename Matrix>
void JDQZ<Matrix>::printProfile(std::string const &filename)
{
    if (!profile_) return;

    // check if timerstack is sane
    bool sane = true;
    while (!timerStack_.empty())
    {
        WRITE("Warning: timerStack_ not empty!");
        timerStack_.top().printLabel();
        timerStack_.pop();
        sane = false;
    }
    assert(sane);


    // setting up a filename
    std::ostringstream profilefile(filename);
    // setup output file
    std::ofstream file(profilefile.str().c_str());

    // Set format flags
    file << std::left;

    // Define line format
#ifndef LINE
# define LINE(s1, s2, s3, s4, s5, s6, s7, s8, s9)               \
    {                                                           \
        int sp = 3;  int it = 5;  int id = 5;                   \
        int db = 12; int st = 45;                               \
        file << std::setw(id) << s1     << std::setw(sp) << s2  \
             << std::setw(st) << s3 << std::setw(sp) << s4      \
             << std::setw(db) << s5     << std::setw(sp) << s6  \
             << std::setw(it) << s7     << std::setw(sp) << s8  \
             << std::setw(db) << s9     << std::endl;           \
    }
#endif

    // Header
    LINE("", "", "", "", "cumul.", "", "calls", "", "average");

    // Display timings of the separate models, summing
    int counter = 0;
    for (auto const &map : profileData_)
        if (map.first.compare(0,8,"_NOTIME_") != 0)
        {
            counter++;
            std::stringstream s;
            s << " (" << counter << ")";
            LINE(s.str(), "", map.first, ":", map.second[0], "",
                 map.second[1], "", map.second[2]);
        }

    // Newline
    file << std::endl;

    // Display iteration information
    for (auto const &map : profileData_)
        if (map.first.compare(0,8,"_NOTIME_") == 0 )
        {
            counter++;
            std::stringstream s;
            s << " (" << counter << ")";
            LINE(s.str(), "", map.first.substr(8), ":", map.second[0], "",
                 map.second[1], "", map.second[2]);
        }
}

#endif
