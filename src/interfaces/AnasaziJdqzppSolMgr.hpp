#ifndef ANASAZI_JDQZPP_SOLMGR_HPP
#define ANASAZI_JDQZPP_SOLMGR_HPP

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCPDecl.hpp"

#include "AnasaziConfigDefs.hpp"
#include "AnasaziTypes.hpp"
#include "AnasaziMultiVecTraits.hpp"
#include "AnasaziEigenproblem.hpp"
#include "AnasaziSolverManager.hpp"

#include "Epetra_Operator.h"
#include "Epetra_MultiVector.h"

#include "Ifpack_Preconditioner.h"

#include "jdqz.H"

#include "EpetraComplexVector.hpp"

#include <algorithm>

namespace Anasazi {

/*!
 * \class JdqzppSolMgr
 * \brief Solver Manager for Jacobi Davidson in phist
 *
 * This class provides a simple interface to the Jacobi Davidson
 * eigensolver.  This manager creates
 * appropriate managers based on user
 * specified ParameterList entries (or selects suitable defaults),
 * provides access to solver functionality, and manages the restarting
 * process.
 *
 * This class is currently only implemented for scalar type double

 \ingroup anasazi_solver_framework

*/
template <class ScalarType, class MV, class OP, class PREC>
class JdqzppSolMgr : public SolverManager<ScalarType,MV,OP>
{
    using ComplexVectorType = EpetraComplexVector<Epetra_MultiVector>;

private:
    typedef MultiVecTraits<ScalarType,MV>        MVT;
    typedef OperatorTraits<ScalarType,MV,OP>     OPT;
    typedef Teuchos::ScalarTraits<ScalarType>    ST;
    typedef typename ST::magnitudeType           MagnitudeType;
    typedef Teuchos::ScalarTraits<MagnitudeType> MT;

    class JdqzppOperator
    {
    public:
        using Vector = ComplexVectorType;

    protected:
        Teuchos::RCP<const OP> A_;
        Teuchos::RCP<const OP> B_;
        Teuchos::RCP<const PREC> M_;

    public:
        //! constructor
        JdqzppOperator(const Teuchos::RCP<Eigenproblem<ScalarType,MV,OP> > &problem,
                       const Teuchos::RCP<PREC> &prec)
            :
            M_(prec)
            {
                A_ = problem->getA();
                if (A_ == Teuchos::null)
                    A_ = problem->getOperator();
                B_ = problem->getM();
                TEUCHOS_TEST_FOR_EXCEPTION(A_ == Teuchos::null || B_ == Teuchos::null,
                                           std::invalid_argument,
                                           "One of the operators is not set!");
            }

        // Subroutine to compute r = Aq
        void AMUL(Vector const &q, Vector &r)
            {
                OPT::Apply(*A_, q.real(), r.real());
                OPT::Apply(*A_, q.imag(), r.imag());
            }

        // Subroutine to compute r = Bq
        void BMUL(Vector const &q, Vector &r)
            {
                OPT::Apply(*B_, q.real(), r.real());
                OPT::Apply(*B_, q.imag(), r.imag());
            }

        // Subroutine to compute q = K^-1 q
        void PRECON(Vector &q)
            {
                Vector tmp(q);
                M_->ApplyInverse(q.real(), tmp.real());
                M_->ApplyInverse(q.imag(), tmp.imag());
                q = tmp;
            }
    };

public:

    /*!
     * \brief Basic constructor for JdqzppSolMgr
     *
     * This constructor accepts the Eigenproblem to be solved and a parameter list of options
     * for the solver.
     * The following options control the behavior
     * of the solver:
     * - "Which" -- a string specifying the desired eigenvalues: SR, LR, SI, or LI. Default: "target"
     * - "Maximum Subspace Dimension" -- maximum number of basis vectors for subspace.  Two
     *  (for standard eigenvalue problems) or three (for generalized eigenvalue problems) sets of basis
     *  vectors of this size will be required. Default: 3*problem->getNEV()*"Block Size"
     * - "Restart Dimension" -- Number of vectors retained after a restart.  Default: NEV
     * - "Maximum Restarts" -- an int specifying the maximum number of restarts the underlying solver
     *  is allowed to perform.  Default: 20
     * - "Verbosity" -- a sum of MsgType specifying the verbosity.  Default: AnasaziWarnings
     * - "Convergence Tolerance" -- a MagnitudeType specifying the level that residual norms must
     *  reach to decide convergence.  Default: machine precision
     * - "Inner Iterations" - maximum number of inner GMRES or MINRES iterations allowed
     *   If "User," the value in problem->getInitVec() will be used.  Default: "Random".
     * - "Print Number of Ritz Values" -- an int specifying how many Ritz values should be printed
     *   at each iteration.  Default: "NEV".
     */
    JdqzppSolMgr(const Teuchos::RCP< Eigenproblem<ScalarType,MV,OP> > &problem,
                 const Teuchos::RCP<PREC> &prec,
                 Teuchos::ParameterList &pl);

    //! destructor
    ~JdqzppSolMgr();

    /*!
     * \brief Return the eigenvalue problem.
     */
    const Eigenproblem<ScalarType,MV,OP> & getProblem() const { return *d_problem; }

    /*!
     * \brief Get the iteration count for the most recent call to solve()
     */
    int getNumIters() const { return numIters_; }

    /*!
     * \brief This method performs possibly repeated calls to the underlying eigensolver's iterate()
     *  routine until the problem has been solved (as decided by the StatusTest) or the solver manager decides to quit.
     */
    ReturnType solve();

protected:
    Teuchos::RCP< Eigenproblem<ScalarType,MV,OP> > d_problem;
    Teuchos::RCP<JdqzppOperator> d_operator;
    Teuchos::RCP<JDQZ<JdqzppOperator> > d_solver;

    int numIters_; //! number of iterations performed in previous solve()

    MagnitudeType tol_;

}; // class JdqzppSolMgr

//---------------------------------------------------------------------------//
// Prevent instantiation on complex scalar type
//---------------------------------------------------------------------------//
template <class MagnitudeType, class MV, class OP, class PREC>
class JdqzppSolMgr<std::complex<MagnitudeType>,MV,OP,PREC>
{
public:

    typedef std::complex<MagnitudeType> ScalarType;
    JdqzppSolMgr(
        const Teuchos::RCP<Eigenproblem<ScalarType,MV,OP> > &problem,
        Teuchos::ParameterList &pl )
        {
            // Provide a compile error when attempting to instantiate on complex type
            MagnitudeType::this_class_is_missing_a_specialization();
        }
};

//---------------------------------------------------------------------------//
// Start member definitions
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
// Constructor
//---------------------------------------------------------------------------//
template <class ScalarType, class MV, class OP, class PREC>
JdqzppSolMgr<ScalarType,MV,OP,PREC>::JdqzppSolMgr(
    const Teuchos::RCP<Eigenproblem<ScalarType,MV,OP> > &problem,
    const Teuchos::RCP<PREC> &prec,
    Teuchos::ParameterList &pl )
    :
    d_problem(problem),
    d_operator(Teuchos::rcp(new JdqzppOperator(problem, prec))),
    numIters_(0)
{
    TEUCHOS_TEST_FOR_EXCEPTION(d_problem == Teuchos::null,
                               std::invalid_argument,
                               "Problem not given to solver manager." );
    TEUCHOS_TEST_FOR_EXCEPTION(!d_problem->isProblemSet(),
                               std::invalid_argument,
                               "Problem not set." );
    TEUCHOS_TEST_FOR_EXCEPTION(d_problem->getA() == Teuchos::null &&
                               d_problem->getOperator() == Teuchos::null,
                               std::invalid_argument,
                               "A operator not supplied on Eigenproblem." );
    TEUCHOS_TEST_FOR_EXCEPTION(d_problem->getInitVec() == Teuchos::null,
                               std::invalid_argument,
                               "No vector to clone from on Eigenproblem." );
    TEUCHOS_TEST_FOR_EXCEPTION(d_problem->getNEV() <= 0,
                               std::invalid_argument,
                               "Number of requested eigenvalues must be positive.");

    Teuchos::RCP<Teuchos::ParameterList> params = Teuchos::rcp(new Teuchos::ParameterList());

    tol_ = pl.get<MagnitudeType>("Convergence Tolerance", MT::eps());
    TEUCHOS_TEST_FOR_EXCEPTION(tol_ <= MT::zero(),
                               std::invalid_argument,
                               "Convergence Tolerance must be greater than zero.");

    params->set("Tolerance", tol_);

    if (pl.isType<Anasazi::MsgType>("Verbosity"))
    {
        Anasazi::MsgType verbosity = pl.get<Anasazi::MsgType>("Verbosity", Anasazi::Warnings);
        int jdqzVerbosity = verbosity;
        if (verbosity & Anasazi::IterationDetails)
            jdqzVerbosity = 5;
        if (verbosity & Anasazi::OrthoDetails)
            jdqzVerbosity = 10;

        params->set("Verbosity", jdqzVerbosity);

        params->set("Profile", verbosity & Anasazi::TimingDetails);
    }
    else
    {
        int verbosity = pl.get<int>("Verbosity", 1);
        params->set("Verbosity", verbosity);
    }

    int nev = d_problem->getNEV();
    params->set("Number of eigenvalues", nev);

    int max_dim = pl.get<int>("Maximum Subspace Dimension", 3 * nev);
    params->set("Max size search space", max_dim);

    int min_dim = pl.get<int>("Restart Dimension", 2 * nev);
    params->set("Min size search space", min_dim);

    int maxit_inner = pl.get<int>("Inner Iterations", 100);
    params->set("Max mat-vec mults", maxit_inner);

    int inner_dim = pl.get<int>("Num Blocks", 3 * nev);
    params->set("GMRES search space", inner_dim);

    if(pl.isType<int>("Maximum Restarts"))
    {
        int maxit_outer = (max_dim - min_dim) * pl.get<int>("Maximum Restarts") + max_dim;
        TEUCHOS_TEST_FOR_EXCEPTION(maxit_outer < 0,
                                   std::invalid_argument,
                                   "Maximum Restarts must be non-negative");
        params->set("Max JD iterations", maxit_outer);
    }

    // Get sort type
    std::string which;
    if(pl.isType<std::string>("Which"))
    {
        which = pl.get<std::string>("Which");
        TEUCHOS_TEST_FOR_EXCEPTION(which!="LI" && which!="SI" && which!="LR" && which!="SR",
                                   std::invalid_argument,
                                   "Which must be one of LI, SI, LR, SR.");
        if (which == "LI")
            params->set("Criterion for Ritz values", 2);
        else if (which == "SI")
            params->set("Criterion for Ritz values", -2);
        else if (which == "LR")
            params->set("Criterion for Ritz values", 1);
        else if (which == "SR")
            params->set("Criterion for Ritz values", -1);
    }

    ComplexVectorType initVec(*d_problem->getInitVec());
    d_solver = Teuchos::rcp(new JDQZ<JdqzppOperator>(*d_operator, initVec));
    d_solver->setParameters(*params);
}

template <class ScalarType, class MV, class OP, class PREC>
JdqzppSolMgr<ScalarType,MV,OP,PREC>::~JdqzppSolMgr()
{}

template <class ScalarType>
bool eigSort(Anasazi::Value<ScalarType> const &a, Anasazi::Value<ScalarType> const &b)
{
  return (a.realpart * a.realpart + a.imagpart * a.imagpart) <
         (b.realpart * b.realpart + b.imagpart * b.imagpart);
}

//---------------------------------------------------------------------------//
// Solve
//---------------------------------------------------------------------------//
template <class ScalarType, class MV, class OP, class PREC>
ReturnType JdqzppSolMgr<ScalarType,MV,OP,PREC>::solve()
{
    d_solver->solve();

    Eigensolution<ScalarType,MV> sol;

    int kmax = d_solver->kmax();
    std::vector<ComplexVectorType> evecs = d_solver->getEigenVectors();
    std::vector<std::complex<double> > alpha = d_solver->getAlpha();
    std::vector<std::complex<double> > beta = d_solver->getBeta();
    std::vector<std::complex<double> > evals(kmax);
    std::transform(alpha.begin(), alpha.begin() + kmax,
                   beta.begin(), evals.begin(),
                   std::divides<std::complex<double> >());

    // Now sort the solution so we can store it in MultiVectors
    std::vector<int> indices(kmax);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [evals](const int &a, const int &b) {
                  return evals[a].real() > evals[b].real(); });

    // Now filter out complex conjugate pairs
    std::vector<std::complex<double> > evals2;
    std::vector<ComplexVectorType *> evecs2;
    bool first = true;
    sol.index.resize(0);
    for (int i = 0; i < kmax; i++)
    {
        if (std::abs(evals[indices[i]].imag()) > tol_)
        {
            if (!first && std::abs(
                    evals[indices[i]].imag() + evals[indices[i-1]].imag()) < tol_)
            {
                first = true;
                continue;
            }
            first = false;
            sol.index.push_back(1);
            sol.index.push_back(-1);
        }
        else
        {
            first = true;
            sol.index.push_back(0);
        }
        evals2.push_back(evals[indices[i]]);
        evecs2.push_back(&evecs[indices[i]]);
    }

    sol.numVecs = sol.index.size();
    sol.Evals.resize(sol.numVecs);

    if (sol.numVecs)
    {
        sol.Evecs = MVT::Clone(*d_problem->getInitVec(), sol.numVecs);
        int len = evals2.size();
        int curind = 0;
        std::vector<int> curindvec(1, 0);
        for (int i = 0; i < len; i++)
        {
            std::complex<double> ev = evals2[i];
            sol.Evals[curind].realpart = ev.real();
            sol.Evals[curind].imagpart = ev.imag();

            Teuchos::RCP<MV> eview = MVT::CloneViewNonConst(*sol.Evecs, curindvec);
            MVT::Assign(evecs2[i]->real(), *eview);

            if (sol.index[curind] != 0)
            {
                curindvec[0] = ++curind;

                sol.Evals[curind].realpart = ev.real();
                sol.Evals[curind].imagpart = -ev.imag();

                eview = MVT::CloneViewNonConst(*sol.Evecs, curindvec);
                MVT::Assign(evecs2[i]->imag(), *eview);
            }
            curindvec[0] = ++curind;
        }
    }
    d_problem->setSolution(sol);

    // Return convergence status
    if (sol.numVecs < d_problem->getNEV())
        return Unconverged;

    return Converged;
}

} // namespace Anasazi

#endif // ANASAZI_JDQZPP_SOLMGR_HPP
