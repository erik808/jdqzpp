#include "gtest/gtest.h" // google test framework

#include <vector>
#include <complex>
#include <iostream>
#include <assert.h>

#include "jdqz.H"

#include "Epetra_SerialComm.h"
#include "Epetra_Map.h"
#include "Epetra_Vector.h"
#include "Epetra_CrsMatrix.h"

#include "AnasaziBasicEigenproblem.hpp"
#include "AnasaziEpetraAdapter.hpp"

#include "EpetraComplexVector.hpp"
#include "AnasaziJdqzppSolMgr.hpp"

#include "test.H"

std::vector<double>  TestVector::norms = std::vector<double>();
std::vector<complex> TestVector::alphas = std::vector<complex>();
std::vector<complex> TestVector::betas = std::vector<complex>();
std::vector<complex> TestVector::dotresults = std::vector<complex>();

//------------------------------------------------------------------
TEST(JDQZ, Anasazi)
{
    int n = 50;
    Teuchos::RCP<Epetra_SerialComm> comm = Teuchos::rcp(new Epetra_SerialComm);
    Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(new Epetra_Map(n, 0, *comm));

    Teuchos::RCP<Epetra_CrsMatrix> A = Teuchos::rcp(new Epetra_CrsMatrix(Copy, *map, 3));

    for (int i = 0; i < n; i++)
    {
        int A_idx[3] = {i-2, i, i+1};
        double A_val[3] = {1.0, 2.0, 1.0};
        if (i < 2)
            A->InsertGlobalValues(i, 2, A_val+1, A_idx+1);
        else if (i > n-2)
            A->InsertGlobalValues(i, 2, A_val, A_idx);
        else
            A->InsertGlobalValues(i, 3, A_val, A_idx);
    }
    A->FillComplete();

    Teuchos::RCP<Epetra_CrsMatrix> B = Teuchos::rcp(new Epetra_CrsMatrix(Copy, *map, 1));

    for (int i = 0; i < n; i++)
    {
        int B_idx[1] = {i};
        double B_val[1] = {1};
        B->InsertGlobalValues(i, 1, B_val, B_idx);
    }
    B->FillComplete();

    Teuchos::RCP<Epetra_Vector> x = Teuchos::rcp(new Epetra_Vector(*map));

    Teuchos::ParameterList params;
    params.set("Convergence Tolerance", 1e-9);

    Teuchos::RCP<Anasazi::BasicEigenproblem<
        double, Epetra_MultiVector, Epetra_Operator> > problem
        = Teuchos::rcp(new Anasazi::BasicEigenproblem<
                       double, Epetra_MultiVector, Epetra_Operator>());
    problem->setA(A);
    problem->setM(B);
    problem->setInitVec(x);
    problem->setNEV(10);

    int ret = problem->setProblem();
    EXPECT_EQ(ret, true);

    Anasazi::JdqzppSolMgr<double, Epetra_MultiVector,
                          Epetra_Operator, Epetra_Operator> jdqz(
                              problem, B, params);

    Anasazi::ReturnType returnCode = jdqz.solve();
    EXPECT_EQ(returnCode, Anasazi::Converged);

    const Anasazi::Eigensolution<double, Epetra_MultiVector>& sol =
        problem->getSolution();

    EXPECT_GE(sol.numVecs, 10);

    for (int i = 0; i < 10; i++)
    {
        Epetra_Vector rhs(*(*sol.Evecs)(i));
        if (sol.index[i] == 1)
            rhs.Update(-sol.Evals[i].imagpart, *(*sol.Evecs)(i+1), sol.Evals[i].realpart);
        else if(sol.index[i] == -1)
            rhs.Update(-sol.Evals[i].imagpart, *(*sol.Evecs)(i-1), sol.Evals[i].realpart);
        else
            rhs.Scale(sol.Evals[i].realpart);

        Epetra_Vector out(*(*sol.Evecs)(i));
        double result = -1;
        out.Norm2(&result);
        EXPECT_GT(result, 1e-9);

        A->Apply(*(*sol.Evecs)(i), out);
        out.Update(1.0, rhs, -1.0);
        out.Norm2(&result);
        EXPECT_LT(result, 1e-9);
    }
}
