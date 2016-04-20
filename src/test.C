#include "gtest/gtest.h" // google test framework
#include "jdqz.H"
#include <vector>
#include <complex>
#include <iostream>
#include <assert.h>



class TestVector : public std::vector<std::complex<double> >
{
public:

	TestVector(size_t n) :
		std::vector<std::complex<double> >(n, 0){}		   

	TestVector(size_t n, double num) :
		std::vector<std::complex<double> >(n,num) {}		   
	
	static std::vector<double> norms;
	
	friend std::ostream &operator<<(std::ostream &out, TestVector const &vec)
		{
			for (auto &el: vec)
				out << el << '\n';
			return out;
		}
	
	double norm()
		{
			double sum = 0.0;
			for (auto &el: *this)
				sum += pow(el.real(), 2) + pow(el.imag(), 2);

			norms.push_back(sqrt(sum)); // keeping track for testing purposes
			std::cout << norms.size()-1 << ": " << sqrt(sum) << '\n';
			return sqrt(sum);
		}

	std::complex<double> dot(TestVector const &other)
		{
			assert(this->size() == other.size());
			std::complex<double> result(0,0);
			for (size_t i = 0; i != other.size(); ++i)
				result += std::conj((*this)[i]) * other[i];			
			return result;
		}

	// y =  a * x + y
	void axpy(std::complex<double> a, TestVector const &x)
		{
			assert(this->size() == x.size());
			for (size_t i = 0; i != x.size(); ++i)
				(*this)[i] += a * x[i];
		}
	
	// y =  a * x + b * y
	void axpby(std::complex<double> a, TestVector const &x,
			   std::complex<double> b)
		{
			assert(this->size() == x.size());
			for (size_t i = 0; i != x.size(); ++i)
			{
				(*this)[i] *= b;
				(*this)[i] += a * x[i];
			}
		}

	// this = a * this
	void scale(std::complex<double> a)
		{
			for (auto &el: *this)
				el *= a;
		}

	// this = 0
	void zero()
		{
			for (auto &el: *this)
				el = 0;
		}
	
	// for now we let this set the real parts to 1
	void random()
		{
			for (auto &el: *this)
				el = 1;
		}
};

std::vector<double> TestVector::norms = std::vector<double>();
 
//------------------------------------------------------------------
// Example matrix wrapper
class TestMatrix
{
public:
	// Define a Vector type
	using Vector = TestVector;

private:	
	// Problem size
	size_t n_;
	
public:
	TestMatrix(int size) : n_(size) {};

	// Subroutine to compute r = Aq
	void AMUL(Vector const &q, Vector &r)
		{
			// being careful with 0-based indexing
			for (size_t i = 1; i <= n_; ++i)
				r[i-1] = ((double) i) * q[i-1];
		}

	// Subroutine to compute r = Bq
	void BMUL(Vector const &q, Vector &r)
		{
			for (size_t i = 1; i <= n_; ++i)
				r[i-1] =  q[i-1] / ((double) i);
		}

	// Subroutine to compute q = K^-1 q
	//   here we use that the target in JDQZ is 31
	void PRECON(Vector &q)
		{
			for (size_t i = 1; i <= n_; ++i)
				q[i-1] = ((double) i) * q[i-1] / ((double) i*i - 31);
		}
	size_t size() { return n_; }
};

//------------------------------------------------------------------
TEST(Vector, General)
{
	TestVector vec1(4, 0.0);
	TestVector vec2(4, 0.0);
	vec1[0].real(1.0); vec1[0].imag(2.0);
	vec1[1].real(1.0); vec1[1].imag(0.0);
	vec1[2].real(1.0); vec1[2].imag(0.0);
	vec1[3].real(2.0); vec1[3].imag(0.0);

	vec2[0].real(0.0); vec2[0].imag(2.0);
	vec2[1].real(0.0); vec2[1].imag(1.0);
	vec2[2].real(1.0); vec2[2].imag(-1.0);
	vec2[3].real(1.0); vec2[3].imag(2.0);

	std::complex<double> result = vec1.dot(vec2);
	
	EXPECT_EQ(result.real(), 7);
	EXPECT_EQ(result.imag(), 6);

	std::complex<double> a(4,3);
	vec1.axpy(a, vec2);
	EXPECT_NEAR(vec1.norm(), 18.1934, 1e-4);
	
	std::complex<double> b(2,1);
	vec1.axpby(a, vec2, b);
	EXPECT_NEAR(vec1.norm(), 57.3149, 1e-4);
}

//------------------------------------------------------------------
TEST(Matrix, Operators)
{
	size_t n = 100;
	TestVector vec1(n, 1.0);
	TestVector vec2(n, 0.0);
	TestVector vec3(n, 0.0);
		
	TestMatrix testmat(n);

	testmat.AMUL(vec1, vec2);
	EXPECT_EQ(vec2[49].real(), 50);
	EXPECT_EQ(vec2[77].real(), 78);
	testmat.BMUL(vec2, vec3);
	EXPECT_EQ(vec1 == vec3, true);
	testmat.PRECON(vec1);
	EXPECT_EQ(vec1[9], (double) 10 / (100-31));
}

//------------------------------------------------------------------
TEST(JDQZ, Setup)
{
	size_t size = 100;
	TestMatrix testmat(size);
	JDQZ<TestMatrix> jdqz(testmat);
	EXPECT_EQ(jdqz.size(), size);
}

//------------------------------------------------------------------
TEST(JDQZ, Run)
{
	TestVector::norms.clear();
	size_t size = 100;
	TestMatrix testmat(size);
	JDQZ<TestMatrix> jdqz(testmat);
	jdqz.solve();
	
	// testing rnrm in the inner "attempt" loop
	EXPECT_NEAR(TestVector::norms[7],   7.370188617854, 1e-9);
	EXPECT_NEAR(TestVector::norms[20],  0.808903713729, 1e-9);
	EXPECT_NEAR(TestVector::norms[35],  0.671927897667, 1e-9);
	EXPECT_NEAR(TestVector::norms[52],  0.467505198259, 1e-9);
	EXPECT_NEAR(TestVector::norms[196], 0.777032215617, 1e-9);
	EXPECT_NEAR(TestVector::norms[238], 0.158442752163, 1e-9);
	EXPECT_NEAR(TestVector::norms[282], 1.1112136E-004, 1e-9);	
}

//------------------------------------------------------------------
int main(int argc, char **argv)
{	
	::testing::InitGoogleTest(&argc, argv);
	int out = RUN_ALL_TESTS();
	std::cout << "TEST exit code " << out << std::endl;
	return out;
}
