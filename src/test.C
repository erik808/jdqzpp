#include "gtest/gtest.h" // google test framework
#include "jdqz.H"
#include <vector>
#include <complex>

typedef std::vector<std::complex<double> > vector;

//------------------------------------------------------------------
// Example matrix wrapper
class TestMatrix
{
public:
	// Define a Vector type
	using Vector = std::vector<std::complex<double> >;

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
TEST(Matrix, Operators)
{
	size_t n = 100;
	vector vec1(n, 1.0);
	vector vec2(n, 0.0);
	vector vec3(n, 0.0);
		
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
int main(int argc, char **argv)
{	
	::testing::InitGoogleTest(&argc, argv);
	int out = RUN_ALL_TESTS();
	std::cout << "TEST exit code " << out << std::endl;
	return out;
}
