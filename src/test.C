#include "gtest/gtest.h" // google test framework
#include "jdqz.H"
#include <vector>

typedef std::vector<double> vector;

//------------------------------------------------------------------
class TestMatrix
{
public:
	TestMatrix(){};

	// Subroutine to compute r = Aq
	void AMUL(size_t n, vector const &q, vector &r)
		{
			// being careful with 0-based indexing
			for (size_t i = 1; i <= n; ++i)
				r[i-1] = ((double) i) * q[i-1];
		}

	// Subroutine to compute r = Bq
	void BMUL(size_t n, vector const &q, vector &r)
		{
			for (size_t i = 1; i <= n; ++i)
				r[i-1] =  q[i-1] / ((double) i);
		}

	// Subroutine to compute q = K^-1 q
	//   here we use that the target in JDQZ is 31
	void PRECON(size_t n, vector &q)
		{
			for (size_t i = 1; i <= n; ++i)
				q[i-1] = ((double) i) * q[i-1] / ((double) i*i - 31);
		}
};

//------------------------------------------------------------------
TEST(Matrix, Operators)
{
	size_t n = 100;
	vector vec1(n, 1.0);
	vector vec2(n, 0.0);
	vector vec3(n, 0.0);
		
	TestMatrix testmat;

	testmat.AMUL(n, vec1, vec2);
	EXPECT_EQ(vec2[49], 50);
	EXPECT_EQ(vec2[77], 78);
	testmat.BMUL(n, vec2, vec3);
	EXPECT_EQ(vec1 == vec3, true);
	testmat.PRECON(n, vec1);
	EXPECT_EQ(vec1[9], (double) 10 / (100-31));
}

//------------------------------------------------------------------
TEST(JDQZ, Setup)
{
	// TODO
}

//------------------------------------------------------------------
int main(int argc, char **argv)
{	
	::testing::InitGoogleTest(&argc, argv);
	int out = RUN_ALL_TESTS();
	std::cout << "TEST exit code " << out << std::endl;
	return out;
}

