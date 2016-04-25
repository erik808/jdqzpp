#ifndef TEST_H
#define TEST_H

class TestVector : public std::vector<std::complex<double> >
{
public:

	TestVector(size_t n) :
		std::vector<std::complex<double> >(n, 0){}		   

	TestVector(size_t n, double num) :
		std::vector<std::complex<double> >(n,num) {}		   
	
	static std::vector<double> norms;
	static std::vector<complex> alphas;
	static std::vector<complex> betas;
	static std::vector<complex> dotresults;
	
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
			return sqrt(sum);
		}

	std::complex<double> dot(TestVector const &other)
		{
			assert(this->size() == other.size());
			std::complex<double> result(0,0);
			for (size_t i = 0; i != other.size(); ++i)
				result += std::conj((*this)[i]) * other[i];


			dotresults.push_back(result);
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

			alphas.push_back(-b); // for testing
			betas.push_back(a);
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

#endif
