#ifndef EPETRA_COMPLEX_VECTOR_HPP
#define EPETRA_COMPLEX_VECTOR_HPP

template<typename Vector>
class EpetraComplexVector
{
    Vector tmp_;
    Vector real_;
    Vector imag_;

public:

    //! constructor accepting a single Vector, which will form the
    //! real_ part
    EpetraComplexVector(Vector const &re)
        :
        tmp_(re),
        real_(re),
        imag_(re)
        {
            imag_.PutScalar(0.0);
        }

    //! constructor accepting a real_ and an imag_inary part
    EpetraComplexVector(Vector const &re, Vector const &im)
        :
        tmp_(re),
        real_(re),
        imag_(im)
        {
            assert(real_.GlobalLength() == imag_.GlobalLength());
        }

    //! constructor accepting a real_ and an imag_inary part
    EpetraComplexVector(EpetraComplexVector const &other)
        :
        tmp_(other.real_),
        real_(other.real_),
        imag_(other.imag_)
        {
            assert(real_.GlobalLength() == imag_.GlobalLength());
        }

    //! destructor
    ~EpetraComplexVector()
        {}

    //! assignment operator
    void operator=(EpetraComplexVector const &other)
        {
            real_ = other.real_;
            imag_ = other.imag_;
        }

    //! obtain global length
    int length() const {return real_.GlobalLength();}

    //! calculate 2-norm
    double norm() const
        {
            return sqrt(dot(*this).real());
        }

    //! calculate dot product
    std::complex<double> dot(EpetraComplexVector const &other) const
        {
            double imPart;
            double rePart;
            double tmp;
            real_.Dot(other.real_, &rePart);
            imag_.Dot(other.imag_, &tmp);
            rePart += tmp;

            real_.Dot(other.imag_, &imPart);
            imag_.Dot(other.real_, &tmp);
            imPart -= tmp;

            std::complex<double> result(rePart, imPart);
            return result;
        }

    //! this = a*x + this
    void axpy(std::complex<double> a, EpetraComplexVector const &x)
        {
            assert(this->length() == x.length());

            real_.Update( a.real(), x.real_, 1.0);
            real_.Update(-a.imag(), x.imag_, 1.0);

            imag_.Update( a.imag(), x.real_, 1.0);
            imag_.Update( a.real(), x.imag_, 1.0);

        }

    //! this = a*x + b*this
    void axpby(std::complex<double> a, EpetraComplexVector const &x,
               std::complex<double> b)
        {
            assert(this->length() == x.length());
            scale(b);
            axpy(a, x);
        }

    //! scale vector with complex number
    void scale(std::complex<double> a)
        {
            tmp_ = real_; // copy
            real_.Update(-a.imag(), imag_, a.real());
            imag_.Update( a.imag(), tmp_, a.real());
        }

    //! zero-out vector
    void zero()
        {
            real_.PutScalar(0.0);
            imag_.PutScalar(0.0);
        }

    //! randomize real_ part
    //! zero imag_inary part
    void random()
        {
            real_.Random();
            imag_.PutScalar(0.0);
        }

    void print(){ std::cerr << "Print not implemented" << std::endl; }

    Vector &real() { return real_; }
    Vector &imag() { return imag_; }

    Vector const &real() const { return real_; }
    Vector const &imag() const { return imag_; }


};

#endif
