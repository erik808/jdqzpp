#ifndef JDQZ_DECL_H
#define JDQZ_DECL_H

// This is a port of the original code by Fokkema and van Gijzen.

// Matrix members:
//   AMUL(N, Vector, Vector)
//   BMUL(N, Vector, Vector)
//   PRECON(N, Vector, Vector)

// Vector members:
//   TODO

template<typename Matrix, typename Vector>
class JDQZ
{
	
public:
	JDQZ(Model model);
	void setup();	
};

#endif
