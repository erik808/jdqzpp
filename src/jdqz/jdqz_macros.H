#ifndef JDQZMACROS_H
#define JDQZMACROS_H

// PARALLEL PRINT MACRO
#ifndef WRITE
# ifdef HAVE_MPI
#  define WRITE(msg)							\
	{											\
		int rank;								\
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);	\
		if (rank == 0)							\
			std::cout << msg << std::endl;		\
	}
# else
#  define WRITE(msg) std::cout << msg << std::endl;
# endif
#endif

// Number of entries in profile
#ifndef PROFILE_ENTRIES
# define PROFILE_ENTRIES 4
#endif

#endif
