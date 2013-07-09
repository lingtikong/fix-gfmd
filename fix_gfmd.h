/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   www.cs.sandia.gov/~sjplimp/lammps.html
   Steve Plimpton, sjplimp@sandia.gov, Sandia National Laboratories

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
   ------------------------------------------------------------------------- */
#ifdef FIX_CLASS

FixStyle(gfmd,FixGFMD)

#else

#ifndef FIX_GFMD_H
#define FIX_GFMD_H

#include <complex>
#include "fix.h"
#include <map>
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
	 
namespace LAMMPS_NS {

class FixGFMD : public Fix {
 
 public:
  FixGFMD(class LAMMPS *, int, char **);
  ~FixGFMD();

  int  setmask();
  void init();
  void post_run();
  void post_force(int);
  void end_of_step();
  double memory_usage();
  double compute_vector(int);

 private:
  int me,nprocs;
  int instyle;                   // denotes how initial configuration and Phi_q are supplied;
  char *file_xorg, *file_phi;    // files for initial configuration and Phi_q
  int  itag,idx,idq;             // current tag; index variable
  int  origin_tag;               // atomic tag of the surface origin for FFT
  int  nGFatoms;                 // total # of GF atoms
  double masstotal, rmasstotal;  // total mass of GF atoms and its inverse; to get the center of mass

  int noutfor;                   // frequency to output Green's Function force on GF atoms;
  char *prefix;                  // prefix for output files in GFMD
  FILE *gfmdlog;                 // file to output GF related info

  double load;                   // extra force added to GF atoms perpendicular to GF layer
  double foriginal[3], foriginal_all[3]; // to store the original forces
  int    collect_flag;           // indicates if all forces have been collected or not
  int    reset_force;            // flag indicate whether to reset force on GF layer to zero (1) or not (0)
                                 // before adding elastic force

  int nx,ny,nucell;              // dimension of FFT mesh
  int sysdim, surfdim;           // system dimension and surface dimension
  double surfvec[2][2];          // surface vectors

  void readxorg();               // method to read initial configuration for GF atoms from file file_xorg
  void compxorg();               // method to compute the initial configuration
  double **xeq;                  // 2D double array stores the equilibrium configuration
  int reset_xeq;                 // flag to indicate whether/how to reset xeq according to surface lattice info from gfc
  std::map<int,int> tag2surf,surf2tag;// map between tag and FFT index
  char *mapfile;                 // file to read in mapping info, if preferred
  void readmap();                // method to read mapping info from mapfile
  void compmap(int);             // method to compute mapping info from initial configuration based on surface vectors

  int nfind;
  int *findings;
  int *recvcnts, *displs;        

  int nasr;

  // FFT variables: Forward (FW) transforms of displacments and backward (BW) transforms of forces
  class FFT3d *fft;
  double *fft_data;
  int nxlo,nxhi,mysize;          // job assignment info
  int mynpt, mynq;               // total real and reciprocal points assigned to local proc
  int fft_nsend;                 // send count for local proc
  int fft_dim, fft_dim2;
  int *fft_cnts, *fft_disp;      // help to gather F(r) on local to all procs

  double **UIrLoc;               // 2D array storing U(r) and index on local proc
  double **UIrAll, **FrAll;      // 2D array storing the gathered U(r) and/or F(r)
  double **UFrnow, **UFrSort;    // 2D array storing either U(r) or F(r) assigned to FFT
  std::complex<double> **UFqnow; // 2D array storing either U(q) or F(q) assigned to FFT

  void readphi();                // Phi related methods and variables
  void Phi_q0_ASR(std::complex<double> *);
  void Phi_analytic();
  void Phi_analytic_3D(double , double, std::complex<double> **);
  void Phi_analytic_2D(double , std::complex<double> **);
  std::complex<double> **Phi_q;

  // bicubic interpolation of Phi_q
  void bcucof (std::complex<double>*, std::complex<double>*, std::complex<double>*, std::complex<double>*,
               double, double, std::complex<double>*);
  void bicuint(std::complex<double>*, std::complex<double>*, std::complex<double>*, std::complex<double>*,
               double, double, double, double, double, double, std::complex<double> *);

  // private methods to do matrix operation
  void GaussJordan(int,double *);                                           // matrix inversion
  void GaussJordan(int,std::complex<double>*);
  void MatMulVec(int,double*,double*,double*);                              // matrix-vector multiplication
  void MatMulVec(int,std::complex<double>*,std::complex<double>*,std::complex<double>*);
  void MatMulMat(int,double*,double*,double*);                              // matrix-matrix multiplication
  void MatMulMat(int,std::complex<double>*,std::complex<double>*,std::complex<double>*);

};

}
#endif
#endif

/* ERROR/WARNING messages:

E: Illegal fix gfmd command: number of arguments < 5

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Wrong command line option

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Insufficient command line options

Self-explanatory.

E: Insufficient command option for ...

Self-explanatory. The command option needs extra arguments but not provided.

E: Wrong command option for reset_xeq.

Self-explanatory. The allowed options are: none, all, last.

E: Unknown command line option: %s

Self-explanatory.

E: No atom is passed to gfmd

Self-explanatory, number of atoms in the group that is passed to fix-gfmd is 0.

E: Cannot open log file: gfmd.%s.log

Self-explanatory.

E: Analytical Phi works only for simple cubic system

Self-explanatory.

E: Cannot open original position file: %s

Self-explanatory.

E: Error while reading initial configuration from file: %s

Self-explanatory.

E: Cannot open mapping file: %s

Self-explanatory.

E: Error %d while reading header info from file: %s

Self-explanatory. Expect to read three positive integer numbers from the first line
of the map file.

E: Number of atoms from FFT mesh and group mismatch

Self-explanatory.

E: Error %d while reading comment line from file: %s

Self-explanatory. The second line of the map file should be comment

E: Error %d while reading mapping info from file: %s

Self-explanatory.

E: Atom in group not found in mapping info

Self-explanatory.

E: No lattice defined while keyword su and/or sv not set

When no map file provided, and that neither su nor sv is set, lattice info
will be used to compute the mapping info. In that case, a lattice should be
defined before invoke this fix.

E: Either U or V must be on the box side

Self-explanatory. One of the surface unit cell vectors must be on the box edge.

E: Surface vector U must be along the +x direction!

Self-explanatory.

E: Surface vector V must point to the +y direction!

Self-explanatory.

E: Error encountered while getting FFT dimensions

Self-explanatory. Either of the fft size is guessed to be less than 1.
In this case one should better provided the mapping info via the map keyword.

E: Mapping info cannot be computed for nucell > 2!

Self-explanatory. The mapping info could be computed based on surface unit cell
vectors only when there are no more than two atoms in each unit cell. Otherwise one
has to provide the mapping info via the map file.

E: Surface origin given by user not found

Self-explanatory; when computing the mapping info based on surface unit cell
vectors, one can define an atom as the origin of the surface lattice; and this
atom must be in the group that is passed to fix-gfc.

E: Error encountered while computing mapping info

Self-explanatory. In this case one should better provided the mapping info via
the map keyword.

E: Atom in group not found in mapping info

Self-explanatory.

E: Error while opening binary Phi file

Self-explanatory.

E: Error while reading ... from binary file

Self-explanatory.

E: System dimension from GFC is %d, current is %d

Self-explanatory. System dimension info from the binary elastic stiffness coefficients
file generated by fix-gfc and this run mismatch.

E: # of atom per cell from GFC is %d, current is %d

Self-explanatory. Number of atoms per surface unit cell from the binary elastic
stiffness coefficients file generated by fix-gfc and this run mismatch.

E: Units used by GFC were LJ, conversion is not possible

Self-explanatory. When LJ unit was used in fix-gfc when measuring the elastic stiffness
coefficients, while a different one is used by fix-gfmd, it is not possible to use the
elastic stiffness coefficients from fix-gfc to run a gfmd simulation.

E: Failed to read ... info from binary file

Self-explanatory.

E: Surface lattice from gfc and this run mismatch

Self-explanatory.

E: Singular matrix in double GaussJordan!

Self-explanatory.

E: Singular matrix in complex GaussJordan!

Self-explanatory.

W: Units used by GFC differ from current one, converted!

Self-explanatory.

W: Surface lattice from gfc and this run mismatch, xeq not reset

Self-explanatory.

W: More than one fix gfmd defined

Self-explanatory. Just to warn that more than one fix-gfmd is defined, but allowed.

*/
