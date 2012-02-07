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

FixStyle(gfc,FixGFC)

#else

#ifndef FIX_GFC_H
#define FIX_GFC_H

#include <complex>
#include "fix.h"
#include <map>
#include "stdio.h"
#include "stdlib.h"

namespace LAMMPS_NS {

class FixGFC : public Fix {
 public:
  FixGFC(class LAMMPS *, int, char **);
  ~FixGFC();

  int  setmask();
  void init();
  void setup(int);
  void post_run();
  void end_of_step();
  double memory_usage();
  int modify_param(int, char **);

 private:
  int me,nprocs;
  bigint waitsteps;                             // wait these number of timesteps before recording atom positions
  bigint prev_nstep;                            // number of steps from previous run(s); to judge if waitsteps is reached.
  int nfreq, ifreq;                             // after this number of measurement (nfreq), the result will be output once
  int nx,ny,nucell;                             // surface dimensions in x- and y-direction, number of atom per unit surface cell
  int origin_tag;                               // tag of the surface origin
  int GFcounter;                                // counter variables
  int sysdim;                                   // system dimension (2D or 3D simulations) !
  int nGFatoms, nfind;                          // total number of GF atoms; total number of GF atom on this proc
  char *prefix, *file_log;                      // prefix of output file names
  FILE *gfclog;
  
  class FFT3d *fft;                             // to do fft via the fft3d wraper
  int nxlo,nxhi,mysize;                         // size info for local MPI_FFTW
  int mynpt,mynq,fft_nsend;
  int *fft_cnts, *fft_disp;
  int fft_dim, fft_dim2;
  double *fft_data;
  
  int  itag, idx, idq;                          // index variables
  std::map<int,int> tag2surf, surf2tag;         // Mapping info

  double **RIloc;                               // R(r) and index on local proc
  double **RIall;                               // gathered R(r) and index
  double **Rsort;                               // sorted R(r)
  double **Rnow;                                // Current R(r) on local proc for GF atoms
  double **Rsum;                                // Accumulated R(r) on local proc for GF atoms

  double surfvec[2][2];                         // 2x2 surface vectors
  double **surfbasis;

  int *recvcnts, *displs;                       // MPI related variables

  std::complex<double> **Rqnow;                 // Current R(q) on local proc
  std::complex<double> **Rqsum;                 // Accumulator for conj(R(q)_alpha)*R(q)_beta
  std::complex<double> **Phi_q;                 // Phi's on local proc
  std::complex<double> **Phi_all;               // Phi for all

  void readmap();                               // to read the mapping of gf atoms
  void compmap(int);                            // try to compute the mapping of gf atoms from SurfVectors
  char *mapfile;                                // file name of the map file

  void postprocess();                           // to post process the data
  
  char *id_temp;                                // compute id for temperature
  double *TempSum;                              // to get the average temperature vector
  double inv_nTemp;                             // inverse of number of atoms in temperature group
  class Compute *temperature;                   // compute that computes the temperature

  // private methods to do matrix inversion
  void GaussJordan(int, double *);
  void GaussJordan(int, std::complex<double>*);

};
}
#endif
#endif

/* ERROR/WARNING messages:

E: Illegal fix gfc command: number of arguments < 7

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Illegal fix gfc command: nevery must >0

Self-explanatory.

E: Illegal fix gfc command: nfreq must >0

Self-explanatory.

E: Illegal fix gfc command: waitsteps < 0

Self-explanatory.

E: Insufficient command line options for fix gfc.

Self-explanatory. Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Unknown command line option for fix gfc

Self-explanatory. Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: No atom found for GFC evaluation!

Self-explanatory, number of atoms in the group that passed to fix-gfc
is 0.

E: Can not open output file %s

Self-explanatory.

E: Error while reading header of mapping file

Self-explanatory, the header line of the map file expects three integer numbers.

E: FFT mesh and number of atoms in group mismatch!

Self-explanatory; numbers read from the header of the map file mismatch the total
number of atoms in the group.

E: Error while reading comment of mapping file

Self-explanatory; second line of the map file is comment.

E: The mapping is incomplete!

Self-explanatory; mapping info read from the map file is incomplete.

E: Error while reading mapping file!

Self-explanatory.

E: The mapping info read is incorrect!

Self-explanatory.

E: No lattice defined while keyword su and/or sv is not set.

When no map file provided, and that neither su nor sv is set, lattice info
will be used to compute the mapping info. In that case, a lattice should be
defined before invoke this fix.

E: Either U or V must be on the box side!

Self-explanatory. One of the surface unit cell vectors must be on the box edge.

E: Surface vector U must be along the +x direction!

Self-explanatory.

E: Surface vector V must point to the +y direction!

Self-explanatory.

E: Error encountered while getting FFT dimensions!

Self-explanatory.

E: Mapping info cannot be computed for nucell > 2!

Self-explanatory. The mapping info could be computed based on surface unit cell
vectors only when there are less than two atoms in each unit cell. Otherwise one
has to provide the mapping info via the map file.

E: Number of atoms from FFT mesh and group mismatch!

Self-explanatory.

E: Specified surface origin atom not in group!

Self-explanatory; when computing the mapping info based on surface unit cell
vectors, one can define an atom as the origin of the surface lattice; and this
atom must be in the group that is passed to fix-gfc.

E: Surface origin not found!

Self-explanatory; see above.

E: Mapping info is incompleted/incorrect!

Self-explanatory.

E: Singular matrix in double GaussJordan!

Self-explanatory.

E: Singular matrix in complex GaussJordan!

Self-explanatory.

W: More than one fix gfc defined

Self-explanatory. Just to warn that more than one fix-gfc is defined, but allowed.

*/
