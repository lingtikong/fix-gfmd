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
