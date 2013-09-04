/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */
/* ----------------------------------------------------------------------
   Contributing authors:
     L.T. Kong(a,b), G. Bartels(a), C. Campana(a),
     C. Denniston(a) and M. Muser (a,c)

   Contact:
     a) Department of Applied Mathematics, University of Western Ontario
        London, ON, Canada N6A 5B7
     b) School of Materials Science and Engineering, Shanghai Jiao Tong Univ.,
        800 Dongchuan Road, Minghang, Shanghai 200240, China
     c) Department of Materials Science, Universitat des Saarlandes,
        Saarbrucken, Germany

     mmuser@uwo.ca, cdennist@uwo.ca, konglt@gmail.com

   References:
     1) Phys. Rev. B 74(7):075420, 2006.
     2) Computer Physics Communications 180(6):1004-1010, 2009.
     3) Computer Physics Communications 182(2):540-541, 2011.
------------------------------------------------------------------------- */

#include "atom.h"
#include "domain.h"
#include "error.h"
#include "fft3d_wrap.h"
#include "fix_gfmd.h"
#include "force.h"
#include "group.h"
#include "lattice.h"
#include "memory.h"
#include "modify.h"
#include "update.h"
	 
#define MAXJACOBI 100
#define MAXLINE   256
#define SIGN(a) ((a)>0.?1.:((a)<0.?-1.:0.))

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixGFMD::FixGFMD(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg)
{
  MPI_Comm_rank(world,&me);
  MPI_Comm_size(world,&nprocs);

  if (narg<5) error->all(FLERR,"Illegal fix gfmd command: number of arguments < 5");
  
  int iarg = 3;
  int n = strlen(arg[iarg]) + 1;
  prefix = new char[n];             // get prefix
  strcpy(prefix, arg[iarg++]);

  instyle = atoi(arg[iarg]);        // get instyle
  if (instyle<0||instyle>3) error->all(FLERR,"Wrong command line option");

  if (instyle & 1){                 // get original position file name, if supplied
    if ((iarg+2)>narg) error->all(FLERR,"Insufficient command line options");
    n = strlen(arg[++iarg])+1;
    file_xorg = new char[n];
    strcpy(file_xorg,arg[iarg]);
  }
  if (instyle & 2){                 // get binary Phi file name, if supplied
    if ((iarg+2)>narg) error->all(FLERR,"Insufficient command line options");
    n = strlen(arg[++iarg])+1;
    file_phi = new char[n];
    strcpy(file_phi,arg[iarg]);
  }
  iarg++;

  // to read other command line options, they are optional
  nasr = 15;
  load = 0.;
  noutfor = 0;
  mapfile = NULL;
  reset_xeq = 0;
  origin_tag = -1;
  reset_force = 0;
  int fsurfmap = 0;

  while (iarg < narg){
    // surface vector U. if not given, will be determined from lattice info
    if (strcmp(arg[iarg],"su") == 0){
      if (iarg+3 > narg) error->all(FLERR,"Insufficient command option for su.");
      surfvec[0][0] = atof(arg[++iarg]);
      surfvec[0][1] = atof(arg[++iarg]);
      fsurfmap |= 1;

    // surfactor vector V. if not given for 3D, will be determined from lattice info
    } else if (strcmp(arg[iarg],"sv") == 0){
      if (iarg+3 > narg) error->all(FLERR,"Insufficient command option for sv.");
      surfvec[1][0] = atof(arg[++iarg]);
      surfvec[1][1] = atof(arg[++iarg]);
      fsurfmap |= 2;

    // to get the tag of surface origin atom
    } else if (strcmp(arg[iarg],"origin") == 0){
      if (iarg+2 > narg) error->all(FLERR,"Insufficient command option for origin.");
      origin_tag = atoi(arg[++iarg]);

    // read the mapping of surface atoms from file, no surface vector is needed now
    } else if (strcmp(arg[iarg],"map") == 0){
      if (iarg+2 > narg) error->all(FLERR,"Insufficient command option for map.");
      if (mapfile) delete []mapfile;
      n = strlen(arg[++iarg]) + 1;
      mapfile = new char [n];
      strcpy(mapfile, arg[iarg]);
      fsurfmap |= 4;

    // extra load added to GF atoms
    } else if (strcmp(arg[iarg],"load") ==0){
      if (iarg+2 > narg) error->all(FLERR,"Insufficient command option for load.");
      load = atof(arg[++iarg]);

    // frequency to output elastic force
    } else if (strcmp(arg[iarg],"output") ==0){
      if (iarg+2 > narg) error->all(FLERR,"Insufficient command option for output.");
      noutfor = atoi(arg[++iarg]);

    // whether/how to reset xeq based on surface lattice info from gfc
    } else if (strcmp(arg[iarg],"reset_xeq") ==0){
      if (iarg+2 > narg) error->all(FLERR,"Insufficient command option for reset_xeq.");
      iarg++;
      if (strcmp(arg[iarg],"none")==0) reset_xeq = 0;
      else if (strcmp(arg[iarg],"all")==0) reset_xeq = 3;
      else if (strcmp(arg[iarg],"last") ==0) reset_xeq = 1;
      else error->all(FLERR,"Wrong command option for reset_xeq.");

    } else if (strcmp(arg[iarg],"reset_f") ==0){
      if (iarg+2 > narg) error->all(FLERR,"Insufficient command option for reset_f.");
      if (strcmp(arg[++iarg],"yes")==0) reset_force = 1;
      
    } else if (strcmp(arg[iarg],"nasr") ==0){
      if (iarg+2 > narg) error->all(FLERR,"Insufficient command option for nasr.");
      nasr = atoi(arg[++iarg]);
      
    } else {
      char str[MAXLINE];
      sprintf(str,"Unknown command line option: %s", arg[iarg]);
      error->all(FLERR,str);
    }

    iarg++;
  } // end of reading command line options

  sysdim = domain->dimension; // find the system and surface dimension
  surfdim = sysdim-1;
  if (sysdim == 2){
    surfvec[1][0] = 0.;
    surfvec[1][1] = 1.;
    fsurfmap |=2;
  }
  nasr = MAX(0, nasr);

  // get the total number and mass (inverse) of atoms in group
  nGFatoms   = static_cast<int>(group->count(igroup));
  masstotal  = group->mass(igroup);
  rmasstotal = 1./masstotal;
  if (nGFatoms < 1) error->all(FLERR,"No atom is passed to gfmd");

  // open the log file on root
  if (me == 0){
    char str[MAXLINE];
    sprintf(str,"gfmd.%s.log",prefix);
    gfmdlog = fopen(str,"w");
    if (gfmdlog == NULL){
      sprintf(str,"Cannot open log file: gfmd.%s.log",prefix);
      error->one(FLERR,str);
    }
    fprintf(gfmdlog, "GFMD initialization...\n");
    fprintf(gfmdlog, "Number of processors used    : %d\n", nprocs);
    fprintf(gfmdlog, "Total number of GF atoms     : %d\n", nGFatoms);
    fprintf(gfmdlog, "Extra load added to GF atoms : %lg\n", load);
    fflush(gfmdlog);
  }

  // allocating real space working arrays; UIrLoc and UIrAll will be used in
  // determining the original position as well as the mapping
  memory->create(xeq    ,nGFatoms,sysdim,"fix_gfmd:xeq");
  memory->create(UIrLoc ,nGFatoms,sysdim+1,"fix_gfmd:UIrLoc");
  memory->create(UIrAll ,nGFatoms,sysdim+1,"fix_gfmd:UIrAll");
  memory->create(UFrSort,nGFatoms,sysdim,"fix_gfmd:UFrSort");
  memory->create(FrAll  ,nGFatoms,sysdim,"fix_gfmd:FrAll");

  tag2surf.clear(); // clear map info
  surf2tag.clear();

  findings = new int [nprocs];
  recvcnts = new int [nprocs];
  displs   = new int [nprocs];

  // to get the original positions of atoms in the manifold; temporarily in UIrAll.
  // might be reset in readphi later on
  if (instyle & 1){ readxorg(); delete []file_xorg; } // read from file file_xorg
  else compxorg();                                    // get from initial configuration

  // To get mapping info
  if (fsurfmap & 4){ readmap(); delete []mapfile; }
  else compmap(fsurfmap);

  // To sort xeq according to FFT mesh
  for (int i = 0; i < nGFatoms; ++i){
    itag = static_cast<int>(UIrAll[i][sysdim]);
    idx  = tag2surf[itag];
    for (int idim = 0; idim < sysdim; ++idim) xeq[idx][idim] = UIrAll[i][idim];
  }

  // reset surface vectors based on box info
  surfvec[0][0] = domain->h[0]/double(nx);
  surfvec[0][1] = domain->h[5]/double(nx);
  surfvec[1][0] = 0.;
  surfvec[1][1] = domain->h[1]/double(ny);
  if (sysdim == 2) surfvec[1][1] = 1.;

  // output mapping info to log file
  if (me == 0){
    fprintf(gfmdlog,"Total # of atoms in manifold : %d\n", nGFatoms);
    fprintf(gfmdlog,"Surface vector along U       : [ %lg, %lg ]\n", surfvec[0][0],surfvec[0][1]);
    fprintf(gfmdlog,"Surface vector along V       : [ %lg, %lg ]\n", surfvec[1][0],surfvec[1][1]);
    fprintf(gfmdlog,"# of atoms per unit cell     : %d\n", nucell);
    fprintf(gfmdlog,"Dimensions of the FFT mesh   : %d x %d\n\n", nx,ny);
    fprintf(gfmdlog,"Mapping between lattice indices and atom ID:\n");
    fprintf(gfmdlog,"# nx ny nucell\n");
    fprintf(gfmdlog," %d %d %d\n",nx,ny,nucell);
    fprintf(gfmdlog,"# l1 l2 k atom_id\n");
    idx=0;
    for (int ix = 0; ix < nx; ++ix)
    for (int iy = 0; iy < ny; ++iy)
    for (int iu = 0; iu < nucell; ++iu){
      itag = surf2tag[idx++];
      fprintf(gfmdlog,"%d %d %d %d\n",ix,iy,iu,itag);
    }
    fflush(gfmdlog);
  }

  // create FFT and initialize related variables
  int *nx_loc = new int [nprocs];
  nxlo = 0;
  for (int i = 0; i < nprocs; ++i){
    nx_loc[i] = nx/nprocs;
    if (i < nx%nprocs) ++nx_loc[i];
  }
  for (int i = 0; i < me; ++i) nxlo += nx_loc[i];
  nxhi  = nxlo + nx_loc[me] - 1;
  mynpt = nx_loc[me]*ny;
  mynq  = mynpt;
  fft_dim   = nucell*sysdim;
  fft_dim2  = fft_dim*fft_dim;
  fft_nsend = mynpt*fft_dim;

  fft_cnts = new int[nprocs];
  fft_disp = new int[nprocs];
  fft_disp[0] = 0;
  for (int i = 0; i < nprocs; ++i) fft_cnts[i] = nx_loc[i]*ny*fft_dim;
  for (int i = 1; i < nprocs; ++i) fft_disp[i] = fft_disp[i-1] + fft_cnts[i-1];

  fft = new FFT3d(lmp,world,1,ny,nx,0,0,0,ny-1,nxlo,nxhi,0,0,0,ny-1,nxlo,nxhi,0,0,&mysize);
  memory->create(fft_data, mynq*2, "fix_gfc:fft_data");

  // write FFT assignment info to log file
  if (me == 0){
    fprintf(gfmdlog,"\nGFMD FFTW assignment:\n");
    for (int i = 0; i < nprocs; ++i) fprintf(gfmdlog,"  On proc %d mynx = %d;\n", i, nx_loc[i]);
    fflush(gfmdlog);
  }
  delete []nx_loc;

  if (nucell == 1) nasr = 1;
  // allocate memory and get Phi_q, only store those relevant to local proc
  memory->create(Phi_q,MAX(1,mynq), fft_dim2, "fix_gfmd:Phi_q");
  if (instyle & 2){ readphi(); delete []file_phi; } // read Phi_q from binary file from FixGFC run
  else Phi_analytic();                              // Claculate analytic Phi_q for simple cubic lattice

  // divide Phi_q by (nx*ny) to reduce float operation after FFT
  double inv_nxny = 1./double(nx*ny);
  for (idq = 0; idq < mynq; ++idq)
  for (int idim = 0; idim < fft_dim2; ++idim) Phi_q[idq][idim] *= inv_nxny;

  // allocating remaining working arrays; MAX(1,.. is used to avoid MPI buffer error
  memory->create(UFrnow,MAX(1,mynpt),fft_dim,"fix_gfmd:UFrnow");
  memory->create(UFqnow,MAX(1,mynq), fft_dim, "fix_gfmd:UFqnow");

  // enable to return original forces before they are changed
  extvector = 1;
  vector_flag = 1;
  size_vector = sysdim;
  global_freq = 1;

  // output xeq to log file
  if (me == 0){
    fprintf(gfmdlog,"\nOriginal/equilibrium position of atoms in the manifold\n");
    if (sysdim == 2){
      fprintf(gfmdlog,"#index atom_ID x  y\n");
      for (idx = 0; idx < nGFatoms; ++idx){
        itag = surf2tag[idx];
        fprintf(gfmdlog,"%d %d %lg %lg\n",idx,itag,xeq[idx][0],xeq[idx][1]);
      }
    } else {
      fprintf(gfmdlog,"#index atom_ID x  y  z\n");
      for (idx = 0; idx < nGFatoms; ++idx){
        itag = surf2tag[idx];
        fprintf(gfmdlog,"%d %d %lg %lg %lg\n",idx,itag,xeq[idx][0],xeq[idx][1],xeq[idx][2]);
      }
    }
    fprintf(gfmdlog,"\nInitialization done!\n");
    fclose(gfmdlog);
  }

return;
} // end of constructor

/* ---------------------------------------------------------------------- */

void FixGFMD::post_run()
{
  // always write the elastic force at the final step
  if (noutfor<=0 || update->ntimestep%nevery!=0) end_of_step();

}

/* ---------------------------------------------------------------------- */

FixGFMD::~FixGFMD()
{
  // delete arrays allocated by new
  delete []prefix;
  delete []fft_cnts;
  delete []fft_disp;

  delete []findings;
  delete []recvcnts;
  delete []displs;

  // delete arrays grow by memory->..
  memory->destroy(UIrLoc);
  memory->destroy(UIrAll);
  memory->destroy(UFrSort);
  memory->destroy(FrAll);
  memory->destroy(UFrnow);
  memory->destroy(UFqnow);
  memory->destroy(Phi_q);
  memory->destroy(xeq);

  // clear map
  tag2surf.clear();
  surf2tag.clear();

  // destroy FFT plan and worksapce
  delete fft;
  memory->sfree(fft_data);

return;
}

/* ---------------------------------------------------------------------- */

int FixGFMD::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  if (noutfor>0){
    mask  |= END_OF_STEP;
    nevery = noutfor;
  }

return mask;
}

/* ---------------------------------------------------------------------- */

void FixGFMD::init()
{
  int count = 0;
  for (int i = 0; i < modify->nfix; ++i){
    if (strcmp(modify->fix[i]->style,"gfmd") == 0) ++count;
  }
  if (count > 1 && me == 0) error->warning(FLERR,"More than one fix gfmd."); // just warn, but it is allowed

return;
} 

/* ---------------------------------------------------------------------- */

void FixGFMD::post_force(int vflag)
{
  double **x  = atom->x;
  double **f  = atom->f;
  double xprd = domain->xprd;
  double yprd = domain->yprd;
  double zprd = domain->zprd;
  double *h   = domain->h;
  double xcur[3];
  double xbox, ybox, zbox;

  int *tag   = atom->tag;
  int *mask  = atom->mask;
  int *image = atom->image;
  int nlocal = atom->nlocal;
  int i,idim;
  int nusend, nfrecv;

  std::complex<double> F_q[fft_dim];

  // get U(r) on local proc
  nfind = 0;
  for (i = 0; i < nlocal; ++i){
    if (mask[i] & groupbit){
      itag = tag[i];
      idx  = tag2surf[itag];

      domain->unmap(x[i], image[i], xcur);
        
      for (idim = 0; idim < sysdim; ++idim) UIrLoc[nfind][idim] = xcur[idim];
      UIrLoc[nfind++][sysdim] = double(idx);
    }
  }

  nfrecv = nfind * sysdim;
  nusend = nfind * (sysdim+1);
  displs[0] = 0;
  for (i = 0; i < nprocs; ++i) findings[i] = 0;

  // Gather U(r) from all procs, then sort and redistribute to all procs for FFT
  MPI_Gather(&nfind,1,MPI_INT,findings,1,MPI_INT,0,world);
  for (i = 0; i < nprocs; ++i) recvcnts[i] = findings[i] * (sysdim+1);
  for (i = 1; i < nprocs; ++i) displs[i]   = displs[i-1] + recvcnts[i-1];
  MPI_Gatherv(UIrLoc[0],nusend,MPI_DOUBLE,UIrAll[0],recvcnts,displs,MPI_DOUBLE,0,world);
  if (me == 0){
    for (i = 0; i < nGFatoms; ++i){
      idx = static_cast<int>(UIrAll[i][sysdim]);
      for (idim = 0; idim < sysdim; ++idim) UFrSort[idx][idim] = UIrAll[i][idim] - xeq[idx][idim];
    }
  }
  MPI_Scatterv(UFrSort[0],fft_cnts,fft_disp, MPI_DOUBLE, UFrnow[0], fft_nsend, MPI_DOUBLE, 0,world);

  // Filling and perform FFT on local data
  for (idim = 0; idim < fft_dim; ++idim){
    int m = 0;
    for (idx = 0; idx < mynpt; ++idx){
      fft_data[m++] = UFrnow[idx][idim];
      fft_data[m++] = 0.;
    }
    fft->compute(fft_data,fft_data,-1);
    m = 0;
    for (idq = 0; idq < mynq; ++idq){
      UFqnow[idq][idim] = std::complex<double>(fft_data[m], fft_data[m+1]);
      m += 2;
    }
  }

  // Perform matrix operation: F(q) = -Phi(q) x U(q)
  for (idq = 0; idq < mynq; ++idq){
    MatMulVec(fft_dim,Phi_q[idq],UFqnow[idq],F_q);
    for (idim = 0; idim < fft_dim; ++idim) UFqnow[idq][idim] = -F_q[idim];
  }

  // Now transform F(q) to F(r)
  for (idim = 0; idim < fft_dim; ++idim){
    int m = 0;
    for (idq = 0; idq < mynq; ++idq){
      fft_data[m++] = real(UFqnow[idq][idim]);
      fft_data[m++] = imag(UFqnow[idq][idim]);
    }
    fft->compute(fft_data, fft_data, 1);
    m = 0;
    for (idx = 0; idx < mynpt; ++idx){
      UFrnow[idx][idim] = fft_data[m];
      m += 2;
    }
  }

  // gather F(r) from local proc to root and then scatter the corresponding parts back
  MPI_Gatherv(UFrnow[0], fft_nsend, MPI_DOUBLE, UFrSort[0], fft_cnts, fft_disp, MPI_DOUBLE, 0, world);
  if (me == 0){
    for (i = 0; i < nGFatoms; ++i){
      idx = static_cast<int>(UIrAll[i][sysdim]);
      for (idim = 0; idim < sysdim; ++idim) FrAll[i][idim] = UFrSort[idx][idim];
    }
  }
  for (i = 0; i < nprocs; ++i) recvcnts[i] = findings[i]*sysdim;
  for (i = 1; i < nprocs; ++i) displs[i]   = displs[i-1] + recvcnts[i-1];
  MPI_Scatterv(FrAll[0], recvcnts, displs, MPI_DOUBLE, UFrSort[0], nfrecv, MPI_DOUBLE, 0,world);

  // to collect original force
  collect_flag = 0;
  foriginal[0] = foriginal[1] = foriginal[2] = 0.;

  // add elastic force and extra load to local atoms
  nfind = 0;
  if (reset_force){ // replace force with elastic force
    for (int i = 0; i < nlocal; ++i)
    if (mask[i] & groupbit){
      for (idim = 0; idim < sysdim; ++idim){
        foriginal[idim] += f[i][idim];
        f[i][idim] = UFrSort[nfind][idim];
      }
      f[i][surfdim] += load;
      ++nfind;
    }
  } else { // keep original force and add elastic force
    for (i = 0; i < nlocal; ++i)
    if (mask[i] & groupbit){
      for (idim = 0; idim < sysdim; ++idim){
        foriginal[idim] += f[i][idim];
        f[i][idim] += UFrSort[nfind][idim];
      }
      f[i][surfdim] += load;
      ++nfind;
    }
  }

return;
} // end post_force(int)

/* ----------------------------------------------------------------------
   return components of total force on fix group before force was changed
------------------------------------------------------------------------- */

double FixGFMD::compute_vector(int n)
{
  if (collect_flag == 0){
    MPI_Allreduce(foriginal,foriginal_all,sysdim,MPI_DOUBLE,MPI_SUM,world);
    collect_flag = 1;
  }

return foriginal_all[n];
}

/* ----------------------------------------------------------------------
 * private method, to get the analytic elastic stiffness coefficients
 * for simple cubic systems
 * --------------------------------------------------------------------*/

void FixGFMD::Phi_analytic()
{
  if (me == 0) fprintf(gfmdlog,"\nPhi_q is computed for simple cubic system.\n");
  if (nucell !=1) error->all(FLERR,"Analytical Phi works only for simple cubic system");

  double qx, qy;
  std::complex<double> **G_q;
  memory->create(G_q,sysdim,sysdim, "FixGFMD:G_q");

  if (sysdim == 2){    // (1+1) D system
    idq = 0;
    for (int i = nxlo; i <= nxhi; ++i){
      qx = (i <= (nx/2)) ? (2.0*M_PI*i/nx) : (2.0*M_PI*(i-nx)/nx);

      Phi_analytic_2D(qx, G_q);
      int ndim = 0;
      for (int idim = 0; idim < sysdim; ++idim)
      for (int jdim = 0; jdim < sysdim; ++jdim) Phi_q[idq][ndim++] = G_q[idim][jdim];

      ++idq;
    }
  } else {               // (2+1) D system
    idq = 0;
    for (int i = nxlo; i <= nxhi; ++i){
      qx = (i <= int((nx)/2)) ? (2.0*M_PI*(i)/nx) : (2.0*M_PI*(i-nx)/nx);
      for (int j=0; j<ny; j++){
        qy = (j <= int((ny)/2)) ? (2.0*M_PI*(j)/ny) : (2.0*M_PI*(j-ny)/ny);

        Phi_analytic_3D(qx, qy, G_q);
        int ndim = 0;
        for (int idim = 0; idim < sysdim; ++idim)
        for (int jdim = 0; jdim < sysdim; ++jdim) Phi_q[idq][ndim++] = G_q[idim][jdim];
        ++idq;
      }
    }
  } // end of if (sysdim ==
  memory->destroy(G_q);

return;
}

/* ----------------------------------------------------------------------
 * private method, to get the analytic elastic stiffness coefficients
 * for 1+1 dimentional system.
 * --------------------------------------------------------------------*/
void FixGFMD::Phi_analytic_2D(double qx, std::complex<double>** Gq)
{
  if (qx == 0) qx = 1.e-6;
  double cx = cos(qx);
  double D  = sqrt(3.0-cx) - 2.0/(sqrt(5.0 + cx)) * ( (cx+3)/ ( 2.0*sqrt(3.0)+sqrt((5.0+cx)*(3.0-cx)) ) )  + sqrt(1.0-cx);
 
  Gq[0][0] = std::complex<double> ( (1.0/(2.0*sqrt(1-cx)))*1.0/D, 0.0);
  Gq[0][1] = std::complex<double> ( 0.0, - (sin(qx) / ( 2.0*(1.0 - cx)) * ( 1.0/( sqrt(5.0 +cx))) * ( (3.0 + cx)
                               /( 2.0*sqrt(3.0)+sqrt((5.0+cx)*(3.0-cx)) ) )  * 1.0/D) );
  Gq[1][0] = -Gq[0][1];
  Gq[1][1] = std::complex<double>( (1.0/(2.0*sqrt(1.0-cx)) * sqrt(3.0/(5.0+cx))*( sqrt(3.0-cx) + sqrt(1.0-cx) + sqrt( (1.0-cx)/3.0)
                             *(  (3.0 + cx)/( 2.0*sqrt(3.0)+sqrt((5.0+cx)*(3.0-cx)) ) ) ) * 1.0/D), 0.0);
  GaussJordan(sysdim,Gq[0]);

return;
}

/* ----------------------------------------------------------------------
 * private method, to get the analytic stiffness coefficients for
 * 2+1 dimentional simple cubic system.
 * --------------------------------------------------------------------*/

void FixGFMD::Phi_analytic_3D(double qx, double qy, std::complex<double>** Gq )
{
  std::complex<double> U_q[sysdim][sysdim];
  std::complex<double> F_q[sysdim][sysdim];

  if (qx == 0) qx = 1.e-6;
  if (qy == 0) qy = 1.e-6;
  if ( (fabs(qx) == M_PI) && (fabs(qy) == M_PI) )  qx=qy=M_PI-1.e-6;

  double c_x = cos(qx);
  double c_y = cos(qy);

  double s_x = sin(qx);
  double s_y = sin(qy);

  double c_plus = 1.0+ c_x + c_y - 2.*c_x*c_y;
  double c_minus = 4.0*sin(qx/2.0)*sin(qy/2.0)*sqrt(1.0-c_x*c_y);

  double s_plus = sqrt(  ( sqrt(pow ( (pow(c_plus,2.0)-pow(c_minus,2.0)-1.),2.0) + 4.*pow(c_plus,2.0)*pow(c_minus,2.0) )
            + ( pow(c_plus,2.0)-pow(c_minus,2.0)-1.) )  / 2.0 )*SIGN(c_plus);
  double s_minus = sqrt( ( sqrt( pow((pow(c_plus,2.0)-pow(c_minus,2.0)-1.),2.0) + 4.*pow(c_plus,2.0)*pow(c_minus,2.0) )
            - ( pow(c_plus,2.0)-pow(c_minus,2.0)-1.) ) /2.0 )*SIGN(c_minus);

  double e_plus = c_plus - s_plus;
  double e_minus = c_minus - s_minus;
  double e_one = (c_x + c_y +c_x*c_y)/(4.-c_x*c_y+sqrt((4.+c_x+c_y)*( 4.-c_x-c_y-2.*c_x*c_y)));

  U_q[0][0] = std::complex<double>(sin(qx/2.0)*cos(qy/2.0)/(c_x+c_x*c_y-2.) ,0.);
  U_q[1][0] = std::complex<double>(-1.*cos(qx/2.0)*sin(qy/2.0)/(c_y + c_x*c_y - 2.),0.);
  U_q[2][0] = std::complex<double>(cos(qx/2.0)*cos(qy/2.0)*(c_x-c_y)/((c_y + c_x*c_y - 2.)*(c_x + c_x*c_y - 2.)),0.);

  real(U_q[2][0]) = real(U_q[2][0])*(sqrt((4.-c_x-c_y-2.*c_x*c_y)/(4. + c_x + c_y)));

  U_q[0][1] = std::complex<double>(s_x/(2.*(1.-c_x+pow(c_x,2.0)-c_x*c_y)),0.);
  U_q[1][1] = std::complex<double>(s_y/(2.*(1.-c_y+pow(c_y,2.0)-c_x*c_y)),0.);
  U_q[2][1] = std::complex<double>(  ( (2.-2.*c_x*c_y*c_plus-pow(c_minus,2.0))*s_plus + (c_minus*(c_plus-2.*c_x*c_y))*s_minus)
            /( pow((2.-2.*c_x*c_y*c_plus-pow(c_minus,2.0)),2.0) + pow(c_minus*(c_plus-2.*c_x*c_y),2.0)),0.);

  U_q[0][2] = std::complex<double> (c_x*cos(qx/2.)*sin(qy/2.)/(sqrt(1.-c_x*c_y)*(1.-c_x+pow(c_x,2.)-c_x*c_y)), 0.);
  U_q[1][2] = std::complex<double> (c_y*sin(qx/2.)*cos(qy/2.)/(sqrt(1.-c_x*c_y)*(1.-c_y+pow(c_y,2.)-c_x*c_y)), 0.);

  U_q[2][2] = std::complex<double> (((2.-2.*c_x*c_y*c_plus-pow(c_minus,2.0))*s_minus -(c_minus*(c_plus-2.*c_x*c_y))*s_plus)/
           (pow((2.-2.*c_x*c_y*c_plus-pow(c_minus,2.0)),2.0) +pow((c_minus*(c_plus-2.*c_x*c_y)),2.0) ),0.);

  F_q[0][0] = std::complex<double> ((5.-2.*c_x-2.*c_x*c_y-e_one*c_x)*real(U_q[0][0]) + 2*s_x*s_y*real(U_q[1][0]) + s_x*e_one*real(U_q[2][0]),0.);
  F_q[0][1] = std::complex<double> ((5.-2.*c_x-2.*c_x*c_y-e_plus*c_x)*real(U_q[0][1]) + 2*s_x*s_y*real(U_q[1][1]) + c_x*e_minus*real(U_q[0][2])
           + s_x*(e_plus*real(U_q[2][1])-e_minus*real(U_q[2][2])),0.);
  F_q[0][2] = std::complex<double> ((5.-2.*c_x-2.*c_x*c_y-e_plus*c_x)*real(U_q[0][2]) + 2*s_x*s_y*real(U_q[1][2]) - c_x*e_minus*real(U_q[0][1])
           + s_x*(e_minus*real(U_q[2][1])+e_plus*real(U_q[2][2])),0.);
  F_q[1][0] = std::complex<double> (2.*s_x*s_y*real(U_q[0][0]) + (5.-2.*c_y-2.*c_x*c_y-e_one*c_y)*real(U_q[1][0]) + s_y*e_one*real(U_q[2][0]),0.);
  F_q[1][1] = std::complex<double> (2.*s_x*s_y*real(U_q[0][1]) + (5.-2.*c_y-2.*c_x*c_y-e_plus*c_y)*real(U_q[1][1]) + e_minus*c_y*real(U_q[1][2])
           +  s_y*(e_plus*real(U_q[2][1])-e_minus*real(U_q[2][2])),0.);
  F_q[1][2] = std::complex<double> (2.*s_x*s_y*real(U_q[0][2]) + (5.-2.*c_y-2.*c_x*c_y-e_plus*c_y)*real(U_q[1][2]) -
           e_minus*c_y*real(U_q[1][1]) + s_y*(e_minus*real(U_q[2][1])+e_plus*real(U_q[2][2])),0.);
  F_q[2][0] = std::complex<double> (-s_x*e_one*real(U_q[0][0])-s_y*e_one*real(U_q[1][0]) + real(U_q[2][0])*(3.-(1.+c_x+c_y)*e_one),0.);

  F_q[2][1] = std::complex<double> (-s_x*(e_plus*real(U_q[0][1])-e_minus*real(U_q[0][2])) - s_y*(e_plus*real(U_q[1][1])-e_minus*real(U_q[1][2]))
           + (3.-(1.+c_x+c_y)*e_plus)*real(U_q[2][1]) + (1.+c_x+c_y)*e_minus*real(U_q[2][2]),0.);

  F_q[2][2] = std::complex<double> (-s_x*(e_minus*real(U_q[0][1])+e_plus*real(U_q[0][2]))-s_y*(e_plus*real(U_q[1][2])+e_minus*real(U_q[1][1])) +
           real(U_q[2][2])*(3.-(1.+c_x+c_y)*e_plus) - e_minus*(1.+c_x+c_y)*real(U_q[2][1]),0.);

  GaussJordan(sysdim,F_q[0]);
  MatMulMat(sysdim,U_q[0],F_q[0],Gq[0]);

  Gq[0][2] = std::complex<double>(0,real(Gq[0][2]));
  Gq[2][0] = std::complex<double>(0,-1.0*real(Gq[2][0]));
  Gq[1][2] = std::complex<double>(0,real(Gq[1][2]));
  Gq[2][1] = std::complex<double>(0,-1.0*real(Gq[2][1]));

  GaussJordan(sysdim,Gq[0]);

return;
}

/* ----------------------------------------------------------------------
 * private method, to read the equilibrium positions of atoms in GFMD 
 * layer; usually used when you restart from previous run so as to 
 * correctly evaluate the dispacement
 * --------------------------------------------------------------------*/
void FixGFMD::readxorg()
{
  int info = 0, indx;
  char strtmp[MAXLINE];
  double xcur[3];
  FILE *fp;

  fp = fopen(file_xorg, "r");
  if (fp == NULL){
    char str[MAXLINE];
    sprintf(str,"Cannot open original position file: %s", file_xorg);
    if (me == 0){ fprintf(gfmdlog,"\n%s\n",str); fflush(gfmdlog); }
    error->one(FLERR,str);
  }
  
  xcur[2] = 0.;
  for (int i = 0; i < nGFatoms; ++i){
    if (fgets(strtmp,MAXLINE,fp) == NULL){info = 1; break;}
    if (sysdim == 2) sscanf(strtmp,"%d %d %lg %lg", &indx, &itag, &xcur[0], &xcur[1]);
    else sscanf(strtmp,"%d %d %lg %lg %lg", &indx, &itag, &xcur[0], &xcur[1], &xcur[2]);
     
    if (itag < 1 || itag > static_cast<int>(atom->natoms)) {info = 2; break;} // 1 <= itag <= natoms
    for (int idim = 0; idim < sysdim; ++idim) UIrAll[i][idim] = xcur[idim];
    UIrAll[i][sysdim] = itag;
  }
  fclose(fp);

  if (info){
    char str[MAXLINE];
    sprintf(str,"Error while reading initial configuration from file: %s",file_xorg);
    error->one(FLERR,str);
  }
  if (me == 0) fprintf(gfmdlog, "\nOriginal positions of GF atoms are read from file: %s\n", file_xorg);

return;
}

/* ----------------------------------------------------------------------
 * private method, to extract the equilibrium positions of atoms in GFMD
 * layer from the initial configuration.
 * For restart run, the initial configuration frequently does not
 * corresponds to the equilibrium position, one should provide the 
 * informaiton by file, instead.
 * --------------------------------------------------------------------*/
void FixGFMD::compxorg()
{
  double **x = atom->x;
  int *mask  = atom->mask;
  int *tag   = atom->tag;
  int *image = atom->image;
  int nlocal = atom->nlocal;

  double xcur[3];
  int i, idim;

  nfind = 0;
  for (i = 0; i < nlocal; ++i){
    if (mask[i] & groupbit){
      domain->unmap(x[i], image[i], xcur);

      for (idim=0; idim<sysdim; idim++) UIrLoc[nfind][idim] = xcur[idim];
      UIrLoc[nfind++][sysdim] = tag[i];
    }
  }
  nfind *= (sysdim+1);

  displs[0] = 0;
  MPI_Allgather(&nfind, 1, MPI_INT,recvcnts,1,MPI_INT,world);
  for (i = 1; i < nprocs; ++i) displs[i] = displs[i-1] + recvcnts[i-1];

  MPI_Allgatherv(UIrLoc[0],nfind,MPI_DOUBLE,UIrAll[0],recvcnts,displs,MPI_DOUBLE,world);

  if (me == 0) fprintf(gfmdlog,"\nOriginal positions of GF atoms determined from initial configuraiton!\n");

return;
}

/* ----------------------------------------------------------------------
 * private method, to read the mapping info from file.
 * --------------------------------------------------------------------*/

void FixGFMD::readmap()
{
  int info = 0;
  char strtmp[MAXLINE];
  FILE *fp;

  fp = fopen(mapfile, "r");
  if (fp == NULL){
    char str[MAXLINE];
    sprintf(str,"Cannot open mapping file: %s", mapfile);
    if (me == 0){ fprintf(gfmdlog,"\n%s\n",str); fflush(gfmdlog); }
    error->one(FLERR,str);
  }

  // first line carries nx,ny and nucell; for (1+1)D system, ny = 1
  if (fgets(strtmp,MAXLINE,fp) == NULL){
    char str[MAXLINE];
    sprintf(str,"Error %d while reading header info from file: %s",info,mapfile);
    error->one(FLERR,str);
  }
  nx = atoi(strtok(strtmp, " \n\t\r\f"));
  ny = atoi(strtok(NULL,   " \n\t\r\f"));
  nucell = atoi(strtok(NULL,   " \n\t\r\f"));
  if (nx*ny*nucell != nGFatoms) error->all(FLERR,"Number of atoms from FFT mesh and group mismatch");

  if (fgets(strtmp,MAXLINE,fp) == NULL){ // second line of mapfile is comment
    char str[MAXLINE];
    sprintf(str,"Error %d while reading comment line from file: %s",info,mapfile);
    error->one(FLERR,str);
  }

  int ix, iy, iu;
  for (int i = 0; i < nGFatoms; ++i){
    if (fgets(strtmp,MAXLINE,fp) == NULL){info = 1; break;}
    ix = atoi(strtok(strtmp, " \n\t\r\f"));
    iy = atoi(strtok(NULL,   " \n\t\r\f"));
    iu = atoi(strtok(NULL,   " \n\t\r\f"));
    itag = atoi(strtok(NULL,   " \n\t\r\f"));
    // check if index is in correct range
    if (ix < 0 || ix >= nx || iy < 0 || iy >= ny || iu < 0 || iu >= nucell) {info = 2; break;} // check if index is in correct range
    if (itag < 1 || itag > static_cast<int>(atom->natoms)) {info = 3; break;}      // 1 <= itag <= natoms
    idx = (ix*ny+iy)*nucell + iu;
    tag2surf[itag] = idx;
    surf2tag[idx]  = itag;
  }
  fclose(fp);

  if (surf2tag.size() != tag2surf.size() || tag2surf.size() != static_cast<std::size_t>(nGFatoms)) info = 4;
  if (info != 0){
    char str[MAXLINE];
    sprintf(str,"Error %d while reading mapping info from file: %s",info,mapfile);
    error->one(FLERR,str);
  }

  // check the correctness of mapping
  int *mask  = atom->mask;
  int *tag   = atom->tag;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit){
      itag = tag[i];
      idx  = tag2surf[itag];
      if (itag != surf2tag[idx]) error->one(FLERR,"Atom in group not found in mapping info");
    }
  }
  origin_tag = surf2tag[0];
  if (me == 0) fprintf(gfmdlog,"\nMapping info read from file  : %s\n\n", mapfile);

return;
}

/* ----------------------------------------------------------------------
 * private method, to calculate the mapping info fo surface atoms.
 * For restart run, usually it is not possible to calculate this from
 * the initial configuration; instead, one would better provide the
 * mapping information by file.
 * --------------------------------------------------------------------*/

void FixGFMD::compmap(int flag)
{

  if ((flag & 3) != 3){
    // Check if lattice information is available from input file!
    if (domain->lattice == NULL){ // get surface vectors from lattice info; orthogonal lattice is assumed
      error->all(FLERR,"No lattice defined while keyword su and/or sv not set");
    }
    if ((flag & 1) == 0){
      surfvec[0][0] = domain->lattice->xlattice;
      surfvec[0][1] = 0.0;
    }
    if ((flag & 2) == 0){
      surfvec[1][0] = 0.0;
      surfvec[1][1] = domain->lattice->ylattice;
    }
  }
  // check the validity of the surface vectors read from command line
  if (fabs(surfvec[0][1]) > 0.0 && fabs(surfvec[1][0]) > 0.0)
    error->all(FLERR,"Either U or V must be on the box side");
  if (surfvec[0][0] <= 0.0)
    error->all(FLERR,"Surface vector U must be along the +x direction");
  if (surfvec[1][1] <= 0.0)
    error->all(FLERR,"Surface vector V must point to the +y direction");

  double invSurfV[2][2];
  for (int i = 0; i < 2; ++i){ invSurfV[i][0] = surfvec[i][0]; invSurfV[i][1] = surfvec[i][1];}

  // get the inverse transpose of surfvec
  GaussJordan(2,invSurfV[0]);
  double dswap = invSurfV[0][1];
  invSurfV[0][1] = invSurfV[1][0];
  invSurfV[1][0] = dswap;

  // get FFT dimensions
  nx = int(domain->xprd*invSurfV[0][0]+0.1);
  ny = (sysdim == 2)?1:int(domain->yprd*invSurfV[1][1]+0.1);
  if (nx < 1 || nx > nGFatoms || ny < 1 || ny > nGFatoms) error->all(FLERR,"Error encountered while getting FFT dimensions");

  nucell = nGFatoms / (nx*ny);
  if (nucell > 2) error->all(FLERR,"Mapping info cannot be computed for nucell > 2");

  int ix, iy, iu;
  double vx[2], vi[2], SurfOrigin[2];

  if (origin_tag > 0){
    nfind = 0;
    for (int i = 0; i < nGFatoms; ++i){
      if (static_cast<int>(UIrAll[i][sysdim]) == origin_tag){
        SurfOrigin[0] = UIrAll[i][0];
        SurfOrigin[1] = UIrAll[i][1];
        nfind++;
        break;
      }
    }
    if (nfind < 1) error->all(FLERR,"Surface origin given by user not found");

  } else {
    SurfOrigin[0] = UIrAll[0][0];
    SurfOrigin[1] = UIrAll[0][1];
  }

  // now to calculate the mapping info
  for (int i = 0; i < nGFatoms; ++i) {
    vx[0] = UIrAll[i][0]-SurfOrigin[0]; // relative coordination on the surface of atom i
    vx[1] = UIrAll[i][1]-SurfOrigin[1];

    // to get the fractional coordination of atom i with the basis of surface vectors
    MatMulVec(2,invSurfV[0],vx,vi);

    ix = (int)floor(vi[0]+0.1); ix %= nx; ix = (ix < 0)? (ix+nx):ix;
    iy = (int)floor(vi[1]+0.1); iy %= ny; iy = (iy < 0)? (iy+ny):iy;
    iu = ((int)floor(vi[0]+vi[0]+0.1))%2 | ((int)floor(vi[1]+vi[1]+0.1))%2;

    itag = static_cast<int>(UIrAll[i][sysdim]);
    idx  = (ix*ny+iy)*nucell+abs(iu);
    tag2surf[itag] = idx;
    surf2tag[idx]  = itag;
  }
  if (tag2surf.size() != surf2tag.size() || (int)tag2surf.size() != nGFatoms)
    error->one(FLERR,"Error encountered while computing mapping info");

  // check the correctness of mapping
  int *mask  = atom->mask;
  int *tag   = atom->tag;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit){
      itag = tag[i];
      idx  = tag2surf[itag];
      if (itag != surf2tag[idx]) error->one(FLERR,"Atom in group not found in mapping info");
    }
  }
  origin_tag = surf2tag[0];
  if (me == 0) fprintf(gfmdlog,"\nMapping info computed from initial configuration.\n");

return;
}

/* ----------------------------------------------------------------------
 * private method, to read the Phi matrix from binary file.
 * Interpolation is done with bilinear for elements near the boundary
 * and bicubic elsewhere. 
 * --------------------------------------------------------------------*/
void FixGFMD::readphi()
{
  int  Nx, Ny, Nucell, idim, ndim;
  double boltz, old2new = 1.;
  double svec_gfc[2][2], sb_gfc[fft_dim];
  FILE *gfc_in;
  std::size_t nread;

  gfc_in = fopen(file_phi, "rb");
  if (gfc_in == NULL) error->all(FLERR,"Error while opening binary Phi file");

  nread = fread(&ndim,  sizeof(int),   1, gfc_in);
  if (nread != 1) error->one(FLERR,"Error while reading ndim from binary file");
  nread = fread(&Nx,    sizeof(int),   1, gfc_in);
  if (nread != 1) error->one(FLERR,"Error while reading Nx from binary file");
  nread = fread(&Ny,    sizeof(int),   1, gfc_in);
  if (nread != 1) error->one(FLERR,"Error while reading Ny from binary file");
  nread = fread(&Nucell,sizeof(int),   1, gfc_in);
  if (nread != 1) error->one(FLERR,"Error while reading Nucell from binary file");
  nread = fread(&boltz, sizeof(double),1, gfc_in);
  if (nread != 1) error->one(FLERR,"Error while reading boltz from binary file");
 
  if (ndim != sysdim){
    char str[MAXLINE];
    sprintf(str,"System dimension from GFC is %d, current is %d", ndim, sysdim);
    if (me == 0){ fprintf(gfmdlog,"\n%s\n",str); fflush(gfmdlog); }
    error->all(FLERR,str);
  }
  if (Nucell != nucell){
    char str[MAXLINE];
    sprintf(str,"# of atom per cell from GFC is %d, current is %d", Nucell, nucell);
    if (me == 0){ fprintf(gfmdlog,"\n%s\n",str); fflush(gfmdlog); }
    error->all(FLERR,str);
  }
  if (boltz != force->boltz){
    char str[MAXLINE];
    if (boltz == 1.){
      sprintf(str,"Units used by GFC were LJ, conversion is not possible");
      if (me == 0){ fprintf(gfmdlog,"\n%s\n",str); fflush(gfmdlog); }
      error->all(FLERR,str);
    } else {
      old2new = force->boltz / boltz;
      sprintf(str,"Units used by GFC differ from current one, converted!");
      if (me == 0) fprintf(gfmdlog,"\n%s\n",str);
      error->warning(FLERR,str);
    }
  }

  if (me == 0){
    fprintf(gfmdlog,"\nTo read  Phi_q info from file: %s\n", file_phi);
    fprintf(gfmdlog,"FFT mesh from GFC measurement: %d x %d\n", Nx, Ny);
    fprintf(gfmdlog,"FFT mesh from the current run: %d x %d\n", nx, ny);
  }

  if (nx == Nx && ny == Ny){ // Dimension from GFC run and current match, read in Phi directly
    std::complex<double> cmdum;
    for (idq=0; idq<nxlo*ny; idq++){
      for (idim=0; idim<fft_dim2; idim++){
        nread = fread(&cmdum, sizeof(std::complex<double>), 1, gfc_in);
        if (nread != 1) error->one(FLERR,"Error while reading Phi from binary file");
      }
    }
    nread = fread(Phi_q[0], sizeof(std::complex<double>), mynq*fft_dim2, gfc_in);
    if (nread != static_cast<std::size_t>(mynq*fft_dim2)) error->one(FLERR,"Error while reading Phi from binary file");
    if (me == 0) fprintf(gfmdlog,"\nPhi_q read successfully from file: %s\n", file_phi);

    if (nxlo==0 && mynq>0) Phi_q0_ASR(Phi_q[0]);

  } else { // Dimension from GFC run and current mismatch, interpolation needed!

    std::complex<double> **Phi_in;  // read in Phi_q from file
    memory->create(Phi_in,Nx*Ny, fft_dim2, "fix_gfmd:Phi_in");
    idq = 0;
    for (int i = 0; i < Nx; ++i)
    for (int j = 0; j < Ny; ++j){
      for (idim = 0; idim < fft_dim2; ++idim){
        nread = fread(&Phi_in[idq][idim], sizeof(std::complex<double>), 1, gfc_in);
        if (nread != 1) error->one(FLERR,"Error while reading Phi from binary file");
      }
      ++idq;
    }

    Phi_q0_ASR(Phi_in[0]);

    /* Now to interpolate the Phi_q we need!
       Bicubic interpolation is used. */
    int ix, iy, xP1, xM1, yP1, yM1, Idx, Ix, Iy;
    int ppx, ppy, pmx, pmy, mpx, mpy, mmx, mmy;
    double dx1, dx2, facx, facy;
    std::complex<double> y[4], y1[4], y2[4], y12[4];
    std::complex<double> *Phi_p1, *Phi_p2, *Phi_p12;
    Phi_p1  = new std::complex<double>[(Nx+1)*(Ny+1)];
    Phi_p2  = new std::complex<double>[(Nx+1)*(Ny+1)];
    Phi_p12 = new std::complex<double>[(Nx+1)*(Ny+1)];
    dx1 = double(Nx)/2.;
    dx2 = double(Ny)/2.;
    
    for (idim = 0; idim < fft_dim2; ++idim){
      // get the gradients by finite element method
      for (Ix = 0; Ix <= Nx; ++Ix)
      for (Iy = 0; Iy <= Ny; ++Iy){
        int Cx = Ix%Nx, Cy = Iy%Ny;

        xP1 = (Cx+1)%Nx;
        xM1 = (Nx+Cx-1)%Nx;
        yP1 = (Cy+1)%Ny;
        yM1 = (Ny+Cy-1)%Ny;
        ppx = pmx = xP1;
        ppy = mpy = yP1;
        mpx = mmx = xM1;
        pmy = mmy = yM1;
        facx = 1.;
        facy = 1.;
       
        if (Ix == 0){
          xM1  = Cx;
          facx = 2.;
          mpx  = mmx = Cx;
          mpy  = mmy = Cy;

        } else if (Ix == Nx){
          xP1  = Cx;
          facx = 2.;
          ppx = pmx = Cx;
          ppy = pmy = Cy;
        }

        if (Iy == 0){
          yM1  = Cy;
          facy = 2.;
          pmx  = mmx = Cx;
          pmy  = mmy = Cy;

        } else if (Iy == Ny){
          yP1  = Cy;
          facy = 2.;
          ppx  = mpx = Cx;
          ppy  = mpy = Cy;
        }

        Idx = Ix * (Ny+1) + Iy;
        Phi_p1 [Idx] = (Phi_in[xP1*Ny+Cy ][idim] - Phi_in[xM1*Ny+Cy][idim]) * dx1 * facx;
        Phi_p2 [Idx] = (Phi_in[Cx*Ny+yP1 ][idim] - Phi_in[Cx*Ny+yM1][idim]) * dx2 * facy;
        Phi_p12[Idx] = (Phi_in[ppx*Ny+ppy][idim] - Phi_in[pmx*Ny+pmy][idim]
                     -  Phi_in[mpx*Ny+mpy][idim] + Phi_in[mmx*Ny+mmy][idim]) * dx1 * dx2 * facx * facy;
      }

      // to do interpolation
      idq = 0;
      for (ix = nxlo; ix <= nxhi; ++ix)
      for (iy = 0; iy < ny; ++iy){
        Ix = (int)(double(ix)/double(nx)*Nx);
        Iy = (int)(double(iy)/double(ny)*Ny);
        xP1 = (Ix+1)%Nx;
        yP1 = (Iy+1)%Ny;

        y[0] = Phi_in[Ix*Ny+Iy][idim];
        y[1] = Phi_in[xP1*Ny+Iy][idim];
        y[2] = Phi_in[xP1*Ny+yP1][idim];
        y[3] = Phi_in[Ix*Ny+yP1][idim];

        xP1   = Ix+1;
        yP1   = Iy+1;
        y1[0] = Phi_p1[Ix *(Ny+1)+Iy ];
        y1[1] = Phi_p1[xP1*(Ny+1)+Iy ];
        y1[2] = Phi_p1[xP1*(Ny+1)+yP1];
        y1[3] = Phi_p1[Ix *(Ny+1)+yP1];

        y2[0] = Phi_p2[Ix *(Ny+1)+Iy];
        y2[1] = Phi_p2[xP1*(Ny+1)+Iy];
        y2[2] = Phi_p2[xP1*(Ny+1)+yP1];
        y2[3] = Phi_p2[Ix *(Ny+1)+yP1];

        y12[0] = Phi_p12[Ix *(Ny+1)+Iy];
        y12[1] = Phi_p12[xP1*(Ny+1)+Iy];
        y12[2] = Phi_p12[xP1*(Ny+1)+yP1];
        y12[3] = Phi_p12[Ix *(Ny+1)+yP1];

        bicuint(y, y1, y2, y12, double(Ix)/Nx, double(Ix+1)/Nx, double(Iy)/Ny,
                double(Iy+1)/Ny, double(ix)/nx, double(iy)/ny, &Phi_q[idq][idim]);
        idq++;
      } // end of interpolation on current idim
    } // end of for (idim ...
     
    delete []Phi_p1;
    delete []Phi_p2;
    delete []Phi_p12;
    memory->destroy(Phi_in);
    if (me == 0) fprintf(gfmdlog,"\nPhi_q interpolated from file: %s\n", file_phi);
  }

  // to read surface lattice info from gfc measurements
  // in old versions, this info is not stored.
  int info_read = 1;
  nread = fread(svec_gfc[0],sizeof(double),4,gfc_in);
  if (nread != 4){
    error->warning(FLERR,"Failed to read surface vector info from binary file");
    info_read = 0;
  }
  nread = fread(sb_gfc,sizeof(double),fft_dim, gfc_in);
  if (nread != static_cast<std::size_t>(fft_dim) ){
    error->warning(FLERR,"Failed to read surface basis info from binary file");
    info_read = 0;
  }
  fclose(gfc_in);

  // unit conversion for the elastic stiffness coefficients
  if (old2new != 1.){
    for (idq = 0; idq < mynq; ++idq)
    for (idim = 0; idim < fft_dim2; ++idim) Phi_q[idq][idim] *= old2new;
  }

  // compare equilibrium surface lattice based on gfc measurement and this run
  if (me == 0 && info_read){
    double sb_eq[fft_dim],d2o[sysdim];
    for (idim = 0; idim < fft_dim; ++idim) sb_eq[idim] = 0.;
    for (int ix = 0; ix < nx; ++ix)
    for (int iy = 0; iy < ny; ++iy){
      int idx = (ix*ny+iy)*nucell;
      for (int iu = 1; iu < nucell; ++iu){
        for (idim = 0; idim < sysdim; ++idim) d2o[idim] = xeq[idx+iu][idim] - xeq[idx][idim];
        domain->minimum_image(d2o);
        ndim = iu*sysdim;
        for (idim = 0; idim < sysdim; ++idim) sb_eq[ndim+idim] += d2o[idim];
      }
    }
    for (idim = sysdim; idim < fft_dim; ++idim) sb_eq[idim] /= nx*ny;

    fprintf(gfmdlog,"\nSurface vector from this run: [%lg %lg], [%lg %lg]\n", surfvec[0][0], surfvec[0][1], surfvec[1][0], surfvec[1][1]);
    fprintf(gfmdlog,"Surface vector from gfc  run: [%lg %lg], [%lg %lg]\n", svec_gfc[0][0], svec_gfc[0][1], svec_gfc[1][0], svec_gfc[1][1]);
    fprintf(gfmdlog,"Surface basis  from this run: ");
    for (idim=0; idim<fft_dim; idim++) fprintf(gfmdlog,"%lg ",sb_eq[idim]);
    fprintf(gfmdlog,"\nSurface basis  from gfc  run: ");
    for (idim=0; idim<fft_dim; idim++) fprintf(gfmdlog,"%lg ",sb_gfc[idim]);
    fprintf(gfmdlog, "\n");
  }
  // to reset xeq if required
  if (reset_xeq && info_read){
    double lx2now, ly2now, lx2gfc, ly2gfc, xs[3], dx2, dy2;
    lx2now = surfvec[0][0]*surfvec[0][0] + surfvec[0][1]*surfvec[0][1];
    ly2now = surfvec[1][0]*surfvec[1][0] + surfvec[1][1]*surfvec[1][1];
    lx2gfc = svec_gfc[0][0]*svec_gfc[0][0] + svec_gfc[0][1]*svec_gfc[0][1];
    ly2gfc = svec_gfc[1][0]*svec_gfc[1][0] + svec_gfc[1][1]*svec_gfc[1][1];
    dx2 = fabs(lx2now-lx2gfc); dy2 = fabs(ly2now-ly2gfc);
    if (dx2 > lx2now*2.5e-3 || dy2 > ly2now*2.5e-3){
      if (me == 0) error->warning(FLERR,"Surface lattice from gfc and this run mismatch, xeq not reset!");
      return;
    }
    xs[0] = sqrt(lx2now/lx2gfc); xs[1] = sqrt(ly2now/ly2gfc);
    xs[2] = sqrt((lx2now+ly2now)/(lx2gfc+ly2gfc));
    for (idim = 0; idim < fft_dim; ++idim) sb_gfc[idim] *= xs[idim%sysdim];

    idx = 0;
    for (int ix = 0; ix < nx; ++ix)
    for (int iy = 0; iy < ny; ++iy){
      ndim = 0;
      for (int iu = 0; iu < nucell; ++iu){
        if (reset_xeq == 3) xeq[idx][0] = double(ix)*svec_gfc[0][0]+double(iy)*svec_gfc[1][0]+sb_gfc[ndim];
        if (reset_xeq == 3 || sysdim == 2) xeq[idx][1] = double(iy)*svec_gfc[1][1]+sb_gfc[ndim+1];
        if (sysdim == 3) xeq[idx][2] = sb_gfc[ndim+2];
        ++idx;
        ndim += sysdim;
      }
    }
    if (me == 0) fprintf(gfmdlog,"\nEquilibrium positions reset based on surface lattice info from: %s\n",file_phi);
  }

return;
}

/* ----------------------------------------------------------------------
 * private method, to get the coefficient for bicubic interpolation
 * --------------------------------------------------------------------*/

void FixGFMD::bcucof(std::complex<double> *y, std::complex<double> *y1, std::complex<double> *y2,
                     std::complex<double> *y12, double d1, double d2, std::complex<double> *c)
{
  int i, j;
  double d1d2;
  std::complex<double> x[16], xx;
  const double wt[16][16] = {1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0, -3,0,0,3,0,0,0,0,-2,0,0,-1,0,0,0,0,
    2,0,0,-2,0,0,0,0,1,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0, 0,0,0,0,-3,0,0,3,0,0,0,0,-2,0,0,-1,
    0,0,0,0,2,0,0,-2,0,0,0,0,1,0,0,1, -3,3,0,0,-2,-1,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,-3,3,0,0,-2,-1,0,0, 9,-9,9,-9,6,3,-3,-6,6,-6,-3,3,4,2,1,2,
    -6,6,-6,6,-4,-2,2,4,-3,3,3,-3,-2,-1,-1,-2, 2,-2,0,0,1,1,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,2,-2,0,0,1,1,0,0, -6,6,-6,6,-3,-3,3,3,-4,4,2,-2,-2,-2,-1,-1,
    4,-4,4,-4,2,2,-2,-2,2,-2,-2,2,1,1,1,1};
  /*-------------------------------------------------------------------*/
  d1d2 = d1 * d2;
  for (i = 0; i < 4; ++i){
    x[i]   = y[i];
    x[i+4] = y1[i] * d1;
    x[i+8] = y2[i] * d2;
    x[i+12]= y12[i]* d1d2;
  }
  for (i = 0; i < 16; ++i){
    xx = 0.;
    for (j = 0; j < 16; ++j) xx += wt[i][j] * x[j];
    c[i] = xx;
  }

return;
}

/* ----------------------------------------------------------------------
 * To do bicubic interpolation. Adapted from Numerical Recipes in Fortran
 * --------------------------------------------------------------------*/

void FixGFMD::bicuint(std::complex<double> *y, std::complex<double> *y1, std::complex<double> *y2,
                      std::complex<double> *y12, double x1l, double x1u, double x2l, 
                      double x2u, double x1, double x2, std::complex<double> *ansy)
{
  int i;
  double t, u;
  std::complex<double> c[4][4];

  bcucof(&y[0],&y1[0], &y2[0], &y12[0], x1u-x1l, x2u-x2l, &c[0][0]);

  t = (x1-x1l)/(x1u-x1l);
  u = (x2-x2l)/(x2u-x2l);

  *ansy = 0.;
  for (i = 3; i >= 0; --i) *ansy = t * (*ansy) + (c[i][3]*u+c[i][2]*u+c[i][1])*u + c[i][0];

return;
}

/* ----------------------------------------------------------------------
 * Private method, to apply acoustic sum rule on the Phi matrix at q = 0.
 * --------------------------------------------------------------------*/

void FixGFMD::Phi_q0_ASR(std::complex<double> * Phi_q0)
{
  if (nasr < 1) return;

  for (int iit = 0; iit < nasr; ++iit){
    // simple ASR; the resultant matrix might not be symmetric
    for (int a = 0; a < sysdim; ++a)
    for (int b = 0; b < sysdim; ++b){
      for (int k = 0; k < nucell; ++k){
        double sum = 0.;
        for (int kp = 0; kp < nucell; ++kp){
          int idx = (k*sysdim+a)*fft_dim+kp*sysdim+b;
          sum += real(Phi_q0[idx]);
        }
        sum /= double(nucell);
        for (int kp = 0; kp < nucell; ++kp){
          int idx = (k*sysdim+a)*fft_dim+kp*sysdim+b;
          real(Phi_q0[idx]) -= sum;
        }
      }
    }
   
    // symmetrize
    for (int k = 0; k < nucell; ++k)
    for (int kp = k; kp < nucell; ++kp){
      double csum = 0.;
      for (int a = 0; a < sysdim; ++a)
      for (int b = 0; b < sysdim; ++b){
        int idx = (k*sysdim+a)*fft_dim + kp*sysdim + b;
        int jdx = (kp*sysdim+b)*fft_dim + k*sysdim + a;
        csum = (real(Phi_q0[idx])+real(Phi_q0[jdx]))*0.5;
        real(Phi_q0[idx]) = real(Phi_q0[jdx]) = csum;
      }
    }
  }

  // symmetric ASR
  for (int a = 0; a < sysdim; ++a)
  for (int b = 0; b < sysdim; ++b){
    for (int k = 0; k < nucell; ++k){
      double sum = 0.;
      for (int kp = 0; kp < nucell; ++kp){
        int idx = (k*sysdim+a)*fft_dim + kp*sysdim + b;
        sum += real(Phi_q0[idx]);
      }
      sum /= double(nucell-k);
      for (int kp = k; kp < nucell; ++kp){
        int idx = (k*sysdim+a)*fft_dim+kp*sysdim+b;
        int jdx = (kp*sysdim+b)*fft_dim+k*sysdim+a;
        real(Phi_q0[idx]) -= sum;
        real(Phi_q0[jdx]) = real(Phi_q0[idx]);
      }
    }
  }

return;
}

/* ----------------------------------------------------------------------
 * To output the elastic force, if keyword output is set.
 * --------------------------------------------------------------------*/
void FixGFMD::end_of_step()
{
  if (me == 0){
    char file_for[MAXLINE];
    FILE *fp;

    sprintf(file_for,"%s." BIGINT_FORMAT, prefix, update->ntimestep);
    fp = fopen(file_for, "w");

    fprintf(fp,"# Elastic forces acting on GF layer, timestep= " BIGINT_FORMAT "\n",update->ntimestep);
    fprintf(fp,"# Size Info: %lg %lg %lg %d %d\n", domain->xprd, domain->yprd, domain->xy, nGFatoms, sysdim);
    fprintf(fp,"# Extra force added on each GF atom: %lg\n", load);
    if (sysdim == 3) fprintf(fp,"#index atom x y z fx fy fz\n");
    else fprintf(fp,"# index atom x y fx fy\n");

    for (int i = 0; i < nGFatoms; ++i){
      idx  = static_cast<int>(UIrAll[i][sysdim]);
      itag = surf2tag[idx];
      fprintf(fp,"%d %d", idx, itag);
      for (int idim = 0; idim < sysdim; ++idim) fprintf(fp," %lg", UIrAll[i][idim]);
      for (int idim = 0; idim < sysdim; ++idim) fprintf(fp," %lg", FrAll[i][idim]);
      fprintf(fp,"\n");
    }
    fclose(fp);
  }

return;
}

/* ---------------------------------------------------------------------- */

double FixGFMD::memory_usage()
{
  double bytes = sizeof(double) * nGFatoms * (6*sysdim+2)
               + sizeof(double) * (MAX(1,mynpt)*sysdim + mynq*nucell)
               + sizeof(int)    * 5 * nprocs
               + sizeof(std::map<int,int>) * 2 * nGFatoms
               + sizeof(std::complex<double>) * MAX(1,mynq) * fft_dim * (1+fft_dim)
               + sizeof(double) * 2 * mynq;
return bytes;
}

/* ----------------------------------------------------------------------
 * private method, to do matrix-vector multiplication for real (double)
 * matrix and vector; square matrix is required.
 * --------------------------------------------------------------------*/
void FixGFMD::MatMulVec(int dim, double *Mat, double *Vin, double *Vout)
{
  int m = 0;
  for (int i = 0; i < dim; ++i){
    Vout[i] = 0.;
    for (int j = 0; j < dim; ++j) Vout[i] += Mat[m++]*Vin[j];
  }

return;
}

/* ----------------------------------------------------------------------
 * private method, to do matrix-vector multiplication for complex
 * matrix and vector; square matrix is required.
 * --------------------------------------------------------------------*/
void FixGFMD::MatMulVec(int dim, std::complex<double> *Mat,
              std::complex<double> *Vin, std::complex<double> *Vout)
{
  int m = 0;
  for (int i = 0; i < dim; ++i){
    Vout[i] = std::complex<double>(0.,0.);
    for (int j = 0; j < dim; ++j) Vout[i] += Mat[m++]*Vin[j];
  }

return;
}

/* ----------------------------------------------------------------------
 * private method, to do matrix-matrix multiplication for real (double)
 * matric; square matrix is required.
 * --------------------------------------------------------------------*/
void FixGFMD::MatMulMat(int dim, double *MatA, double *MatB, double *MatC)
{
  int m = 0;
  int idim = 0;
  for (int i = 0; i < dim; ++i){
    for (int j=0; j < dim; ++j){
      MatC[m] =0.;
      for (int k = 0; k < dim; ++k) MatC[m] += MatA[idim+k]*MatB[k*dim+j];
      ++m;
    }
    idim += dim;
  }

return;
}

/* ----------------------------------------------------------------------
 * private method, to do matrix-matrix multiplication for complex
 * matric; square matrix is required.
 * --------------------------------------------------------------------*/
void FixGFMD::MatMulMat(int dim, std::complex<double> *MatA,
              std::complex<double> *MatB, std::complex<double> *MatC)
{
  int m = 0;
  int idim = 0;
  for (int i = 0; i < dim; ++i){
    for (int j = 0; j < dim; ++j){
      MatC[m] =std::complex<double>(0.,0.);
      for (int k = 0; k < dim; ++k) MatC[m] += MatA[idim+k]*MatB[k*dim+j];
      ++m;
    }
    idim += dim;
  }

return;
}

/* ----------------------------------------------------------------------
 * private method, to get the inverse of a double precision matrix
 * by means of Gaussian-Jordan Elimination with full pivoting.
 *
 * Adapted from the Numerical Recipes in Fortran.
 * --------------------------------------------------------------------*/
void FixGFMD::GaussJordan(int n, double *Mat)
{
  int i,icol,irow,j,k,l,ll,idr,idc;
  int *indxc,*indxr,*ipiv;
  double big, dum, pivinv;

  indxc = new int[n];
  indxr = new int[n];
  ipiv  = new int[n];

  for (i = 0; i < n; ++i) ipiv[i] = 0;
  for (i = 0; i < n; ++i){
    big = 0.;
    for (j = 0; j < n; ++j){
      if (ipiv[j] != 1){
        for (k = 0; k < n; ++k){
          if (ipiv[k] == 0){
            idr = j*n+k;
            if (fabs(Mat[idr]) >= big){
              big  = fabs(Mat[idr]);
              irow = j;
              icol = k;
            }
          } else if (ipiv[k] >1) error->one(FLERR,"Singular matrix in double GaussJordan!");
        }
      }
    }

    ipiv[icol] += 1;
    if (irow != icol){
      for (l = 0; l < n; ++l){
        idr  = irow*n+l;
        idc  = icol*n+l;
        dum  = Mat[idr];
        Mat[idr] = Mat[idc];
        Mat[idc] = dum;
      }
    }
    indxr[i] = irow;
    indxc[i] = icol;
    idr = icol*n + icol;
    if (Mat[idr] == 0.) error->one(FLERR,"Singular matrix in double GaussJordan!");
    
    pivinv = 1./ Mat[idr];
    Mat[idr] = 1.;
    idr = icol*n;
    for (l = 0; l < n; ++l) Mat[idr+l] *= pivinv;
    for (ll = 0; ll < n; ++ll){
      if (ll != icol){
        idc = ll*n+icol;
        dum = Mat[idc];
        Mat[idc] = 0.;
        idc -= icol;
        for (l = 0; l < n; ++l) Mat[idc+l] -= Mat[idr+l]*dum;
      }
    }
  }
  for (l = n-1; l >= 0; --l){
    int rl = indxr[l];
    int cl = indxc[l];
    if (rl != cl){
      for (k = 0; k < n; ++k){
        idr = k*n+rl;
        idc = k*n+cl;
        dum = Mat[idr];
        Mat[idr] = Mat[idc];
        Mat[idc] = dum;
      }
    }
  }
  delete []indxr;
  delete []indxc;
  delete []ipiv;

return;
}

/* ----------------------------------------------------------------------
 * private method, to get the inverse of a complex matrix by means of
 * Gaussian-Jordan Elimination with full pivoting.
 *
 * Adapted from the Numerical Recipes in Fortran.
 * --------------------------------------------------------------------*/
void FixGFMD::GaussJordan(int n, std::complex<double> *Mat)
{
  int i,icol,irow,j,k,l,ll,idr,idc;
  int *indxc,*indxr,*ipiv;
  double big, nmjk;
  std::complex<double> dum, pivinv;

  indxc = new int[n];
  indxr = new int[n];
  ipiv  = new int[n];

  for (i = 0; i < n; ++i) ipiv[i] = 0;
  for (i = 0; i < n; ++i){
    big = 0.;
    for (j = 0; j < n; ++j){
      if (ipiv[j] != 1){
        for (k = 0; k < n; ++k){
          if (ipiv[k] == 0){
            idr = j*n+k;
            nmjk = norm(Mat[idr]);
            if (nmjk >= big){
              big  = nmjk;
              irow = j;
              icol = k;
            }
          } else if (ipiv[k]>1) error->one(FLERR,"Singular matrix in complex GaussJordan!");
        }
      }
    }
    ipiv[icol] += 1;
    if (irow != icol){
      for (l = 0; l < n; ++l){
        idr  = irow*n+l;
        idc  = icol*n+l;
        dum  = Mat[idr];
        Mat[idr] = Mat[idc];
        Mat[idc] = dum;
      }
    }
    indxr[i] = irow;
    indxc[i] = icol;
    idr = icol*n+icol;
    if (Mat[idr] == std::complex<double>(0.,0.)) error->one(FLERR,"Singular matrix in complex GaussJordan!");
    
    pivinv = 1./ Mat[idr];
    Mat[idr] = std::complex<double>(1.,0.);
    idr = icol*n;
    for (l = 0; l < n; ++l) Mat[idr+l] *= pivinv;
    for (ll = 0; ll < n; ++ll){
      if (ll != icol){
        idc = ll*n+icol;
        dum = Mat[idc];
        Mat[idc] = 0.;
        idc -= icol;
        for (l = 0; l < n; ++l) Mat[idc+l] -= Mat[idr+l]*dum;
      }
    }
  }
  for (l = n-1; l >= 0; --l){
    int rl = indxr[l];
    int cl = indxc[l];
    if (rl != cl){
      for (k = 0; k < n; ++k){
        idr = k*n+rl;
        idc = k*n+cl;
        dum = Mat[idr];
        Mat[idr] = Mat[idc];
        Mat[idc] = dum;
      }
    }
  }
  delete []indxr;
  delete []indxc;
  delete []ipiv;

return;
}

/* ---------------------------------------------------------------------- */
