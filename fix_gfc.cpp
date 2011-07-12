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
#include "compute.h"
#include "domain.h"
#include "error.h"
#include "fix_gfc.h"
#include "fft3d_wrap.h"
#include "force.h"
#include "group.h"
#include "lattice.h"
#include "memory.h"
#include "modify.h"
#include "update.h"

using namespace LAMMPS_NS;

#define INVOKED_SCALAR 1
#define INVOKED_VECTOR 2
#define MAXLINE 256
#define MAX(a,b) ((a) > (b) ? (a) : (b))

FixGFC::FixGFC(LAMMPS *lmp,  int narg, char **arg) : Fix(lmp, narg, arg)
{
  MPI_Comm_rank(world,&me);
  MPI_Comm_size(world,&nprocs);
  
  if (narg<7) error->all("Illegal fix gfc command: number of arguments < 7");

  nevery = atoi(arg[3]);   // Calculate this fix every n steps!
  if (nevery <= 0) error->all("Illegal fix gfc command");

  nfreq  = atoi(arg[4]);   // frequency to output result
  if (nfreq <=0) error->all("Illegal fix gfc command");

  waitsteps = ATOBIGINT(arg[5]); // Wait this many timesteps before actually measuring GFC's
  if (waitsteps < 0) error->all("fix gfc: waitsteps < 0 ! Please provide non-negative number!");

  int n = strlen(arg[6]) + 1;
  prefix = new char[n];
  strcpy(prefix, arg[6]);
  file_log = new char[n+4];
  sprintf(file_log,"%s.log",prefix);
  
  int iarg = 7;
  int fsurfmap = 0;
  mapfile = NULL;
  origin_tag = -1;

  // to read other command line options, they are optional
  while (iarg < narg){
    // surface vector U. if not given, will be determined from lattice info
    if (strcmp(arg[iarg],"su") == 0){
      if (iarg+3 > narg) error->all("fix gfc: Insufficient command line options.");
      surfvec[0][0] = atof(arg[++iarg]);
      surfvec[0][1] = atof(arg[++iarg]);
      fsurfmap |= 1;

    // surfactor vector V. if not given for 3D, will be determined from lattice info
    } else if (strcmp(arg[iarg],"sv") == 0){
      if (iarg+3 > narg) error->all("fix gfc: Insufficient command line options.");
      surfvec[1][0] = atof(arg[++iarg]);
      surfvec[1][1] = atof(arg[++iarg]);
      fsurfmap |= 2;

    // tag of surface origin atom
    } else if (strcmp(arg[iarg],"origin") == 0){
      if (iarg+2 > narg) error->all("fix gfc: Insufficient command line options.");
      origin_tag = atoi(arg[++iarg]);

    // read the mapping of surface atoms from file! no surface vector is needed now
    } else if (strcmp(arg[iarg],"map") == 0){
      if (iarg+2 > narg) error->all("fix gfc: Insufficient command line options.");
      if (mapfile) delete []mapfile;
      int n = strlen(arg[++iarg]) + 1;
      mapfile = new char [n];
      strcpy(mapfile, arg[iarg]);
      fsurfmap |= 4;

    } else {
      error->all("fix gfc: Unknown command line option!");
    }
    ++iarg;
  } // end of reading command line options
  
  sysdim  = domain->dimension; // find the system dimension
  if (sysdim == 2) {
    surfvec[1][0] = 0.;
    surfvec[1][1] = 1.;
    fsurfmap |=2;
  }

  // get the total number of atoms in group
  nGFatoms = static_cast<int>(group->count(igroup));
  if (nGFatoms<1) error->all("fix gfc: no atom found for GFC evaluation!");

  // MPI gatherv related variables
  recvcnts = new int[nprocs];
  displs   = new int[nprocs];

  // mapping index
  tag2surf.clear(); // clear map info
  surf2tag.clear();

  // get the mapping between FFT mesh and surface atoms
  if (fsurfmap & 4){ readmap(); delete []mapfile; }
  else compmap(fsurfmap);

  // create FFT and allocate memory for FFT
  nxlo = 0;
  int *nx_loc = new int [nprocs];
  for (int i=0; i<nprocs;i++){
    nx_loc[i] = nx/nprocs;
    if (i < nx%nprocs) nx_loc[i]++;
  }
  for (int i=0; i<me; i++) nxlo += nx_loc[i];
  nxhi  = nxlo + nx_loc[me] - 1;
  mynpt = nx_loc[me] * ny;
  mynq  = mynpt;

  fft_dim   = nucell*sysdim;
  fft_dim2  = fft_dim*fft_dim;
  fft_nsend = mynpt*fft_dim;

  fft_cnts  = new int[nprocs];
  fft_disp  = new int[nprocs];
  fft_disp[0] = 0;
  for (int i=0; i<nprocs; i++) fft_cnts[i] = nx_loc[i]*ny*fft_dim;
  for (int i=1; i<nprocs; i++) fft_disp[i] = fft_disp[i-1] + fft_cnts[i-1];
  delete []nx_loc;

  fft   = new FFT3d(lmp,world,1,ny,nx,0,0,0,ny-1,nxlo,nxhi,0,0,0,ny-1,nxlo,nxhi,0,0,&mysize);
  fft_data = (double *) memory->smalloc(MAX(1,mynq)*2*sizeof(double),"fix_gfc:fft_data");

  // allocate variables; MAX(1,... is used because NULL buffer will result in error for MPI
  RIloc = memory->create(RIloc,nGFatoms,(sysdim+1),"fix_gfc:RIloc");
  RIall = memory->create(RIall,nGFatoms,(sysdim+1),"fix_gfc:RIall");
  Rsort = memory->create(Rsort,nGFatoms, sysdim, "fix_gfc:Rsort");

  Rnow  = memory->create(Rnow ,MAX(1,mynpt),fft_dim,"fix_gfc:Rnow");
  Rsum  = memory->create(Rsum ,MAX(1,mynpt),fft_dim,"fix_gfc:Rsum");

  surfbasis = memory->create(surfbasis,nucell, sysdim, "fix_gfc:surfbasis");

  // because of hermit, only nearly half of q points are stored
  Rqnow = memory->create(Rqnow,MAX(1,mynq),fft_dim, "fix_gfc:Rqnow");
  Rqsum = memory->create(Rqsum,MAX(1,mynq),fft_dim2,"fix_gfc:Rqsum");
  Phi_q = memory->create(Phi_q,MAX(1,mynq),fft_dim2,"fix_gfc:Phi_q");
  if (me == 0) // variable to collect all local Phi to root
    Phi_all = memory->create(Phi_all,nx*ny,fft_dim2,"fix_gfc:Phi_all");
  else
    Phi_all = memory->create(Phi_all,1,1,"fix_gfc:Phi_all");

  // output some information on the system to log file
  surfvec[0][0] = domain->h[0]/double(nx);
  surfvec[0][1] = domain->h[5]/double(nx);
  surfvec[1][0] = 0.;
  surfvec[1][1] = domain->h[1]/double(ny);
  if (sysdim == 2) surfvec[1][1] = 1.;

  if (me == 0){
    gfclog = fopen(file_log, "w");
    if (gfclog == NULL) {
      char str[MAXLINE];
      sprintf(str,"fix gfc: Can not open output file %s",file_log);
      error->one(str);
    }

    for (int i=0; i<60; i++) fprintf(gfclog,"#"); fprintf(gfclog,"\n");
    fprintf(gfclog,"# group name of the Green's Function layer : %s\n", group->names[igroup]);
    fprintf(gfclog,"# total number of atoms in the GF layer    : %d\n", nGFatoms);
    fprintf(gfclog,"# dimension of the system                  : %d D\n", sysdim);
    fprintf(gfclog,"# number of atoms per unit surface cell    : %d\n", nucell);
    fprintf(gfclog,"# dimension of the FFT mesh                : %d x %d\n", nx, ny);
    fprintf(gfclog,"# atomic tag of the surface origin         : %d\n", origin_tag);
    fprintf(gfclog,"# number of wait steps before measurement  : " BIGINT_FORMAT "\n", waitsteps);
    fprintf(gfclog,"# frequency of GFC measurement             : %d\n", nevery);
    fprintf(gfclog,"# output result after this many measurement: %d\n", nfreq);
    fprintf(gfclog,"# number of processors used by this run    : %d\n", nprocs);
    for (int i=0; i<60; i++) fprintf(gfclog,"#"); fprintf(gfclog,"\n");
    fprintf(gfclog,"# Surface vectors: [ %lg, %lg ] [ %lg, %lg ]\n", surfvec[0][0],
      surfvec[0][1], surfvec[1][0], surfvec[1][1]);

    fprintf(gfclog,"# mapping information between FFT mesh and atom id\n");
    fprintf(gfclog,"# nx ny nucell\n");
    fprintf(gfclog,"%d %d %d\n", nx, ny, nucell);
    fprintf(gfclog,"# l1 l2 k atom_id\n");
    int ix, iy, iu;
    for (idx =0; idx<nGFatoms; idx++){
      itag = surf2tag[idx];
      iu   = idx%nucell;
      iy   = (idx/nucell)%ny;
      ix   = idx/(ny*nucell);
      fprintf(gfclog,"%d %d %d %d\n", ix, iy, iu, itag);
    }
    for (int i=0; i<60; i++) fprintf(gfclog,"#"); fprintf(gfclog,"\n");
    fflush(gfclog);
  }
 
  // default temperature is from thermo
  TempSum = new double[sysdim];
  id_temp = new char[12];
  strcpy(id_temp,"thermo_temp");
  int icompute = modify->find_compute(id_temp);
  temperature = modify->compute[icompute];
  inv_nTemp = 1.0/group->count(temperature->igroup);

} // end of constructor

/* ---------------------------------------------------------------------- */

void FixGFC::post_run()
{
  // compute and output final GFC results
  if (GFcounter%nfreq) postprocess();
  if (me == 0) fclose(gfclog);

}

/* ---------------------------------------------------------------------- */

FixGFC::~FixGFC()
{
  // delete locally stored array
  memory->destroy(RIloc);
  memory->destroy(RIall);
  memory->destroy(Rsort);
  memory->destroy(Rnow);
  memory->destroy(Rsum);
  memory->destroy(surfbasis);

  memory->destroy(Rqnow);
  memory->destroy(Rqsum);
  memory->destroy(Phi_q);
  memory->destroy(Phi_all);

  delete []recvcnts;
  delete []displs;
  delete []prefix;
  delete []file_log;
  delete []fft_cnts;
  delete []fft_disp;
  delete []id_temp;
  delete []TempSum;

  // destroy FFT
  delete fft;
  memory->sfree(fft_data);
  
  // clear map info
  tag2surf.clear();
  surf2tag.clear();

}

/* ---------------------------------------------------------------------- */

int FixGFC::setmask()
{
  int mask = 0;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixGFC::init()
{
  // warn if more than one gfc fix
  int count = 0;
  for (int i=0;i<modify->nfix;i++) if (strcmp(modify->fix[i]->style,"gfc") == 0) count++;
  if (count > 1 && me == 0) error->warning("More than one fix gfc defined"); // just warn, but allowed.
}

/* ---------------------------------------------------------------------- */

void FixGFC::setup()
{
  // initialize accumulating variables
  for (int i=0; i<sysdim; i++) TempSum[i] = 0.;
  for (int i=0; i<mynpt; i++){
    for (int j=0; j<fft_dim;  j++) Rsum[i][j] = 0.;
  }
  for (int i=0; i<mynq; i++){
    for (int j=0; j<fft_dim2; j++) Rqsum[i][j] = std::complex<double> (0.,0.);
  }
  for (int i=0; i<nucell; i++){
    for (int j=0; j<sysdim; j++) surfbasis[i][j] = 0.;
  }
  prev_nstep = update->ntimestep;
  GFcounter  = 0;
  ifreq      = 0;
}

/* ---------------------------------------------------------------------- */

void FixGFC::end_of_step()
{
  if ( (update->ntimestep-prev_nstep) <= waitsteps) return;

  double **x = atom->x;
  int *mask  = atom->mask;
  int *tag   = atom->tag;
  int *image = atom->image;
  int nlocal = atom->nlocal;

  double xprd = domain->xprd;
  double yprd = domain->yprd;
  double zprd = domain->zprd;
  double *h   = domain->h;
  double xbox, ybox, zbox;

  int i,idim,jdim,ndim;
  double xcur[3], dist2orig[3];

  // to get the current temperature
  if (!(temperature->invoked_flag & INVOKED_VECTOR)) temperature->compute_vector();
  for (idim=0; idim<sysdim; idim++) TempSum[idim] += temperature->vector[idim];

  // evaluate R(r) on local proc
  nfind = 0;
  if (domain->triclinic == 0) { // for orthogonal lattice
    for (i=0; i<nlocal; i++){
      if (mask[i] & groupbit){
        itag = tag[i];
        idx  = tag2surf[itag];
        
        xbox = (image[i] & 1023) - 512;
        ybox = (image[i] >> 10 & 1023) - 512;
        zbox = (image[i] >> 20) - 512;
        xcur[0] = x[i][0] + xprd*xbox;
        xcur[1] = x[i][1] + yprd*ybox;
        xcur[2] = x[i][2] + zprd*zbox;
        for (idim=0; idim<sysdim; idim++) RIloc[nfind][idim] = xcur[idim];
        RIloc[nfind++][sysdim] = idx;
      }
    }
  }else{                      // for non-orthogonal lattice
    for (i=0; i<nlocal; i++){
      if (mask[i] & groupbit){
        itag = tag[i];
        idx  = tag2surf[itag];

        xbox = (image[i] & 1023) - 512;
        ybox = (image[i] >> 10 & 1023) - 512;
        zbox = (image[i] >> 20) - 512;
        xcur[0] = x[i][0] + h[0]*xbox + h[5]*ybox + h[4]*zbox;
        xcur[1] = x[i][1] + h[1]*ybox + h[3]*zbox;
        xcur[2] = x[i][2] + h[2]*zbox;
        for (idim=0; idim<sysdim; idim++) RIloc[nfind][idim] = xcur[idim];
        RIloc[nfind++][sysdim] = idx;
      }
    }
  }

  // gather R(r) on local proc, then sort and redistribute to all procs for FFT
  nfind *= (sysdim+1);
  displs[0] = 0;
  for (i=0; i<nprocs;i++) recvcnts[i] = 0;
  MPI_Gather(&nfind,1,MPI_INT,recvcnts,1,MPI_INT,0,world);
  for (i=1; i<nprocs; i++) displs[i] = displs[i-1] + recvcnts[i-1];

  MPI_Gatherv(RIloc[0],nfind,MPI_DOUBLE,RIall[0],recvcnts,displs,MPI_DOUBLE,0,world);
  if (me == 0){
    for (i=0; i<nGFatoms; i++){
      idx = static_cast<int>(RIall[i][sysdim]);
      for (idim=0; idim<sysdim; idim++) Rsort[idx][idim] = RIall[i][idim];
    }
  }
  MPI_Scatterv(Rsort[0],fft_cnts,fft_disp, MPI_DOUBLE, Rnow[0], fft_nsend, MPI_DOUBLE,0,world);

  // get Rsum
  for (idx=0; idx<mynpt; idx++){
    for (idim=0; idim<fft_dim; idim++){
      Rsum[idx][idim] += Rnow[idx][idim];
    }
  }

  // FFT R(r) to get R(q)
  for (idim=0; idim<fft_dim; idim++){
    int m=0;
    for (idx=0; idx<mynpt; idx++){
      fft_data[m++] = Rnow[idx][idim];
      fft_data[m++] = 0.;
    }
    fft->compute(fft_data, fft_data, -1);
    m = 0;
    for (idq=0; idq<mynq; idq++){
      Rqnow[idq][idim] = std::complex<double>(fft_data[m], fft_data[m+1]);
      m += 2;
    }
  }
  // to get sum(R(q).R(q)*)
  for (idq=0; idq<mynq; idq++){
    ndim = 0;
    for (idim=0; idim<fft_dim; idim++){
      for (jdim=0; jdim<fft_dim; jdim++){
        Rqsum[idq][ndim++] += Rqnow[idq][idim]*conj(Rqnow[idq][jdim]);
      }
    }
  }

  // get surfbasis
  if (fft_dim > sysdim){
    for (idx=0; idx<mynpt; idx++){
      ndim = sysdim;
      for (i=1; i<nucell; i++){
        for (idim=0; idim<sysdim; idim++) dist2orig[idim] = Rnow[idx][ndim++] - Rnow[idx][idim];
        domain->minimum_image(dist2orig);
        for (idim=0; idim<sysdim; idim++) surfbasis[i][idim] += dist2orig[idim];
      }
    }
  }

  // increment counter
  GFcounter++;

  // compute and output Phi_q after every nfreq evaluations
  if (++ifreq == nfreq) postprocess();

}   // end of end_of_step()

/* ---------------------------------------------------------------------- */

double FixGFC::memory_usage()
{
  double bytes = sizeof(double)*(2*mynq+nucell*sysdim)
               + sizeof(std::map<int,int>)*2*nGFatoms
               + sizeof(double)*(nGFatoms*(3*sysdim+2)+mynpt*fft_dim*2)
               + sizeof(std::complex<double>)*MAX(1,mynq)*fft_dim *(1+2*fft_dim)
               + sizeof(std::complex<double>)*nx*ny*fft_dim2
               + sizeof(int) * nprocs * 4;
  return bytes;
}

/* ---------------------------------------------------------------------- */

int FixGFC::modify_param(int narg, char **arg)
{
  if (strcmp(arg[0],"temp") == 0) {
    if (narg < 2) error->all("Illegal fix_modify command");
    delete [] id_temp;
    int n = strlen(arg[1]) + 1;
    id_temp = new char[n];
    strcpy(id_temp,arg[1]);

    int icompute = modify->find_compute(id_temp);
    if (icompute < 0) error->all("Could not find fix_modify temp ID");
    temperature = modify->compute[icompute];

    if (temperature->tempflag == 0)
      error->all("Fix_modify temp ID does not compute temperature");
    inv_nTemp = 1.0/group->count(temperature->igroup);

    return 2;
  }
  return 0;
}

/* ----------------------------------------------------------------------
 * private method, to read the mapping info from file
 * --------------------------------------------------------------------*/

void FixGFC::readmap()
{
  int info = 0;
  char strtmp[MAXLINE];
  FILE *fp;

  fp = fopen(mapfile, "r");
  if (fp == NULL){
    char str[MAXLINE];
    sprintf(str,"fix gfc: cannot open input map file %s", mapfile);
    error->one(str);
  }

  if (fgets(strtmp,MAXLINE,fp) == NULL)
    error->all("fix gfc: Error while reading header of mapping file!");
  sscanf(strtmp,"%d %d %d", &nx, &ny, &nucell);
  if (nx*ny*nucell != nGFatoms) error->all("fix gfc: FFT mesh and number of atoms in group mismatch!");
  
  if (fgets(strtmp,MAXLINE,fp) == NULL)    // second line of mapfile is comment
    error->all("fix gfc: Error while reading comment of mapping file!");

  int ix, iy, iu;
  for (int i=0; i<nGFatoms; i++){
    if (fgets(strtmp,MAXLINE,fp) == NULL) {info = 1; break;}
    sscanf(strtmp,"%d %d %d %d", &ix, &iy, &iu, &itag); // the remaining lines carry the mapping info

    if (ix<0 || ix>=nx || iy<0 || iy>=ny || iu<0 || iu>=nucell) {info = 2; break;} // check if index is in correct range
    if (itag<1 || itag>static_cast<int>(atom->natoms)) {info = 3; break;}     // 1 <= itag <= natoms
    idx = (ix*ny+iy)*nucell+iu;
    tag2surf[itag] = idx;
    surf2tag[idx]  = itag;
  }
  fclose(fp);

  if (tag2surf.size() != surf2tag.size() || tag2surf.size() != static_cast<std::size_t>(nGFatoms) )
    error->all("fix gfc: the mapping is incomplete!");
  if (info) error->all("fix gfc: Error while reading mapping file!");
  
  // check the correctness of mapping
  int *mask  = atom->mask;
  int *tag   = atom->tag;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit){
      itag = tag[i];
      idx  = tag2surf[itag];
      if (itag != surf2tag[idx]) error->one("fix gfc: the mapping info read is incorrect!");
    }
  }
  origin_tag = surf2tag[0];

  return;
}

/* ----------------------------------------------------------------------
 * private method, to calculate the mapping info fo surface atoms
 * --------------------------------------------------------------------*/

void FixGFC::compmap(int flag)
{
  if ((flag & 3) != 3){
    // Check if lattice information is available from input file!
    if (domain->lattice == NULL) // need surface vectors from lattice info; orthogonal lattice is assumed
      error->all("fix gfc: No lattice defined while keyword su and/or sv is not set.");

    if ((flag & 1) == 0){
      surfvec[0][0] = domain->lattice->xlattice;
      surfvec[0][1] = 0.0;
    }
    if ((flag & 2) == 0){
      surfvec[1][0] = 0.0;
      surfvec[1][1] = domain->lattice->ylattice;
    }
  }
  // check the validity of the surface vectors read from command line.
  if (fabs(surfvec[0][1]) > 0.0 && fabs(surfvec[1][0]) > 0.0)
    error->all("fix gfc: Either U or V must be on the box side!");
  if (surfvec[0][0] <= 0.0)
    error->all("fix gfc: Surface vector U must be along the +x direction!");
  if (surfvec[1][1] <= 0.0)
    error->all("fix gfc: Surface vector V must point to the +y direction!");
  
  double invSurfV[2][2];
  for (int i=0;i<2;i++){
    invSurfV[i][0] = surfvec[i][0];
    invSurfV[i][1] = surfvec[i][1];
  }
  // get the inverse of surfvec
  GaussJordan(2,invSurfV[0]);
  double dswap = invSurfV[0][1];
  invSurfV[0][1] = invSurfV[1][0];
  invSurfV[1][0] = dswap;

  double **x = atom->x;
  int *mask  = atom->mask;
  int *image = atom->image;
  int *tag   = atom->tag;
  int nlocal = atom->nlocal;

  nx = int(domain->xprd*invSurfV[0][0]+0.1);
  ny = (sysdim == 2)?1:int(domain->yprd*invSurfV[1][1]+0.1);

  if (nx<1 || nx>nGFatoms || ny<1 || ny>nGFatoms)
    error->all("fix gfc: error encountered while getting FFT dimensions!");

  nucell = nGFatoms / (nx*ny);
  if (nucell > 2) error->all("fix gfc: mapping info cannot be computed for nucell > 2!");
  if (nx*ny*nucell != nGFatoms) error->all("fix gfc: number of atoms from FFT mesh and group mismatch!");

  double vx[3], vi[2], SurfOrigin[2];
  // determining surface origin
  nfind =0;
  if ( origin_tag > 0 ){
    for (int i=0; i<nlocal; i++){
      if (tag[i] == origin_tag){
        domain->unmap(x[i],image[i],vx);
        if ( (mask[i] & groupbit) == 0) {
          error->one("fix gfc: specified surface origin atom not in group!");
          return;
        }
        nfind++;
        break;
      }
    }
  } else{
    for (int i=0; i<nlocal; i++){
      if (mask[i] & groupbit){
        domain->unmap(x[i],image[i],vx);
        nfind++;
        break;
      }
    }
  }
  
  nfind *= 2;
  MPI_Gather(&nfind, 1, MPI_INT, recvcnts, 1, MPI_INT, 0, world);
  if (me == 0){
    displs[0]=0;
    for (int i=1; i<nprocs; i++) displs[i] = displs[i-1] + recvcnts[i-1];
    int ntotal=0;
    for (int i=0; i<nprocs; i++) ntotal += recvcnts[i];
    if (ntotal<2) error->one("fix gfc: surface origin not found!");
  }

  double recvbuf[nprocs][2];
  MPI_Gatherv(vx, nfind, MPI_DOUBLE, recvbuf[0], recvcnts, displs, MPI_DOUBLE, 0, world);
  if (me == 0){
     SurfOrigin[0] = recvbuf[0][0];
     SurfOrigin[1] = recvbuf[0][1];
  }
  MPI_Bcast(SurfOrigin, 2, MPI_DOUBLE, 0, world);

  // now to calculate the mapping info
  int ix, iy, iu;
  int **IndxLoc, **IndxAll;
  IndxLoc = memory->create(IndxLoc,nGFatoms,2,"fix_gfc:IndxLoc");
  IndxAll = memory->create(IndxAll,nGFatoms,2,"fix_gfc:IndxAll");

  nfind =0;
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit){
      domain->unmap(x[i],image[i],vx);
      itag  = tag[i];

      vx[0] -= SurfOrigin[0]; // relative coordination on the surface of atom i
      vx[1] -= SurfOrigin[1];

      vi[0] = invSurfV[0][0]*vx[0] + invSurfV[0][1]*vx[1];
      vi[1] = invSurfV[1][0]*vx[0] + invSurfV[1][1]*vx[1];

      ix = (int)floor(vi[0]+0.1); ix %= nx; ix = (ix < 0)? (ix+nx):ix;
      iy = (int)floor(vi[1]+0.1); iy %= ny; iy = (iy < 0)? (iy+ny):iy;
      // usually, if one unit cell has two atoms, the second one lies in the center
      iu = ((int)floor(vi[0]+vi[0]+0.1))%2 | ((int)floor(vi[1]+vi[1]+0.1))%2;

      idx = ((ix*ny)+iy)*nucell+abs(iu);

      IndxLoc[nfind][0] = itag;
      IndxLoc[nfind][1] = idx;

      nfind++;
    }
  }

  nfind *= 2;
  // gather mapping info to all procs
  MPI_Allgather(&nfind, 1, MPI_INT, recvcnts, 1, MPI_INT, world);
  displs[0]=0;
  for (int i=1; i<nprocs; i++) displs[i] = displs[i-1] + recvcnts[i-1];
  MPI_Allgatherv(IndxLoc[0], nfind, MPI_INT, IndxAll[0], recvcnts, displs, MPI_INT, world);

  for (int i=0; i<nGFatoms; i++){
    itag = IndxAll[i][0];
    idx  = IndxAll[i][1];
    tag2surf[itag] = idx;
    surf2tag[idx]  = itag;
  }
  memory->destroy(IndxLoc);
  memory->destroy(IndxAll);

  origin_tag = surf2tag[0];

  if (tag2surf.size() != surf2tag.size() || tag2surf.size() != static_cast<std::size_t>(nGFatoms) )
    error->all("fix gfc: mapping info is incompleted/incorrect!");

  return;
}

/* ----------------------------------------------------------------------
 * private method, to get the elastic stiffness coefficients.
 * --------------------------------------------------------------------*/

void FixGFC::postprocess( )
{
  if (GFcounter<1) return;

  ifreq =0;
  int idim, jdim, ndim;
  double invGFcounter = 1.0 /double(GFcounter);

  // to get <Rq.Rq*>
  for (idq=0; idq<mynq; idq++){
    for (idim=0; idim<fft_dim2; idim++) Phi_q[idq][idim] = Rqsum[idq][idim]*invGFcounter;
  }

  // to get <R>
  for (idx=0; idx<mynpt; idx++){
    for (idim=0; idim<fft_dim; idim++) Rnow[idx][idim] = Rsum[idx][idim] * invGFcounter;
  }

  // to get <R>q
  for (idim=0; idim<fft_dim; idim++){
    int m = 0;
    for (idx=0; idx<mynpt; idx++){
      fft_data[m++] = Rnow[idx][idim];
      fft_data[m++] = 0.;
    }
    fft->compute(fft_data,fft_data,-1);
    m = 0;
    for (idq=0; idq<mynq; idq++){
      Rqnow[idq][idim]  = std::complex<double>(fft_data[m], fft_data[m+1]);
      m += 2;
    }
  }

  // to get G(q) = <Rq.Rq*> - <R>q.<R*>q
  for (idq=0; idq<mynq; idq++){
    ndim = 0;
    for (idim=0; idim<fft_dim; idim++){
      for (jdim=0; jdim<fft_dim; jdim++) Phi_q[idq][ndim++] -= Rqnow[idq][idim]*conj(Rqnow[idq][jdim]);
    }
  }

  // to get Phi = KT.G^-1; normalization of FFTW data is done here
  double boltz = force->boltz, kbtsqrt[sysdim], TempAve = 0.;
  double TempFac = invGFcounter*inv_nTemp;
  double NormFac = TempFac*nx*ny;

  for (idim=0; idim<sysdim; idim++){
    kbtsqrt[idim] = sqrt(TempSum[idim]*NormFac);
    TempAve += TempSum[idim]*TempFac;
  }
  TempAve /= sysdim*boltz;
  
  for (idq=0; idq<mynq; idq++){
    GaussJordan(fft_dim, Phi_q[idq]);
    ndim =0;
    for (idim=0; idim<fft_dim; idim++){
      for (jdim=0; jdim<fft_dim; jdim++){
        Phi_q[idq][ndim++] *= kbtsqrt[idim%sysdim]*kbtsqrt[jdim%sysdim];
      }
    }
  }

  // to collect all local Phi_q to root
  displs[0]=0;
  for (int i=0; i<nprocs; i++) recvcnts[i] = fft_cnts[i]*fft_dim*2;
  for (int i=1; i<nprocs; i++) displs[i] = displs[i-1] + recvcnts[i-1];
  MPI_Gatherv(Phi_q[0],mynq*fft_dim2*2,MPI_DOUBLE,Phi_all[0],recvcnts,displs,MPI_DOUBLE,0,world);
  
  // to collect all surfbasis and averaged it on root
  double sb_root[fft_dim];
  if (fft_dim > sysdim)
    MPI_Reduce (&surfbasis[1][0], &sb_root[sysdim], fft_dim-sysdim, MPI_DOUBLE, MPI_SUM, 0, world);

  if (me == 0){ // output by root
    for (idim=0;      idim<sysdim;  idim++) sb_root[idim]  = 0.;
    for (idim=sysdim; idim<fft_dim; idim++) sb_root[idim] /= double(nx)*double(ny)*double(GFcounter);

    // write binary file
    char fname[MAXLINE];
    FILE *GFC_bin;

    sprintf(fname,"%s.bin." BIGINT_FORMAT, prefix,update->ntimestep);
    GFC_bin = fopen(fname,"wb");

    fwrite(&sysdim, sizeof(int),    1, GFC_bin);
    fwrite(&nx,     sizeof(int),    1, GFC_bin);
    fwrite(&ny,     sizeof(int),    1, GFC_bin);
    fwrite(&nucell, sizeof(int),    1, GFC_bin);
    fwrite(&boltz,  sizeof(double), 1, GFC_bin);

    fwrite(Phi_all[0],sizeof(double),nx*ny*fft_dim2*2,GFC_bin);

    fwrite(surfvec[0], sizeof(double),4,GFC_bin);
    fwrite(sb_root, sizeof(double),fft_dim, GFC_bin);
    fclose(GFC_bin);

    // write log file
    for (int i=0; i<60; i++) fprintf(gfclog,"#"); fprintf(gfclog,"\n");
    fprintf(gfclog, "# Current time step                      : " BIGINT_FORMAT "\n", update->ntimestep);
    fprintf(gfclog, "# Total number of GFC measurements       : %d\n", GFcounter);
    fprintf(gfclog, "# Average temperature of the measurement : %lg\n", TempAve);
    fprintf(gfclog, "# Boltzmann constant under current units : %lg\n", boltz);
    fprintf(gfclog, "# Basis of the surface unit cell         : ");
    for (idim=0; idim<fft_dim; idim++) fprintf(gfclog,"%lg ", sb_root[idim]);
    fprintf(gfclog, "\n");
    for (int i=0; i<60; i++) fprintf(gfclog,"#"); fprintf(gfclog,"\n");
    fprintf(gfclog, "# ix\t iy \t qx \t qy\t\t\t Phi(q)\n");

    int ix, iy;
    double qx, qy;
    idq =0;
    for (ix=0; ix<nx; ix++){
      qx = (ix<=(nx/2))?(2.0*M_PI*ix/nx):(2.0*M_PI*(ix-nx)/nx);
      for (iy=0; iy<ny; iy++){
        qy = (iy<=(ny/2))?(2.0*M_PI*iy/ny):(2.0*M_PI*(iy-ny)/ny);
        fprintf(gfclog,"%d %d %lg %lg", ix, iy, qx, qy);
        for (idim=0; idim<fft_dim2; idim++) fprintf(gfclog, " %lg %lg", real(Phi_all[idq][idim]), imag(Phi_all[idq][idim]));
        fprintf(gfclog, "\n");
        idq++;
      }
    }
    fflush(gfclog);
  }

}   // end of postprocess

/* ----------------------------------------------------------------------
 * private method, to get the inverse of a double precision matrix
 * by means of Gaussian-Jordan Elimination with full pivoting, square
 * matrix required.
 *
 * Adapted from the Numerical Recipes in Fortran.
 * --------------------------------------------------------------------*/
void FixGFC::GaussJordan(int n, double *Mat)
{
  int i,icol,irow,j,k,l,ll,idr,idc;
  int *indxc,*indxr,*ipiv;
  double big, dum, pivinv;

  indxc = new int[n];
  indxr = new int[n];
  ipiv  = new int[n];

  for (i=0; i<n; i++) ipiv[i] = 0;
  for (i=0; i<n; i++){
    big = 0.;
    for (j=0; j<n; j++){
      if (ipiv[j] != 1){
        for (k=0; k<n; k++){
          if (ipiv[k] == 0){
            idr = j*n+k;
            if (fabs(Mat[idr]) >= big){
              big  = fabs(Mat[idr]);
              irow = j;
              icol = k;
            }
          }else if (ipiv[k] >1){
            error->one("FixGFC: Singular matrix in double GaussJordan!");
          }
        }
      }
    }
    ipiv[icol] += 1;
    if (irow != icol){
      for (l=0; l<n; l++){
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
    if (Mat[idr] == 0.) error->one("FixGFC: Singular matrix in double GaussJordan!");
    
    pivinv = 1./ Mat[idr];
    Mat[idr] = 1.;
    idr = icol*n;
    for (l=0; l<n; l++) Mat[idr+l] *= pivinv;
    for (ll=0; ll<n; ll++){
      if (ll != icol){
        idc = ll*n+icol;
        dum = Mat[idc];
        Mat[idc] = 0.;
        idc -= icol;
        for (l=0; l<n; l++) Mat[idc+l] -= Mat[idr+l]*dum;
      }
    }
  }
  for (l=n-1; l>=0; l--){
    int rl = indxr[l];
    int cl = indxc[l];
    if (rl != cl){
      for (k=0; k<n; k++){
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
 * Gaussian-Jordan Elimination with full pivoting; square matrix required.
 *
 * Adapted from the Numerical Recipes in Fortran.
 * --------------------------------------------------------------------*/
void FixGFC::GaussJordan(int n, std::complex<double> *Mat)
{
  int i,icol,irow,j,k,l,ll,idr,idc;
  int *indxc,*indxr,*ipiv;
  double big, nmjk;
  std::complex<double> dum, pivinv;

  indxc = new int[n];
  indxr = new int[n];
  ipiv  = new int[n];

  for (i=0; i<n; i++) ipiv[i] = 0;
  for (i=0; i<n; i++){
    big = 0.;
    for (j=0; j<n; j++){
      if (ipiv[j] != 1){
        for (k=0; k<n; k++){
          if (ipiv[k] == 0){
            idr = j*n+k;
            nmjk = norm(Mat[idr]);
            if (nmjk >= big){
              big  = nmjk;
              irow = j;
              icol = k;
            }
          }else if (ipiv[k]>1){
            error->one("FixGFC: Singular matrix in complex GaussJordan!");
          }
        }
      }
    }
    ipiv[icol] += 1;
    if (irow != icol){
      for (l=0; l<n; l++){
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
    if (Mat[idr] == std::complex<double>(0.,0.)) error->one("FixGFC: Singular matrix in complex GaussJordan!");
    
    pivinv = 1./ Mat[idr];
    Mat[idr] = std::complex<double>(1.,0.);
    idr = icol*n;
    for (l=0; l<n; l++) Mat[idr+l] *= pivinv;
    for (ll=0; ll<n; ll++){
      if (ll != icol){
        idc = ll*n+icol;
        dum = Mat[idc];
        Mat[idc] = 0.;
        idc -= icol;
        for (l=0; l<n; l++) Mat[idc+l] -= Mat[idr+l]*dum;
      }
    }
  }
  for (l=n-1; l>=0; l--){
    int rl = indxr[l];
    int cl = indxc[l];
    if (rl != cl){
      for (k=0; k<n; k++){
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

/* --------------------------------------------------------------------*/
