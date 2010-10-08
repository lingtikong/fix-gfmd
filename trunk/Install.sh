# Install/unInstall package classes in LAMMPS

if (test $1 = 1) then

  cp -p fix_gfc.cpp ..
  cp -p fix_gfmd.cpp ..

  cp -p fix_gfc.h ..
  cp -p fix_gfmd.h ..

elif (test $1 = 0) then

  rm ../fix_gfc.cpp
  rm ../fix_gfmd.cpp

  rm ../fix_gfc.h
  rm ../fix_gfmd.h

fi
