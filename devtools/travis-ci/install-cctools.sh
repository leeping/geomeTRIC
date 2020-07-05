#!/bin/bash

#=======================================#
#| Step 1 : Before downloading package |#
#=======================================#

#----
# Provide install prefix for cctools as well as
# locations of Swig and Python packages (i.e. the
# executable itself is inside the bin subdirectory).
#
# This is to ensure that we can call the correct
# versions of Python and Swig since the version
# installed for the OS might be too old.
#----
if [[ x$(which swig) == x || x$(which python) == x ]] ; then
    echo "Installation cannot continue because python or swig executables cannot be found"
    echo "Please install these packages first."
    echo "Press Enter to continue or Ctrl-C to quit"
    read
fi

swgpath=$(dirname $(dirname $(which swig)))
pypath=$(dirname $(dirname $(which python)))

PYTHON_SITEPACKAGES=`python -c "import site; print(site.getsitepackages()[0])"`
if [[ "$PYTHON_SITEPACKAGES" != "$HOME"* ]]; then
    echo "Python sitepackages appears to be $PYTHON_SITEPACKAGES which does not contain $HOME"
    echo "Please make sure your Python site_packages folder is in your home folder and try again."
    echo "If you are using the system Python, consider installing Python into your home folder."
    echo "Press Enter to continue or Ctrl-C to quit"
    read
fi

if [ ! -f /usr/include/zlib.h ]; then
    echo "Warning: /usr/include/zlib.h not found, compilation may fail."
    echo "  You may install by running sudo apt-get install zlib1g-dev on Ubuntu Linux"
    echo "  or a corresponding command on your operating system."
    echo "Press Enter to continue or Ctrl-C to quit"
    read
fi


#=======================================#
#| Step 2 : Download package, extract, |#
#|          modify some source codes   |#
#=======================================#

# Delete existing Python module.
echo "== Deleting existing Python module =="
rm -fv $PYTHON_SITEPACKAGES/work_queue* $PYTHON_SITEPACKAGES/_work_queue*

# Create the prefix folder and remove previous "current" symlink.
prefix=$HOME/opt/cctools
mkdir -p $prefix
echo "== Deleting existing $HOME/opt/cctools/current symlink =="
rm -fv $prefix/current

# Download latest version from website.
echo "== Downloading source. =="
version="7.0.15"

# Delete existing archives and extracted folders.
echo "== Deleting existing source code archive and extracted folders =="
rm -rfv cctools-${version} v${version}.tar.gz*

wget https://github.com/lpwgroup/cctools/archive/v${version}.tar.gz
echo "== Extracting archive. =="
tar xzf v${version}.tar.gz
cd cctools-${version}

# sed -i on Mac OS works differently
if [[ "$OSTYPE" == "darwin"* ]] ; then 
    sedsuffix="'.original'"
fi

# Increase all sorts of timeouts.
sed -i$sedsuffix s/"timeout = 5;"/"timeout = 7200;"/g work_queue/src/*.c
sed -i$sedsuffix s/"timeout = 10;"/"timeout = 7200;"/g work_queue/src/*.c
sed -i$sedsuffix s/"timeout = 15;"/"timeout = 7200;"/g work_queue/src/*.c
sed -i$sedsuffix s/"foreman_transfer_timeout = 3600"/"foreman_transfer_timeout = 86400"/g work_queue/src/work_queue.c
sed -i$sedsuffix s/"long_timeout = 3600"/"long_timeout = 86400"/g work_queue/src/work_queue.c

# Disable perl
sed -i$sedsuffix s/"config_perl_path=auto"/"config_perl_path=no"/g configure

# Disable globus
sed -i$sedsuffix s/"config_globus_path=auto"/"config_globus_path=no"/g configure

#=======================================#
#| Step 3 : Build and install library  |#
#=======================================#

# Configure script depends on the Python version.
PYTHON_VERSION=`python -c 'import sys; print(sys.version_info[0])'`
if [ "$PYTHON_VERSION" -eq "3" ]
then
    echo "== Configuring for Python 3 =="
    ./configure --prefix $prefix/$version --with-python3-path $pypath --with-python-path no --with-swig-path $swgpath --with-perl-path no --with-globus-path no
else
    echo "== Configuring for Python 2 =="
    ./configure --prefix $prefix/$version --with-python-path $pypath --with-swig-path $swgpath
fi
echo "== Compilation and installation =="
make && make install && cd work_queue && make install
echo "== Work Queue library installed =="

#----
# Make symbolic link from installed version, i.e. $HOME/opt/cctools/<version> to $HOME/opt/cctools/current folder.
# This allows you to add $HOME/opt/cctools/current/bin to your PATH.
#----
cd $prefix/
ln -s $version current

#=======================================#
#| Step 4 : Install Python module      |#
#=======================================#

PNAME=$(basename $(dirname $PYTHON_SITEPACKAGES))

echo "== Installing Python module =="
cp -r $prefix/$version/lib/$PNAME/site-packages/* $PYTHON_SITEPACKAGES

echo "== Python module installed into $PYTHON_SITEPACKAGES: =="
ls -ltr $PYTHON_SITEPACKAGES/*work_queue*
